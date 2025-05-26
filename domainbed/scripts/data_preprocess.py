import os
import torch
import argparse
import pickle
import random
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from torch_geometric.data import Data
from torch_geometric.utils import degree
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset import LinkPropPredDataset
from domainbed.scripts.utils import is_directed, generate_node_times, bucketize_and_concat, add_degree_features, add_time_features
import pdb

class DataSplitter:
    def __init__(self, args, data):
        self.data = data
        self.dataname = args.dataset
        self.end_time = data.end_time
        self.start_time = data.start_time
        self.test_negative_sampling_ratio = getattr(args, "test_negative_sampling_ratio", None)
        self.save_path = args.save_path

    def _split_edges(self):
        edge_index = self.data.edge_index
        if self.dataname in [
            "tgbl-wiki",
            "tgbl-review",
            "tgbl-coin",
            "tgbl-flight",
            "tgbl-comment",
            "ogbl-collab",
        ]:
            edge_times = self.data.time.squeeze()
            test_mask = (edge_times <= self.end_time) & (edge_times >= self.start_time)
            train_mask = edge_times < self.start_time
            test_edges = edge_index[:, test_mask]
        return train_mask, test_edges

    def _split_nodes(self):
        if self.dataname in ['ogbn-arxiv']:
            node_times = self.data.node_time.squeeze()
            test_mask = (node_times <= self.end_time) & (node_times >= self.start_time)
            edge_times = self.data.time.squeeze()
            train_edge_mask = edge_times < self.start_time
            train_mask = node_times < self.start_time
        return train_mask, test_mask, train_edge_mask

    def _negative_sampling(self, negative_sampling_ratio, edge_index, test_edges):
        num_nodes = self.data.num_nodes
        num_samples = int(num_nodes * negative_sampling_ratio)

        time_mask = self.data.time.squeeze() < self.start_time
        historical_edges = edge_index[:, time_mask]
        historical_edges_dict = {i: set() for i in range(num_nodes)}

        src, tgt = historical_edges
        for s, t in zip(src.tolist(), tgt.tolist()):
            historical_edges_dict[s].add(t)
        existing_edges = set(map(tuple, edge_index.t().tolist()))

        neg_edges = []

        for source, _ in test_edges.t():
            source = source.item()
            sampled_neg_edges = []

            historical_sampled = []
            if source in historical_edges_dict:
                historical_targets = list(historical_edges_dict[source])
                historical_sampled = random.sample(
                    historical_targets, min(len(historical_targets), num_samples // 2)
                )
                sampled_neg_edges.extend([[source, tgt] for tgt in historical_sampled])

            while len(sampled_neg_edges) < num_samples:
                target = random.randint(0, num_nodes - 1)
                if (
                    target != source
                    and target not in historical_edges_dict[source]
                    and (source, target) not in existing_edges
                ):
                    sampled_neg_edges.append([source, target])

            neg_edges.extend(sampled_neg_edges[:num_samples])

        return torch.tensor(neg_edges, dtype=torch.long).t()

    def _generate_save_path(self):
        if self.dataname in [
            "tgbl-wiki",
            "tgbl-review",
            "tgbl-coin",
            "tgbl-flight",
            "tgbl-comment",
            "ogbl-collab",
        ]:
            file_name = f"start_time_{self.start_time}_end_time_{self.end_time}_neg_{self.test_negative_sampling_ratio}.pkl"
        elif self.dataname in ['ogbn-arxiv']:
            file_name = f"start_time_{self.start_time}_end_time_{self.end_time}.pkl"
        return os.path.join(self.save_path, file_name)

    def load_or_create_splits(self):
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message="You are using `torch.load` with `weights_only=False`",
        )
        if os.path.exists(self._generate_save_path()):
            with open(self._generate_save_path(), "rb") as f:
                splits = pickle.load(f)
            return splits
        if self.dataname in [
            "tgbl-wiki",
            "tgbl-review",
            "tgbl-coin",
            "tgbl-flight",
            "tgbl-comment",
            "ogbl-collab",
        ]:
            train_mask, test_edges = self._split_edges()
            test_neg_edges = self._negative_sampling(
                self.test_negative_sampling_ratio, self.data.edge_index, test_edges
            )

            splits = {
                "train_mask": train_mask,
                "test_edges": test_edges,
                "test_neg_edges": test_neg_edges,
            }

        elif self.dataname in ['ogbn-arxiv']:
            train_mask, test_mask, train_edge_mask = self._split_nodes()
            splits = {
                "train_mask": train_mask,
                "test_mask": test_mask,
                "train_edge_mask": train_edge_mask
            }
        os.makedirs(self.save_path, exist_ok=True)
        with open(self._generate_save_path(), "wb") as f:
            pickle.dump(splits, f)

        return splits


class EvolvingDataset:
    def __init__(self, args):
        self.dataset = args.dataset
        self.data_path = args.data_path
        self.edges = []
        self.edge_times = {}
        self.feature_dimension = args.input_dim
        self.time_dim = args.time_dim
        self.degree_dim = args.degree_dim
        self.load_data()

    def load_data(self):
        warnings.filterwarnings("ignore", category=FutureWarning)
        if self.dataset in [
            "tgbl-wiki",
            "tgbl-review",
            "tgbl-coin",
            "tgbl-flight",
            "tgbl-comment",
        ]:
            dataset = LinkPropPredDataset(
                name=self.dataset, root="datasets", preprocess=True
            )
            data = dataset.full_data
            ### load edges, edge_features and edge_times
            edge_features = dataset.edge_feat
            self.edges = [
                [source, target]
                for source, target in zip(data["sources"], data["destinations"])
            ]
            self.edge_times = data["timestamps"]
            ### sort edge and edge_times
            sorted_edges = sorted(
                zip(self.edge_times, self.edges, edge_features), key=lambda x: x[0]
            )
            ### set max_edges 500_000
            max_edges = 500_000
            if len(sorted_edges) > max_edges:
                sorted_edges = sorted_edges[:max_edges]
            self.edge_times, self.edges, self.edge_feature = zip(*sorted_edges)
            ### normalize edge and edge_times
            min_time = min(self.edge_times)
            self.edge_times = [t - min_time for t in self.edge_times]
        elif self.dataset in ["ogbn-arxiv"]:
            from ogb.nodeproppred import PygNodePropPredDataset

            dataset = PygNodePropPredDataset(name=self.dataset)
            graph = dataset[0]
            ### load node features
            self.node_features = graph.x
            self.node_label = graph.y
            ### load edges and edge_times
            edge_index = graph.edge_index
            edges = edge_index.t().tolist()
            self.node_times = graph.node_year.squeeze().tolist()
            ### normalize node_times
            min_time = min(self.node_times)
            self.node_times = [t - min_time for t in self.node_times]
            edge_times = [
                max(self.node_times[source], self.node_times[target]) for source, target in edges
            ]
            ### sort edge and edge_times
            sorted_edges = sorted(zip(edge_times, edges), key=lambda x: x[0])
            ### set max_edges 500_000
            max_edges = 500_000
            if len(sorted_edges) > max_edges:
                sorted_edges = sorted_edges[:max_edges]
            self.edge_times, self.edges = zip(*sorted_edges)
        elif self.dataset in ["ogbl-collab"]:
            from ogb.linkproppred import PygLinkPropPredDataset

            dataset = PygLinkPropPredDataset(name="ogbl-collab")
            graph = dataset[0]
            ### load node features
            self.node_features = graph.x
            ### load edges, edge_features and edge_times
            edge_index = graph.edge_index
            edges = edge_index.t().tolist()
            edge_times = graph.edge_year.squeeze().tolist()
            edge_features = graph.edge_weight
            ### sort edge and edge_times
            sorted_edges = sorted(
                zip(edge_times, edges, edge_features), key=lambda x: x[0]
            )
            ### set max_edges 500_000
            max_edges = 500_000
            if len(sorted_edges) > max_edges:
                sorted_edges = sorted_edges[:max_edges]
            self.edge_times, self.edges, self.edge_feature = zip(*sorted_edges)
            ### normalize edge and edge_times
            min_time = min(self.edge_times)
            self.edge_times = [t - min_time for t in self.edge_times]
        else:
            raise ValueError("Dataset not found: " + self.dataset)

    def build_graph(self, start_time, end_time):
        if self.dataset in [
            "tgbl-wiki",
            "tgbl-review",
            "tgbl-coin",
            "tgbl-flight",
            "tgbl-comment",
        ]:
            edges = np.array(self.edges)
            times = np.array(self.edge_times)

            mask = times <= end_time
            valid_edges = edges[mask]
            valid_times = times[mask]
            edge_features = torch.tensor(
                np.array(self.edge_feature, dtype=np.float32)[mask], dtype=torch.float32
            )
            if edge_features.dim() == 1:
                edge_features = edge_features.unsqueeze(1)
            unique_nodes, node_mapping = np.unique(valid_edges, return_inverse=True)
            num_nodes = len(unique_nodes)

            mapped_edges = node_mapping.reshape(-1, 2)

            edge_index = torch.tensor(mapped_edges.T, dtype=torch.long)
            time_features = torch.tensor(valid_times, dtype=torch.float32).unsqueeze(1)

            x = torch.randn(num_nodes, self.feature_dimension)
            # x = add_time_features(x, edge_index, time_features, num_nodes, self.time_dim)
            # x = add_degree_features(x, edge_index, num_nodes, self.degree_dim)
            # node_time_features = generate_node_times(edge_index, time_features, num_nodes).clone().detach().to(torch.float32).unsqueeze(1)
            # x = torch.cat([x, node_time_features/max(node_time_features)], dim=1)
            # x = bucketize_and_concat(x, node_time_features, self.time_dim)

            self.graph = Data(
                x=x,
                edge_index=edge_index,
                time=time_features,
                end_time=end_time,
                start_time=start_time,
                edge_features=edge_features,
            )
        elif self.dataset in [
            "ogbn-arxiv",
        ]:
            edges = np.array(self.edges)
            edge_times = np.array(self.edge_times)
            node_times = np.array(self.node_times)

            node_mask = node_times <= end_time
            valid_nodes = np.where(node_mask)[0]

            node_mapping = {node: i for i, node in enumerate(valid_nodes)}
            num_nodes = len(valid_nodes)

            mask = edge_times <= end_time
            valid_edges = edges[mask]
            valid_times = edge_times[mask]

            mapped_edges = np.array([[node_mapping[src], node_mapping[dst]] for src, dst in valid_edges])

            edge_index = torch.tensor(mapped_edges.T, dtype=torch.long)
            edge_time_features = torch.tensor(valid_times, dtype=torch.float32).unsqueeze(1)
            node_time_features = torch.tensor(node_times[valid_nodes], dtype=torch.float32).unsqueeze(1)
            x = self.node_features[valid_nodes].clone().detach().float()
            # x = add_time_features(x, edge_index, edge_time_features, num_nodes, self.time_dim)
            # x = add_degree_features(x, edge_index, num_nodes, self.degree_dim)
            # x = bucketize_and_concat(x, node_time_features, self.time_dim)
            # x = torch.cat([x, node_time_features/max(node_time_features)], dim=1)

            node_label = self.node_label[valid_nodes].clone().detach().long()

            self.graph = Data(
                x=x,
                edge_index=edge_index,
                node_time=node_time_features,
                time=edge_time_features,
                end_time=end_time,
                start_time=start_time,
                y=node_label
            )
        elif self.dataset in [
            "ogbl-collab",
        ]:
            edges = np.array(self.edges)
            times = np.array(self.edge_times)

            mask = times <= end_time
            valid_edges = edges[mask]
            valid_times = times[mask]
            # pdb.set_trace()
            edge_features = torch.tensor(np.array(self.edge_feature, dtype=np.float32)[mask], dtype=torch.float32)
            if edge_features.dim() == 1:
                edge_features = edge_features.unsqueeze(1)
            unique_nodes, node_mapping = np.unique(valid_edges, return_inverse=True)
            num_nodes = len(unique_nodes)

            mapped_edges = node_mapping.reshape(-1, 2)

            edge_index = torch.tensor(mapped_edges.T, dtype=torch.long)
            time_features = torch.tensor(valid_times, dtype=torch.float32).unsqueeze(1)
            
            x = self.node_features[unique_nodes].clone().detach().float()
            # x = add_time_features(x, edge_index, time_features, num_nodes, self.time_dim)
            # x = add_degree_features(x, edge_index, num_nodes, self.degree_dim)
            # node_time_features = generate_node_times(edge_index, time_features, num_nodes).clone().detach().to(torch.float32).unsqueeze(1)
            # x = torch.cat([x, node_time_features/max(node_time_features)], dim=1)
            # x = bucketize_and_concat(x, node_time_features, self.time_dim)

            self.graph = Data(
                x=x,
                edge_index=edge_index,
                time=time_features,
                end_time=end_time,
                start_time=start_time,
                edge_features=edge_features,
            )
        return self.graph
        # Create one-hot vectors for the node degrees
        # node_degrees = degree(edge_index[0], num_nodes=num_nodes).long()
        # max_degree = node_degrees.max().item()
        # degree_one_hot = torch.zeros((num_nodes, max_degree + 1))
        # degree_one_hot[torch.arange(num_nodes), node_degrees] = 1
        # # One-hot encode the times from 1993 to 2002
        # time_one_hot = torch.zeros((num_nodes, self.end_time - self.start_time + 1))
        # time_one_hot[torch.arange(num_nodes), time_features - self.start_time] = 1
        # Concatenate the original features, time features, and degree one-hot vectors
        # x = torch.randn(num_nodes, self.feature_dimension - time_one_hot.shape[1])
        # x = torch.cat([x, time_one_hot], dim=1)
        # x = torch.randn(num_nodes, self.feature_dimension - degree_one_hot.shape[1])
        # x = torch.cat([x, degree_one_hot], dim=1)
        # x = torch.randn(num_nodes, self.feature_dimension - degree_one_hot.shape[1] - time_one_hot.shape[1])
        # x = torch.cat([x, degree_one_hot, time_one_hot], dim=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training a Simple Model with Cached Data Splitting"
    )
    parser.add_argument(
        "--dataset", type=str, default="ogbn-arxiv", help="Dataset name"
    )
    parser.add_argument(
        "--data_path", type=str, default="../dataset/arxiv_HepTh", help="Data path"
    )
    parser.add_argument(
        "--save_path", type=str, default="split_data/ogbn-arxiv", help="split path"
    )
    parser.add_argument(
        "--input_dim", type=int, default=16, help="Number of hidden units in the model"
    )
    parser.add_argument(
        "--test_negative_sampling_ratio",
        type=float,
        default=0.01,
        help="Ratio of negative samples to positive samples",
    )

    args = parser.parse_args()
    dataset = EvolvingDataset(args)
    pdb.set_trace()
    data = dataset.build_graph(40, 47)
    
    print(data)
    print(max(data.time), min(data.time))
    print(max(data.y), min(data.y))
    print(max(data.node_time), min(data.node_time))
    print(is_directed(data.edge_index))
    splitter = DataSplitter(args, data)
    splits = splitter.load_or_create_splits()
    print(f"Train nodes shape is {sum(splits['train_mask'].tolist())}.")
    print(f"Test nodes shape is {sum(splits['test_mask'].tolist())}.")
    print(f"Train edges shape is {sum(splits['train_edge_mask'].tolist())}.")
    # print(f"Test edges shape is {splits['test_edges'].shape}.")
    # print(f"Test negative edges shape is {splits['test_neg_edges'].shape}.")
