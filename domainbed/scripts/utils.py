import os
import torch
import yaml
import argparse
import random
import numpy as np
from torch_geometric.utils import degree
import math
import torch.nn.functional as F

def add_time_features(x, edge_index, time_features, num_nodes, time_dim):
    """
    Add time-based features to the input tensor x by:
    1. Bucketizing time features into time_dim buckets.
    2. Marking nodes as 1 if they appear in an edge in a specific time bucket, otherwise 0.
    
    Parameters:
        x (torch.Tensor): Node feature tensor, shape (num_nodes, feature_dim).
        edge_index (torch.Tensor): Edge index tensor, shape (2, num_edges).
        time_features (torch.Tensor): Time features for edges, shape (num_edges, 1).
        num_nodes (int): Number of nodes.
        time_dim (int): Number of time buckets.

    Returns:
        torch.Tensor: Updated node features with time-based features appended.
    """
    # Normalize time features to [0, 1]
    time_features_norm = time_features / time_features.max()

    # Create time buckets
    bins = torch.linspace(0, 1, steps=time_dim + 1)
    bucket_indices = torch.bucketize(time_features_norm.squeeze(-1), bins) - 1

    # Initialize time-based features for nodes
    time_based_features = torch.zeros((num_nodes, time_dim), dtype=torch.float32)

    # Populate time-based features
    for edge, bucket in zip(edge_index.T, bucket_indices):
        source, target = edge[0].item(), edge[1].item()
        time_based_features[source, bucket] = 1
        time_based_features[target, bucket] = 1

    # Concatenate the new features with existing features
    x = torch.cat([x, time_based_features], dim=1)
    return x


def compute_node_degrees(edge_index, num_nodes):
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    for node in edge_index.flatten():
        degrees[node] += 1
    return degrees

def add_degree_features(x, edge_index, num_nodes, degree_dim):
    degrees = compute_node_degrees(edge_index, num_nodes)
    bins = torch.linspace(0, degrees.max() + 1, steps=degree_dim + 1)
    bucket_indices = torch.bucketize(degrees, bins) - 1
    bucket_indices = torch.clamp(bucket_indices, min=0)
    one_hot_features = torch.nn.functional.one_hot(bucket_indices, num_classes=degree_dim).float()
    x = torch.cat([x, one_hot_features], dim=1)
    return x

def generate_node_times(edge_index, time_features, num_nodes):
    """
    Generate node times based on the edge times. Each node's time is the earliest time it appears on any edge.

    Parameters:
        edge_index (torch.Tensor): Shape (2, num_edges), the source and target nodes for each edge.
        time_features (torch.Tensor): Shape (num_edges,), the time associated with each edge.
        num_nodes (int): The total number of nodes in the graph.

    Returns:
        torch.Tensor: Shape (num_nodes,), the earliest time each node appears.
    """
    # Initialize node times with infinity
    node_times = torch.full((num_nodes,), float('-inf'))

    # Update node times based on edge times
    for i in range(edge_index.shape[1]):
        source, target = edge_index[0, i], edge_index[1, i]
        time = time_features[i].item()
        node_times[source] = max(node_times[source], time)
        node_times[target] = max(node_times[target], time)

    # Replace infinity with 0 for nodes that have no edges (optional)
    node_times[node_times == float('inf')] = 0
    return node_times

def bucketize_and_concat(x, node_time_features, d):
    node_time_features_norm = node_time_features / node_time_features.max()
    bins = torch.linspace(0, 1, steps=d + 1)
    
    # Clamp normalized features to ensure they are within [0, 1]
    node_time_features_norm = torch.clamp(node_time_features_norm, min=0.0, max=1.0)
    
    bucket_indices = torch.bucketize(node_time_features_norm.squeeze(-1), bins) - 1
    
    # Ensure bucket indices are non-negative
    bucket_indices = torch.clamp(bucket_indices, min=0)
    
    one_hot_features = torch.nn.functional.one_hot(bucket_indices, num_classes=d).float()
    x = torch.cat([x, one_hot_features], dim=1)
    return x


def generate_one_hot_degree_vectors(data):
    """
    Generates one-hot vectors for node degrees in a graph.

    Parameters:
        data (torch_geometric.data.Data): The input graph data in PyG format.

    Returns:
        torch.Tensor: A matrix where each row is a one-hot vector representing the degree of a node.
    """
    # Compute the degree of each node
    node_degrees = degree(data.edge_index[0], num_nodes=data.num_nodes).long()

    # Create one-hot vectors for the node degrees
    max_degree = node_degrees.max().item()
    one_hot_vectors = torch.zeros((data.num_nodes, max_degree + 1))
    one_hot_vectors[torch.arange(data.num_nodes), node_degrees] = 1

    return one_hot_vectors

class TemporalDataSplitter:
    def __init__(self, args, dataset):
        self.span = args.span
        if args.dataset in [
            "tgbl-wiki",
            "tgbl-review",
            "tgbl-coin",
            "tgbl-flight",
            "tgbl-comment",
            "ogbl-collab",
        ]:
            self.start_time = min(dataset.edge_times)
            self.end_time = max(dataset.edge_times)
        elif args.dataset in ["ogbn-arxiv"]:
            self.end_time = max(dataset.edge_times)
            self.start_time = 38
            
    def split_by_time(self):
        num_time = self.end_time - self.start_time + 1
        print(num_time)
        assert num_time >= self.span, "The total time span must be at least {self.span}."

        split_size = num_time // self.span
        extra = num_time % self.span

        train_time_end = self.start_time + split_size - 1
        val_time = [train_time_end + 1, train_time_end + split_size]

        test_time_list = []
        for i in range(self.span-2):
            start = val_time[1] + 1 + i * split_size
            end = start + split_size - 1
            if i == self.span-3:
                end += extra
            test_time_list.append([start, end])

        return val_time, test_time_list


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_directed(edge_index):
    edge_index_reversed = edge_index[[1, 0], :]
    directed = not torch.equal(edge_index, edge_index_reversed)
    return directed


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[: pos_edge_index.size(1)] = 1
    return link_labels

def compute_entropy(probabilities):
    """Compute entropy for a given probability distribution."""
    return -np.sum(probabilities * np.log(probabilities))


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, path="checkpoint.pth", verbose=False, loss=True):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        self.loss = loss
    def __call__(self, val_loss, model):
        if self.loss:
            if self.best_loss is None:
                self.best_loss = val_loss
                self.save_checkpoint(model)
            elif val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
                self.save_checkpoint(model)
            else:
                self.counter += 1
                # if self.verbose:
                #     print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            if self.best_loss is None:
                self.best_loss = val_loss
                self.save_checkpoint(model)
            elif val_loss > self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
                self.save_checkpoint(model)
            else:
                self.counter += 1
                # if self.verbose:
                #     print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True            

    def save_checkpoint(self, model):
        self.best_model_state = model.state_dict()
        torch.save(self.best_model_state, self.path)
        # if self.verbose:
        #     print(f"Validation loss decreased. Saving model to {self.path}")

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.path))


def split_range(start, end, num_parts):
    # 计算每部分的基本大小
    total_length = end - start + 1
    part_size = total_length // num_parts
    remainder = total_length % num_parts  # 余数

    # 切分区间
    ranges = []
    current_start = start
    for i in range(num_parts):
        current_end = current_start + part_size - 1
        if remainder > 0:  # 如果有余数，分配一个额外的元素
            current_end += 1
            remainder -= 1
        ranges.append([current_start, current_end])
        current_start = current_end + 1  # 更新起始点

    return ranges

# # 示例
# start, end = 8, 20
# num_parts = 3
# ranges = split_range(start, end, num_parts)
# print(ranges)

