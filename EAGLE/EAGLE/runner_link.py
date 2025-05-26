from EAGLE.utils.mutils import *
from EAGLE.utils.inits import prepare
from EAGLE.utils.loss import EnvLoss
from EAGLE.utils.util import init_logger, logger
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch.nn import functional as F
from torch import nn
from tqdm import tqdm
import pickle
import warnings

import os
import sys
import time
import torch
import numpy as np
import torch.optim as optim
import pandas as pd
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from torch_geometric.utils import negative_sampling
from torch.nn import functional as F
from tqdm import tqdm
import pandas as pd

TEST_K = 10
TEST_BATCH_SIZE = 1024

class DataSplitter:
    def __init__(self, args, data):
        self.data = data
        self.dataset = args.dataset
        self.end_time = data.end_time
        self.start_time = data.start_time
        self.save_path = args.save_path

        ### For link prediction task
        self.test_negative_sampling_ratio = getattr(args, "test_negative_sampling_ratio", None)

        ### For synthetic dataset
        self.initial_nodes = getattr(args, "initial_nodes", None)
        self.link_probability = getattr(args, "link_probability", None)
        self.nodes_per_step = getattr(args, "nodes_per_step", None)
        self.time_step = getattr(args, "time_step", None)
        
    def _split_edges(self):
        if self.dataset in ["tgbl-wiki", "tgbl-review", "tgbl-coin", "tgbl-flight", "tgbl-comment", "ogbl-collab"]:
            edge_time = self.data.edge_time.squeeze()
            test_edge_mask = (edge_time <= self.end_time) & (edge_time >= self.start_time)
            train_edge_mask = edge_time < self.start_time
            test_edges = self.data.edge_index[:, test_edge_mask]
        return train_edge_mask, test_edges

    def _split_nodes(self):
        if self.dataset in ["ogbn-arxiv", "BA-random", 'penn', 'Amherst41', 'Cornell5', 'Johns_Hopkins55', 'Reed98']:
            node_time = self.data.node_time.squeeze()
            test_node_mask = (node_time <= self.end_time) & (node_time >= self.start_time)
            edge_time = self.data.edge_time.squeeze()
            train_edge_mask = edge_time < self.start_time
            train_node_mask = node_time < self.start_time
        return train_node_mask, test_node_mask, train_edge_mask

    def _negative_sampling(self, negative_sampling_ratio, edge_index, test_edges):
        num_nodes = self.data.num_nodes
        num_samples = int(num_nodes * negative_sampling_ratio)

        time_mask = self.data.edge_time.squeeze() < self.start_time
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
                historical_sampled = random.sample(historical_targets, min(len(historical_targets), num_samples // 2))
                sampled_neg_edges.extend([[source, tgt] for tgt in historical_sampled])

            while len(sampled_neg_edges) < num_samples:
                target = random.randint(0, num_nodes - 1)
                if (target != source and target not in historical_edges_dict[source] and (source, target) not in existing_edges):
                    sampled_neg_edges.append([source, target])
            neg_edges.extend(sampled_neg_edges[:num_samples])

        return torch.tensor(neg_edges, dtype=torch.long).t()

    def _generate_save_path(self):
        if self.dataset in ["tgbl-wiki", "tgbl-review", "tgbl-coin", "tgbl-flight", "tgbl-comment", "ogbl-collab"]:
            file_name = f"start_time_{self.start_time}_end_time_{self.end_time}_neg_{self.test_negative_sampling_ratio}.pkl"
        elif self.dataset in ["ogbn-arxiv", 'penn', 'Amherst41', 'Cornell5', 'Johns_Hopkins55', 'Reed98']:
            file_name = f"start_time_{self.start_time}_end_time_{self.end_time}.pkl"
        elif self.dataset in ["BA-random"]:
            file_name = f"start_time_{self.start_time}_end_time_{self.end_time}_node_{self.initial_nodes}_p_{self.link_probability}_newnode{self.nodes_per_step}_timestep_{self.time_step}.pkl"
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
        if self.dataset in ["tgbl-wiki", "tgbl-review", "tgbl-coin", "tgbl-flight", "tgbl-comment", "ogbl-collab"]:
            train_mask, test_edges = self._split_edges()
            test_neg_edges = self._negative_sampling(self.test_negative_sampling_ratio, self.data.edge_index, test_edges)

            splits = {
                "train_mask": train_mask,
                "test_edges": test_edges,
                "test_neg_edges": test_neg_edges,
            }

        elif self.dataset in ["ogbn-arxiv", "BA-random", 'penn', 'Amherst41', 'Cornell5', 'Johns_Hopkins55', 'Reed98']:
            train_mask, test_mask, train_edge_mask = self._split_nodes()
            splits = {
                "train_mask": train_mask,
                "test_mask": test_mask,
                "train_edge_mask": train_edge_mask,
            }
        os.makedirs(self.save_path, exist_ok=True)
        with open(self._generate_save_path(), "wb") as f:
            pickle.dump(splits, f)

        return splits

class NeibSampler:
    def __init__(self, graph, nb_size, include_self=False):
        n = graph.number_of_nodes()
        assert 0 <= min(graph.nodes()) and max(graph.nodes()) < n
        if include_self:
            nb_all = torch.zeros(n, nb_size + 1, dtype=torch.int64)
            nb_all[:, 0] = torch.arange(0, n)
            nb = nb_all[:, 1:]
        else:
            nb_all = torch.zeros(n, nb_size, dtype=torch.int64)
            nb = nb_all
        popkids = []
        for v in range(n):
            nb_v = sorted(graph.neighbors(v))
            if len(nb_v) <= nb_size:
                nb_v.extend([-1] * (nb_size - len(nb_v)))
                nb[v] = torch.LongTensor(nb_v)
            else:
                popkids.append(v)
        self.include_self = include_self
        self.g, self.nb_all, self.pk = graph, nb_all, popkids

    def to(self, dev):
        self.nb_all = self.nb_all.to(dev)
        return self

    def sample(self):
        nb = self.nb_all[:, 1:] if self.include_self else self.nb_all
        nb_size = nb.size(1)
        pk_nb = np.zeros((len(self.pk), nb_size), dtype=np.int64)
        for i, v in enumerate(self.pk):
            pk_nb[i] = np.random.choice(sorted(self.g.neighbors(v)), nb_size)
        nb[self.pk] = torch.from_numpy(pk_nb).to(nb.device)
        return self.nb_all


class Runner(object):
    def __init__(self, args, model, cvae, dataset, train_time, test_time_list, save_path, writer=None, **kwargs):
        seed_everything(args.seed)
        self.device = args.device
        self.args = args
        self.data = dataset
        self.model = model
        self.cvae = cvae
        self.writer = writer
        self.save_path = save_path
        self.train_time = train_time
        self.test_time_list = test_time_list
        self.test_negative_sampling_ratio = args.test_negative_sampling_ratio
        self.nbsz = args.nbsz
        self.n_factors = args.n_factors
        self.delta_d = args.delta_d
        self.d = self.n_factors * self.delta_d
        self.interv_size_ratio = args.interv_size_ratio
        train_data = dataset.build_graph(train_time[0], train_time[1])
        x = train_data.x.to(args.device).clone().detach()
        self.x = [x for _ in range(4)] if len(x.shape) <= 2 else x

        splitter = DataSplitter(self.args, train_data)
        splits = splitter.load_or_create_splits()
        all_edges = train_data.edge_index[:,splits["train_mask"]]
        all_time = train_data.edge_time[splits["train_mask"]]

        time_min, time_max = all_time.min().item(), all_time.max().item()
        time_bins = torch.linspace(time_min, time_max, 4)
        time_intervals = torch.bucketize(all_time, time_bins) 
        edge_index_list = []
        for i in range(4):
            mask = (time_intervals.squeeze(-1) <= i)
            edge_index_list.append(all_edges[:, mask])  # Group edges by time intervals

            self.edge_index_list_pre = [
                edge_index.long().to(self.device) for edge_index in edge_index_list
            ]

        neighbors_all = []
        for t in range(4):
            graph_data = Data(x=self.x[t], edge_index=self.edge_index_list_pre[t])
            graph = to_networkx(graph_data)
            sampler = NeibSampler(graph, self.nbsz)
            neighbors = sampler.sample().to(args.device)
            neighbors_all.append(neighbors)
        self.neighbors_all = torch.stack(neighbors_all).to(args.device)

        self.loss = EnvLoss(args)

    def cal_fact_rank(self, x_all):
        times = len(x_all)
        n = len(x_all[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        x_all = x_all.view(times, n, k, delta_d)
        x_all_trans = x_all.permute(1, 2, 0, 3)
        points = torch.var(x_all_trans, dim=[2, 3])
        rank = torch.argsort(points, 0, descending=True)
        return rank

    def cal_fact_var(self, x_all):
        times = len(x_all)
        n = len(x_all[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        x_all_trans = x_all.permute(1, 2, 0, 3)
        points = torch.var(x_all_trans, dim=[2, 3]).view(n, k)
        return points

    def intervention(self, x_all):
        times = len(x_all)
        n = len(x_all[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        m = int(self.interv_size_ratio * n)
        indices = torch.randperm(n)[:m]
        x_m = x_all[:, indices, :].view(times, m, k, delta_d)
        mask = self.cal_mask(x_m)
        mask_expand = torch.repeat_interleave(mask, delta_d, dim=1).view(m, k * delta_d)
        mask_expand = torch.stack([mask_expand] * times)

        saved_env = self.saved_env(x_m, mask)
        sampled_env = self.gen_env(saved_env)

        x_m = x_m.view(times, m, k * delta_d)
        embeddings_interv = x_m * mask_expand + sampled_env * (1 - mask_expand)
        x_all[:, indices, :] = embeddings_interv.to(torch.float32)
        return x_all, indices

    def intervention_faster(self, x_all):
        times = len(x_all)
        n = len(x_all[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        x_all = x_all.view(times, n, k, delta_d)
        mask = self.cal_mask_faster(x_all)
        mask_expand = torch.repeat_interleave(mask, delta_d, dim=1).view(n, k * delta_d)
        mask_expand = torch.stack([mask_expand] * times)

        saved_env = self.saved_env(x_all, mask)
        sampled_env = self.gen_env(saved_env)

        x_all = x_all.view(times, n, k * delta_d)
        embeddings_interv = x_all * mask_expand + sampled_env * (1 - mask_expand)
        embeddings_interv = embeddings_interv.to(torch.float32)
        return embeddings_interv

    def intervention_final(self, x_all_original):
        x_all = x_all_original.clone()
        times = len(x_all)
        n = len(x_all[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        m = int(self.interv_size_ratio * n)
        indices = torch.randperm(n, device=x_all.device)[:m]

        x_m = x_all[:, indices, :].view(times, m, k, delta_d)
        mask = self.cal_mask_faster(x_m)
        mask_expand = torch.repeat_interleave(mask, delta_d, dim=1).view(m, k * delta_d)
        mask_expand = torch.stack([mask_expand] * times)

        saved_env = self.saved_env(x_m, mask)
        sampled_env = self.gen_env(saved_env)

        x_m = x_m.view(times, m, k * delta_d)
        embeddings_interv = x_m * mask_expand + sampled_env * (1 - mask_expand)
        x_all[:, indices, :] = embeddings_interv.to(torch.float32)
        return x_all, indices

    def cal_mask(self, x_m):
        times = len(x_m)
        m = len(x_m[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        var = self.cal_fact_var(x_m).cpu().detach().numpy()

        def split_array(arr):
            n = len(arr)
            total_sum = sum(arr)
            dp = [[False for _ in range(total_sum + 1)] for __ in range(n + 1)]
            dp[0][0] = True
            for i in range(1, n + 1):
                for j in range(total_sum + 1):
                    if j >= arr[i - 1]:
                        dp[i][j] = dp[i - 1][j] or dp[i - 1][j - arr[i - 1]]
                    else:
                        dp[i][j] = dp[i - 1][j]
            min_diff = float("inf")
            for j in range(total_sum // 2, -1, -1):
                if dp[n][j]:
                    min_diff = total_sum - 2 * j
                    break
            return min_diff

        def process_matrix(matrix):
            matrix = adjust_matrix(matrix)
            n, k = matrix.shape
            result = np.zeros((n, k))
            for i in range(n):
                row = matrix[i]
                min_diff = split_array(row)
                avg = sum(row) / k
                for j in range(k):
                    if row[j] <= avg - min_diff / 2:
                        result[i][j] = 1
                    else:
                        result[i][j] = 0
                if np.sum(result[i]) == 0:
                    index = np.argmin(result[i])
                    result[i][index] = 1
            return result

        def adjust_matrix(matrix):
            for row in matrix:
                while np.min(row) < 1:
                    row *= 10
            matrix_min = matrix.min(axis=1).astype(int)
            matrix_min = np.expand_dims(matrix_min, axis=1)
            matrix_min = np.tile(matrix_min, (1, len(matrix[1])))
            matrix = matrix - matrix_min

            for row in matrix:
                while np.min(row) < 1:
                    row *= 10

            return matrix.astype(int)

        mask = process_matrix(var)
        return torch.from_numpy(mask).to(self.args.device)

    def cal_mask_faster(self, x_m):
        times = len(x_m)
        m = len(x_m[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        var = self.cal_fact_var(x_m)

        def max_avg_diff_index(sorted_tensor):
            n, k = sorted_tensor.shape
            result = np.zeros(n)
            for i in range(n):
                row = sorted_tensor[i]
                max_diff = 0
                max_index = 0
                for j in range(1, k):
                    avg1 = sum(row[:j]) / j
                    avg2 = sum(row[j:]) / (k - j)
                    diff = abs(avg1 - avg2)
                    if diff <= max_diff:
                        break
                    if diff > max_diff:
                        max_diff = diff
                        max_index = j
                result[i] = max_index - 1
            return result

        var_sorted = torch.sort(var, dim=1)
        var_sorted_index = var_sorted.indices
        indices = max_avg_diff_index(var_sorted.values).astype(int)
        for i in range(var.shape[0]):
            sort_indices = var_sorted_index[i]
            values = var[i, sort_indices]
            mask = torch.zeros_like(values)
            mask[: indices[i] + 1] = 1
            var[i, sort_indices] = mask

        return var

    def saved_env(self, x_all, mask):
        times = len(x_all)
        n = len(x_all[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        m = mask.shape[0]
        x_all = x_all.view(times, n, d)
        mask_env = 1 - mask
        mask_expand = torch.repeat_interleave(mask_env, delta_d, dim=1).view(
            m, k * delta_d
        )
        mask_expand = torch.stack([mask_expand] * times)

        extract_env = (
            (x_all * mask_expand).view(times, n, k, delta_d).permute(2, 0, 1, 3)
        )
        extract_env = extract_env.view(k, times * n, delta_d)
        extract_env = extract_env[:, torch.randperm(times * n), :]
        for i in range(k):
            zero_rows = (extract_env[i].sum(dim=1) == 0).nonzero(as_tuple=True)[0]
            non_zero_rows = (extract_env[i].sum(dim=1) != 0).nonzero(as_tuple=True)[0]
            if len(non_zero_rows) > 0:
                replacement_rows = non_zero_rows[
                    torch.randint(0, len(non_zero_rows), (len(zero_rows),))
                ]
                extract_env[i][zero_rows] = extract_env[i][replacement_rows]

        return extract_env.view(times, n, d)

    def gen_env(self, extract_env):
        times = len(extract_env)
        n = len(extract_env[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        n_gen = int(self.args.gen_ratio * n)

        z = torch.randn(n_gen * k, self.args.d_for_cvae).to(self.args.device)
        y = torch.ones(n_gen, k)
        for i in range(k):
            y[:, i : i + 1] = y[:, i : i + 1] * i
        y_T = y.transpose(0, 1)
        y = (F.one_hot(y_T.reshape(-1).to(torch.int64))).to(self.args.device)
        gen_env = self.cvae.decode(z, y).view(n_gen, k * delta_d)

        random_indices = torch.randperm(n)[:n_gen]
        extract_env[:, random_indices] = gen_env
        return extract_env.view(times, n, d)

    def loss_cvae(self, recon, x, mu, log_std) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, x, reduction="sum")
        kl_loss = -0.5 * (1 + 2 * log_std - mu.pow(2) - torch.exp(2 * log_std))
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss + kl_loss
        return loss

    def train(self, epoch, data, train_time, test_time_list):
        args = self.args
        self.model.train()
        optimizer = self.optimizer

        embeddings = self.model(
            [
                self.edge_index_list_pre[ix].long().to(args.device)
                    for ix in range(4)
            ],
            self.x,
            self.neighbors_all,
        )

        y = torch.ones(len(embeddings[0]) * len(embeddings), args.n_factors)
        for i in range(args.n_factors):
            y[:, i : i + 1] = y[:, i : i + 1] * i
        y_T = y.transpose(0, 1)
        y = (F.one_hot(y_T.reshape(-1).to(torch.int64))).to(args.device)

        embeddings_view = embeddings.view(
            len(embeddings), len(embeddings[0]), args.n_factors, args.delta_d
        )
        embeddings_trans = embeddings_view.permute(2, 0, 1, 3)
        x_flatten = torch.flatten(embeddings_trans, start_dim=0, end_dim=2)
        recon, mu, log_std = self.cvae(x_flatten, y)
        cvae_loss = self.loss_cvae(recon, x_flatten, mu, log_std) / (
            len(embeddings[0] * len(embeddings[0]))
        )

        device = embeddings[0].device

        ### validation
        z = embeddings[-1]
        train_data = data.build_graph(train_time[0], train_time[1])
        splitter = DataSplitter(self.args, train_data)
        splits = splitter.load_or_create_splits()
        pos_edge = splits["test_edges"].to(args.device)
        neg_edge = splits["test_neg_edges"].to(args.device)
        val_score, _ = self.loss.predict_link(z, pos_edge, neg_edge,
                                                self.model.edge_decoder, self.test_negative_sampling_ratio)
                    

        edge_index = []
        pos_edge_index_all = []
        neg_edge_index_all = []
        edge_label = []
        tsize = []
        for t in range(3):
            z = embeddings[t]
            pos_edge_index = self.edge_index_list_pre[t]
            if args.dataset == "yelp":
                neg_edge_index = bi_negative_sampling(
                    pos_edge_index, args.num_nodes, args.shift
                )
            else:
                neg_edge_index = negative_sampling(
            edge_index=self.edge_index_list_pre[t],
            num_nodes=train_data.num_nodes,
            num_neg_samples=int(
                self.edge_index_list_pre[t].size(1) * args.sampling_times
            ),
        ).to(args.device)
            edge_index.append(torch.cat([pos_edge_index, neg_edge_index], dim=-1))
            pos_edge_index_all.append(pos_edge_index)
            neg_edge_index_all.append(neg_edge_index)
            pos_y = z.new_ones(pos_edge_index.size(1)).to(device)
            neg_y = z.new_zeros(neg_edge_index.size(1)).to(device)
            edge_label.append(torch.cat([pos_y, neg_y], dim=0))
            tsize.append(pos_edge_index.shape[1] * 2)

        edge_label = torch.cat(edge_label, dim=0)

        def generate_edge_index_interv(pos_edge_index_all, neg_edge_index_all, indices):
            edge_label = []
            edge_index = []
            pos_edge_index_interv = pos_edge_index_all.copy()
            neg_edge_index_interv = neg_edge_index_all.copy()
            index = indices.cpu().numpy()
            for t in range(3):
                mask_pos = np.logical_and(
                    np.isin(pos_edge_index_interv[t].cpu()[0], index),
                    np.isin(pos_edge_index_interv[t].cpu()[1], index),
                )
                pos_edge_index = pos_edge_index_interv[t][:, mask_pos]
                mask_neg = np.logical_and(
                    np.isin(neg_edge_index_interv[t].cpu()[0], index),
                    np.isin(neg_edge_index_interv[t].cpu()[1], index),
                )
                neg_edge_index = neg_edge_index_interv[t][:, mask_neg]
                pos_y = z.new_ones(pos_edge_index.size(1)).to(device)
                neg_y = z.new_zeros(neg_edge_index.size(1)).to(device)
                edge_label.append(torch.cat([pos_y, neg_y], dim=0))
                edge_index.append(torch.cat([pos_edge_index, neg_edge_index], dim=-1))
            edge_label = torch.cat(edge_label, dim=0)
            return edge_label, edge_index

        def cal_y(embeddings, decoder):
            preds = torch.tensor([]).to(device)
            for t in range(3):
                z = embeddings[t]
                pred = decoder(z, edge_index[t])
                preds = torch.cat([preds, pred])
            return preds

        def cal_y_interv(embeddings, decoder, edge_index_interv, indices):
            index = indices.cpu().numpy()
            preds = torch.tensor([]).to(device)
            for t in range(3):
                z = embeddings[t]
                pred = decoder(z, edge_index_interv[t])
                preds = torch.cat([preds, pred])
            return preds

        criterion = torch.nn.BCELoss()
        criterion_var = torch.nn.MSELoss()

        def cal_loss(y, label):
            return criterion(y, label)

        def cal_loss_var(y, label):
            return criterion_var(y, label)

        pred_y = cal_y(embeddings, self.model.edge_decoder)
        main_loss = cal_loss(pred_y, edge_label)

        intervention_times = args.n_intervene
        env_loss = torch.tensor([]).to(device)
        for i in range(intervention_times):
            
            embeddings_interv, indices = self.intervention_final(embeddings)
            edge_label_interv, edge_index_interv = generate_edge_index_interv(
                pos_edge_index_all, neg_edge_index_all, indices
            )

            pred_y_interv = cal_y_interv(
                embeddings_interv, self.model.edge_decoder, edge_index_interv, indices
            )
            env_loss = torch.cat(
                [env_loss, cal_loss(pred_y_interv, edge_label_interv).unsqueeze(0)]
            )

        var_loss = torch.var(env_loss)

        alpha = args.alpha
        beta = args.beta

        if epoch % args.every_epoch == 0:
            loss = main_loss + alpha * var_loss + beta * cvae_loss
        else:
            loss = main_loss + alpha * var_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        average_epoch_loss = loss.item()

        return average_epoch_loss, val_score

    def run(self):
        args = self.args
        min_epoch = args.min_epoch
        max_patience = args.patience
        patience = 0

        self.optimizer = optim.SGD(
            [p for n, p in self.model.named_parameters() if "ss" not in n]
            + [p for n, p in self.cvae.named_parameters() if "lr" not in n],
            weight_decay=args.weight_decay,
        )
        max_score = 0

        with tqdm(range(1, args.max_epoch + 1)) as bar:
            for epoch in bar:
                t0 = time.time()
                epoch_losses, var_score,  = self.train(
                    epoch, self.data, self.train_time, self.test_time_list)

                if var_score > max_score:
                    max_score = var_score
                    test_results = self.test(epoch, self.data, [self.train_time]+self.test_time_list)
                    self.model = self.model.to(args.device)
                    patience = 0
                    filepath = "./checkpoint/" + self.args.dataset + ".pth"
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "cvae_state_dict": self.cvae.state_dict(),
                        },
                        filepath,
                    )

                else:
                    patience += 1
                    if epoch > min_epoch and patience > max_patience:
                        break
                if epoch == 1 or epoch % self.args.log_interval == 0:
                    print(f"Epoch {epoch+1}/{args.max_epoch}, Training Loss: {epoch_losses}, Validation score: {var_score}")

        return test_results

    def test(self, epoch, data, time_list):
        args = self.args
        score_list = {}
        self.model.eval()
        self.model = self.model.to('cpu')
        for i, period in enumerate(time_list):
            test_data = data.build_graph(period[0], period[1])
            x = test_data.x.to('cpu')
            x_list = [x for _ in range(4 + i)] if len(x.shape) <= 2 else x

            splitter = DataSplitter(self.args, test_data)
            splits = splitter.load_or_create_splits()
            all_edges = test_data.edge_index[:,splits["train_mask"]]
            all_time = test_data.edge_time[splits["train_mask"]]

            time_min, time_max = all_time.min().item(), all_time.max().item()
            time_bins = torch.linspace(time_min, time_max, 4 + i)
            time_intervals = torch.bucketize(all_time, time_bins) 
            
            edge_index_list = []
            for j in range(4 + i):
                mask = (time_intervals.squeeze(-1) <= j)
                edge_index_list.append(all_edges[:, mask])  # Group edges by time intervals

            edge_index_list_pre = [
                edge_index.long().to('cpu') for edge_index in edge_index_list
            ]

            neighbors_all = []
            for t in range(4 + i):
                graph_data = Data(x=x_list[t], edge_index=edge_index_list_pre[t])
                graph = to_networkx(graph_data)
                sampler = NeibSampler(graph, self.nbsz)
                neighbors = sampler.sample()
                neighbors_all.append(neighbors)
            neighbors_all = torch.stack(neighbors_all)
            embeddings = self.model(
                [
                    edge_index_list_pre[ix].long()
                    for ix in range(4 + i)
                ],
                x_list,
                neighbors_all.to('cpu'),
            )
            z = embeddings[-1].to('cpu')
            pos_edge = splits["test_edges"].to('cpu')
            neg_edge = splits["test_neg_edges"].to('cpu')

            pos_scores = []
            neg_scores = []
            num_nodes = z.size(0)
            num_neg_samples = int(num_nodes * self.test_negative_sampling_ratio)
            
            # print(next(self.model.parameters()).device)
            decoder = self.model.edge_decoder
            for start in range(0, pos_edge.size(1), TEST_BATCH_SIZE):
                end = start + TEST_BATCH_SIZE
                pos_scores.append(decoder(z, pos_edge[:, start:end]))

            for start in range(0, neg_edge.size(1), TEST_BATCH_SIZE * num_neg_samples):
                end = start + TEST_BATCH_SIZE * num_neg_samples
                neg_scores.append(decoder(z, neg_edge[:, start:end]))

            pos_scores_cpu = [score.to('cpu') for score in pos_scores]
            neg_scores_cpu = [score.to('cpu') for score in neg_scores]
            positive_scores = torch.cat(pos_scores_cpu).view(-1, 1).to('cpu')
            negative_scores = torch.cat(neg_scores_cpu).view(-1, num_neg_samples).to('cpu')
            scores = torch.cat([positive_scores, negative_scores], dim=1).to('cpu')
            ranks = torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1)

            reciprocal_ranks = 1.0 / (ranks[:, 0] + 1).float()
            mrr = reciprocal_ranks.mean().item()
            score_list[f"test_period_{i}_mrr"] = mrr

            top_k_hits = (ranks[:, :TEST_K] == 0).float().sum(dim=1)  
            top_k_accuracy = top_k_hits.mean().item()
            score_list[f"test_period_{i}_hit@k"] = top_k_accuracy
            scores_at_k = scores[:, :TEST_K]
            dcg_at_k = torch.sum((2.0 ** scores_at_k - 1) / torch.log2(ranks[:, :TEST_K].float() + 2), dim=1)
            sorted_scores = torch.sort(scores, dim=1, descending=True)[0]  
            ideal_scores_at_k = sorted_scores[:, :TEST_K]  
            ideal_dcg_at_k = torch.sum((2.0 ** ideal_scores_at_k - 1) / torch.log2(torch.arange(1, TEST_K + 1, device=scores.device).float() + 1), dim=1)
            ndcg_at_k = (dcg_at_k / ideal_dcg_at_k).mean().item()
            score_list[f"test_period_{i}_ndcg@k"] = ndcg_at_k

        return score_list

