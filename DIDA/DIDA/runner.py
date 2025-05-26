import os
import sys
import time
import torch
import pickle
import warnings
import numpy as np
import torch.optim as optim
from DIDA.utils.mutils import *

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from DIDA.utils.inits import prepare
from DIDA.utils.loss import EnvLoss
from DIDA.utils.util import init_logger, logger

from torch_geometric.utils import negative_sampling
from torch.nn import functional as F
from tqdm import tqdm
import pandas as pd

class DataSplitter:
    def __init__(self, data, dataname, save_path, test_negative_sampling_ratio):
        self.data = data
        self.dataname = dataname
        self.end_time = data.end_time
        self.start_time = data.start_time
        self.test_negative_sampling_ratio = test_negative_sampling_ratio
        self.save_path = save_path

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
class Runner(object):
    def __init__(self, args, model, dataset, train_time, test_time_list, save_path, test_negative_sampling_ratio, writer=None, **kwargs):
        seed_everything(args.seed)
        self.args = args
        self.data = dataset
        self.model = model
        self.writer = writer
        self.train_time = train_time
        self.test_time_list = test_time_list
        self.loss = EnvLoss(args)
        self.save_path = save_path
        self.test_negative_sampling_ratio = test_negative_sampling_ratio

    def train(self, epoch, data, train_time, test_time_list):
        args = self.args
        self.model.train()
        optimizer = self.optimizer
        conf_opt = self.conf_opt

        train_data = data.build_graph(train_time[0], train_time[1])
        splitter = DataSplitter(train_data, args.dataset, self.save_path, self.test_negative_sampling_ratio)
        splits = splitter.load_or_create_splits()
        train_mask = splits["train_mask"]
        edge_index_list = [train_data.edge_index[:,splits["train_mask"]].to(args.device), train_data.edge_index[:,splits["train_mask"]].to(args.device), splits["test_edges"].to(args.device)]
        x_list = [train_data.x.to(args.device), train_data.x.to(args.device)]
        for i, period in enumerate(test_time_list):
            test_data = data.build_graph(period[0], period[1])
            splitter = DataSplitter(test_data, args.dataset, self.save_path, self.test_negative_sampling_ratio)
            splits = splitter.load_or_create_splits()
            if i != (len(test_time_list)-1):
                edge_index_list.append(splits["test_edges"].to(args.device))
            x_list.append(test_data.x.to(args.device)) 
        embeddings, cs, ss = self.model(edge_index_list, x_list)

        device = cs[0].device
        ss = [s.detach() for s in ss]

        # test
        val_mrr = 0
        score_list = {}
        for i, period in enumerate([train_time] + test_time_list[:-3]):
            z = cs[i+1]
            test_data = data.build_graph(period[0], period[1])
            splitter = DataSplitter(test_data, args.dataset, self.save_path, self.test_negative_sampling_ratio)
            splits = splitter.load_or_create_splits()
            pos_edge = splits["test_edges"].to(args.device)
            neg_edge = splits["test_neg_edges"].to(args.device)
            mrr, top_k_accuracy, ndcg_at_k, ap = self.loss.predict(z, pos_edge, neg_edge,
                                        self.model.cs_decoder, self.test_negative_sampling_ratio)
            if i == 0:
                val_mrr = mrr
            score_list[f"test_period_{i}_mrr"] = mrr
            score_list[f"test_period_{i}_hit@k"] = top_k_accuracy
            score_list[f"test_period_{i}_ndcg@k"] = ndcg_at_k

        # train
        edge_index = []
        edge_label = []
        epoch_losses = []
        tsize = []

        z = embeddings[0]
        pos_edge_index = train_data.edge_index[:,train_mask]
        if args.dataset == 'yelp':
            neg_edge_index = bi_negative_sampling(pos_edge_index,
                                                    args.num_nodes,
                                                    args.shift)
        else:
            neg_edge_index = negative_sampling(
                pos_edge_index,
                num_neg_samples=pos_edge_index.size(1) *
                args.sampling_times)
        edge_index.append(
            torch.cat([pos_edge_index, neg_edge_index], dim=-1))
        pos_y = z.new_ones(pos_edge_index.size(1)).to(device)
        neg_y = z.new_zeros(neg_edge_index.size(1)).to(device)
        edge_label.append(torch.cat([pos_y, neg_y], dim=0))
        tsize.append(pos_edge_index.shape[1] * 2)

        edge_label = torch.cat(edge_label, dim=0)

        def cal_y(embeddings, decoder):
            preds = torch.tensor([]).to(device)
            z = embeddings[0]
            pred = decoder(z, edge_index[0])
            preds = torch.cat([preds, pred])
            return preds

        criterion = torch.nn.BCELoss()

        def cal_loss(y, label):
            return criterion(y, label)

        cy = cal_y(cs, self.model.cs_decoder)
        sy = cal_y(ss, self.model.ss_decoder)

        conf_loss = cal_loss(sy, edge_label)
        causal_loss = cal_loss(cy, edge_label)

        env_loss = torch.tensor([]).to(device)
        intervention_times = args.n_intervene
        la = args.la_intervene

        if epoch < args.warm_epoch:
            la = 0

        if intervention_times > 0 and la > 0:
            if args.intervention_mechanism == 0:
                # slower version of spatial-temporal
                for i in range(intervention_times):
                    s1 = np.random.randint(len(sy))
                    s = torch.sigmoid(sy[s1]).detach()
                    conf = s * cy
                    # conf=self.model.comb_pred(cs,)
                    env_loss = torch.cat(
                        [env_loss,
                         cal_loss(conf, edge_label).unsqueeze(0)])
                env_mean = env_loss.mean()
                env_var = torch.var(env_loss * intervention_times)
                penalty = env_mean + env_var
            elif args.intervention_mechanism == 1:
                # only spatial
                sy = torch.sigmoid(sy).detach().split(tsize)
                cy = cy.split(tsize)
                for i in range(intervention_times):
                    conf = []
                    for j, t in enumerate(tsize):
                        s1 = np.random.randint(len(sy[j]))
                        s1 = sy[j][s1]
                        conf.append(cy[j] * s1)
                    conf = torch.cat(conf, dim=0)
                    env_loss = torch.cat(
                        [env_loss,
                         cal_loss(conf, edge_label).unsqueeze(0)])
                env_mean = env_loss.mean()
                env_var = torch.var(env_loss * intervention_times)
                penalty = env_mean + env_var
            elif args.intervention_mechanism == 2:
                # only temporal 
                alle = torch.cat(edge_index, dim=-1)
                v, idxs = torch.sort(alle[0])
                c = v.bincount()
                tsize = c[c.nonzero()].flatten().tolist()

                sy = torch.sigmoid(sy[idxs]).detach().split(tsize)
                cy = cy[idxs].split(tsize)
                edge_label = edge_label[idxs].split(tsize)

                crit = torch.nn.BCELoss(reduction='none')
                elosses = []
                for j, t in tqdm(enumerate(tsize)):
                    s1 = torch.randint(len(sy[j]),
                                       (intervention_times, 1)).flatten()
                    alls = sy[j][s1].unsqueeze(-1)
                    allc = cy[j].expand(intervention_times, cy[j].shape[0])
                    conf = allc * alls
                    alle = edge_label[j].expand(intervention_times,
                                                edge_label[j].shape[0])
                    env_loss = crit(conf.flatten(), alle.flatten()).view(
                        intervention_times, sy[j].shape[0])
                    elosses.append(env_loss)
                env_loss = torch.cat(elosses, dim=-1).mean(dim=-1)
                env_mean = env_loss.mean()
                env_var = torch.var(env_loss * intervention_times)
                penalty = env_mean + env_var
            elif args.intervention_mechanism == 3:
                # faster approximate version of spatial-temporal
                select=torch.randperm(len(sy))[:intervention_times].to(sy.device)
                alls=torch.sigmoid(sy).detach()[select].unsqueeze(-1) # [I,1]
                allc=cy.expand(intervention_times,cy.shape[0]) # [I,E]
                conf=allc*alls
                alle=edge_label.expand(intervention_times,edge_label.shape[0])
                crit=torch.nn.BCELoss(reduction='none')
                env_loss=crit(conf.flatten(),alle.flatten())
                env_loss=env_loss.view(intervention_times,sy.shape[0]).mean(dim=-1)
                env_mean = env_loss.mean()
                env_var = torch.var(env_loss*intervention_times)
                penalty = env_mean+env_var
            else:
                raise NotImplementedError('intervention type not implemented')
        else:
            penalty = 0

        loss = causal_loss + la * penalty

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        average_epoch_loss = loss.item()

        return average_epoch_loss, val_mrr, score_list

    def run(self):
        args = self.args
        
        minloss = 10
        min_epoch = args.min_epoch
        max_patience = args.patience
        patience = 0
        self.optimizer = optim.Adam(
            [p for n, p in self.model.named_parameters() if 'ss' not in n],
            lr=args.lr,
            weight_decay=args.weight_decay)
        if args.learns:
            self.conf_opt = optim.Adam(
                [p for n, p in self.model.named_parameters() if 'ss' in n],
                lr=args.lr,
                weight_decay=args.weight_decay)

        max_mrr = 0

        with tqdm(range(1, args.max_epoch + 1)) as bar:
            for epoch in bar:
                t0 = time.time()
                epoch_loss, val_mrr, test_mrr_list = self.train(
                    epoch, self.data, self.train_time, self.test_time_list)

                # update the best results.
                if val_mrr > max_mrr:
                    max_mrr = val_mrr
                    results = test_mrr_list
                    patience = 0
                else:
                    patience += 1
                    if epoch > min_epoch and patience > max_patience:
                        break
                if epoch == 1 or epoch % self.args.log_interval == 0:
                    print(f"Epoch {epoch+1}/{args.max_epoch}, Training Loss: {epoch_loss}, Validation mrr: {val_mrr}")

        return results


