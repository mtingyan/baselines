import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from DIDA.config import args
from torch_geometric.utils import negative_sampling
from DIDA.utils.util import logger
from DIDA.utils.mutils import *

device = args.device

EPS = 1e-15
MAX_LOGVAR = 10
TEST_K = 10
TEST_BATCH_SIZE = 256

class EnvLoss(nn.Module):
    
    def __init__(self,args) -> None:
        super().__init__()
        self.args = args
        
    @staticmethod
    def maybe_num_nodes(index, num_nodes=None):
        return index.max().item() + 1 if num_nodes is None else num_nodes

    def decoder(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value
    
    def forward(self, z, pos_edge_index, neg_edge_index=None, decoder=None):
        if not decoder:
            decoder = self.decoder
        pos_loss = -torch.log(decoder(z, pos_edge_index) + EPS).mean()
        if neg_edge_index == None:
            args = self.args
            if args.dataset == 'yelp':
                neg_edge_index = bi_negative_sampling(pos_edge_index,
                                                      args.num_nodes,
                                                      args.shift)
            else:
                neg_edge_index = negative_sampling(
                    pos_edge_index,
                    num_neg_samples=pos_edge_index.size(1) *
                    self.sampling_times)
        neg_loss = -torch.log(1 - decoder(z, neg_edge_index) + EPS).mean()

        return pos_loss + neg_loss

    def predict(self, z, pos_edge_index, neg_edge_index, decoder, test_negative_sampling_ratio):
        pos_scores = []
        neg_scores = []
        num_nodes = z.size(0)
        num_neg_samples = int(num_nodes * test_negative_sampling_ratio)

        # batch_size = 256

        # for start in range(0, pos_edge_index.size(1), batch_size):
        #     end = start + batch_size
        #     pos_scores.append(decoder(z, pos_edge_index[:, start:end]).to("cpu"))

        # for start in range(0, neg_edge_index.size(1), batch_size * num_neg_samples):
        #     end = start + batch_size * num_neg_samples
        #     neg_scores.append(decoder(z, neg_edge_index[:, start:end]).to("cpu"))

        # positive_scores = torch.cat(pos_scores).view(-1, 1)
        # negative_scores = torch.cat(neg_scores).view(-1, num_neg_samples)

        # scores = torch.cat([positive_scores, negative_scores], dim=1)
        # ranks = torch.argsort(
        #     torch.argsort(scores, dim=1, descending=True), dim=1
        # )
        # reciprocal_ranks = 1.0 / (ranks[:, 0] + 1).float()
        # mrr = reciprocal_ranks.mean().item()

        # torch.cuda.empty_cache()
        for start in range(0, pos_edge_index.size(1), TEST_BATCH_SIZE):
            end = start + TEST_BATCH_SIZE
            pos_scores.append(decoder(z, pos_edge_index[:, start:end]).to("cpu"))

        for start in range(0, neg_edge_index.size(1), TEST_BATCH_SIZE * num_neg_samples):
            end = start + TEST_BATCH_SIZE * num_neg_samples
            neg_scores.append(decoder(z, neg_edge_index[:, start:end]).to("cpu"))

        positive_scores = torch.cat(pos_scores).view(-1, 1)
        negative_scores = torch.cat(neg_scores).view(-1, num_neg_samples)
        scores = torch.cat([positive_scores, negative_scores], dim=1)
        ranks = torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1)

        reciprocal_ranks = 1.0 / (ranks[:, 0] + 1).float()
        mrr = reciprocal_ranks.mean().item()

        top_k_hits = (ranks[:, :TEST_K] == 0).float().sum(dim=1)  
        top_k_accuracy = top_k_hits.mean().item()
        scores_at_k = scores[:, :TEST_K]
        dcg_at_k = torch.sum((2.0 ** scores_at_k - 1) / torch.log2(ranks[:, :TEST_K].float() + 2), dim=1)
        sorted_scores = torch.sort(scores, dim=1, descending=True)[0]  
        ideal_scores_at_k = sorted_scores[:, :TEST_K]  
        ideal_dcg_at_k = torch.sum((2.0 ** ideal_scores_at_k - 1) / torch.log2(torch.arange(1, TEST_K + 1, device=scores.device).float() + 1), dim=1)
        ndcg_at_k = (dcg_at_k / ideal_dcg_at_k).mean().item()
        torch.cuda.empty_cache()
        return mrr, top_k_accuracy, ndcg_at_k, 0
    
    def predict_label(self, z, decoder):
        return decoder(z)
