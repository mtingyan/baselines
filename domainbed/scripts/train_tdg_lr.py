# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import math

import pandas as pd
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torch_geometric.utils import negative_sampling
from domainbed.scripts.data_preprocess import EvolvingDataset, DataSplitter
from domainbed.scripts.utils import load_config, TemporalDataSplitter, set_seed, EarlyStopping, get_link_labels, split_range
from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
import pdb

TEST_K = 10
TEST_BATCH_SIZE = 1024
# @torch.no_grad()
# def evaluate(args, model, dataset, test_time, test_device):
#     model.eval()
#     # dataset = dataset.to(test_device)

#     score_list = {}
#     for i, period in enumerate(test_time):
#         print(period)
#         test_data = dataset.build_graph(period[0], period[1])
#         splitter = DataSplitter(args, test_data)
#         splits = splitter.load_or_create_splits()
#         edge_index = test_data.edge_index[:,splits["train_mask"]].to(test_device)
#         if args.dataset in [
#             "tgbl-review",
#             "tgbl-coin",
#             "tgbl-flight",
#             "tgbl-comment",
#             "ogbl-collab",
#         ]:
#             edge_features = test_data.edge_features[splits["train_mask"]].to(test_device)
#             z = model.network.encode(test_data.x.to(test_device), edge_index, edge_features)
#         elif args.dataset in ["tgbl-wiki",]:
#             z = model.network.encode(test_data.x.to(test_device), edge_index)
#         test_edge_index = splits["test_edges"].to(test_device)
#         test_neg_edge_index = splits["test_neg_edges"].to(test_device)

#         pos_scores = []
#         neg_scores = []
#         num_nodes=test_data.num_nodes
#         num_neg_samples = int(num_nodes * args.test_negative_sampling_ratio)

#         batch_size = 512
#         for start in range(0, test_edge_index.size(1), batch_size):
#             end = start + batch_size
#             pos_scores.append(model.network.decode(z, test_edge_index[:, start:end]))
#         for start in range(
#             0, test_neg_edge_index.size(1), batch_size * num_neg_samples
#         ):
#             end = start + batch_size * num_neg_samples
#             neg_scores.append(model.network.decode(z, test_neg_edge_index[:, start:end]))

#         positive_scores = torch.cat(pos_scores).view(-1, 1)
#         negative_scores = torch.cat(neg_scores).view(-1, num_neg_samples)

#         scores = torch.cat([positive_scores, negative_scores], dim=1)
#         ranks = torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1)
#         reciprocal_ranks = 1.0 / (ranks[:, 0] + 1).float()

#         mrr = reciprocal_ranks.mean().item()
#         score_list[f"test_period_{i}_mrr"] = mrr

#         # torch.cuda.empty_cache()

#     return score_list

@torch.no_grad()
def evaluate(args, model, dataset, test_time, test_device):
    model.eval()
    # dataset = dataset.to(test_device)

    score_list = {}
    for i, period in enumerate(test_time):
        print(period)
        test_data = dataset.build_graph(period[0], period[1])
        splitter = DataSplitter(args, test_data)
        splits = splitter.load_or_create_splits()
        edge_index = test_data.edge_index[:,splits["train_mask"]].to(test_device)
        if args.dataset in [
            "tgbl-review",
            "tgbl-coin",
            "tgbl-flight",
            "tgbl-comment",
            "ogbl-collab",
        ]:
            edge_features = test_data.edge_features[splits["train_mask"]].to(test_device)
            z = model.network.encode(test_data.x.to(test_device), edge_index, edge_features)
        elif args.dataset in ["tgbl-wiki",]:
            z = model.network.encode(test_data.x.to(test_device), edge_index)
        test_edge_index = splits["test_edges"].to(test_device)
        test_neg_edge_index = splits["test_neg_edges"].to(test_device)

        pos_scores = []
        neg_scores = []
        num_nodes=test_data.num_nodes
        num_neg_samples = int(num_nodes * args.test_negative_sampling_ratio)

        for start in range(0, test_edge_index.size(1), TEST_BATCH_SIZE):
            end = start + TEST_BATCH_SIZE
            pos_scores.append(model.network.decode(z, test_edge_index[:, start:end]))

        for start in range(0, test_neg_edge_index.size(1), TEST_BATCH_SIZE * num_neg_samples):
            end = start + TEST_BATCH_SIZE * num_neg_samples
            neg_scores.append(model.network.decode(z, test_neg_edge_index[:, start:end]))

        positive_scores = torch.cat(pos_scores).view(-1, 1)
        negative_scores = torch.cat(neg_scores).view(-1, num_neg_samples)
        scores = torch.cat([positive_scores, negative_scores], dim=1)
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
        torch.cuda.empty_cache()

        # torch.cuda.empty_cache()

    return score_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default="./domainbed/data/MNIST/")
    parser.add_argument('--datasets', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=[42, 123, 456, 66],
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0, help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--num_envs", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--model_save_path", type=str, default='/home/tingyan/TDG/domainbed/results', help="Path to the config file")
    parser.add_argument("--device", type=str, default='cuda', help="Path to the config file")
       
    args = parser.parse_args()
    config = load_config(args.config)
    args.datasets = "RotatedMNIST"
    # pdb.set_trace()
    result_list = []
    for seed in args.seed:
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
        # pdb.set_trace()
        test_save_path = args.save_path
        train_spilt_path = 'domainbed/data/'+ args.dataset +'/'
        args.save_path = train_spilt_path
        # If we ever want to implement checkpointing, just persist these values
        # every once in a while, and then load them from disk here.
        start_step = 0
        algorithm_dict = None

        os.makedirs(args.output_dir, exist_ok=True)
        sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
        sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

        print("Environment:")
        print("\tPython: {}".format(sys.version.split(" ")[0]))
        print("\tPyTorch: {}".format(torch.__version__))
        print("\tTorchvision: {}".format(torchvision.__version__))
        print("\tCUDA: {}".format(torch.version.cuda))
        print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
        print("\tNumPy: {}".format(np.__version__))
        print("\tPIL: {}".format(PIL.__version__))

        print('Args:')
        for k, v in sorted(vars(args).items()):
            print('\t{}: {}'.format(k, v))

        if args.hparams_seed == 0:
            hparams = hparams_registry.default_hparams(args.algorithm, args.datasets)
        else:
            hparams = hparams_registry.random_hparams(args.algorithm, args.datasets,
                misc.seed_hash(args.hparams_seed, args.trial_seed))
        if args.hparams:
            hparams.update(json.loads(args.hparams))
        hparams['hidden_dim'] = args.hidden_dim
        hparams['output_dim'] = args.output_dim
        print('HParams:')
        for k, v in sorted(hparams.items()):
            print('\t{}: {}'.format(k, v))

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        # pdb.set_trace() 
        # 这里是把dataset切成了k个环境，每个环境有自己的数据集
        if args.datasets in vars(datasets):
            dataset = vars(datasets)[args.datasets](args.data_dir,
                args.test_envs, hparams)
        else:
            raise NotImplementedError

        in_splits = []
        out_splits = []
        uda_splits = []
        tg_dataset = EvolvingDataset(args)
        train_time, test_time_list = TemporalDataSplitter(
            args, tg_dataset
        ).split_by_time()
        # pdb.set_trace()
        time_range_list = split_range(train_time[0], train_time[1], 3)

        train_env1 = tg_dataset.build_graph(time_range_list[0][0], time_range_list[0][1])
        train_env2 = tg_dataset.build_graph(time_range_list[1][0], time_range_list[1][1])
        train_env3 = tg_dataset.build_graph(time_range_list[2][0], time_range_list[2][1])
        splitter_1 = DataSplitter(args, train_env1)
        splitter_2 = DataSplitter(args, train_env2)
        splitter_3 = DataSplitter(args, train_env3)
        splits_1 = splitter_1.load_or_create_splits()
        splits_2 = splitter_2.load_or_create_splits()
        splits_3 = splitter_3.load_or_create_splits()
        train_edge_index_1 = train_env1.edge_index[:, splits_1["train_mask"]]
        train_edge_index_2 = train_env2.edge_index[:, splits_2["train_mask"]]
        train_edge_index_3 = train_env3.edge_index[:, splits_3["train_mask"]]
        algorithm_class = algorithms.get_algorithm_class(args.algorithm)
        algorithm = algorithm_class(args.input_dim, args.output_dim, 3, hparams)

        early_stopping = EarlyStopping(
            patience=args.patience,
            path=f"{args.model_save_path}/{args.dataset}/{args.algorithm}_{seed}.pt",
            verbose=True,
            loss=True,
        )


        algorithm.to(device)

        checkpoint_vals = collections.defaultdict(lambda: [])


        n_steps = args.epochs
        patience = args.patience
        checkpoint_freq = 3


        last_results_keys = None
        best_val_loss = 0
        patience = 0 
        for step in range(start_step, n_steps):
            step_start_time = time.time()
            # pdb.set_trace()
            neg_edge_index_1 = negative_sampling(
                edge_index=train_edge_index_1,
                num_nodes=train_env1.num_nodes,
                num_neg_samples=int(
                    train_edge_index_1.size(1) * args.train_negative_sampling_ratio
                ),
            ).to(args.device)
            neg_edge_index_2 = negative_sampling(
                edge_index=train_edge_index_2,
                num_nodes=train_env2.num_nodes,
                num_neg_samples=int(
                    train_edge_index_2.size(1) * args.train_negative_sampling_ratio
                ),
            ).to(args.device)
            neg_edge_index_3 = negative_sampling(
                edge_index=train_edge_index_3,
                num_nodes=train_env3.num_nodes,
                num_neg_samples=int(
                    train_edge_index_3.size(1) * args.train_negative_sampling_ratio
                ),
            ).to(args.device)
            link_labels_1 = get_link_labels(train_edge_index_1, neg_edge_index_1).to(args.device)
            link_labels_2 = get_link_labels(train_edge_index_2, neg_edge_index_2).to(args.device)
            link_labels_3 = get_link_labels(train_edge_index_3, neg_edge_index_3).to(args.device)
            if args.dataset in [
                "tgbl-review",
                "tgbl-coin",
                "tgbl-flight",
                "tgbl-comment",
                "ogbl-collab",
            ]:
                train_edge_features_1 = train_env1.edge_features[splits_1["train_mask"]].to(args.device)  
                train_edge_features_2 = train_env2.edge_features[splits_2["train_mask"]].to(args.device)
                train_edge_features_3 = train_env3.edge_features[splits_3["train_mask"]].to(args.device)      
            else:
                train_edge_features_1 = train_edge_features_2 = train_edge_features_3 = None
            
            train_data = [(train_env1.to('cuda'), train_edge_index_1.to('cuda'), neg_edge_index_1, link_labels_1, train_edge_features_1), (train_env2.to('cuda'), train_edge_index_2.to('cuda'), neg_edge_index_2, link_labels_2, train_edge_features_2), (train_env3.to('cuda'), train_edge_index_3.to('cuda'), neg_edge_index_3, link_labels_3, train_edge_features_3)]

            uda_device = None
            step_vals = algorithm.update(train_data, args.num_envs)
            checkpoint_vals['step_time'].append(time.time() - step_start_time)

            for key, val in step_vals.items():
                checkpoint_vals[key].append(val)

            # eval
            if (step % checkpoint_freq == 0) or (step == n_steps - 1):
                results = {
                    'step': step,
                    'epoch': step ,
                }
                val_edge_index_1 = splits_1["test_edges"].to(args.device)
                val_edge_index_2 = splits_2["test_edges"].to(args.device)
                val_edge_index_3 = splits_3["test_edges"].to(args.device)
                val_neg_edge_index_1 = splits_1["test_neg_edges"].to(args.device)
                val_neg_edge_index_2 = splits_2["test_neg_edges"].to(args.device)
                val_neg_edge_index_3 = splits_3["test_neg_edges"].to(args.device)
                val_link_labels_1 = get_link_labels(val_edge_index_1, val_neg_edge_index_1).to(args.device)
                val_link_labels_2 = get_link_labels(val_edge_index_2, val_neg_edge_index_2).to(args.device)
                val_link_labels_3 = get_link_labels(val_edge_index_3, val_neg_edge_index_3).to(args.device)
                eval_data = [(train_env1.to('cuda'), train_edge_index_1.to('cuda'), train_edge_features_1, val_edge_index_1, val_neg_edge_index_1, val_link_labels_1), (train_env2.to('cuda'), train_edge_index_2.to('cuda'), train_edge_features_2, val_edge_index_2, val_neg_edge_index_2, val_link_labels_2),(train_env2.to('cuda'), train_edge_index_2.to('cuda'), train_edge_features_2, val_edge_index_2, val_neg_edge_index_2, val_link_labels_2)]
                for key, val in checkpoint_vals.items():
                    results[key] = np.mean(val)
                env_name = 0
                val_loss = 0
                for data, train_edge_index, train_edge_features, val_edge_index, val_neg_edge_index, val_link_labels in eval_data:
                    loss = misc.accuracy_gnn_lr(algorithm, data, train_edge_index, train_edge_features, val_edge_index, val_neg_edge_index, val_link_labels)
                    results['env_'+str(env_name) + '_loss'] = loss
                    env_name += 1
                    val_loss += loss
                val_loss = val_loss/(env_name + 1)

                results_keys = sorted(results.keys())
                if results_keys != last_results_keys:
                    misc.print_row(results_keys, colwidth=12)
                    last_results_keys = results_keys
                misc.print_row([results[key] for key in results_keys],
                    colwidth=12)

                results.update({
                    'hparams': hparams,
                    'args': vars(args)
                })

                early_stopping(val_loss, algorithm)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    # algorithm = early_stopping.load_checkpoint(algorithm)
                    break
                # if args.save_model_every_checkpoint:
                #     save_checkpoint(f'model_step{step}.pkl')

                # save_checkpoint('model.pkl')




        # test
        # test_time_list.pop()
        test_device = 'cpu'        
        checkpoint = torch.load(f"{args.model_save_path}/{args.dataset}/{args.algorithm}_{seed}.pt")
        algorithm.load_state_dict(checkpoint)
        algorithm.to(test_device)
        for i, test_time in enumerate(test_time_list):
            print(f"Test time {i + 1}: {test_time}")
        args.save_path = test_save_path
        scores_list = evaluate(args, algorithm, tg_dataset, test_time_list, test_device)
        for i, score in scores_list.items():
        # wandb.log({f"ACC_{i}": score})
            print({f"ACC_{i}": score})
        result_list.append(scores_list)

    # for j in range(len(test_time_list)):
    #     result = []
    #     for i in range(len(args.seed)):
    #         result.append(result_list[i][f"test_period_{j}_mrr"])
    #     result = np.array(result)
    #     print({f"Average_MRR_{j}": np.mean(result)})
    #     print({f"Std_MRR_{j}": np.std(result)})

    final_scores = {}

    for period_metric in result_list[0].keys():
        period = "_".join(period_metric.split("_")[0:3])
        metric = period_metric.split("_")[-1]
        if period not in final_scores:
            final_scores[period] = {}

        values = [result[period_metric] for result in result_list]
        mean_val = sum(values) / len(values)
        variance_val = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_val = math.sqrt(variance_val)
        final_scores[period][metric] = {"mean": mean_val, "std": std_val}
    
    save_model = args.algorithm

    periods = list(final_scores.keys())
    metrics = list(next(iter(final_scores.values())).keys())
    columns = ["Model"]
    for metric in metrics:
        for period in periods:
            columns.append(f"{period}_{metric}")

    rows = []
    row = [save_model]  
    for metric in metrics: 
        for period in periods:
            stats = final_scores[period].get(metric, {"mean": "N/A", "std": "N/A"})
            value = f"{stats['mean'] * 100:.4f}%±{stats['std'] * 100:.4f}%" if "mean" in stats and "std" in stats else "N/A"
            row.append(value)

    rows.append(row)
    new_data_df = pd.DataFrame(rows, columns=columns)
    result_path = os.path.join("/home/tingyan/TDG/domainbed/result_table", f'{args.dataset}.xlsx')
    print(result_path)

    if os.path.exists(result_path):
        existing_df = pd.read_excel(result_path)
        updated_df = pd.concat([existing_df, new_data_df], ignore_index=True)
    else:
        updated_df = new_data_df

    updated_df.to_excel(result_path, index=False)
