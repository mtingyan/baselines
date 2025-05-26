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
import numpy as np
import pandas as pd
import PIL
import torch
import torchvision
import torch.utils.data
from domainbed.scripts.data_preprocess import EvolvingDataset, DataSplitter
from domainbed.scripts.utils import load_config, TemporalDataSplitter, set_seed, EarlyStopping
from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
import pdb

# @torch.no_grad()
# def evaluate(args, model, dataset, test_time):
#     model.eval()
#     score_list = {}
#     for i, period in enumerate(test_time):
#         test_data = dataset.build_graph(period[0], period[1])
#         splitter = DataSplitter(args, test_data)
#         splits = splitter.load_or_create_splits()
#         labels = test_data.y.to('cuda')[splits['test_mask']].squeeze(1)
#         edge_index = test_data.edge_index.to('cuda')
#         if args.dataset in ["ogbn-arxiv",]:
#             logits = model.predict(test_data.x.to('cuda'), edge_index)
#         preds = logits.argmax(dim=1)  
#         acc = (preds[splits['test_mask']] == labels).sum().item() / labels.size(0)  
#         score_list[f"test_period_{i}_acc"] = acc
#         torch.cuda.empty_cache()
#     return score_list

@torch.no_grad()
def evaluate(args, model, dataset, test_time):
    model.eval()
    score_list = {}
    for i, period in enumerate(test_time):
        test_data = dataset.build_graph(period[0], period[1])
        splitter = DataSplitter(args, test_data)
        splits = splitter.load_or_create_splits()
        edge_index = test_data.edge_index.to('cuda')
        logits = model.predict(test_data.x.to('cuda'), edge_index)
        if args.dataset in ["ogbn-arxiv", "SBM", 'penn', 'Amherst41', 'Cornell5', 'Johns_Hopkins55', 'Reed98']:
            labels = test_data.y.to('cuda')[splits['test_mask']].squeeze(1)
            preds = logits.argmax(dim=1)[splits["test_mask"]]
            acc = (preds == labels).sum().item() / labels.size(0)
            score_list[f"test_period_{i}_MicroACC"] = acc
            unique_classes = torch.unique(labels)
            class_accuracies = {}
            for cls in unique_classes:
                cls_mask = (labels == cls)
                cls_acc = (preds[cls_mask] == labels[cls_mask]).sum().item() / labels[cls_mask].size(0)
                class_accuracies[cls] = cls_acc
            score_list[f"test_period_{i}_MacroACC"] = np.mean(list(class_accuracies.values()))
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
        if args.dataset == 'ogbn-arxiv':
            train_env1 = tg_dataset.build_graph(38, 39)
            train_env2 = tg_dataset.build_graph(39, 40)
            train_env3 = tg_dataset.build_graph(40, 41)
        splitter_1 = DataSplitter(args, train_env1)
        splitter_2 = DataSplitter(args, train_env2)
        splitter_3 = DataSplitter(args, train_env3)
        splits_1 = splitter_1.load_or_create_splits()
        splits_2 = splitter_2.load_or_create_splits()
        splits_3 = splitter_3.load_or_create_splits()
        train_mask_1 = splits_1["train_mask"]
        train_mask_2 = splits_2["train_mask"]
        train_mask_3 = splits_3["train_mask"]
        val_mask_1 = splits_1['test_mask']
        val_mask_2 = splits_2['test_mask']
        val_mask_3 = splits_3['test_mask']
        train_edge_index_1 = train_env1.edge_index[:,splits_1["train_edge_mask"]]
        train_edge_index_2 = train_env2.edge_index[:,splits_2["train_edge_mask"]]
        train_edge_index_3 = train_env3.edge_index[:,splits_3["train_edge_mask"]]
        algorithm_class = algorithms.get_algorithm_class(args.algorithm)
        algorithm = algorithm_class(args.input_dim, args.output_dim, 3, hparams)

        early_stopping = EarlyStopping(
            patience=args.patience,
            path=f"{args.model_save_path}/{args.dataset}/{args.algorithm}_{seed}.pt",
            verbose=True,
            loss=False,
        )
        # def save_checkpoint(filename):
        #     if args.skip_model_save:
        #         return
        #     save_dict = {
        #         "args": vars(args),
        #         "model_input_shape": dataset.input_shape,
        #         "model_num_classes": dataset.num_classes,
        #         "model_num_domains": len(dataset) - len(args.test_envs),
        #         "model_hparams": hparams,
        #         "model_dict": algorithm.state_dict()
        #     }
        #     torch.save(save_dict, os.path.join(args.output_dir, filename))


        algorithm.to(device)

        checkpoint_vals = collections.defaultdict(lambda: [])


        n_steps = args.epochs
        patience = args.patience
        checkpoint_freq = 3
        train_data = [(train_env1.to('cuda'), train_edge_index_1.to('cuda'), train_mask_1.to('cuda')), (train_env2.to('cuda'), train_edge_index_2.to('cuda'), train_mask_2.to('cuda')), (train_env3.to('cuda'), train_edge_index_3.to('cuda'), train_mask_3.to('cuda'))]
        
        eval_data = [(train_env1.to('cuda'), train_env1.edge_index.to('cuda'), val_mask_1.to('cuda')), (train_env2.to('cuda'), train_env2.edge_index.to('cuda'), val_mask_2.to('cuda')), (train_env3.to('cuda'), train_env3.edge_index.to('cuda'), val_mask_3.to('cuda'))]

        last_results_keys = None
        best_val_acc = 0
        patience = 0 
        for step in range(start_step, n_steps):
            step_start_time = time.time()
            # pdb.set_trace()
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

                for key, val in checkpoint_vals.items():
                    results[key] = np.mean(val)
                env_name = 0
                val_acc = 0
                for x, edge, mask in eval_data:
                    acc = misc.accuracy_gnn_class_cls(algorithm, x, edge, mask,device)
                    results['env_'+str(env_name) + '_acc'] = acc
                    env_name += 1
                    # acc = misc.accuracy(algorithm, loader, weights, device)
                    # results[name+'_acc'] = acc
                    val_acc += acc
                val_acc = val_acc/(env_name + 1)

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

                early_stopping(val_acc, algorithm)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    # algorithm = early_stopping.load_checkpoint(algorithm)
                    break
                # if args.save_model_every_checkpoint:
                #     save_checkpoint(f'model_step{step}.pkl')

                # save_checkpoint('model.pkl')

        # test
        checkpoint = torch.load(f"{args.model_save_path}/{args.dataset}/{args.algorithm}_{seed}.pt")
        algorithm.load_state_dict(checkpoint)
        for i, test_time in enumerate(test_time_list):
            print(f"Test time {i + 1}: {test_time}")
        args.save_path = test_save_path
        scores_list = evaluate(args, algorithm, tg_dataset, test_time_list)
        for i, score in scores_list.items():
        # wandb.log({f"ACC_{i}": score})
            print({f"ACC_{i}": score})
        result_list.append(scores_list)
    # for j in range(len(test_time_list)):
    #     result = []
    #     for i in range(len(args.seed)):
    #         result.append(result_list[i][f"test_period_{j}_acc"])
    #     result = np.array(result)
    #     print({f"Average_ACC_{j}": np.mean(result)})
    #     print({f"Std_ACC_{j}": np.std(result)})
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