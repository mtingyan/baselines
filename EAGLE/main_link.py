import sys
import math

sys.path.append("..")

from EAGLE.config import args
from EAGLE.utils.mutils import *
from EAGLE.utils.data_util import *
from EAGLE.utils.util import init_logger

import warnings
import networkx as nx

import wandb
from data_transform_utils import EvolvingDataset, DataSplitter
from add_utils import TemporalDataSplitter

warnings.simplefilter("ignore")

# load data
# args, data = load_data(args)
datasets_by_name = {
    "ogbl-collab": {
        "save_path": "split_data/ogbl-collab",
        "data_path": None,
        "edge_dim": 1,
        "input_dim": 128,
        "hidden_dim": 32,
        "output_dim": 16,
        "degree_dim": 64,
        "time_dim": 32,
        "span": 5,
        'test_negative_sampling_ratio': 0.01
    },
    "tgbl-review": {
        "save_path": "split_data/tgbl-review",
        "data_path": None,
        "edge_dim": 1,
        "input_dim": 32,
        "hidden_dim": 32,
        "output_dim": 16,
        "degree_dim": 32,
        "time_dim": 32,
        "span": 10,
        'test_negative_sampling_ratio': 0.01
    },
    "tgbl-comment": {
        "save_path": "split_data/tgbl-comment",
        "data_path": None,
        "edge_dim": 2,
        "input_dim": 32,
        "hidden_dim": 32,
        "output_dim": 16,
        "degree_dim": 32,
        "time_dim": 64,
        "span": 10,
        'test_negative_sampling_ratio': 0.01
    },
    "tgbl-coin": {
        "save_path": "split_data/tgbl-coin",
        "data_path": None,
        "edge_dim": 1,
        "input_dim": 32,
        "hidden_dim": 128,
        "output_dim": 8,
        "degree_dim": 32,
        "time_dim": 32,
        "span": 10,
        'test_negative_sampling_ratio': 0.01
    }
}
if args.dataset in datasets_by_name:
    for key, value in datasets_by_name[args.dataset].items():
        setattr(args, key, value)
else:
    raise ValueError(f"Dataset '{args.dataset}' is not defined in datasets_by_name")

# pre-logs
log_dir = args.log_dir
init_logger(prepare_dir(log_dir) + "log_" + args.dataset + ".txt")
info_dict = get_arg_dict(args)

# Runner
from EAGLE.runner_link import Runner
from EAGLE.model import EADGNN
from EAGLE.model import ECVAE

wandb_run_id = f"{args.project_name}_{args.dataset}_EAGLE"
wandb.init(project=args.project_name, config=args, resume=None, id=wandb_run_id)
all_results = []
seed_list = [42, 123, 456, 66]
all_results = []

for seed in seed_list:
    print(f"Running experiment with seed {seed}")
    args.seed = seed
    args.nfeat = args.input_dim
    args.nhid = args.hidden_dim
    args.d_for_cvae = args.hidden_dim

    model = EADGNN(args=args).to(args.device)
    cvae = ECVAE(args=args).to(args.device)
    dataset = EvolvingDataset(args)
    train_time, test_time_list = TemporalDataSplitter(args, dataset).split_by_time()
    runner = Runner(args,model,cvae,dataset,train_time, test_time_list, args.save_path)
    results = runner.run()
    all_results.append(results)

final_scores = {}
for period_metric in all_results[0].keys():
    period = "_".join(period_metric.split("_")[0:3])
    metric = period_metric.split("_")[-1]
    if period not in final_scores:
        final_scores[period] = {}

    values = [result[period_metric] for result in all_results]
    mean_val = sum(values) / len(values)
    variance_val = sum((x - mean_val) ** 2 for x in values) / len(values)
    std_val = math.sqrt(variance_val)
    final_scores[period][metric] = {"mean": mean_val, "std": std_val}

print("Final Results:")
for period, metrics in final_scores.items():
    metric_strings = []
    for metric, stats in metrics.items():
        metric_strings.append(f"{metric.upper()}: Mean = {stats['mean']:.4f}, Std = {stats['std']:.4f}")
        wandb.log(
            {
                f"{period}_{metric}_mean": stats["mean"],
                f"{period}_{metric}_std": stats["std"],
            }
        )
    print(f"{period}: {', '.join(metric_strings)}")
wandb.finish()

