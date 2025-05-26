from DIDA.config import args
from DIDA.utils.mutils import *
from DIDA.utils.data_util import *
from DIDA.utils.util import init_logger
import warnings
warnings.simplefilter("ignore")
import wandb
from data_transform_utils import EvolvingDataset, DataSplitter
from add_utils import TemporalDataSplitter

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
    },
    "ogbn-arxiv": {
    "save_path": "split_data/ogbn-arxiv",
    "data_path": None,
    "input_dim": 128,
    "hidden_dim":16,
    "output_dim": 40,
    "degree_dim": 32,
    "time_dim": 32,
    "span": 5
    }
}

# load data
# args,_=load_data(args)

# pre-logs
log_dir=args.log_dir
init_logger(prepare_dir(log_dir) + 'log.txt')
info_dict=get_arg_dict(args)

# Runner
from DIDA.runner_node import Runner
from DIDA.model import DGNN
wandb_run_id = f"{args.project_name}_{args.dataset}_DIDA"
wandb.init(project=args.project_name, config=args, resume=None, id=wandb_run_id)
all_results = []
seed_list = [42, 123, 456, 66]
all_results = []
dataname = args.dataset 
dataset = EvolvingDataset(dataname, datasets_by_name[dataname]['data_path'], datasets_by_name[dataname]['input_dim'], datasets_by_name[dataname]['time_dim'], datasets_by_name[dataname]['degree_dim'])
train_time, test_time_list = TemporalDataSplitter(dataname, datasets_by_name[dataname]['span'], dataset).split_by_time()
if dataname in ['ogbl-collab', 'tgbl-review']:
    test_time_list.pop()
for seed in seed_list:
    print(f"Running experiment with seed {seed}")
    args.seed = seed
    args.nfeat = datasets_by_name[dataname]['input_dim']
    args.nhid = datasets_by_name[dataname]['hidden_dim']
    args.max_epoch = 500
    args.n_layers = 2
    args.output_dim = datasets_by_name[dataname]['output_dim']
    model = DGNN(args=args).to(args.device)
    runner = Runner(args,model,dataset,train_time, test_time_list, datasets_by_name[dataname]['save_path'])
    results = runner.run()
    all_results.append(results)

final_scores = {}
for metric in all_results[0].keys():
    values = [result[metric] for result in all_results]
    mean_val = sum(values) / len(values)
    variance_val = sum((x - mean_val) ** 2 for x in values) / len(values)
    final_scores[metric] = {"mean": mean_val, "variance": variance_val}

print("Final Results:")
for metric, stats in final_scores.items():
    print(f"{metric}: Mean = {stats['mean']}, Variance = {stats['variance']}")
for metric, stats in final_scores.items():
    wandb.log(
        {f"{metric}_mean": stats["mean"], f"{metric}_variance": stats["variance"]}
    )
wandb.finish()

