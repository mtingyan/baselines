import os
from symbol import shift_expr
import numpy as np
import torch
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import Data
import pickle
from .mutils import seed_everything


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def prepare_dir(output_folder):
    mkdirs(output_folder)
    log_folder = mkdirs(output_folder)
    return log_folder


def select_by_field(edges, fields=[0, 1]):
    # field [0,1,2,3,4]
    res = []
    for f in fields:
        e = edges[edges[:, 4] == f]
        res.append(e)
    edges = torch.cat(res, dim=0)
    res = []
    for i in range(16):
        e = edges[edges[:, 2] == i]
        e = e[:, :2]
        res.append(e)
    edges = res
    return edges


def select_by_venue(edges, venues=[0, 1]):
    # venue [0-21]
    res = []
    for f in venues:
        e = edges[edges[:, 3] == f]
        res.append(e)
    edges = torch.cat(res, dim=0)
    res = []
    for i in range(16):
        e = edges[edges[:, 2] == i]
        e = e[:, :2]
        res.append(e)
    edges = res
    return edges


def load_data(args):
    seed_everything(0)
    dataset = args.dataset
    if dataset == "collab":
        from ..data_configs.collab import (
            testlength,
            vallength,
            length,
            processed_datafile,
        )

        args.dataset = dataset
        args.testlength = testlength
        args.vallength = vallength
        args.length = length
        data = torch.load(f"{processed_datafile}")
        args.nfeat = data["x"].shape[1]
        args.num_nodes = len(data["x"])

    elif dataset == "yelp":
        from ..data_configs.yelp import (
            testlength,
            vallength,
            length,
            processed_datafile,
            shift,
            num_nodes,
        )

        args.dataset = dataset
        args.testlength = testlength
        args.vallength = vallength
        args.length = length
        args.shift = shift
        args.num_nodes = num_nodes
        data = torch.load(f"{processed_datafile}")
        args.nfeat = data["x"].shape[1]
        args.num_nodes = len(data["x"])

    elif dataset == "act":
        from ..data_configs.act import testlength, vallength, length, processed_datafile

        args.dataset = dataset
        args.testlength = testlength
        args.vallength = vallength
        args.length = length
        data = torch.load(f"{processed_datafile}")
        args.nfeat = data["x"].shape[1]
        args.num_nodes = len(data["x"])

    elif dataset == "collab_04":
        from ..data_configs.collab_04 import testlength, vallength, processed_datafile

        args.dataset = dataset
        args.testlength = testlength
        args.vallength = vallength
        data = torch.load(f"{processed_datafile}")
        args.nfeat = data["x"][0].shape[1]
        args.num_nodes = len(data["x"][0])
        args.length = len(data["x"])

    elif dataset == "collab_06":
        from ..data_configs.collab_06 import testlength, vallength, processed_datafile

        args.dataset = dataset
        args.testlength = testlength
        args.vallength = vallength
        data = torch.load(f"{processed_datafile}")
        args.nfeat = data["x"][0].shape[1]
        args.num_nodes = len(data["x"][0])
        args.length = len(data["x"])

    elif dataset == "collab_08":
        from ..data_configs.collab_08 import testlength, vallength, processed_datafile

        args.dataset = dataset
        args.testlength = testlength
        args.vallength = vallength
        data = torch.load(f"{processed_datafile}")
        args.nfeat = data["x"][0].shape[1]
        args.num_nodes = len(data["x"][0])
        args.length = len(data["x"])

    else:
        raise NotImplementedError(f"Unknown dataset {dataset}")
    print(f"Loading dataset {dataset}")
    return args, data
