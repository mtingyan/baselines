import torch
import numpy as np
from sklearn.cluster import KMeans
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GATConv

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, path="checkpoint.pth", verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        self.best_model_state = model.state_dict()
        torch.save(self.best_model_state, self.path)
        if self.verbose:
            print(f"Validation loss decreased. Saving model to {self.path}")

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.path))


class LinkPredictor(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
        edge_feature_dim=None,
        backbone="GCNConv",
        activation="relu",
        dropout_rate=0.0,
    ):
        super(LinkPredictor, self).__init__()

        conv_layer = {
            "GCNConv": GCNConv,
            "GraphConv": GraphConv,
            "SAGEConv": SAGEConv,
            "GATConv": GATConv,
        }.get(backbone, GCNConv)

        self.activation = getattr(F, activation, F.relu)

        self.convs = torch.nn.ModuleList()
        self.convs.append(conv_layer(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(hidden_dim, hidden_dim))
        self.convs.append(conv_layer(hidden_dim, output_dim))
        self.edge_feature_dim = edge_feature_dim
        self.dropout_rate = dropout_rate

    def encode(self, x, edge_index, edge_feature=None):
        edge_weight = None
        if edge_feature is not None:
            if self.edge_feature_dim == 1:
                edge_weight = edge_feature
            else:
                edge_weight = edge_feature.mean(dim=1)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        return x

    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index)
        pos_scores = self.decode(z, pos_edge_index)
        neg_scores = self.decode(z, neg_edge_index)
        return pos_scores, neg_scores

class NodeClassifier(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
        edge_feature_dim=None,
        backbone="GCNConv",
        activation="relu",
        dropout_rate=0.0,
    ):
        super(NodeClassifier, self).__init__()

        conv_layer = {
            "GCNConv": GCNConv,
            "GraphConv": GraphConv,
            "SAGEConv": SAGEConv,
            "GATConv": GATConv,
        }.get(backbone, GCNConv)

        self.activation = getattr(F, activation, F.relu)
        self.convs = torch.nn.ModuleList()
        self.convs.append(conv_layer(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(hidden_dim, hidden_dim))
        self.convs.append(conv_layer(hidden_dim, output_dim))
        self.edge_feature_dim = edge_feature_dim
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, edge_feature=None):
        edge_weight = None
        if edge_feature is not None:
            if self.edge_feature_dim == 1:
                edge_weight = edge_feature
            else:
                edge_weight = edge_feature.mean(dim=1)  
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight=edge_weight)
            # if i == len(self.convs[:-1]) - 1:
            #     last_layer_output = x.clone()  
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        last_layer_output = x.clone()  
        return x, last_layer_output


    
class NodeRegressor(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
        edge_feature_dim=None,
        backbone="GCNConv",
        activation="relu",
        dropout_rate=0.0,
    ):
        super(NodeRegressor, self).__init__()

        conv_layer = {
            "GCNConv": GCNConv,
            "GraphConv": GraphConv,
            "SAGEConv": SAGEConv,
            "GATConv": GATConv,
        }.get(backbone, GCNConv)

        self.activation = getattr(F, activation, F.relu)

        self.convs = torch.nn.ModuleList()
        self.convs.append(conv_layer(input_dim, hidden_dim))

        for i in range(num_layers - 1):
            in_dim = hidden_dim // (2 ** (i))
            out_dim = hidden_dim // (2 ** (i+1))
            self.convs.append(conv_layer(in_dim, out_dim))
        self.convs.append(conv_layer(out_dim, output_dim))
        self.edge_feature_dim = edge_feature_dim
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, edge_feature=None):
        edge_weight = None
        if edge_feature is not None:
            if self.edge_feature_dim == 1:
                edge_weight = edge_feature
            else:
                edge_weight = edge_feature.mean(dim=1)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        return x.squeeze(1)  
    
class TSSNodeRegressor(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
        edge_feature_dim=None,
        backbone="GCNConv",
        activation="relu",
        dropout_rate=0.0,
    ):
        super(TSSNodeRegressor, self).__init__()

        conv_layer = {
            "GCNConv": GCNConv,
            "GraphConv": GraphConv,
            "SAGEConv": SAGEConv,
            "GATConv": GATConv,
        }.get(backbone, GCNConv)

        self.activation = getattr(F, activation, F.relu)
        # Linear transformation for the first layer
        self.linear = torch.nn.Linear(input_dim, hidden_dim)
        # Graph convolutional layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            in_dim = hidden_dim // (2 ** (i))
            out_dim = hidden_dim // (2 ** (i+1))
            self.convs.append(conv_layer(in_dim, out_dim))
        self.convs.append(conv_layer(out_dim, output_dim))

        # MLP layers
        self.mlps = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            in_dim = hidden_dim // (2 ** i)
            out_dim = hidden_dim // (2 ** (i + 1))
            self.mlps.append(torch.nn.Sequential(
                torch.nn.Linear(in_dim, out_dim),
                # torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout_rate),
            ))
        self.mlps.append(torch.nn.Linear(hidden_dim // (2 ** (num_layers - 1)), output_dim))

        self.edge_feature_dim = edge_feature_dim
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, edge_feature=None):
        edge_weight = None
        if edge_feature is not None:
            if self.edge_feature_dim == 1:
                edge_weight = edge_feature
            else:
                edge_weight = edge_feature.mean(dim=1)

        # First layer: linear transformation
        x_linear = self.linear(x)

        # Second layer onwards: non-linear transformation and MLP
        x_conv = x_linear.clone()
        for conv in self.convs:
            x_conv = conv(x_conv, edge_index, edge_weight=edge_weight)
            x_conv = self.activation(x_conv)
            x_conv = F.dropout(x_conv, p=self.dropout_rate, training=self.training)

        # MLP transformation
        x_mlp = x_linear.clone()
        for mlp in self.mlps:
            x_mlp = mlp(x_mlp)

        # Combine the linear and non-linear parts
        x = x_conv + x_mlp
        return x.squeeze(1)
