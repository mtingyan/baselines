import pickle
import torch
import yaml
import random
import numpy as np
import networkx as nx

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[: pos_edge_index.size(1)] = 1
    return link_labels

def get_node_labels(node_label_method, edge_index, num_nodes):
    G = nx.Graph()
    G.add_edges_from(edge_index)
    if node_label_method == "closeness":
        closeness_dict = nx.closeness_centrality(G)
        closeness_dict = {int(k.item()): v for k, v in closeness_dict.items()}
        values = [closeness_dict.get(node, 0.0) * 1000 for node in range(num_nodes)]
    elif node_label_method == "degree":
        degree_dict = dict(G.degree())
        degree_dict = {int(k.item()): v for k, v in degree_dict.items()}
        values = [degree_dict.get(node, 0) for node in range(num_nodes)]
    elif node_label_method == "betweenness":
        betweenness_dict = nx.betweenness_centrality(G)
        betweenness_dict = {int(k.item()): v for k, v in betweenness_dict.items()}
        values = [betweenness_dict.get(node, 0.0) for node in range(num_nodes)]
    elif node_label_method == "clustering":
        clustering_dict = nx.clustering(G)  
        clustering_dict = {int(k.item()): v for k, v in clustering_dict.items()}
        values = [clustering_dict.get(node, 0.0) for node in range(num_nodes)]  
    else:
        raise ValueError(f"Unknown node label method: {node_label_method}")
    labels = torch.tensor(values, dtype=torch.float32)
    return labels

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
            "BA-random",
        ]:
            self.start_time = min(dataset.edge_time)
            self.end_time = max(dataset.edge_time)
        elif args.dataset in ["penn", 'Reed98', "Amherst41", 'Johns_Hopkins55', "Cornell5"]:
            self.end_time = max(dataset.edge_time)
            self.start_time = 1998
        elif args.dataset in ["ogbn-arxiv"]:
            self.end_time = max(dataset.edge_time)
            self.start_time = 38        
            
    def split_by_time(self):
        num_time = self.end_time - self.start_time + 1
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

def add_time_features(x, edge_index, time_feature, num_nodes, time_dim):
    time_features_norm = time_feature / time_feature.max()
    bins = torch.linspace(0, 1, steps=time_dim + 1)
    bucket_indices = torch.bucketize(time_features_norm.squeeze(-1), bins) - 1
    time_based_features = torch.zeros((num_nodes, time_dim), dtype=torch.float32)
    for edge, bucket in zip(edge_index.T, bucket_indices):
        source, target = edge[0].item(), edge[1].item()
        time_based_features[source, bucket] = 1
        time_based_features[target, bucket] = 1
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

def add_temporal_structure_features(x, edge_index, time_feature, num_nodes, tss_dim):
    time_features_norm = time_feature / time_feature.max()
    bins = torch.linspace(0, 1, steps=tss_dim + 1)
    bucket_indices = torch.bucketize(time_features_norm.squeeze(-1), bins) - 1
    time_based_features = torch.zeros((num_nodes, tss_dim), dtype=torch.float32)
    for edge, bucket in zip(edge_index.T, bucket_indices):
        source, target = edge[0].item(), edge[1].item()
        time_based_features[source, bucket] += 1
        time_based_features[target, bucket] += 1
    col_sums = time_based_features.sum(dim=0, keepdim=True)
    time_based_features = torch.div(time_based_features, col_sums + 1e-8)  # Avoid division by zero
    x = torch.cat([x, time_based_features], dim=1)
    return x

def create_ba_random_graph(args, raw_data_path):
    """Create a Barabási-Albert random graph with evolving structure."""
    G = nx.barabasi_albert_graph(args.initial_nodes, int(args.link_probability * args.initial_nodes))

    # Initialize node and edge time tracking
    node_time = {node: 0 for node in G.nodes}
    edge_time = {tuple(sorted(edge)): 0 for edge in G.edges}

    # Evolve the graph over timesteps
    current_node = max(G.nodes) + 1
    for t in range(1, args.time_step):
        for _ in range(args.nodes_per_step):
            G.add_node(current_node)
            node_time[current_node] = t

            degrees = dict(G.degree())
            total_degree = sum(degrees.values())
            attachment_probs = [degrees[node] / total_degree for node in G.nodes]

            targets = random.choices(
                population=list(G.nodes),
                weights=attachment_probs,
                k=int(args.link_probability * args.initial_nodes)
            )
            for target in targets:
                if current_node != target:  # Avoid self-loops
                    edge = tuple(sorted((current_node, target)))
                    G.add_edge(*edge)
                    edge_time[edge] = t
            current_node += 1

    # Generate graph data
    node_feature = np.random.rand(len(G.nodes), args.input_dim)
    node_time = np.array([node_time[node] for node in sorted(G.nodes)])
    edges = np.array(sorted(tuple(edge) for edge in G.edges))
    edge_time = np.array([edge_time.get(tuple(edge), 0) for edge in edges])

    # Save all data as a pickle file
    data = {
        "node_feature": node_feature,
        "node_time": node_time,
        "edges": edges,
        "edge_time": edge_time,
    }
    with open(raw_data_path, "wb") as f:
        pickle.dump(data, f)
    print("Graph generation complete.")
    return data