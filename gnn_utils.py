import pandas as pd
import torch
from torch_geometric.data import Data


def load_sample(sample_dir):
    nodes = pd.read_csv(f"{sample_dir}/nodes.csv")
    edges = pd.read_csv(f"{sample_dir}/edges.csv")
    labels = pd.read_csv(f"{sample_dir}/labels.csv")

    node_index = {name: i for i, name in enumerate(nodes['station_name'])}
    features = torch.tensor(nodes[['init_seed']].values, dtype=torch.float)
    labels = torch.tensor(labels['label'].values, dtype=torch.float)

    edge_index = torch.tensor(
        [[node_index[s], node_index[t]] for s, t in zip(edges['source'], edges['target'])],
        dtype=torch.long
    ).t().contiguous()

    max_weight = edges['weight'].max()
    edge_weight = torch.tensor(edges['weight'].values / max_weight, dtype=torch.float)

    return Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=labels)
