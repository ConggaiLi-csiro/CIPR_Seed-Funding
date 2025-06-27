import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

<<<<<<< HEAD
# class GNN(torch.nn.Module):
#     def __init__(self, input_dim=1, hidden_dim=32):
#         super().__init__()
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, 1)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         return torch.sigmoid(x).squeeze(-1)

=======
>>>>>>> 7e7d846 (Initial commit)
class GNN(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return torch.sigmoid(x).squeeze(-1)
