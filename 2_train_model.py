import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from gnn_model import GNN
from gnn_utils import load_sample

os.makedirs("models", exist_ok=True)

sample_dirs = sorted([d for d in os.listdir("samples") if os.path.isdir(os.path.join("samples", d))])
train_dirs = sample_dirs[:int(0.8 * len(sample_dirs))]
val_dirs = sample_dirs[int(0.8 * len(sample_dirs)):]

train_dataset = [load_sample(os.path.join("samples", d)) for d in train_dirs]
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

val_dataset = [load_sample(os.path.join("samples", d)) for d in val_dirs]
val_loader = DataLoader(val_dataset, batch_size=4)

model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_weight)
        loss = F.binary_cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch.x, batch.edge_index, batch.edge_weight)
            val_loss += F.binary_cross_entropy(out, batch.y).item()

    print(f"Epoch {epoch}: Train Loss {total_loss/len(train_loader):.4f}, Val Loss {val_loss/len(val_loader):.4f}")

torch.save(model.state_dict(), "models/gnn_model.pth")
print("✅ Model saved.")
