import os
import torch
import pandas as pd
from gnn_model import GNN
from gnn_utils import load_sample

model = GNN()
model.load_state_dict(torch.load("models/gnn_model.pth"))
model.eval()

sample_dirs = sorted([d for d in os.listdir("samples") if os.path.isdir(os.path.join("samples", d))])

os.makedirs("outputs", exist_ok=True)

for i, d in enumerate(sample_dirs):
    data = load_sample(os.path.join("samples", d))
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_weight).cpu().numpy()
    station_names = pd.read_csv(f"samples/{d}/nodes.csv")['station_name']
    df = pd.DataFrame({'station_name': station_names, 'predicted_failure_prob': out})
    df.to_csv(f"outputs/prediction_{i+1}.csv", index=False)
    print(f"✅ Prediction saved for sample {i+1}")
