import os
import random
import pandas as pd
from gnn_cascading_pipeline import GNNPropagationCascadingPipeline

gtfs_folder = "./DataSet/"
sample_output_dir = "./samples/"
os.makedirs(sample_output_dir, exist_ok=True)

GENERATE_GIF = True
SEED_SAMPLE_NUM = 50
SEED_RANGE = (1, 3)

pipeline = GNNPropagationCascadingPipeline(gtfs_folder, output_dir="./outputs/")
G_full = pipeline.build_directed_graph()
pipeline.save_data(G_full, "sydney_train")
print("✅ Full Sydney Train Network saved.")


city_circle = [
    "Central Station", "Town Hall Station", "Wynyard Station",
    "Circular Quay Station", "St James Station", "Museum Station",
    "Redfern Station", "Martin Place Station"
]
seeds = [s for s in city_circle if s in G_full.nodes]
subG = pipeline.sample_subgraph(G_full, seeds, target_size=30)

node_list = list(subG.nodes())

for i in range(SEED_SAMPLE_NUM):
    num_seeds = random.randint(*SEED_RANGE)
    init_seeds = random.sample(node_list, num_seeds)
    print(f"Generating sample {i+1}: seeds={init_seeds}")

    history = pipeline.simulate_cascading_failure(subG, init_seeds, threshold=0.4)
    final_failed = set().union(*history)

    sample_dir = os.path.join(sample_output_dir, f"sample_{i+1}")
    os.makedirs(sample_dir, exist_ok=True)

    pd.DataFrame([
        {'station_name': n, 'init_seed': int(n in init_seeds)} for n in subG.nodes()
    ]).to_csv(os.path.join(sample_dir, "nodes.csv"), index=False)

    pd.DataFrame([
        {'source': u, 'target': v, 'weight': subG[u][v]['weight']} for u, v in subG.edges()
    ]).to_csv(os.path.join(sample_dir, "edges.csv"), index=False)


    pd.DataFrame([
        {'station_name': n, 'label': int(n in final_failed)} for n in subG.nodes()
    ]).to_csv(os.path.join(sample_dir, "labels.csv"), index=False)

    if GENERATE_GIF:
        pos = pipeline.visualize(subG, f"sample_{i+1}")
        pipeline.visualize_cascading_gif(subG, history, pos, f"sample_{i+1}", initial_failed=init_seeds)


print("\n✅ All multi-seed samples generated (Single-threaded Stable Version).")
