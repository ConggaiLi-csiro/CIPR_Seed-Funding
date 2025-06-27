import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import random
import imageio

class GNNPropagationCascadingPipeline:
    def __init__(self, gtfs_folder, output_dir):
        self.gtfs_folder = gtfs_folder
        self.output_dir = output_dir
        self._load_data()
        self._preprocess()

    def _load_data(self):
        self.routes = pd.read_csv(os.path.join(self.gtfs_folder, "routes.txt"), low_memory=False)
        self.trips = pd.read_csv(os.path.join(self.gtfs_folder, "trips.txt"), low_memory=False)
        self.stop_times = pd.read_csv(os.path.join(self.gtfs_folder, "stop_times.txt"), low_memory=False)
        self.stops = pd.read_csv(os.path.join(self.gtfs_folder, "stops.txt"), low_memory=False)

    def _preprocess(self):
        self.stops['station_name'] = self.stops['stop_name'].str.split(',').str[0].str.strip()
        self.stops_dict = self.stops.set_index('stop_id').to_dict('index')

        train_routes = self.routes[self.routes['route_type'] == 2]
        exclude_keywords = ['Temporary', 'Coach', 'Replacement', 'Bus', 'Regional']
        def exclude(name):
            if pd.isnull(name): return False
            for kw in exclude_keywords:
                if kw.lower() in name.lower():
                    return True
            return False
        self.filtered_routes = train_routes[~train_routes['route_long_name'].apply(exclude)]
        trips_filtered = self.trips[self.trips['route_id'].isin(self.filtered_routes['route_id'])]
        self.stop_times = self.stop_times[self.stop_times['trip_id'].isin(trips_filtered['trip_id'])]
        self.trip_stop_sequences = self.stop_times.sort_values(['trip_id','stop_sequence']) \
            .groupby('trip_id')['stop_id'].apply(list).to_dict()

    def build_directed_graph(self):
        G = nx.DiGraph()
        for trip_id, stop_list in self.trip_stop_sequences.items():
            for i in range(len(stop_list)-1):
                u_id, v_id = stop_list[i], stop_list[i+1]
                if u_id in self.stops_dict and v_id in self.stops_dict:
                    u_station = self.stops_dict[u_id]['station_name']
                    v_station = self.stops_dict[v_id]['station_name']
                    if u_station != v_station:
                        if G.has_edge(u_station, v_station):
                            G[u_station][v_station]['weight'] += 1
                        else:
                            G.add_edge(u_station, v_station, weight=1)
        print(f"\nDirected graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def sample_subgraph(self, G, seeds, target_size=40, depth_limit=3):
        bfs_nodes = set(seeds)
        queue = list(seeds)
        while queue and len(bfs_nodes) < target_size:
            current = queue.pop(0)
            for neighbor in nx.bfs_tree(G, current, depth_limit=depth_limit).nodes:
                if neighbor not in bfs_nodes:
                    bfs_nodes.add(neighbor)
                    queue.append(neighbor)
                    if len(bfs_nodes) >= target_size:
                        break
        subG = G.subgraph(bfs_nodes).copy()
        print(f"  Sampled subgraph: {subG.number_of_nodes()} nodes, {subG.number_of_edges()} edges")
        return subG

    def save_data(self, G, tag):
        out_path = os.path.join(self.output_dir, tag)
        os.makedirs(out_path, exist_ok=True)
        nodes_data = [{'station_name': node} for node in G.nodes()]
        pd.DataFrame(nodes_data).to_csv(os.path.join(out_path, "nodes.csv"), index=False)
        edges_data = [{'source': u, 'target': v, 'weight': d['weight']} for u, v, d in G.edges(data=True)]
        pd.DataFrame(edges_data).to_csv(os.path.join(out_path, "edges.csv"), index=False)
        print(f"  Data saved to: {out_path}")

    def visualize(self, G, tag):
        plt.figure(figsize=(8, 6))
        # pos = nx.spring_layout(G, seed=42, k=2.0)
        # pos = nx.kamada_kawai_layout(G)
        pos = nx.spring_layout(G, seed=42, k=20.0)

        nx.draw_networkx_nodes(G, pos, node_size=400, node_color='#66b3ff', alpha=0.9)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=15, edge_color='gray', width=1.2, alpha=0.7)
        clean_labels = {node: node.replace(" Station", "") for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=clean_labels, font_size=8)
        plt.title(f"{tag} Subgraph", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, f"{tag}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  Figure saved to: {out_path}")
        return pos

    def simulate_cascading_failure(self, G, initial_failed, threshold=0.4):
        failed = set(initial_failed)
        history = [set(initial_failed)]
        steps = 0
        while True:
            next_failed = set()
            for node in G.nodes():
                if node in failed:
                    continue
                neighbors = list(G.predecessors(node)) + list(G.successors(node))
                if not neighbors:
                    continue
                failed_neighbors = sum(1 for nbr in neighbors if nbr in failed)
                if failed_neighbors / len(neighbors) >= threshold:
                    next_failed.add(node)
            if not next_failed:
                break
            failed.update(next_failed)
            history.append(set(next_failed))
            steps += 1
            print(f"Step {steps}: {len(next_failed)} newly failed, total failed so far: {len(failed)}")
        print(f"\n  Cascading complete after {steps} steps; total failed nodes: {len(failed)}")
        return history
    def visualize_cascading_gif_(self, G, history, pos, tag):
        fig, ax = plt.subplots(figsize=(10, 8))
        clean_labels = {node: node.replace(" Station", "") for node in G.nodes()}

        def update(step):
            ax.clear()
            failed_total = set().union(*history[:step+1])
            node_colors = ['red' if node in failed_total else '#66b3ff' for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400, alpha=0.9, ax=ax)
            nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=20, edge_color='gray', width=1.2, alpha=0.7)
            nx.draw_networkx_labels(G, pos, labels=clean_labels, font_size=10, font_weight='bold', ax=ax)

            new_fail_num = len(history[step])
            total_fail_num = sum(len(h) for h in history[:step+1])
            ax.set_title(f"Step {step}: {new_fail_num} new failed, {total_fail_num} total failed", fontsize=14)
            ax.axis('off')

        ani = animation.FuncAnimation(fig, update, frames=len(history), interval=1000, repeat_delay=2000)

        gif_path = os.path.join(self.output_dir, f"{tag}_cascade.gif")
        ani.save(gif_path, writer='pillow', dpi=150, fps=1)
        plt.close()
        print(f"  ✅ GIF has been saved: {gif_path}")

    def visualize_cascading_gif(self, G, history, pos, tag, initial_failed):
        clean_labels = {node: node.replace(" Station", "") for node in G.nodes()}
        frames = []

        for step in range(len(history)):
            fig, ax = plt.subplots(figsize=(10, 8))
            failed_total = set().union(*history[:step+1])

            node_colors = []
            for node in G.nodes():
                if node in initial_failed:
                    node_colors.append('#003399')
                elif node in failed_total:
                    node_colors.append('red')
                else:
                    node_colors.append('#99ccff')

            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400, alpha=0.9, ax=ax)
            nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=20, edge_color='gray', width=1.2, alpha=0.7)
            nx.draw_networkx_labels(G, pos, labels=clean_labels, font_size=10, font_weight='bold', ax=ax)

            new_fail_num = len(history[step])
            total_fail_num = sum(len(h) for h in history[:step+1])
            ax.set_title(f"Step {step}: {new_fail_num} new failed, {total_fail_num} total failed", fontsize=14)
            ax.axis('off')

            frame_path = os.path.join(self.output_dir, f"{tag}_frame_{step}.png")
            plt.savefig(frame_path, dpi=150)
            plt.close()
            frames.append(imageio.imread(frame_path))
            os.remove(frame_path)

        gif_path = os.path.join(self.output_dir, f"{tag}_cascade.gif")
        imageio.mimsave(gif_path, frames, duration=800)
        print(f"✅ GIF saved: {gif_path}")


        

if __name__ == "__main__":
    gtfs_folder = "./DataSet/"
    output_dir = "./outputs/"
    os.makedirs(output_dir, exist_ok=True)

    pipeline = GNNPropagationCascadingPipeline(gtfs_folder, output_dir)
    G_full = pipeline.build_directed_graph()

    city_circle = [
        "Central Station", "Town Hall Station", "Wynyard Station",
        "Circular Quay Station", "St James Station", "Museum Station",
        "Redfern Station", "Martin Place Station"
    ]
    seeds = [s for s in city_circle if s in G_full.nodes]
    subG = pipeline.sample_subgraph(G_full, seeds, target_size=30)
    pipeline.save_data(subG, "city_circle")
    pos = pipeline.visualize(subG, "city_circle")

    initial_fail = ["Central Station"]
    history = pipeline.simulate_cascading_failure(subG, initial_fail, threshold=0.4)
    pipeline.visualize_cascading_gif(subG, history, pos, "city_circle")
