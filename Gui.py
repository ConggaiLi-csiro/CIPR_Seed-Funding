import sys
import os
import subprocess
import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                             QVBoxLayout, QWidget, QListWidget, QListWidgetItem,
                             QTableWidget, QTableWidgetItem)
from PyQt5.QtGui import QPixmap, QImage, QMovie
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QComboBox, QStyledItemDelegate
from PyQt5.QtGui import QStandardItemModel, QStandardItem

from PIL import Image
import imageio

from gnn_model import GNN
from gnn_utils import load_sample

from PyQt5.QtWidgets import QToolButton, QMenu, QAction

class MultiSelectDropdown(QToolButton):
    def __init__(self, items, label_widget):
        super().__init__()
        self.setText("Select Nodes ▼")
        self.setPopupMode(QToolButton.InstantPopup)
        self.setFixedWidth(300)

        self.selected_items = []
        self.label_widget = label_widget

        self.menu = QMenu()
        self.actions = []
        for item in items:
            action = QAction(item, self, checkable=True)
            action.toggled.connect(self.update_selection)
            self.menu.addAction(action)
            self.actions.append(action)

        self.setMenu(self.menu)

    def update_selection(self):
        self.selected_items = [a.text() for a in self.actions if a.isChecked()]
        self.label_widget.setText("Selected: [" + ", ".join(self.selected_items) + "]")
        self.label_widget.setStyleSheet("""
            color: red;
            font-weight: bold;
            text-decoration: underline;
            font-size: 14px;
        """)
    def get_selected(self):
        return self.selected_items

class GNNApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sydney Train GNN Demo")
        self.setGeometry(100, 100, 1200, 800)

        if not os.path.exists("samples"):
            subprocess.run(["python", "1_generate_samples_multiseed.py"])
            subprocess.run(["python", "2_train_model.py"])
            subprocess.run(["python", "3_predict_probabilities.py"])

        self.model = GNN(input_dim=1, hidden_dim=32)
        self.model.load_state_dict(torch.load("models/gnn_model.pth"))
        self.model.eval()

        sample_dir = "samples/sample_1"
        self.data = load_sample(sample_dir)
        self.nodes = pd.read_csv(f"{sample_dir}/nodes.csv")
        self.edges = pd.read_csv(f"{sample_dir}/edges.csv")
        self.node_list = list(self.nodes['station_name'])
        self.node_index = {name: idx for idx, name in enumerate(self.node_list)}
        self.G = nx.from_pandas_edgelist(self.edges, source='source', target='target', create_using=nx.DiGraph())
        self.pos = nx.spring_layout(self.G, seed=42, k=2.0)
        self.selected_seeds = []

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.selector_label = QLabel("Selected: None")
        self.selector_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.selector_label)

        self.node_selector = MultiSelectDropdown(self.node_list, self.selector_label)
        layout.addWidget(self.node_selector)

        self.predict_button = QPushButton("Prediction")
        self.predict_button.setFixedHeight(40)
        self.predict_button.clicked.connect(self.predict)
        layout.addWidget(self.predict_button)

        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(['station_name', 'predicted_prob'])
        self.result_table.setColumnWidth(0, 200)
        self.result_table.setColumnWidth(1, 150)
        # self.result_table.setMaximumHeight(300)
        self.result_table.setMinimumHeight(180)
        self.result_table.setMaximumHeight(180)

        layout.addWidget(self.result_table)

        self.simulate_button = QPushButton("Cascading Simulation")
        self.simulate_button.setFixedHeight(40)
        self.simulate_button.clicked.connect(self.simulate)
        layout.addWidget(self.simulate_button)

        self.loading_label = QLabel("")
        self.loading_label.setStyleSheet("color: red; font-weight: bold; font-size: 16px;")
        self.loading_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.loading_label)

        self.image_label = QLabel()
        self.image_label.setMinimumSize(1000, 800)
        self.image_label.setAlignment(Qt.AlignCenter)


        layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        default_image_path = "outputs/Sydney_Rail_Graph.png"
        if os.path.exists(default_image_path):
            self.display_image(default_image_path)

    def predict(self):
        self.selected_seeds = self.node_selector.get_selected()
        self.image_label.clear()

        features = np.zeros((len(self.node_list), 1))
        for s in self.selected_seeds:
            features[self.node_index[s], 0] = 1.0
        features_tensor = torch.tensor(features, dtype=torch.float)
        with torch.no_grad():
            preds = self.model(features_tensor, self.data.edge_index, self.data.edge_weight).numpy()

        self.nodes['predicted_prob'] = preds
        top10 = self.nodes.sort_values(by='predicted_prob', ascending=False).head(10).reset_index(drop=True)

        self.result_table.setRowCount(len(top10))
        for i, row in top10.iterrows():
            self.result_table.setItem(i, 0, QTableWidgetItem(row['station_name']))
            self.result_table.setItem(i, 1, QTableWidgetItem(f"{row['predicted_prob']:.6f}"))

        fig, ax = plt.subplots(figsize=(12, 10), dpi=500)
        node_colors = []
        for node in self.G.nodes():
            if node in self.selected_seeds:
                node_colors.append('#003399')
            elif node in top10['station_name'].values and node not in self.selected_seeds:
                node_colors.append('red')
            else:
                node_colors.append('#66b3ff')
        nx.draw_networkx_nodes(self.G, self.pos, node_color=node_colors, node_size=500, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(self.G, self.pos, arrows=True, edge_color='gray', ax=ax)
        nx.draw_networkx_labels(self.G, self.pos, ax=ax, font_size=9)
        ax.set_title("Top 10 Predicted Failure Nodes")
        ax.axis('off')
        fig.savefig("outputs/predict.png", bbox_inches='tight')
        plt.close()
        self.display_image("outputs/predict.png")

        self.loading_label.setText("")
        self.simulate_button.setText("Cascading Simulation")
        self.simulate_button.setEnabled(True)

    def simulate(self):
        self.selected_seeds = self.node_selector.get_selected()
        self.image_label.clear()


        self.loading_label.setText("Running cascading simulation, please wait...")
        QApplication.processEvents()
        self.simulate_button.setText("Simulating...")
        self.simulate_button.setEnabled(False)
        QApplication.processEvents()

        threshold = 0.4
        failed = set(self.selected_seeds)
        history = [set(self.selected_seeds)]
        while True:
            next_failed = set()
            for node in self.G.nodes():
                if node in failed:
                    continue
                neighbors = list(self.G.predecessors(node)) + list(self.G.successors(node))
                if not neighbors:
                    continue
                failed_neighbors = sum(1 for nbr in neighbors if nbr in failed)
                if failed_neighbors / len(neighbors) >= threshold:
                    next_failed.add(node)
            if not next_failed:
                break
            failed.update(next_failed)
            history.append(next_failed)
            
            # Create GIF
            frames = []
            total_failed = set().union(*history)
            summary_text = f"Total failed nodes: {len(total_failed)}"

            for step, failed_now in enumerate(history):
                fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
                current_failed = set().union(*history[:step + 1])

                node_colors = []
                for node in self.G.nodes():
                    if node in self.selected_seeds:
                        node_colors.append('#003399')  # seed
                    elif node in current_failed:
                        node_colors.append('red')      # failed
                    else:
                        node_colors.append('#99ccff')  # normal

                nx.draw_networkx_nodes(self.G, self.pos, node_color=node_colors, node_size=500, alpha=0.9, ax=ax)
                nx.draw_networkx_edges(self.G, self.pos, arrows=True, edge_color='gray', ax=ax)
                nx.draw_networkx_labels(self.G, self.pos, ax=ax, font_size=8)

                ax.set_title(f"{summary_text}\nStep {step}: {len(failed_now)} new failed, {len(current_failed)} total",
                            fontsize=12, weight='bold')

                ax.axis('off')
                path = f"outputs/frame_{step}.png"
                fig.savefig(path, bbox_inches='tight')
                frames.append(imageio.v2.imread(path))
                plt.close()

        gif_path = "outputs/cascade.gif"
        imageio.mimsave(gif_path, frames, duration=1.2)
        self.display_gif(gif_path)

        self.loading_label.setText("")
        self.simulate_button.setText("Cascading Simulation")
        self.simulate_button.setEnabled(True)

    def display_image(self, path):
        pixmap = QPixmap(path)
        scaled_pixmap = pixmap.scaled(900, 700, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)


    def display_gif(self, path):
        self.movie = QMovie(path)
        self.movie.setCacheMode(QMovie.CacheAll)
        self.movie.setSpeed(30)

        self.movie.jumpToFrame(0)
        original_image = self.movie.currentImage()
        original_size = original_image.size()

        target_width = 900
        w = original_size.width()
        h = original_size.height()
        if w > 0 and h > 0:
            scale_factor = target_width / w
            target_height = int(h * scale_factor)
            scaled_size = QSize(target_width, target_height)
            self.movie.setScaledSize(scaled_size)

        if hasattr(self.movie, 'setLoopCount'):
            self.movie.setLoopCount(-1)
        else:
            self.movie.finished.connect(self.movie.start)

        self.image_label.setMinimumSize(scaled_size)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMovie(self.movie)

        self.movie.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GNNApp()
    window.show()
    sys.exit(app.exec_())
