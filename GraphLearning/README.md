# 🧬 HCCExplorer: Multi-modal graph learning for biomarker discovery

This repository implements a supernode-based graph learning pipeline for histological image analysis. It includes modules for graph construction, feature aggregation, and deep learning model training.

# 🛠️ Environment Setup
To set up the development environment, please ensure you have Python 3.8+ installed. You can install all necessary dependencies using the provided requirements file:

pip install -r requirements.txt

## 🚀 Workflow

### 1. Graph Construction & Visualization
Before training, use the ★ Tutorials ★ to prepare the data and verify the graph topology.

* **Construction**: Run [hccexplorer_graph_construction.ipynb](Tutorial_hccexplorer_graph_construction.ipynb) to generate super patches and build the graph structure.
* **Visualization**: Run [hccexplorer_graph_visualization.ipynb](Tutorial_hccexplorer_graph_visualization.ipynb) to inspect the generated graphs and attention maps.

### 2. Model Training
Once the graphs are constructed and saved to the `data/` directory, use the shell script to initiate the training pipeline.

```bash
# Run the training pipeline
bash scripts/train.sh
```

## 📂 Project Structure

```text
.
├── config/                                              # Model configurations
├── data/                                                # Processed graph storage
├── dataset/                                             # Raw dataset
├── models/                                              # Model definitions
│   └── hccexplorer_graph.py
├── scripts/
│   └── train.sh                                         # Training execution script
├── utils/                                               # Utility functions
├── Trained_weight/                                      # Model checkpoints
├── Tutorial_hccexplorer_graph_construction.ipynb        # ★ Tutorial: Graph Construction ★
├── Tutorial_hccexplorer_graph_visualization.ipynb       # ★ Tutorial: Visualization ★
├── requirements.txt                                     # Python dependencies
└── README.md
```

## 🤝 Contributing
Thanks to the following work for improving our project：
- UNI: [https://github.com/mahmoodlab/uni](https://github.com/mahmoodlab/uni)
- TEA-Graph: [https://github.com/taliq/TEA-graph](https://github.com/taliq/TEA-graph)
- Pytorch_Geometric: [http://github.com/pyg-team/pytorch_geometric](http://github.com/pyg-team/pytorch_geometric)
