# 🧬 Hccexplorer Graph Learning Framework

This repository implements a supernode-based graph learning pipeline for histological image analysis. It includes modules for graph construction, feature aggregation, and deep learning model training.

# 🛠️ Environment Setup
To set up the development environment, please ensure you have Python 3.8+ installed. You can install all necessary dependencies using the provided requirements file:

pip install -r requirements.txt

## 🚀 Workflow

### 1. Graph Construction & Visualization
Before training, use the interactive notebooks to prepare the data and verify the graph topology.

* **Construction**: Run `hccexplorer_graph_construction.ipynb` to generate super patches and build the graph structure.
* **Visualization**: Run `hccexplorer_graph_visualization.ipynb` to inspect the generated graphs and attention maps.

### 2. Model Training
Once the graphs are constructed and saved to the `data/` directory, use the shell script to initiate the training pipeline.

```bash
# Run the training pipeline
bash scripts/train.sh
```

## 📂 Project Structure

```text
.
├── config/                      # Model configurations
├── data/                        # Processed graph storage
├── dataset/                     # Raw dataset
├── models/                      # Model definitions
│   └── hccexplorer_graph.py
├── scripts/
│   └── train.sh                 # Training execution script
├── utils/                       # Utility functions
├── Trained_weight/              # Model checkpoints
├── hccexplorer_graph_construction.ipynb  # Notebook: Graph Construction
├── hccexplorer_graph_visualization.ipynb # Notebook: Visualization
├── requirements.txt             # Python dependencies
└── README.md