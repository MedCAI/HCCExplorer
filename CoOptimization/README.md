# 🖌️ Hccexplorer: H&E-Virtual mIF Contrastive Learning

This repository implements a multi-modal contrastive learning framework designed to fuse morphological information from H&E images with molecular insights from virtual mIF data.

## 🛠️ Environment Setup

Please ensure you have **Python 3.8+** installed. We recommend using a virtual environment (e.g., Conda) to manage dependencies.

```bash
# Install dependencies
pip install -r requirements.txt
```

## 🚀 Workflow

### 1. Model Training
```bash
# Run the training pipeline
bash launch_pretrain_withStainEncodings_256.sh
```

### 2. Model Inference 
```bash
# Run the training pipeline
bash launch_inference_withStainEncodings.sh
```


## 📂 Project Structure

```text
.
├── bin/                                     # Training implementation
├── madeline/                                # Core algorithm implementation
├── results/                                 # Model checkpoints
├── launch_inference_withStainEncodings.sh   # Inference script
├── launch_pretrain_withStainEncodings.sh    # Training script
├── requirements.txt                         # Python dependencies
└── README.md
```

## 🤝 Contributing
Thanks to the following work for improving our project：

* MADELEINE: https://github.com/mahmoodlab/MADELEINE
