# CellFilter: Cell Segmentation & Extraction Toolkit

![Demo](seg.gif)

## 📖 Introduction

**CellFilter** is a comprehensive repository designed for high-precision cell segmentation and mask extraction in histopathology and multiplex immunofluorescence (mIF) images. This project integrates state-of-the-art deep learning models tailored for specific modalities:

* **H&E Images:** Utilizes **[HoVer-Net](https://github.com/vqdang/hover_net)** for simultaneous segmentation and classification of nuclei in Hematoxylin & Eosin stained tissues.
* **mIF Images:** Utilizes **[StarDist](https://github.com/stardist/stardist)** for robust star-convex object detection to extract cell masks from DAPI/Fluorescence channels.

This repository provides a complete pipeline including preprocessing (patch extraction), inference, and post-processing.

---

## 📂 Repository Structure

```text
CellFilter/
├── data/                       # Data directory
├── dataloader/                 # Data loading mechanisms
├── docs/                       # Documentation
├── examples/                   # Example usage scripts
├── infer/                      # Inference core logic
├── metrics/                    # Evaluation metrics
├── models/                     # Model definitions (HoVer-Net, etc.)
├── run_utils/                  # Runtime utilities
├── test_samples/               # Sample images for testing
├── compute_stats.py            # Dataset statistics computation
├── config.py                   # Configuration file
├── convert_chkpt_tf2pytorch.py # Tool to convert HoVer-Net TF weights to PyTorch
├── dataset.py                  # Dataset class definitions
├── extract_patches.py          # Script for extracting patches from WSIs/Images
├── filter.ipynb                # ★ Tutorial for cell segmentation & image filter ★
├── filter.py                   # Cell segmentation & image filter
├── fluo_seg.py                 # StarDist segmentation script for mIF/DAPI
├── run_infer.py                # Main inference script for HoVer-Net
├── run_tile.sh                 # Shell script for tile-based processing
├── run_wsi.sh                  # Shell script for Whole Slide Image processing
└── type_info.json              # Nuclei type definitions
```

🛠️ Installation Guide
It is recommended to use Anaconda or Miniconda to manage dependencies.

1. Create Environment
```
conda create -n cellfilter python=3.8
conda activate cellfilter
```

2. Install General Dependencies
```
pip install numpy matplotlib opencv-python scikit-image pandas docopt termcolor
```

3. Install HoVer-Net Dependencies (for H&E)
This repository uses a PyTorch implementation of HoVer-Net.

# Install PyTorch
```
pip install torch torchvision # Check [https://pytorch.org/](https://pytorch.org/) for your specific CUDA version
```

Weight Conversion (Optional):
If you have original HoVer-Net weights in TensorFlow format, use the provided converter:

python convert_chkpt_tf2pytorch.py --help
4. Install StarDist (for mIF/DAPI)
StarDist is required for fluorescence image segmentation.

# Install StarDist and CSBDeep
```
pip install stardist csbdeep
```
Note: If you encounter GPU issues with StarDist, please refer to the official [StarDist](https://github.com/stardist/stardist) installation guide.

🚀 Usage
1. H&E Cell Extraction (HoVer-Net)
To extract masks from H&E images, use run_infer.py.

Standard Inference:
```
python run_infer.py \
    --gpu='0' \
    --input_dir='data/he_images/' \
    --output_dir='output/hovernet_pred/' \
    --model_path='models/hovernet_weights.pth' \
    --model_mode='fast'
```
Whole Slide Image (WSI) Processing:
For processing large .svs or .tif files, utilize the shell script:

bash run_wsi.sh
2. mIF/DAPI Nuclei Extraction (StarDist)
To extract masks from DAPI channels or fluorescence images, use fluo_seg.py.
```
python fluo_seg.py
```
2D_versatile_fluo is the default pretrained model provided by StarDist, which performs exceptionally well on DAPI nuclei.*

3. Preprocessing (Patch Extraction)
If your images are too large for direct inference, you can tile them first:
```
python extract_patches.py \
    --input_dir 'raw_data/' \
    --output_dir 'data/patches/' \
    --patch_size 1024 \
    --step_size 1024
```
⚙️ Configuration
config.py: Global configuration for training parameters and model paths.

type_info.json: JSON file mapping integer class labels to nuclei types (e.g., 1: Neoplastic, 2: Inflammatory). Modify this if using a custom-trained model with different classes.

📝 References
HoVer-Net: Graham, Simon, et al. "HoVer-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images." Medical Image Analysis (2019).

StarDist: Schmidt, Uwe, et al. "Cell detection with star-convex polygons." MICCAI (2018).
