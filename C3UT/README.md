# C3UT: Cell-map-guided H&E-to-mIF image translation

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**C3UT** is a deep learning framework designed for high-resolution histopathology image to multiplex Immunofluorescence image translation. It provides a complete pipeline for Whole Slide Image (WSI) processing, generative modeling (GANs), mIF image synthesis, and clinical quality evaluation.

## 📂 Repository Structure

```text
C3UT/
├── datasets/                 # Data storage (Raw slides & Patches)
├── experiments/              # experiment-related codes
├── models/                   # Neural network architectures
├── options/                  # Configuration (Train/Test options)
├── util/                     # Utility functions
├── weights/                  # Model weights
├── docs/                     # Documentation files
├── imgs/                     # Sample images
├── he2patches.py             # Preprocessing: Crop WSI into patches
├── patch2slide.py            # Postprocessing: Stitch patches into WSI
├── tiff2pyramid.py           # Convert stitched images to Pyramid TIFF
├── train.py                  # Main training script
├── modified_test.py          # Main inference script
├── calculate_fid_patch.py    # Metric: FID Score calculation
├── calculate_css_patch.py    # Metric: CSS Score calculation
├── turing_test_app.py        # Evaluation: Visual Turing Test App
├── run_pipeline.sh           # Shell script for end-to-end execution
├── requirements.txt          # Python dependencies
└── c3ut_tutorial.ipynb       # ★ Jupyter Notebook tutorial ★ 
```

## 🚀 Getting Started

### Prerequisites

- Linux or macOS
- Python 3.8+
- NVIDIA GPU + CUDA

### Installation

```
# Clone the repository
git clone [https://github.com/MedCAI/c3ut.git](https://github.com/MedCAI/c3ut.git)
cd c3ut

# Install dependencies
pip install -r requirements.txt
```

## 🛠️ Data Preparation

1. **Foreground Extraction:** Use [Trident](https://github.com/mahmoodlab/TRIDENT) or [CLAM](https://github.com/mahmoodlab/CLAM) framework to extract the foreground regions into ".h5" file.
1. **Preprocessing (WSI to Patches):** Use `he2patches.py` to crop tissue into patches.

   ```
   python he2patches.py --wsi_path ./datasets/raw_slide.svs --h5_path ./datasets/raw_slide.h5 --save_path ./raw_slide/
   ```


## 🏃‍♂️ Usage

### 1. Training

Train the model using `train.py`. Configure hyperparameters in `options/` or via command line.

```
python train.py --dataroot ./datasets/patches --name c3ut_experiment --model [c3ut | pix2pix | etc]
```

### 2. Inference

Generate synthetic pathology images.

```
python modified_test.py --dataroot ./datasets/test_patches --name c3ut_experiment --model [c3ut | pix2pix | etc]
```

*Note: For specialized testing logic, refer to `modified_test.py`.*

### 3. Post-Processing

reconstruct the generated patches back into a Whole Slide Image (WSI) for visualization.

```
# Stitch patches
python patch2slide.py --input_dir ./results/c3ut_experiment --output_dir ./results/slides

# Convert to Pyramid TIFF (for viewing in ASAP/QuPath)
python tiff2pyramid.py --input_dir ./results/slides
```

### 4. Automated Pipeline

Run the full inference workflow (Preprocess -> Inference --> Post-Processing) using the provided shell script:

```
bash run_pipeline.sh
```

## 📊 Evaluation

We provide tools for both quantitative and qualitative evaluation.

- **FID Score (Fréchet Inception Distance):**

  ```
  python calculate_fid_patch.py --real_dir datasets/real --fake_dir results/fake
  ```

- **CSS Score (Content-Style Similarity):**

  ```
  python calculate_css_patch.py
  ```

- **Visual Turing Test:** Launch the interactive app to evaluate the realism of generated images with pathologists.

  ```
  streamlit run turing_test_app.py
  ```
*Note: You need to install streamlit library and prepare your data in the following format:*
```text
doctor_evaluation/
├── datasets/             # Data storage (Real H&E, Real mIF, and Fake mIF patches)
├── ├── CD3               # CD3 marker dir
├── ├── ├── fake          # fake mIF and real H&E pairs
├── ├── ├── real          # real mIF and real H&E pairs
├── ├── CD4               # CD4 marker dir
├── ├── ├── fake          # fake mIF and real H&E pairs
├── ├── ├── real          # real mIF and real H&E pairs
├── ├── CD19
├── ├── ...
├── results/              # Evaluation results
├── turing_test_app.py/   # turing test web app
```

## 📖 Tutorial

For a detailed walkthrough, including inference samples, translation evaluation, turing test, run the Jupyter Notebook:

```
jupyter notebook c3ut_tutorial.ipynb
```

## 🤝 Contributing
Thanks to the following work for improving our project：
- CLAM: [https://github.com/mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM)
- CUT: [https://github.com/taesungp/contrastive-unpaired-translation/](https://github.com/taesungp/contrastive-unpaired-translation/)
- Trident: [https://github.com/mahmoodlab/TRIDENT](https://github.com/mahmoodlab/TRIDENT)
- 
## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE&authuser=2) file for details.
