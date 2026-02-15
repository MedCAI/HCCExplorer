# 🦠 HCCExplorer: Tumor immune microenvironment analysis

This repository contains the computational pipeline used to quantitatively characterize the Tumor Immune Microenvironment (TIME) in Hepatocellular Carcinoma (HCC). 

This workflow integrates virtual multiplex immunofluorescence (mIF) generation, graph-based spatial feature extraction, deep learning attention analysis (via HCCExplorer), and transcriptomic deconvolution using CIBERSORTx.

## 📂 Repository Structure

### Core Execution Scripts
* **`run_feature.sh`**: Bash script to execute the immune feature extraction pipeline.
* **`run_visualize.sh`**: Bash script to execute the visualization and attention mapping modules.
* **`Tutorial.ipynb`**: A Jupyter notebook demonstrating the workflow and data analysis steps.

### Source Code: Feature Extraction
* **`cell_extraction.py`**: Performs cell segmentation (using StarDist) and cell phenotyping based on marker expression (CD3, CD4, CD8, CD19, CD68, Foxp3).
* **`cell_feature.py`**: Calculates morphological properties and expression statistics for individual cells.
* **`graph_feature.py`**: Constructs patch-level graphs to compute topological metrics (centrality, modularity, connectivity).
* **`calculate_relations.py`**: Quantifies cellular interactions, assortativity, and neighborhood analysis.

### Source Code: Visualization & Attention Analysis
* **`step1_save_mif_pseudocolor.py`**: Generates high-resolution heatmaps and pseudocolor images of the virtual mIF signals.
* **`step2_make_celltype_thumbnail.py`**: Creates visual thumbnails for specific cell types to verify classification.
* **`step3_tumor_mask.py`**: Applies masks to filter valid tissue regions from background/noise.
* **`step4_copy_attention.py`**: Extracts and maps attention weights from the HCCExplorer Graph Transformer to specific tissue patches.

### Data
* **`ImmuneAnalysis:CIBERSORTx_Job1_Results.csv`**: Resulting cell-type fraction data derived from CIBERSORTx deconvolution on TCGA-LIHC bulk RNA-seq data.

---

## 🚀 Pipeline Overview

### 1. Immune Feature Extraction (mIF)
**Goal:** Transform raw mIF images into 244 high-dimensional, interpretable spatial metrics.

* **Cell Detection:** Uses [StarDist](https://github.com/stardist/stardist) for nuclear segmentation via the DAPI channel.
* **Phenotyping:** Classifies cells into lineages (CD4+ T, CD8+ T, Tregs, DN T cells, Macrophages, B cells) based on protein marker intensity.
* **Feature Engineering:**
    1.  **Composition:** Cell densities, relative fractions, and clinical ratios (e.g., CD8-to-suppressive ratio).
    2.  **Graph Topology:** Constructs cell-graphs to measure spatial organization (Degree, Betweenness, Louvain community modularity).
    3.  **Interactions:** Assortativity coefficients and edge fractions to quantify cellular crosstalk.

### 2. Attention-Based Analysis
**Goal:** Link deep learning interpretability to biological features.

* **Attention Extraction:** Extracts attention weights from the HCCExplorer graph transformer.
* **Correlation Analysis:** Correlates high-attention regions (top 10%) with quantitative immune features to identify morphological drivers of prognosis.
* **Survival Analysis:** Aggregates patch-level features to slide-level representations for Cox proportional hazards regression (Overall Survival & Disease-Free Survival).

### 3. Transcriptomic Validation (CIBERSORTx)
**Goal:** Refine cellular subtypes using bulk RNA-seq data (TCGA-LIHC).

* **Deconvolution:** Uses [CIBERSORTx](https://cibersortx.stanford.edu/) with the LM22 signature matrix to infer cell fractions.
* **Integration:** Combines mIF-derived spatial counts with RNA-seq derived proportions to estimate absolute abundances of specific functional states (e.g., Macrophage M0, M1, M2 subtypes).

---

## 🛠️ Usage

### Prerequisites
Ensure you have the necessary Python environment set up. Key dependencies likely include:
* `numpy`, `pandas`, `scikit-image`
* `stardist`, `tensorflow` or `pytorch`
* `networkx` (for graph features)
* `matplotlib`, `seaborn`

### Running the Pipeline

1.  **Feature Extraction:**
    Run the shell script to process images and generate CSV features.
    ```bash
    bash run_feature.sh
    ```

2.  **Visualization & Attention Mapping:**
    Run the visualization steps to generate heatmaps and map attention weights.
    ```bash
    bash run_visualize.sh
    ```
    *This script sequentially executes steps 1 through 4 (pseudocolor generation -> thumbnails -> masking -> attention mapping).*

3.  **Data Analysis:**
    Open `Tutorial.ipynb` to view the downstream analysis, including correlation plots and survival analysis integration.

---

## 📊 Output Description

The pipeline outputs quantitative tables (CSV) and visual maps:
* **Patch-level Features:** 244 metrics covering density, intensity stats, graph topology, and cellular interactions.
* **Slide-level Heatmaps:** Visualizations mapping immune markers (e.g., Macrophage density) back to the WSI grid, transparently overlaid on valid tissue.
* **Deconvolution Results:** `ImmuneAnalysis:CIBERSORTx_Job1_Results.csv` contains the estimated cell-type fractions for the analyzed cohort.

---
*For questions or issues, please open a GitHub Issue.*