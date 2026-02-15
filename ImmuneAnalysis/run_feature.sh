#!/usr/bin/env bash
set -euo pipefail

# ========== Fixed Sample List ==========
NAMES=(
"201232958_HE_5"
"201473849_HE_3")

# Modify this variable if a specific Python interpreter is required
PYTHON_BIN="python"

# ========== Helper Functions ==========
check_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "ERROR: File not found: $f"
    exit 1
  fi
}

check_dir_or_make() {
  local d="$1"
  if [[ ! -d "$d" ]]; then
    echo "Creating directory: $d"
    mkdir -p "$d"
  fi
}

run_one_sample() {
  local NAME="$1"
  echo "=============================="
  echo "[INFO] Processing NAME=${NAME}"
  echo "=============================="

  # Path templates (configured based on your directory structure)
  local CSV_RAW="/ZJU/data1/mIF/immune/Hepatoma_first_trial/dead_within_2_years/${NAME}.csv"
  local WSI_SVS="/data/ceiling/data/ZJU/ZJU/Hepatoma_first_trial/dead_within_2_years/${NAME}.svs"
  local ATTN_IMG="/data/jsh/ZJU/attention_map/heatmaps_results_survival/dead_within_2_years/sgformer_hcc/HEATMAP_OUTPUT/II/${NAME}/${NAME}_attenblock.png"
  local OUT_DIR="/data/ceiling/workspace/HCC/ImmueAnalysis/Visualization/${NAME}"
  local CELLS_RETYPED="${OUT_DIR}/cells_retyped.csv"
  local PATCH_GRAPH_FEAT="${OUT_DIR}/patch_graph_features.csv"
  local PATCH_FEAT="${OUT_DIR}/patch_features.csv"
  local TIFF_IMG="/data/ceiling/workspace/HCC/CUT/results/Hepatoma_first_trial/dead_within_2_years/${NAME}.tiff"
  local CELLTYPE_PNG="${OUT_DIR}/${NAME}_celltypes.png"

  echo "[INFO] Checking inputs..."
  check_file "$CSV_RAW"
  check_file "$WSI_SVS"
  check_file "$ATTN_IMG"
  check_dir_or_make "$OUT_DIR"

  # STEP 1: Cell feature extraction and strict classification
  echo "[STEP 1] Cell feature extraction (strict typing)..."
  set -x
  $PYTHON_BIN cell_feature.py \
    --csv "$CSV_RAW" \
    --outdir "$OUT_DIR"
  set +x
  echo "[STEP 1] Done."

  if [[ ! -f "$CELLS_RETYPED" ]]; then
    echo "ERROR: Expected cells_retyped.csv not found at $CELLS_RETYPED"
    exit 1
  fi

  # STEP 2: Graph-based feature extraction
  echo "[STEP 2] Graph feature extraction..."
  check_file "$CELLS_RETYPED"
  set -x
  $PYTHON_BIN graph_feature.py \
    --input "$CELLS_RETYPED" \
    --outdir "$OUT_DIR"
  set +x
  echo "[STEP 2] Done."

  # STEP 3: Attention-feature correlation analysis
  echo "[STEP 3] Attention-feature correlation..."
  check_file "$PATCH_GRAPH_FEAT"
  check_file "$PATCH_FEAT"
  set -x
  $PYTHON_BIN calculate_relations.py \
    --wsi "$WSI_SVS" \
    --attn "$ATTN_IMG" \
    --graph-feat "$PATCH_GRAPH_FEAT" \
    --patch-feat "$PATCH_FEAT" \
    --outdir "$OUT_DIR"
  set +x
  echo "[STEP 3] Done."

  # STEP 4: Generate cell-type visualization thumbnails
  echo "[STEP 4] Make cell-type thumbnail..."
  check_file "$TIFF_IMG"
  check_file "$CELLS_RETYPED"
  set -x
  $PYTHON_BIN step2_make_celltype_thumbnail.py \
    --tiff "$TIFF_IMG" \
    --csv "$CELLS_RETYPED" \
    --out "$CELLTYPE_PNG"
  set +x
  echo "[STEP 4] Done."

  echo "[DONE] ${NAME} finished. Outputs located in: $OUT_DIR"
}

# ========== Main Loop ==========
for NAME in "${NAMES[@]}"; do
  run_one_sample "$NAME"
done

echo "[ALL DONE] Processed ${#NAMES[@]} sample(s)."