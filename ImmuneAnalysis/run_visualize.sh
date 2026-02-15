#!/usr/bin/env bash
set -euo pipefail

# ========== Batch Sample List ==========
name_list=(
  "201416366_HE_6"
)

# Modify this variable if a specific Python interpreter is required
PY=python

# ========== Path Configuration Function ==========
# This function organizes file naming templates. 
# ${NAME} is automatically replaced for each sample in the loop.
build_paths() {
  local NAME="$1"

  TIFF_IF="/data/ceiling/workspace/HCC/CUT/results/Hepatoma_first_trial/5_year_no_recur/${NAME}.tiff"
  OUT_DIR="/data/ceiling/workspace/HCC/ImmueAnalysis/Visualization/${NAME}"
  mkdir -p "${OUT_DIR}"

  PSEUDO_PNG="${OUT_DIR}/${NAME}.png"

  CSV_MIF="/ZJU/data1/mIF/immune/Hepatoma_first_trial/5_year_no_recur/${NAME}.csv"
  CELLTYPE_PNG="${OUT_DIR}/${NAME}_celltypes.png"

  HE_XML="/data/ceiling/data/ZJU/ZJU/Hepatoma_first_trial/5_year_no_recur/${NAME}.xml"
  HE_WSI="/data/ceiling/data/ZJU/ZJU/Hepatoma_first_trial/5_year_no_recur/${NAME}.svs"
  HE_PNG="${OUT_DIR}/${NAME}_HE.png"            # Generated in Step 3.1
  TUMOR_MASK="${OUT_DIR}/${NAME}_tumor_mask.png"  # Generated in Step 3.1

  TC_MASK="${OUT_DIR}/${NAME}_tumor_mask_TC_mask.png"   # Generated in Step 3.2
  IF_MASK="${OUT_DIR}/${NAME}_tumor_mask_IF_mask.png"   # Generated in Step 3.2
  AS_MASK="${OUT_DIR}/${NAME}_tumor_mask_AS_mask.png"   # Generated in Step 3.2
  TC_OVERLAY="${OUT_DIR}/${NAME}_tumor_overlay.png"

  ATTN_ROOT="/data/jsh/ZJU/attention_map/heatmaps_results_survival/5_year_no_recur/sgformer_hcc/Hepatoma_2_5_years_800/HEATMAP_OUTPUT/II/"
  SAVE_ROOT="/data/ceiling/workspace/HCC/ImmueAnalysis/Visualization/"
}

# ========== Helper Utilities ==========
check_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "[WARN] Missing file: $f"
    return 1
  fi
  return 0
}

# ========== Main Processing Pipeline (Single Sample) ==========
run_one() {
  local NAME="$1"
  echo "=============================="
  echo "[INFO] Starting process for: ${NAME}"
  echo "=============================="

  build_paths "$NAME"

  # 1) Generate pseudocolor images from mIF TIFF files
  echo "[STEP 1] Generating pseudocolor from mIF TIFF"
  if check_file "$TIFF_IF"; then
    set -x
    $PY step1_save_mif_pseudocolor.py \
      --tiff "$TIFF_IF" \
      --out "$PSEUDO_PNG" \
      --down-ratio 32
    set +x
  else
    echo "[SKIP] TIFF not found for Step 1: $TIFF_IF"
  fi

  # 2) Generate cell-type visualization thumbnails
  echo "[STEP 2] Generating cell-type thumbnail"
  if check_file "$TIFF_IF" && check_file "$CSV_MIF"; then
    set -x
    $PY step2_make_celltype_thumbnail.py \
      --tiff "$TIFF_IF" \
      --csv "$CSV_MIF" \
      --out "$CELLTYPE_PNG"
    set +x
  else
    echo "[SKIP] Missing TIFF or CSV for Step 2:"
    echo "       TIFF_IF=$TIFF_IF"
    echo "       CSV_MIF=$CSV_MIF"
  fi

  # 3) Visualize HE and tumor regions (Outputs HE_PNG and TUMOR_MASK)
  echo "[STEP 3.1] Generating HE and tumor mask"
  if check_file "$HE_XML" && check_file "$HE_WSI"; then
    set -x
    $PY step31_tumor_mask.py \
      --xml "$HE_XML" \
      --wsi "$HE_WSI" \
      --outdir "$OUT_DIR"
    set +x
  else
    echo "[SKIP] Missing HE xml/wsi for Step 3.1:"
    echo "       HE_XML=$HE_XML"
    echo "       HE_WSI=$HE_WSI"
  fi

  # 4) Transfer attention maps
  echo "[STEP 4] Copying attention maps"
  if [[ -d "$ATTN_ROOT" && -d "$SAVE_ROOT" ]]; then
    set -x
    $PY step4_copy_attention.py \
      --attn-root "$ATTN_ROOT" \
      --save-root "$SAVE_ROOT" \
      --name "$NAME"
    set +x
  else
    echo "[SKIP] Missing attention or save root directories for Step 4:"
    echo "       ATTN_ROOT=$ATTN_ROOT"
    echo "       SAVE_ROOT=$SAVE_ROOT"
  fi

  echo "[DONE] Finished processing: ${NAME}"
}

# ========== Batch Execution ==========
# Disable 'set -e' temporarily so one sample's failure doesn't stop the entire loop
set +e
for NAME in "${name_list[@]}"; do
  run_one "$NAME" || echo "[ERROR] ${NAME} failed. Proceeding to next sample."
done
set -e

echo "[ALL DONE] Processed ${#name_list[@]} sample(s)."