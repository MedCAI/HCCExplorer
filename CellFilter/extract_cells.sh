# 输入路径为分批后的目录
ulimit -n 24000

BASE_DIR="/data/ceiling/workspace/HCC/patches/HE_parts"
OUTPUT_BASE_DIR="./output_tiles_Pannuke"
SLIDE_LIST_FILE="slide_list.txt"  # 列表文件，每行一个 slide 名称

# HoverNet 参数配置
GPU_ID=7
NR_TYPES=6
TYPE_INFO_PATH="type_info_pannuke.json"
MODEL_PATH="./weights/hovernet_fast_pannuke_type_tf2pytorch.tar"
MODEL_MODE="fast"
BATCH_SIZE=4
NR_INFER_WORKERS=4
NR_POST_PROC_WORKERS=8
MEM_USAGE=0.1

# 读取 slide 列表
while IFS= read -r SLIDE_NAME || [[ -n "$SLIDE_NAME" ]]; do
    SLIDE_DIR="$BASE_DIR/$SLIDE_NAME"
    if [[ ! -d "$SLIDE_DIR" ]]; then
        echo "!! Slide directory not found: $SLIDE_DIR"
        continue
    fi

    echo "==> Processing slide: $SLIDE_NAME"

    for PART_DIR in "$SLIDE_DIR"/part_*/; do
        [[ -d "$PART_DIR" ]] || continue
        PART_NAME=$(basename "$PART_DIR")
        echo "    → Running on $PART_NAME"

        OUTPUT_DIR="$OUTPUT_BASE_DIR/$SLIDE_NAME/$PART_NAME"
        mkdir -p "$OUTPUT_DIR"

        python run_infer.py \
            --gpu="$GPU_ID" \
            --nr_types="$NR_TYPES" \
            --type_info_path="$TYPE_INFO_PATH" \
            --model_path="$MODEL_PATH" \
            --model_mode="$MODEL_MODE" \
            --nr_inference_workers="$NR_INFER_WORKERS" \
            --nr_post_proc_workers="$NR_POST_PROC_WORKERS" \
            --batch_size="$BATCH_SIZE" \
            tile \
            --input_dir="$PART_DIR" \
            --output_dir="$OUTPUT_DIR" \
            --mem_usage="$MEM_USAGE"
    done
done < "$SLIDE_LIST_FILE"