#!/bin/bash

# 源目录和目标目录
INPUT_DIR="/data/ceiling/workspace/HCC/Registration/IF/"
OUTPUT_DIR="/data/ceiling/workspace/HCC/Registration/deal_IF/"
mkdir -p "$OUTPUT_DIR"

# 缩略图最大尺寸
SIZE=2048

# 批量处理
for FILE in "$INPUT_DIR"/*.tif "$INPUT_DIR"/*.tiff; do
  [ -e "$FILE" ] || continue  # 跳过无匹配文件
  FILENAME=$(basename "$FILE")
  BASENAME="${FILENAME%.*}"
  OUTFILE="$OUTPUT_DIR/${BASENAME}_thumb.jpg"

  if [ -f "$OUTFILE" ]; then
    echo "⚠️ 缩略图已存在，跳过: $OUTFILE"
    continue
  fi

  echo "🔧 正在处理: $FILENAME"
  vips thumbnail "$FILE" "$OUTFILE" $SIZE
  echo "✅ 完成: $OUTFILE"
done