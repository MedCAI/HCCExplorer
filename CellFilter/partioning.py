import os
import shutil
from math import ceil

SOURCE_BASE_DIR = "/data/ceiling/workspace/HCC/patches/HE"
TARGET_BASE_DIR = "/data/ceiling/workspace/HCC/patches/HE_parts"
PART_SIZE = 500  # 每批最多多少 patch 文件

def split_slide_patches(slide_dir, slide_name, target_base_dir, part_size=PART_SIZE):
    all_files = sorted([
        f for f in os.listdir(slide_dir)
        if f.endswith(".png") or f.endswith(".jpg")  # 你可以按实际 patch 类型调整
    ])
    total = len(all_files)
    parts = ceil(total / part_size)
    print(f"Splitting {total} files from {slide_name} into {parts} parts")

    for i in range(parts):
        part_dir = os.path.join(target_base_dir, slide_name, f"part_{i}")
        os.makedirs(part_dir, exist_ok=True)
        start = i * part_size
        end = min(start + part_size, total)
        for fname in all_files[start:end]:
            src = os.path.join(slide_dir, fname)
            dst = os.path.join(part_dir, fname)
            shutil.copy2(src, dst)
        print(f"  -> part_{i}: {end - start} files")

# 扫描所有 slide 文件夹
for slide_folder in os.listdir(SOURCE_BASE_DIR):
    full_slide_path = os.path.join(SOURCE_BASE_DIR, slide_folder)
    if os.path.isdir(full_slide_path):
        split_slide_patches(full_slide_path, slide_folder, TARGET_BASE_DIR)