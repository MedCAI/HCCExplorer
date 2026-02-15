import os
import numpy as np
import cv2
from tqdm import tqdm
import tifffile
import argparse
import openslide


def parse_filename(filename):
    name = os.path.splitext(filename)[0]
    x_str, y_str = name.split('_')
    return int(x_str), int(y_str)


def load_patch(path):
    patch = cv2.imread(path, cv2.IMREAD_COLOR)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    patch = cv2.resize(255 - patch, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)  # 反转像素
    return patch


def combine_multichannel_patches_tilewise(
    patch_dirs,
    wsi_width,
    wsi_height,
    save_path,
    tile_size=2048,
    patch_size=512,
    overlap=128
):
    num_channels = 8

    # 1️⃣ 构建 patch 映射
    patch_map = {}  # {(x, y): {dir_path: patch_path}}
    for dir_path in patch_dirs:
        for f in os.listdir(dir_path):
            if not f.endswith(('.png', '.jpg', '.tif')):
                continue
            if 'checkpoint' in f:
                continue
            x, y = parse_filename(f)
            key = (x, y)
            if key not in patch_map:
                patch_map[key] = {}
            patch_map[key][dir_path] = os.path.join(dir_path, f)

    # 2️⃣ 初始化空图
    big_image = np.zeros((wsi_height, wsi_width, num_channels), dtype=np.uint8)

    # 3️⃣ 按 Tile 分块拼接（加入 Overlap）
    for ty in tqdm(range(0, wsi_height, tile_size), desc="Tile rows"):
        for tx in range(0, wsi_width, tile_size):
            # 扩展 Tile 的范围，加入 Overlap
            tile_x_start = max(0, tx - overlap)
            tile_x_end = min(wsi_width, tx + tile_size + overlap)
            tile_y_start = max(0, ty - overlap)
            tile_y_end = min(wsi_height, ty + tile_size + overlap)

            # 扩展范围的尺寸
            tile_h_ext = tile_y_end - tile_y_start
            tile_w_ext = tile_x_end - tile_x_start

            # 初始化扩展范围的 tile_sum 和 tile_weight
            tile_sum_ext = np.zeros((tile_h_ext, tile_w_ext, num_channels), dtype=np.float32)
            tile_weight_ext = np.zeros((tile_h_ext, tile_w_ext, num_channels), dtype=np.float32)

            # 遍历补丁，判断是否在扩展范围内
            for (px, py), patch_dict in patch_map.items():
                # 检查补丁左上角是否在扩展范围内
                if not (tile_x_start <= px < tile_x_end and tile_y_start <= py < tile_y_end):
                    continue

                # 计算补丁在扩展范围中的相对位置
                dx_ext = px - tile_x_start
                dy_ext = py - tile_y_start
                h = w = patch_size

                # 如果补丁超出扩展范围，裁剪尺寸
                if dy_ext + h > tile_h_ext:
                    h = tile_h_ext - dy_ext
                if dx_ext + w > tile_w_ext:
                    w = tile_w_ext - dx_ext

                # 加载补丁到所有通道
                patches = [None] * num_channels
                for step, dir_path in enumerate(patch_dirs):
                    if dir_path not in patch_dict:
                        continue
                    patch = load_patch(patch_dict[dir_path]).astype(np.float32)

                    if 'Foxp3' in dir_path:
                        patches[0] = patch[:h, :w, 0]  # DAPI
                        patches[1] = patch[:h, :w, 1]  # Foxp3
                        patches[7] = patch[:h, :w, 2]  # SampleAF
                    else:
                        patch_green = patch[:h, :w, 1]
                        ch_index = step + 1  # CD19=1, CD68=2...
                        patches[ch_index] = patch_green

                # 更新扩展范围的 tile_sum 和 tile_weight
                for ch in range(num_channels):
                    if patches[ch] is not None:
                        tile_sum_ext[dy_ext:dy_ext+h, dx_ext:dx_ext+w, ch] += patches[ch]
                        tile_weight_ext[dy_ext:dy_ext+h, dx_ext:dx_ext+w, ch] += 1.0

            # 计算扩展范围内的平均值
            tile_weight_ext[tile_weight_ext == 0] = 1.0
            tile_avg_ext = (tile_sum_ext / tile_weight_ext).astype(np.uint8)

            # 提取非重叠区域并写入大图
            tile_x_start_real = overlap if tx > 0 else 0
            tile_y_start_real = overlap if ty > 0 else 0
            tile_x_end_real = tile_x_start_real + tile_size
            tile_y_end_real = tile_y_start_real + tile_size

            # 确保边界不超出 big_image 的范围
            big_image_y_end = min(ty + tile_size, wsi_height)
            big_image_x_end = min(tx + tile_size, wsi_width)

            big_image[ty:big_image_y_end, tx:big_image_x_end, :] = tile_avg_ext[
                tile_y_start_real:tile_y_start_real + (big_image_y_end - ty),
                tile_x_start_real:tile_x_start_real + (big_image_x_end - tx),
                :
            ]

    # 4️⃣ 保存结果
    tifffile.imwrite(
        save_path,
        big_image,
        photometric='minisblack',
        bigtiff=True,
        metadata=None,
        planarconfig='contig',
        compression='deflate',
    )
    print(f"✅ Multi-channel image saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multi-channel patches into a whole slide image.")
    parser.add_argument("--wsi_path", type=str, required=True, help="Path to the whole slide image (WSI).")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the combined output image.")

    args = parser.parse_args()

    # Load WSI dimensions
    slide = openslide.OpenSlide(args.wsi_path)
    wsi_width, wsi_height = slide.dimensions

    # Define patch directories
    patch_dirs = [
        './convert_images/HE2Foxp3_1024/val_7/images/fake_B/',
        './convert_images/HE2CD19_1024/val_7/images/fake_B/',
        './convert_images/HE2CD68_1024/val_7/images/fake_B/',
        './convert_images/HE2CD4_1024/val_7/images/fake_B/',
        './convert_images/HE2CD3_1024/val_7/images/fake_B/',
        './convert_images/HE2CD8_1024/val_7/images/fake_B/',
    ]

    combine_multichannel_patches_tilewise(
        patch_dirs=patch_dirs,
        wsi_width=wsi_width,
        wsi_height=wsi_height,
        save_path=args.save_path,
        tile_size=4096,
        patch_size=1024,
        overlap=1024  # 设置 Overlap
    )