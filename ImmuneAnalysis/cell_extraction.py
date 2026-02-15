import os
import tifffile
import numpy as np
import pandas as pd
from skimage import filters, measure, morphology
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from tqdm import tqdm
import gc

# ===================== 参数 =====================
# image_dir = '/data/ceiling/workspace/HCC/CUT/results/Hepatoma_first_trial/recur_alive_2_years'
image_dir = '/data/ceiling/workspace/HCC/CUT/results/Hepatoma_first_trial/dead_within_2_years/new'

# save_dir = '/ZJU/data1/mIF/immune/Hepatoma_first_trial/recur_alive_2_years'
save_dir = '/ZJU/data1/mIF/immune/Hepatoma_first_trial/dead_within_2_years/new'
os.makedirs(save_dir, exist_ok=True)

# 通道名称，顺序必须与TIFF文件匹配
channel_names = ['DAPI', 'Foxp3', 'CD19', 'CD68', 'CD4', 'CD3', 'CD8', 'SampleAF']

# 阈值（用于分类，不会保存到CSV）
marker_thresholds = {
    'Foxp3': 13,
    'CD19': 13,
    'CD68': 13,
    'CD4': 13,
    'CD3': 13,
    'CD8': 13
}

patch_size = 256
stride = 256
min_cell_size = 20
# ===================== 表型分类函数 =====================
def get_cell_type(row):
    if row['CD3'] > marker_thresholds['CD3'] and row['CD8'] > marker_thresholds['CD8'] and row['CD4'] <= marker_thresholds['CD4']:
        return 'CD8+ T cell'
    elif row['CD3'] > marker_thresholds['CD3'] and row['CD4'] > marker_thresholds['CD4'] and row['CD8'] <= marker_thresholds['CD8']:
        return 'CD4+ T cell'
    elif row['CD3'] > marker_thresholds['CD3'] and row['Foxp3'] > marker_thresholds['Foxp3'] and row['CD4'] > marker_thresholds['CD4']:
        return 'Treg'
    elif row['CD19'] > marker_thresholds['CD19'] and row['CD3'] <= marker_thresholds['CD3']:
        return 'B cell'
    elif row['CD68'] > marker_thresholds['CD68']:
        return 'Macrophage'
    else:
        return 'Other'

# ================== 遍历文件 ==================
tiff_files = [f for f in os.listdir(image_dir) if f.endswith('.tiff') or f.endswith('.tif')]

for file_name in tqdm(tiff_files, desc='Processing files'):
    try:
        if 'new' in file_name:
            continue
        file_path = os.path.join(image_dir, file_name)
        save_path = os.path.join(save_dir, file_name.replace('.tiff', '.csv').replace('ome', ''))
        print("目标将保存:", save_path)

        if os.path.exists(save_path):
            print(f"已存在: {file_name}")
            continue

        all_cells = []

        # === 读取 tiff 文件 ===
        with tifffile.TiffFile(file_path) as tif:
            memmap_array = tif.asarray(out='memmap')  # (H, W, 8)

        if memmap_array.ndim != 3 or memmap_array.shape[2] != len(channel_names):
            print(f"Skip {file_name}: unexpected shape {memmap_array.shape}")
            continue

        H, W, C = memmap_array.shape

        # === 遍历所有 patch ===
        for y in tqdm(range(0, H - patch_size + 1, stride)):
            for x in range(0, W - patch_size + 1, stride):
                patch = np.transpose(memmap_array[y:y+patch_size, x:x+patch_size, :], (2, 0, 1))  # (8, ps, ps)
                dapi = patch[0]

                if np.mean(dapi) < 5:
                    continue

                try:
                    thresh = filters.threshold_otsu(dapi)
                except Exception:
                    continue  # 全黑 patch 跳过

                # 核分割
                mask = dapi > thresh
                mask = morphology.remove_small_objects(mask, min_size=16)  # 去掉很小的核像素块

                # 重新 label，并过滤整块小面积细胞
                labeled = measure.label(mask)
                labeled = morphology.remove_small_objects(labeled, min_size=min_cell_size)

                props = measure.regionprops(labeled)
                if len(props) == 0:
                    continue

                for i, prop in enumerate(props):
                    cy, cx = prop.centroid
                    means = [np.mean(patch[ch][labeled == prop.label]) for ch in range(8)]
                    variances = [np.var(patch[ch][labeled == prop.label]) for ch in range(8)]
                    cell_info = {
                        'cell_id': f'{y}_{x}_{i+1}',
                        'x': int(cx + x),
                        'y': int(cy + y),
                        'nucleus_area': prop.area,
                        **{ch: means[idx] for idx, ch in enumerate(channel_names)},
                        **{f'{ch}_var': variances[idx] for idx, ch in enumerate(channel_names)},
                    }
                    all_cells.append(cell_info)

        # === 保存结果 ===
        if len(all_cells) == 0:
            print(f"No cells found in {file_name}")
            continue

        df = pd.DataFrame(all_cells)
        df['cell_type'] = df.apply(get_cell_type, axis=1)
        df.to_csv(save_path, index=False)
        print(f'保存完成: {save_path}, shape: {df.shape}')

        # 清理
        del memmap_array
        gc.collect()

    except Exception as e:
        print(f"Error processing {file_name}: {e}")