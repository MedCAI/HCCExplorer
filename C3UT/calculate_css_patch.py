# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import color
from scipy.stats import pearsonr

# ------------------------------
# 可配置参数 (保持与你的 FID 脚本一致)
# ------------------------------
marker_list = ['CD19', 'CD3', 'CD4', 'CD8', 'CD68', 'Foxp3']
name_list = ["201547450.3.ome", "201419249.5.ome", "201472230-9.ome"] # 简化列表

gen_tmpl = "/data/ceiling/workspace/HCC/CUT/datasets/Generated_HE2mIF_1024_FastCUT/{name}/HE2{marker}_1024/val_lastest/images/reverse_fake_B"
real_tmpl = "/data/ceiling/workspace/HCC/CUT/datasets/reorganized_val/{name_noext}/{marker}_val"

def calculate_css_metrics(img_gen, img_real):
    """
    计算两幅图像之间的颜色相似度指标
    """
    # 1. PSNR & SSIM (基础结构与噪声比)
    # win_size 设为 7 以适应 1024x1024 或较小 patch
    score_ssim = ssim(img_gen, img_real, channel_axis=2, data_range=255)
    score_psnr = psnr(img_real, img_gen, data_range=255)

    # 2. LAB 空间色差 (Delta E)
    # 将图像转为 LAB 空间，计算欧几里得距离
    gen_lab = color.rgb2lab(img_gen)
    real_lab = color.rgb2lab(img_real)
    
    # Delta E 简化计算 (CIE76): sqrt((L1-L2)^2 + (a1-a2)^2 + (b1-b2)^2)
    delta_e = np.sqrt(np.sum((gen_lab - real_lab) ** 2, axis=2)).mean()

    # 3. 颜色相关性 (Pearson Correlation Coefficient)
    # 拉平后计算 RGB 三通道的平均相关性
    gen_flat = img_gen.flatten().astype(np.float32)
    real_flat = img_real.flatten().astype(np.float32)
    corr, _ = pearsonr(gen_flat, real_flat)

    return {
        'ssim': score_ssim,
        'psnr': score_psnr,
        'delta_e': delta_e,  # 越小越好
        'pcc': corr          # 越大越好
    }

def main():
    all_results = []

    for name in name_list:
        name_noext = name[:-4] if name.endswith('.ome') else name
        for marker in marker_list:
            folder_gen = gen_tmpl.format(name=name, marker=marker)
            folder_real = real_tmpl.format(name_noext=name_noext, marker=marker)

            if not (os.path.isdir(folder_gen) and os.path.isdir(folder_real)):
                continue

            print(f"👉 Calculating CSS: {name} - {marker}")
            
            # 获取两个文件夹中的图片交集（确保文件名匹配）
            gen_files = {os.path.basename(f): f for f in glob(os.path.join(folder_gen, "*.*"))}
            real_files = {os.path.basename(f): f for f in glob(os.path.join(folder_real, "*.*"))}
            common_names = set(gen_files.keys()).intersection(set(real_files.keys()))

            if not common_names:
                print(f"  [warn] No matching filename found in {marker}")
                continue

            metrics_accumulator = []
            
            # 遍历匹配的图片对进行计算
            for fname in tqdm(list(common_names)[:500], desc="Processing patches"): # 限制 500 张以防过慢
                img_gen = cv2.imread(gen_files[fname])
                img_real = cv2.imread(real_files[fname])
                
                # 统一尺寸（如果由于 padding 导致尺寸微差）
                if img_gen.shape != img_real.shape:
                    img_gen = cv2.resize(img_gen, (img_real.shape[1], img_real.shape[0]))
                
                img_gen = cv2.cvtColor(img_gen, cv2.COLOR_BGR2RGB)
                img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)
                
                res = calculate_css_metrics(img_gen, img_real)
                metrics_accumulator.append(res)

            # 计算该 Marker 的平均值
            if metrics_accumulator:
                df_temp = pd.DataFrame(metrics_accumulator)
                summary = df_temp.mean().to_dict()
                summary.update({'slide_id': name, 'marker': marker, 'count': len(metrics_accumulator)})
                all_results.append(summary)
                print(f"  [ok] SSIM: {summary['ssim']:.4f}, DeltaE: {summary['delta_e']:.4f}")

    # 保存结果
    if all_results:
        df_final = pd.DataFrame(all_results)
        df_final.to_csv("css_metrics_results.csv", index=False)
        print("\n✅ CSS Calculation Finished. Saved to 'css_metrics_results.csv'")

if __name__ == '__main__':
    main()