# -*- coding: utf-8 -*-
import os
import torch
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy import linalg
import warnings

# ------------------------------
# 可配置参数
# ------------------------------
marker_list = ['CD19', 'CD3', 'CD4', 'CD8', 'CD68', 'Foxp3']
name_list = [
    "201547450.3.ome",
    "201419249.5.ome",
    "201472230-9.ome",
    "201560526.5.ome",
    "201473864-5.ome",
    "201404940.2.ome",
    "201433764.3.ome",
    "201429858.3.ome",
    "201418604.2.ome",
    "201615614.4.ome",
    # '201647678.4.ome',
    # '201638904.1.ome',
    # '201638887.3.ome'
]

batch_size = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dims = 2048  # InceptionV3 feature dim

# 抽样评测开关与参数
use_sampling = True        # False 表示只计算主FID；True 表示加做抽样稳定性
K_per_wsi = 1000           # 每个文件夹（WSI）随机抽样的 patch 数
R_repeats = 5              # 抽样重复次数
seed_base = 123            # 基础随机种子

# 目录模板（保持你的原路径逻辑）
gen_tmpl = "/data/ceiling/workspace/HCC/CUT/datasets/Generated_HE2mIF_1024_FastCUT/{name}/HE2{marker}_1024/val_lastest/images/reverse_fake_B"
real_tmpl = "/data/ceiling/workspace/HCC/CUT/datasets/reorganized_val/{name_noext}/{marker}_val"

# ------------------------------
# 工具：从路径列表计算 FID（轻量版）
# 说明：pytorch-fid 的 calculate_fid_given_paths 仅支持目录路径。
# 这里提供一个从“图片路径列表”计算 FID 的函数，以便做抽样而无需复制文件。
# ------------------------------
class PathsDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.t = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        return self.t(img)

def get_inception_model(dims):
    from pytorch_fid.inception import InceptionV3
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device).eval()
    return model

def compute_activations_from_paths(paths, model, batch_size=50, num_workers=4):
    tfm = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    ds = PathsDataset(paths, tfm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    acts = []
    with torch.no_grad():
        for x in dl:
            x = x.to(device)
            pred = model(x)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = torch.nn.functional.adaptive_avg_pool2d(pred, (1,1))
            acts.append(pred.squeeze(-1).squeeze(-1).cpu().numpy())
    if len(acts) == 0:
        return np.empty((0, dims), dtype=np.float32)
    return np.concatenate(acts, axis=0)

def compute_stats(acts: np.ndarray):
    if acts.shape[0] < 2:
        # 协方差至少需要2个样本
        mu = acts.mean(axis=0) if acts.size > 0 else np.zeros((acts.shape[1] if acts.ndim==2 else dims,), dtype=np.float32)
        cov = np.eye(mu.shape[0], dtype=np.float64)
        return mu.astype(np.float64), cov
    mu = acts.mean(axis=0)
    X = acts - mu
    cov = (X.T @ X) / (acts.shape[0] - 1)
    return mu.astype(np.float64), cov.astype(np.float64)

def fid_from_stats(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        warnings.warn("fid: adding eps to cov diagonals for stability.")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))

def calculate_fid_given_pathlists(paths1, paths2, batch_size=50, dims=2048):
    model = get_inception_model(dims)
    acts1 = compute_activations_from_paths(paths1, model, batch_size=batch_size)
    acts2 = compute_activations_from_paths(paths2, model, batch_size=batch_size)
    mu1, sigma1 = compute_stats(acts1)
    mu2, sigma2 = compute_stats(acts2)
    return fid_from_stats(mu1, sigma1, mu2, sigma2)

# ------------------------------
# 路径与抽样工具
# ------------------------------
def list_images(folder):
    exts = ('.png','.jpg','.jpeg','.tif','.tiff','.bmp')
    return [p for p in glob(os.path.join(folder, '**/*'), recursive=True) if p.lower().endswith(exts)]

def sample_paths(paths, K, seed):
    rng = np.random.default_rng(seed)
    if len(paths) <= K:
        return paths
    idx = rng.choice(len(paths), size=K, replace=False)
    return [paths[i] for i in idx]

# ------------------------------
# 主流程
# ------------------------------
def calculate_all_fid_scores():
    results = []

    print(f"Using device: {device}, dims={dims}, batch_size={batch_size}")
    print("="*60)

    from pytorch_fid import fid_score  # 延后导入，避免未安装时报错

    for name in name_list:
        if name.startswith('.'):
            continue
        name_noext = name[:-4] if name.endswith('.ome') else os.path.splitext(name)[0]

        for marker in marker_list:
            folder1 = gen_tmpl.format(name=name, marker=marker)
            folder2 = real_tmpl.format(name_noext=name_noext, marker=marker)

            print(f"Processing: slide={name}, marker={marker}")

            if not os.path.isdir(folder1):
                print(f"  [warn] missing: {folder1}")
                continue
            if not os.path.isdir(folder2):
                print(f"  [warn] missing: {folder2}")
                continue

            # 主 FID（全量文件夹）
            try:
                fid_value = fid_score.calculate_fid_given_paths(
                    [folder1, folder2],
                    batch_size=batch_size,
                    device=device,
                    dims=dims
                )
                results.append({
                    'slide_id': name,
                    'marker': marker,
                    'fid_main': float(fid_value),
                })
                print(f"  [ok] main FID: {fid_value:.4f}")
            except Exception as e:
                print(f"  [error] main FID failed: {e}")
                continue

            # 抽样评测（可选）
            if use_sampling:
                try:
                    gen_paths_all = list_images(folder1)
                    real_paths_all = list_images(folder2)
                    if len(gen_paths_all) == 0 or len(real_paths_all) == 0:
                        print("  [warn] empty folder for sampling; skip sampling.")
                        results[-1].update({
                            'fid_sample_mean': np.nan,
                            'fid_sample_std': np.nan,
                            'fid_sample_ci95_lo': np.nan,
                            'fid_sample_ci95_hi': np.nan,
                            'K': 0,
                            'R': 0,
                        })
                    else:
                        fids = []
                        for r in range(R_repeats):
                            seed = seed_base + r
                            gen_paths = sample_paths(gen_paths_all, K_per_wsi, seed)
                            real_paths = sample_paths(real_paths_all, K_per_wsi, seed + 10000)
                            fid_r = calculate_fid_given_pathlists(
                                gen_paths, real_paths,
                                batch_size=batch_size, dims=dims
                            )
                            fids.append(fid_r)
                        fids = np.asarray(fids, dtype=np.float64)
                        mean_fid = fids.mean()
                        std_fid = fids.std(ddof=1) if len(fids) > 1 else 0.0
                        # 使用百分位法近似 95% CI（R 较小时仅作参考）
                        if len(fids) >= 3:
                            lo, hi = np.percentile(fids, [2.5, 97.5])
                        else:
                            # 退化情形：用 mean ± 2*std 粗略给一个区间
                            lo, hi = mean_fid - 2*std_fid, mean_fid + 2*std_fid

                        results[-1].update({
                            'fid_sample_mean': float(mean_fid),
                            'fid_sample_std': float(std_fid),
                            'fid_sample_ci95_lo': float(lo),
                            'fid_sample_ci95_hi': float(hi),
                            'K': int(min(K_per_wsi, len(gen_paths_all), len(real_paths_all))),
                            'R': int(R_repeats),
                        })
                        print(f"  [ok] sampling FID (K={K_per_wsi}, R={R_repeats}): "
                              f"{mean_fid:.4f} ± {std_fid:.4f}, 95% CI [{lo:.4f}, {hi:.4f}]")
                except Exception as e:
                    print(f"  [error] sampling failed: {e}")

    # 保存 CSV
    if len(results) == 0:
        print("No results to save.")
        return
    df = pd.DataFrame(results)
    out_csv = "fid_scores_FastCUT_with_sampling.csv" if use_sampling else "fid_scores_FastCUT.csv"
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print("\n✅ Saved:", out_csv)
    print(df.head())

if __name__ == '__main__':
    calculate_all_fid_scores()