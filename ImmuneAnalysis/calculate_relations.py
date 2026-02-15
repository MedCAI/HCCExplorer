#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import cv2
import math
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, List

# 依赖：pip install openslide-python opencv-python numpy pandas scipy
try:
    import openslide
    from openslide import OpenSlide
except Exception as e:
    raise ImportError("openslide-python is required. Install via: pip install openslide-python") from e

# -------------------------
# 默认参数
# -------------------------
DEFAULT_PATCH_ID_REGEX = r'^(\d+)_(\d+)$'
DEFAULT_PATCH_SIZE = 256
DEFAULT_PATCH_STRIDE = 256
DEFAULT_ATTN_REDUCTION = ['mean', 'median', 'max', 'p90']
DEFAULT_REG_SEARCH_SCALES = [8, 16, 32, 64, 128]
DEFAULT_REG_DOWNSAMPLE_FOR_MASK = 32
DEFAULT_REG_MAX_SHIFT_FRAC = 0.15

# 汇总项（由参数覆盖）
ATTN_REDUCTION: List[str] = []


# -------------------------
# 工具函数
# -------------------------
def read_wsi(wsi_path: str) -> OpenSlide:
    return openslide.OpenSlide(wsi_path)

def get_wsi_dims(slide: OpenSlide) -> Dict:
    return {'level_dims': slide.level_dimensions, 'downsamples': slide.level_downsamples}

def imread_gray_any(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return img

def read_attn_gray(attn_path: str) -> np.ndarray:
    img = imread_gray_any(attn_path)
    return img.astype(np.float32) / 255.0

def make_tissue_mask_from_wsi(slide: OpenSlide, downsample: int = 32) -> Tuple[np.ndarray, float]:
    best_level = int(np.argmin([abs(float(d) - float(downsample)) for d in slide.level_downsamples]))
    img = np.array(slide.read_region((0, 0), best_level, slide.level_dimensions[best_level]).convert('RGB'))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    _, th = cv2.threshold(255 - L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.medianBlur(th, 5)
    kernel = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = (th > 0).astype(np.uint8)
    level_scale = float(slide.level_downsamples[best_level])
    return mask, level_scale

def extract_contours(mask: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(canvas, cnts, -1, 255, thickness=2)
    return canvas

def build_distance_transform(binary: np.ndarray) -> np.ndarray:
    bin_u8 = (binary > 0).astype(np.uint8)
    inv = 1 - bin_u8
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    if dist.max() > 0:
        dist = dist / dist.max()
    return dist

def auto_register_attention_to_wsi(slide: OpenSlide,
                                   attn_img: np.ndarray,
                                   reg_search_scales: List[float],
                                   reg_max_shift_frac: float,
                                   reg_downsample_for_mask: int) -> Dict:
    tissue_mask_low, wsi_low_scale = make_tissue_mask_from_wsi(slide, downsample=reg_downsample_for_mask)
    tissue_dist = build_distance_transform(tissue_mask_low)

    attn_u8 = (attn_img * 255).astype(np.uint8)
    # 使用 Otsu，若太稀疏则用 P60
    otsu_thr, otsu_bin = cv2.threshold(attn_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(otsu_bin) < 0.001 * otsu_bin.size:
        thr = max(8, int(np.percentile(attn_u8, 60)))
        _, attn_bin = cv2.threshold(attn_u8, thr, 255, cv2.THRESH_BINARY)
    else:
        attn_bin = otsu_bin
    attn_edge = extract_contours(attn_bin)

    H_low, W_low = tissue_mask_low.shape
    Ha, Wa = attn_edge.shape

    best = {'score': -1e9, 'scale': None, 'offset_xy': None}

    ys, xs = np.where(attn_edge > 0)
    edge_points = np.stack([xs, ys], axis=1).astype(np.float32) if xs.size > 0 else np.empty((0, 2), np.float32)
    edge_is_dense = False
    if edge_points.shape[0] < 50:
        edge_points = np.stack(np.meshgrid(np.arange(Wa), np.arange(Ha)), axis=-1).reshape(-1, 2).astype(np.float32)
        edge_is_dense = True

    for s in reg_search_scales:
        s_low = float(s) / float(wsi_low_scale)
        attn_w_low = Wa * s_low
        attn_h_low = Ha * s_low
        cx_wsi = W_low / 2
        cy_wsi = H_low / 2
        cx_attn = attn_w_low / 2
        cy_attn = attn_h_low / 2
        base_tx = cx_wsi - cx_attn
        base_ty = cy_wsi - cy_attn

        max_shift = float(reg_max_shift_frac) * max(W_low, H_low)
        shifts = [
            (base_tx, base_ty),
            (base_tx - max_shift, base_ty),
            (base_tx + max_shift, base_ty),
            (base_tx, base_ty - max_shift),
            (base_tx, base_ty + max_shift),
        ]

        for (tx, ty) in shifts:
            pts = edge_points.copy()
            pts[:, 0] = pts[:, 0] * s_low + tx
            pts[:, 1] = pts[:, 1] * s_low + ty
            mask_in = (pts[:, 0] >= 0) & (pts[:, 0] < W_low) & (pts[:, 1] >= 0) & (pts[:, 1] < H_low)
            if mask_in.sum() < 100:
                continue
            pts = pts[mask_in].astype(np.int32)
            if edge_is_dense:
                pts = pts[::50]
            dvals = tissue_dist[pts[:, 1], pts[:, 0]]
            score = -float(dvals.mean())
            if score > best['score']:
                best = {
                    'score': score,
                    'scale': float(s),
                    'offset_xy': (float(tx * wsi_low_scale), float(ty * wsi_low_scale))
                }

    if best['scale'] is None:
        raise RuntimeError('Registration failed. Provide --known-scale and --known-offset to skip registration.')

    return best

def parse_patch_id(pid: str, regex: str) -> Optional[Tuple[int, int]]:
    m = re.match(regex, str(pid))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def attn_stats_on_patch(attn_img: np.ndarray,
                        A_scale: float, A_offset: Tuple[float, float],
                        patch_xy_lvl0: Tuple[int, int],
                        patch_size: int,
                        reductions: List[str]) -> Dict[str, float]:
    x0, y0 = patch_xy_lvl0
    x1, y1 = x0 + patch_size, y0 + patch_size

    ax0 = (x0 - A_offset[0]) / A_scale
    ay0 = (y0 - A_offset[1]) / A_scale
    ax1 = (x1 - A_offset[0]) / A_scale
    ay1 = (y1 - A_offset[1]) / A_scale

    H, W = attn_img.shape
    ax0i = int(math.floor(min(ax0, ax1)))
    ay0i = int(math.floor(min(ay0, ay1)))
    ax1i = int(math.ceil(max(ax0, ax1)))
    ay1i = int(math.ceil(max(ay0, ay1)))

    if ax1i <= 0 or ay1i <= 0 or ax0i >= W or ay0i >= H:
        return {f"attn_{k}": np.nan for k in reductions}

    ax0i = max(0, ax0i); ay0i = max(0, ay0i)
    ax1i = min(W, ax1i); ay1i = min(H, ay1i)

    roi = attn_img[ay0i:ay1i, ax0i:ax1i]
    if roi.size == 0:
        return {f"attn_{k}": np.nan for k in reductions}

    vals = roi.reshape(-1)
    out: Dict[str, float] = {}
    if 'mean' in reductions:
        out['attn_mean'] = float(np.mean(vals))
    if 'median' in reductions:
        out['attn_median'] = float(np.median(vals))
    if 'max' in reductions:
        out['attn_max'] = float(np.max(vals))
    if 'p90' in reductions:
        out['attn_p90'] = float(np.percentile(vals, 90))
    return out

def attach_attention_to_features(features_df: pd.DataFrame,
                                 attn_img: np.ndarray,
                                 A_scale: float, A_offset: Tuple[float, float],
                                 patch_size: int,
                                 patch_id_regex: str,
                                 reductions: List[str],
                                 patch_id_as_index: bool,
                                 patch_stride: int) -> pd.DataFrame:
    coords = features_df['patch_id'].apply(lambda s: parse_patch_id(s, patch_id_regex))
    if coords.isnull().any():
        bad = features_df.loc[coords.isnull(), 'patch_id'].unique()[:10]
        raise ValueError(f'Found unparseable patch_id examples: {bad[:5]} ... Check --patch-id-regex.')

    xs: List[int] = []
    ys: List[int] = []
    for c in coords:
        x, y = c
        if patch_id_as_index:
            x = x * patch_stride
            y = y * patch_stride
        xs.append(x); ys.append(y)

    attn_stats_list = []
    for x, y in zip(xs, ys):
        stats = attn_stats_on_patch(attn_img, A_scale, A_offset, (x, y), patch_size, reductions)
        attn_stats_list.append(stats)
    attn_df = pd.DataFrame(attn_stats_list)
    out = pd.concat([features_df.reset_index(drop=True), attn_df], axis=1)
    return out

def compute_correlations(df: pd.DataFrame,
                         attn_col: str = 'attn_mean') -> pd.DataFrame:
    num_df = df.select_dtypes(include=[np.number]).copy()
    if attn_col not in num_df.columns:
        raise ValueError(f'Attention column {attn_col} not found.')
    targets = [c for c in num_df.columns if c != attn_col]
    from scipy.stats import spearmanr, pearsonr
    rows = []
    for c in targets:
        s = num_df[c]
        valid = ~(s.isna() | num_df[attn_col].isna())
        if valid.sum() < 10:
            rho_s, p_s = np.nan, np.nan
            r_p, p_p = np.nan, np.nan
        else:
            rho_s, p_s = spearmanr(s[valid], num_df.loc[valid, attn_col])
            r_p, p_p = pearsonr(s[valid], num_df.loc[valid, attn_col])
        rows.append({'feature': c, 'spearman_r': rho_s, 'spearman_p': p_s,
                     'pearson_r': r_p, 'pearson_p': p_p})
    return pd.DataFrame(rows).sort_values(by='spearman_r', ascending=False)


# -------------------------
# 参数解析与主流程
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Link attention heatmap to per-patch features and compute correlations.")
    # 四个必需路径
    p.add_argument('--wsi', required=True, type=str, help='Path to WSI file (.svs/.tif/...).')
    p.add_argument('--attn', required=True, type=str, help='Path to attention heatmap image (png/jpg).')
    p.add_argument('--graph-feat', required=True, type=str, help='Path to patch_graph_features.csv.')
    p.add_argument('--patch-feat', required=True, type=str, help='Path to patch_features.csv.')
    # 输出与 patch 参数
    p.add_argument('--outdir', type=str, default='./attn_feature_linkage', help='Output directory.')
    p.add_argument('--patch-size', type=int, default=DEFAULT_PATCH_SIZE, help='Patch size (level-0 pixels).')
    p.add_argument('--patch-stride', type=int, default=DEFAULT_PATCH_STRIDE, help='Patch stride (level-0 pixels).')
    p.add_argument('--patch-id-regex', type=str, default=DEFAULT_PATCH_ID_REGEX, help='Regex to parse patch_id -> x_y.')
    p.add_argument('--patch-id-as-index', action='store_true',
                   help='Interpret patch_id as tile indices (ix_iy). Then pixel coord = ix*stride, iy*stride.')
    # 已知映射（可选，跳过自动配准）
    p.add_argument('--known-scale', type=float, default=None, help='Known mapping: WSI pixels per attention pixel.')
    p.add_argument('--known-offset', type=float, nargs=2, metavar=('OX', 'OY'), default=None,
                   help='Known mapping offset (x0, y0) in WSI level-0.')
    # 自动配准参数（仅在未提供已知映射时使用）
    p.add_argument('--reg-search-scales', type=float, nargs='+', default=DEFAULT_REG_SEARCH_SCALES,
                   help='Candidate scales (WSI px per attn px).')
    p.add_argument('--reg-max-shift-frac', type=float, default=DEFAULT_REG_MAX_SHIFT_FRAC,
                   help='Max shift as fraction of max(image dimension) at low-res.')
    p.add_argument('--reg-downsample-for-mask', type=int, default=DEFAULT_REG_DOWNSAMPLE_FOR_MASK,
                   help='Downsample factor to build tissue mask.')
    # attention 汇总统计
    p.add_argument('--attn-reductions', type=str, nargs='+', default=DEFAULT_ATTN_REDUCTION,
                   choices=['mean', 'median', 'max', 'p90'], help='Aggregations over attention within patch window.')
    # 简易偏移微调（调试映射用）
    p.add_argument('--offset-shift-x', type=float, default=0.0, help='Extra shift added to offset x (WSI px).')
    p.add_argument('--offset-shift-y', type=float, default=0.0, help='Extra shift added to offset y (WSI px).')
    # 调试：打印前若干个 patch 投影窗口
    p.add_argument('--debug-print', action='store_true', help='Print projection info for first 10 patches.')
    return p.parse_args()

def main():
    global ATTN_REDUCTION
    args = parse_args()
    ATTN_REDUCTION = list(args.attn_reductions)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取数据
    print('Loading WSI...')
    slide = read_wsi(args.wsi)
    meta = get_wsi_dims(slide)
    print(f'WSI level-0 size: {meta["level_dims"][0]}')

    print('Loading attention heatmap...')
    attn = read_attn_gray(args.attn)
    print(f'Attention image size: {attn.shape[1]}x{attn.shape[0]}')

    # 估计或使用已知映射
    if args.known_scale is not None and args.known_offset is not None:
        A_scale = float(args.known_scale)
        A_offset = (float(args.known_offset[0]), float(args.known_offset[1]))
        print(f'Using known mapping: scale={A_scale}, offset={A_offset}')
    else:
        print('Estimating attention-to-WSI mapping...')
        reg = auto_register_attention_to_wsi(
            slide, attn,
            reg_search_scales=list(map(float, args.reg_search_scales)),
            reg_max_shift_frac=float(args.reg_max_shift_frac),
            reg_downsample_for_mask=int(args.reg_downsample_for_mask),
        )
        A_scale = reg['scale']
        A_offset = reg['offset_xy']
        print(f'Estimated mapping: scale={A_scale:.3f} (wsi px per attn px), offset={A_offset}')

    # 允许微调偏移（用于避免全 NaN 的情况）
    if args.offset_shift_x != 0.0 or args.offset_shift_y != 0.0:
        A_offset = (A_offset[0] + args.offset_shift_x, A_offset[1] + args.offset_shift_y)
        print(f'Adjusted offset by ({args.offset_shift_x}, {args.offset_shift_y}) -> {A_offset}')

    # 读取特征表
    print('Loading feature tables...')
    df_graph = pd.read_csv(args.graph_feat)
    df_patch = pd.read_csv(args.patch_feat)

    if 'patch_id' not in df_graph.columns or 'patch_id' not in df_patch.columns:
        raise KeyError('Both CSVs must contain a patch_id column.')

    # 调试打印：前 10 个 patch 的投影窗口是否与注意力图相交
    if args.debug_print:
        def project_window(pid_str):
            m = re.match(args.patch_id_regex, str(pid_str))
            if not m:
                return None
            x = int(m.group(1)); y = int(m.group(2))
            if args.patch_id_as_index:
                x = x * args.patch_stride
                y = y * args.patch_stride
            x0,y0 = x,y; x1,y1 = x+int(args.patch_size), y+int(args.patch_size)
            ax0 = (x0 - A_offset[0]) / A_scale
            ay0 = (y0 - A_offset[1]) / A_scale
            ax1 = (x1 - A_offset[0]) / A_scale
            ay1 = (y1 - A_offset[1]) / A_scale
            H,W = attn.shape
            ax0i = int(math.floor(min(ax0, ax1))); ay0i = int(math.floor(min(ay0, ay1)))
            ax1i = int(math.ceil(max(ax0, ax1)));  ay1i = int(math.ceil(max(ay0, ay1)))
            in_x = not (ax1i <= 0 or ax0i >= W)
            in_y = not (ay1i <= 0 or ay0i >= H)
            return dict(pid=str(pid_str), ax0i=ax0i, ay0i=ay0i, ax1i=ax1i, ay1i=ay1i, W=W, H=H, overlap=bool(in_x and in_y))
        sample_patches = df_graph['patch_id'].astype(str).head(10).tolist()
        print('Debug projection of first 10 patches:')
        for sp in sample_patches:
            print(project_window(sp))

    # 拼接注意力统计
    print('Attaching attention to patch_graph_features...')
    df_graph_attn = attach_attention_to_features(
        df_graph, attn, A_scale, A_offset, int(args.patch_size),
        args.patch_id_regex, ATTN_REDUCTION, args.patch_id_as_index, int(args.patch_stride)
    )
    print('Attaching attention to patch_features...')
    df_patch_attn = attach_attention_to_features(
        df_patch, attn, A_scale, A_offset, int(args.patch_size),
        args.patch_id_regex, ATTN_REDUCTION, args.patch_id_as_index, int(args.patch_stride)
    )

    # 保存拼接结果
    out_dir.mkdir(parents=True, exist_ok=True)
    out_graph_csv = Path(args.outdir) / 'patch_graph_features_with_attention.csv'
    out_patch_csv = Path(args.outdir) / 'patch_features_with_attention.csv'
    df_graph_attn.to_csv(out_graph_csv, index=False)
    df_patch_attn.to_csv(out_patch_csv, index=False)
    print(f'Saved: {out_graph_csv}')
    print(f'Saved: {out_patch_csv}')

    # 计算相关性
    print('Computing correlations (Spearman/Pearson) with attn_mean...')
    if 'attn_mean' not in df_graph_attn.columns or 'attn_mean' not in df_patch_attn.columns:
        print('Warning: attn_mean not available; correlations will be skipped.')
    else:
        corr_graph = compute_correlations(df_graph_attn, attn_col='attn_mean')
        corr_patch = compute_correlations(df_patch_attn, attn_col='attn_mean')
        corr_graph_csv = Path(args.outdir) / 'correlations_patch_graph_features_vs_attention.csv'
        corr_patch_csv = Path(args.outdir) / 'correlations_patch_features_vs_attention.csv'
        corr_graph.to_csv(corr_graph_csv, index=False)
        corr_patch.to_csv(corr_patch_csv, index=False)
        print(f'Saved: {corr_graph_csv}')
        print(f'Saved: {corr_patch_csv}')
        print('Top correlated (graph features):')
        print(corr_graph.head(10))
        print('Top correlated (patch features):')
        print(corr_patch.head(10))

if __name__ == '__main__':
    main()