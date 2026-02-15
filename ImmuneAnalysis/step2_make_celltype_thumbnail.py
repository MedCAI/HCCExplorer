#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import imageio.v3 as iio
import tifffile as tiff

# ============ 工具与推断函数 ============

def robust_minmax(series, p_low=1.0, p_high=99.5):
    lo = np.nanpercentile(series, p_low)
    hi = np.nanpercentile(series, p_high)
    if not np.isfinite(lo): lo = np.nanmin(series)
    if not np.isfinite(hi): hi = np.nanmax(series)
    if hi <= lo: hi = lo + 1.0
    x = (series - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0)

def infer_cell_types_by_markers(
    df,
    qlow=1.0, qhigh=99.5,
    win_thresh=0.28,
    af_penalty=0.28
):
    markers = ('Foxp3','CD19','CD68','CD4','CD3','CD8','SampleAF','DAPI')
    missing = [m for m in markers if m not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少列: {missing}")

    F   = robust_minmax(df['Foxp3'].astype(float).values,    p_low=qlow, p_high=qhigh)
    Bc  = robust_minmax(df['CD19'].astype(float).values,     p_low=qlow, p_high=qhigh)
    M   = robust_minmax(df['CD68'].astype(float).values,     p_low=qlow, p_high=qhigh)
    C4  = robust_minmax(df['CD4'].astype(float).values,      p_low=qlow, p_high=qhigh)
    C3  = robust_minmax(df['CD3'].astype(float).values,      p_low=qlow, p_high=qhigh)
    C8  = robust_minmax(df['CD8'].astype(float).values,      p_low=qlow, p_high=qhigh)
    AF  = robust_minmax(df['SampleAF'].astype(float).values, p_low=qlow, p_high=qhigh)
    D   = robust_minmax(df['DAPI'].astype(float).values,     p_low=qlow, p_high=qhigh)

    max_T = np.maximum.reduce([C3, C4, C8])

    s_treg = 0.65*F + 0.35*C4 - af_penalty*AF
    s_cd8  = 0.65*C8 + 0.35*C3 - af_penalty*AF
    s_cd4  = 0.55*C4 + 0.30*C3 + 0.15*(1.0 - F) - af_penalty*AF
    s_b    = 0.80*Bc + 0.20*(1.0 - max_T) - af_penalty*AF
    s_mac  = 0.80*M + 0.20*(M - np.maximum.reduce([C3, C4, C8, Bc, F]) + 0.5) - af_penalty*AF

    scores = np.stack([s_cd8, s_cd4, s_treg, s_b, s_mac], axis=1)
    labels = np.array(['CD8+ T cell','CD4+ T cell','Treg','B cell','Macrophage'])

    best_idx = np.argmax(scores, axis=1)
    best_score = scores[np.arange(len(df)), best_idx]
    pred = labels[best_idx]

    pred[best_score < win_thresh] = 'Other'
    return pred

def read_mif_shape_xyc(tiff_path):
    """
    读取 mIF 多通道 TIFF 的 (H, W, C)。
    优先选择 XY 面积最大的 series；C 轴优先识别 'C'，兼容 'S'，若无通道轴则为 1。
    """
    with tiff.TiffFile(tiff_path) as tf:
        best = None
        best_area = -1
        for s in tf.series:
            if 'Y' in s.axes and 'X' in s.axes:
                H = s.shape[s.axes.index('Y')]
                W = s.shape[s.axes.index('X')]
                area = H * W
                if area > best_area:
                    best = s
                    best_area = area
        if best is None:
            raise ValueError("TIFF 中未找到包含 X/Y 轴的 series。")

        axes = best.axes
        shape = best.shape
        H = shape[axes.index('Y')]
        W = shape[axes.index('X')]
        if 'C' in axes:
            C = shape[axes.index('C')]
        elif 'S' in axes:
            C = shape[axes.index('S')]
        else:
            C = 1
        return H, W, C

def compute_thumb_size(H, W, down_ratio=None, thumb_min_side=None):
    """根据缩放比或目标最短边，计算缩略图 (h, w)。"""
    if down_ratio is not None and down_ratio > 1:
        return (max(1, math.ceil(H / down_ratio)), max(1, math.ceil(W / down_ratio)))
    if thumb_min_side is not None and thumb_min_side > 0:
        ms = min(H, W)
        if ms <= thumb_min_side:
            return (H, W)
        r = ms / float(thumb_min_side)
        return (max(1, math.ceil(H / r)), max(1, math.ceil(W / r)))
    # 默认与原脚本一致：down=32
    return (max(1, math.ceil(H / 32)), max(1, math.ceil(W / 32)))

def make_celltype_thumbnail(
    csv_path,
    out_png,
    src_shape_xyc,                 # (H, W, C)
    thumb_size,                    # (h, w)
    fixed_cell_types=(
        'CD8+ T cell',
        'CD4+ T cell',
        'Treg',
        'B cell',
        'Macrophage',
        'Other',
    ),
    point_mode="dot",              # "dot" or "area"
    dot_radius_px=1.0,
    area_radius_clamp=(0.5, 4.0),
    bg="black",
    alpha=1.0,
    qlow=1.0,
    qhigh=99.5,
    win_thresh=0.28,
    af_penalty=0.28
):
    type2color = {
        'CD8+ T cell': (0.90, 0.10, 0.10, 1.0),
        'CD4+ T cell': (0.10, 0.70, 0.10, 1.0),
        'Treg':        (0.85, 0.00, 0.85, 1.0),
        'B cell':      (0.00, 0.70, 0.90, 1.0),
        'Macrophage':  (1.00, 0.60, 0.00, 1.0),
        'Other':       (0.60, 0.60, 0.60, 1.0),
    }

    h, w = thumb_size
    H, W = src_shape_xyc[0], src_shape_xyc[1]

    canvas = np.ones((h, w, 4), dtype=np.float32) if bg == "white" else np.zeros((h, w, 4), dtype=np.float32)

    df = pd.read_csv(csv_path, sep=None, engine='python')
    df.columns = [str(c).strip() for c in df.columns]

    # 坐标缩放
    for col in ('x', 'y'):
        if col not in df.columns:
            raise ValueError(f"CSV 缺少坐标列 '{col}'")
    scale_x = w / float(W)
    scale_y = h / float(H)
    x_s = np.clip(np.rint(df['x'].values * scale_x).astype(np.int32), 0, w - 1)
    y_s = np.clip(np.rint(df['y'].values * scale_y).astype(np.int32), 0, h - 1)

    # 半径
    if point_mode == "area" and 'nucleus_area' in df.columns:
        area = np.maximum(df['nucleus_area'].values.astype(np.float64), 1.0)
        r_src = np.sqrt(area / np.pi)
        avg_scale = 0.5 * (scale_x + scale_y)
        r = np.clip(r_src * avg_scale, area_radius_clamp[0], area_radius_clamp[1]).astype(np.float32)
    else:
        r = np.full(len(df), float(dot_radius_px), dtype=np.float32)

    # 细胞类型推断
    df['cell_type'] = infer_cell_types_by_markers(
        df, qlow=qlow, qhigh=qhigh, win_thresh=win_thresh, af_penalty=af_penalty
    )

    # 绘制
    for ct in fixed_cell_types:
        idx = np.where(df['cell_type'].values == ct)[0]
        if idx.size == 0:
            continue

        base_rgba = np.array(type2color[ct], dtype=np.float32)
        color_rgba = base_rgba.copy()
        color_rgba[3] = alpha

        xs = x_s[idx]
        ys = y_s[idx]
        rs = r[idx]

        for x0, y0, rad in zip(xs, ys, rs):
            rr = int(np.ceil(rad))
            y_min = max(0, y0 - rr)
            y_max = min(h - 1, y0 + rr)
            x_min = max(0, x0 - rr)
            x_max = min(w - 1, x0 + rr)
            if y_min > y_max or x_min > x_max:
                continue

            yy, xx = np.ogrid[y_min:y_max + 1, x_min:x_max + 1]
            mask = (xx - x0) * (xx - x0) + (yy - y0) * (yy - y0) <= rad * rad
            sub = canvas[y_min:y_max + 1, x_min:x_max + 1]

            a = color_rgba[3]
            sub[mask, :3] = color_rgba[:3] * a + sub[mask, :3] * (1.0 - a)
            sub[mask, 3] = a + sub[mask, 3] * (1.0 - a)

    rgb = np.clip(canvas[:, :, :3], 0, 1)
    Path(os.path.dirname(out_png) or ".").mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_png, (rgb * 255).astype(np.uint8))

    counts = pd.Series(df['cell_type']).value_counts().reindex(list(fixed_cell_types), fill_value=0)
    return {
        "thumb_size": tuple(thumb_size),
        "cells": int(len(df)),
        "counts": {k: int(v) for k, v in counts.to_dict().items()},
        "out": os.path.abspath(out_png)
    }

# ============ 命令行入口 ============

def parse_args():
    p = argparse.ArgumentParser(
        description="Render cell-type thumbnail from per-cell CSV using marker-based inference and mIF TIFF size."
    )
    p.add_argument("--tiff", "-i", required=True, help="输入 mIF TIFF 路径，用于读取原图 H,W")
    p.add_argument("--csv", "-c", required=True, help="输入每细胞 CSV（需含 x,y 以及各 marker 列）")
    p.add_argument("--out", "-o", required=True, help="输出 PNG 路径")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--down-ratio", type=float, default=32.0, help="缩小比例（最短边≈原短边/ratio，默认 32）")
    g.add_argument("--thumb-min-side", type=int, default=None, help="目标最短边像素，与 --down-ratio 互斥")

    p.add_argument("--point-mode", choices=["dot", "area"], default="area", help="绘制点样式")
    p.add_argument("--dot-radius", type=float, default=1.0, help="point-mode=dot 时的像素半径")
    p.add_argument("--area-radius-min", type=float, default=0.5, help="point-mode=area 半径下限")
    p.add_argument("--area-radius-max", type=float, default=2.0, help="point-mode=area 半径上限")
    p.add_argument("--bg", choices=["black", "white"], default="black", help="背景颜色")
    p.add_argument("--alpha", type=float, default=1.0, help="点/圆的透明度 0~1")

    # 推断参数
    p.add_argument("--qlow", type=float, default=1.0, help="robust_minmax 低百分位")
    p.add_argument("--qhigh", type=float, default=99.5, help="robust_minmax 高百分位")
    p.add_argument("--win-thresh", type=float, default=0.28, help="winner 阈值，低于则标记为 Other")
    p.add_argument("--af-penalty", type=float, default=0.28, help="自发荧光惩罚权重")

    p.add_argument("--quiet", action="store_true", help="减少日志输出")

    return p.parse_args()

def main():
    args = parse_args()

    if not os.path.isfile(args.tiff):
        print(f"[error] TIFF 不存在: {args.tiff}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.csv):
        print(f"[error] CSV 不存在: {args.csv}", file=sys.stderr)
        sys.exit(1)

    try:
        H, W, C = read_mif_shape_xyc(args.tiff)
    except Exception as e:
        print(f"[error] 读取 TIFF 失败: {e}", file=sys.stderr)
        sys.exit(2)

    h, w = compute_thumb_size(
        H, W,
        down_ratio=None if args.thumb_min_side is not None else args.down_ratio,
        thumb_min_side=args.thumb_min_side
    )

    if not args.quiet:
        print(f"[info] source HW=({H},{W}), computed thumb=({h},{w})")

    try:
        res = make_celltype_thumbnail(
            csv_path=args.csv,
            out_png=args.out,
            src_shape_xyc=(H, W, C),
            thumb_size=(h, w),
            point_mode=args.point_mode,
            dot_radius_px=args.dot_radius,
            area_radius_clamp=(args.area_radius_min, args.area_radius_max),
            bg=args.bg,
            alpha=args.alpha,
            qlow=args.qlow,
            qhigh=args.qhigh,
            win_thresh=args.win_thresh,
            af_penalty=args.af_penalty
        )
    except Exception as e:
        print(f"[error] 生成缩略图失败: {e}", file=sys.stderr)
        sys.exit(3)

    if not args.quiet:
        print("[done]")
        for k, v in res.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()