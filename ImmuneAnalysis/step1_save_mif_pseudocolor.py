#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import tifffile as tiff
import imageio.v3 as iio

# =========================
# CONFIG: 颜色/百分位/增益
# =========================

channel_names = ['DAPI', 'Foxp3', 'CD19', 'CD68', 'CD4', 'CD3', 'CD8', 'SampleAF']

PSEUDO_COLOR_MAP = {
    'DAPI':     (0.10, 0.10, 1.00, 0.15),
    'Foxp3':    (0.85, 0.25, 0.85, 0.55),
    'CD19':     (0.05, 0.90, 0.70, 0.70),
    'CD68':     (1.00, 0.50, 0.15, 0.85),
    'CD4':      (0.15, 0.95, 0.20, 0.85),
    'CD3':      (0.95, 0.15, 0.55, 0.65),
    'CD8':      (0.85, 0.10, 0.75, 0.60),
    'SampleAF': (0.70, 0.95, 0.20, 0.08),
}

CHANNEL_PERCENTILES = {
    'DAPI':     (3, 99.0),
    'Foxp3':    (2, 99.4),
    'CD19':     (2, 99.4),
    'CD68':     (3, 99.8),
    'CD4':      (2, 99.8),
    'CD3':      (2, 99.6),
    'CD8':      (2, 99.6),
    'SampleAF': (20, 99.9),
}

GAIN = {
    'DAPI':     0.6,
    'Foxp3':    0.85,
    'CD19':     1.00,
    'CD68':     1.10,
    'CD4':      1.15,
    'CD3':      0.95,
    'CD8':      0.90,
    'SampleAF': 0.30,
}

POSTPROC = dict(
    clip_quantile=0.999,
    gain=0.78,
    gamma=1.12,
    contrast=1.05
)

# =========================
# 工具函数
# =========================

def percentile_norm(arr, p_low=1, p_high=99):
    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    x = (arr.astype(np.float32) - lo) / (hi - lo)
    return np.clip(x, 0, 1)

def choose_largest_xy_series(tif: tiff.TiffFile):
    best = None
    best_area = -1
    for s in tif.series:
        if 'Y' in s.axes and 'X' in s.axes:
            H = s.shape[s.axes.index('Y')]
            W = s.shape[s.axes.index('X')]
            area = H * W
            if area > best_area:
                best = s
                best_area = area
    if best is None:
        raise ValueError("No series with X/Y axes found.")
    return best

def compute_step_from_ratio_or_size(H, W, down_ratio=None, thumb_min_side=None):
    ms = min(H, W)
    if down_ratio is not None:
        r = float(down_ratio)
        if r <= 1:
            return 1
        return max(1, int(round(r)))
    if thumb_min_side is None or thumb_min_side <= 0:
        return 1
    if ms <= thumb_min_side:
        return 1
    return max(1, int(np.floor(ms / thumb_min_side)))

# =========================
# 主函数（仅保存 mIF，重叠简单平均）
# =========================

def save_pseudocolor_rgb_memmap(
    tiff_path,
    out_png,
    down_ratio=None,          # 比例：最短边≈原短边/ratio（推荐）
    thumb_min_side=None,      # 兼容：目标最短边像素数（down_ratio 未设时使用）
    normalize="percentile",
    p_low=1,
    p_high=99,
    tile_size=None,
    overlap=0.5
):
    Path(os.path.dirname(out_png) or ".").mkdir(parents=True, exist_ok=True)

    with tiff.TiffFile(tiff_path) as tif:
        s = choose_largest_xy_series(tif)
        axes = s.axes
        shape = s.shape
        print(f"[info] series axes={axes}, shape={shape}")

        arr = s.asarray(out='memmap')

        yi = axes.index('Y'); xi = axes.index('X')
        if 'C' in axes:
            ch_axis_label = 'C'
        elif 'S' in axes:
            ch_axis_label = 'S'
        else:
            ch_axis_label = None

        H = shape[yi]; W = shape[xi]
        step = compute_step_from_ratio_or_size(H, W, down_ratio=down_ratio, thumb_min_side=thumb_min_side)
        print(f"[info] image HW=({H},{W}), down_ratio={down_ratio}, step={step}")

        h_thumb = (H + step - 1)//step
        w_thumb = (W + step - 1)//step
        sumR = np.zeros((h_thumb, w_thumb), dtype=np.float32)
        sumG = np.zeros((h_thumb, w_thumb), dtype=np.float32)
        sumB = np.zeros((h_thumb, w_thumb), dtype=np.float32)
        sumW = np.zeros((h_thumb, w_thumb), dtype=np.float32)

        if tile_size is None:
            tile = max(1024, 8 * step)
        else:
            tile = int(tile_size)
        stride = max(step, int(tile * (1.0 - overlap)))
        print(f"[info] tile={tile}, stride={stride}, overlap={overlap:.2f}")

        for y0 in range(0, H, stride):
            y1 = min(H, y0 + tile)
            for x0 in range(0, W, stride):
                x1 = min(W, x0 + tile)

                indexer = []
                axes_sub = []
                for ax, dim in zip(axes, shape):
                    if ax == 'Y':
                        indexer.append(slice(y0, y1, step)); axes_sub.append('Y')
                    elif ax == 'X':
                        indexer.append(slice(x0, x1, step)); axes_sub.append('X')
                    elif ax == ch_axis_label:
                        indexer.append(slice(None)); axes_sub.append('C')
                    else:
                        indexer.append(0)

                tile_arr = arr[tuple(indexer)]
                cur = ''.join(axes_sub)
                if 'C' not in cur:
                    tile_arr = np.expand_dims(tile_arr, 0); cur = 'C'+cur
                perm = [cur.index('C'), cur.index('Y'), cur.index('X')]
                chw = np.transpose(tile_arr, perm).astype(np.float32)
                C, th, tw = chw.shape
                if th == 0 or tw == 0:
                    continue

                Rt = np.zeros((th, tw), dtype=np.float32)
                Gt = np.zeros((th, tw), dtype=np.float32)
                Bt = np.zeros((th, tw), dtype=np.float32)

                used = min(C, len(channel_names))
                for c in range(used):
                    name = channel_names[c]
                    r, g, b, wgt = PSEUDO_COLOR_MAP.get(name, (1.0, 1.0, 1.0, 0.0))
                    if wgt <= 0:
                        continue
                    ch = np.asarray(chw[c])
                    pl, ph = CHANNEL_PERCENTILES.get(name, (p_low, p_high))
                    if normalize == "percentile":
                        chn = percentile_norm(ch, pl, ph)
                    else:
                        ch = ch.astype(np.float32)
                        lo, hi = float(ch.min()), float(ch.max())
                        if not np.isfinite(hi) or hi <= lo:
                            hi = lo + 1.0
                        chn = np.clip((ch - lo) / (hi - lo), 0, 1)

                    gk = GAIN.get(name, 1.0)
                    Rt += chn * (r * wgt) * gk
                    Gt += chn * (g * wgt) * gk
                    Bt += chn * (b * wgt) * gk

                ty0 = y0 // step; tx0 = x0 // step
                ty1 = ty0 + th;     tx1 = tx0 + tw

                # 简单平均融合
                sumR[ty0:ty1, tx0:tx1] += Rt
                sumG[ty0:ty1, tx0:tx1] += Gt
                sumB[ty0:ty1, tx0:tx1] += Bt
                sumW[ty0:ty1, tx0:tx1] += 1.0

        RGB = np.stack([sumR, sumG, sumB], axis=-1)
        WN = (sumW[..., None] + 1e-8)
        RGB = np.clip(RGB / WN, 0, 1)

        # 暗色调后处理
        q = POSTPROC.get('clip_quantile', 0.999)
        if q is not None and 0 < q < 1:
            for c in range(3):
                qq = np.quantile(RGB[..., c], q)
                if qq > 0:
                    RGB[..., c] = np.clip(RGB[..., c] / qq, 0, 1)

        gain = POSTPROC.get('gain', 1.0)
        if gain is not None and gain != 1.0:
            RGB = np.clip(RGB * float(gain), 0, 1)

        gamma = POSTPROC.get('gamma', 1.0)
        if gamma is not None and gamma != 1.0:
            RGB = np.power(RGB, float(gamma))

        contrast = POSTPROC.get('contrast', 1.0)
        if contrast is not None and contrast != 1.0:
            RGB = np.clip((RGB - 0.5) * float(contrast) + 0.5, 0, 1)

        iio.imwrite(out_png, (RGB * 255).astype(np.uint8))

        return {
            "axes": axes, "shape": shape,
            "step": step, "down_ratio": down_ratio,
            "thumb_size": (RGB.shape[0], RGB.shape[1]),
            "tile": tile, "stride": stride,
            "channels_used": used,
            "out_png": os.path.abspath(out_png),
        }

# =========================
# 命令行入口
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Render pseudo-colored RGB thumbnail from multiplex TIFF with tiling and percentile normalization."
    )
    parser.add_argument("--tiff", "-i", required=True, help="输入 TIFF 路径")
    parser.add_argument("--out", "-o", required=True, help="输出 PNG 路径")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--down-ratio", type=float, default=32, help="缩小比例（最短边≈原短边/ratio）")
    group.add_argument("--thumb-min-side", type=int, default=None, help="目标最短边像素（与 --down-ratio 互斥）")
    parser.add_argument("--normalize", choices=["percentile", "minmax"], default="percentile", help="通道归一化方式")
    parser.add_argument("--p-low", type=float, default=1.0, help="默认低百分位（通道未在 CHANNEL_PERCENTILES 中时）")
    parser.add_argument("--p-high", type=float, default=99.0, help="默认高百分位（通道未在 CHANNEL_PERCENTILES 中时）")
    parser.add_argument("--tile-size", type=int, default=None, help="tile 大小（默认 max(1024, 8*step)）")
    parser.add_argument("--overlap", type=float, default=0.5, help="tile 重叠比 0~0.9")
    parser.add_argument("--quiet", action="store_true", help="减少打印输出")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.isfile(args.tiff):
        print(f"[error] TIFF 文件不存在: {args.tiff}", file=sys.stderr)
        sys.exit(1)

    if args.overlap < 0 or args.overlap >= 1:
        print(f"[error] --overlap 必须在 [0, 1) 范围内", file=sys.stderr)
        sys.exit(1)

    try:
        res = save_pseudocolor_rgb_memmap(
            tiff_path=args.tiff,
            out_png=args.out,
            down_ratio=args.down_ratio,
            thumb_min_side=args.thumb_min_side,
            normalize=args.normalize,
            p_low=args.p_low,
            p_high=args.p_high,
            tile_size=args.tile_size,
            overlap=args.overlap
        )
    except Exception as e:
        print(f"[error] 处理失败: {e}", file=sys.stderr)
        sys.exit(2)

    if not args.quiet:
        print("[done]")
        for k, v in res.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()