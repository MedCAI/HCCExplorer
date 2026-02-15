#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import shutil

def copy_attention_maps(
    attn_root: str,
    save_root: str,
    name: str,
    mask_suffix: str = "_attenblock.png",
    overlay_suffix: str = "_blockmap.png",
    mask_out_suffix: str = "_attn_mask.png",
    overlay_out_suffix: str = "_attn_overlay.png",
):
    # 输入文件路径
    attn_mask_path = os.path.join(attn_root, name, name + mask_suffix)
    attn_overlay_path = os.path.join(attn_root, name, name + overlay_suffix)

    # 输出文件路径
    out_dir = os.path.join(save_root, name)
    os.makedirs(out_dir, exist_ok=True)
    save_mask_path = os.path.join(out_dir, name + mask_out_suffix)
    save_overlay_path = os.path.join(out_dir, name + overlay_out_suffix)

    # 复制
    copied = {"mask": False, "overlay": False}

    if os.path.exists(attn_mask_path):
        shutil.copy(attn_mask_path, save_mask_path)
        print(f"[ok] 复制成功: {attn_mask_path} -> {save_mask_path}")
        copied["mask"] = True
    else:
        print(f"[warn] 文件不存在: {attn_mask_path}")

    if os.path.exists(attn_overlay_path):
        shutil.copy(attn_overlay_path, save_overlay_path)
        print(f"[ok] 复制成功: {attn_overlay_path} -> {save_overlay_path}")
        copied["overlay"] = True
    else:
        print(f"[warn] 文件不存在: {attn_overlay_path}")

    return {
        "name": name,
        "attn_mask_src": attn_mask_path,
        "attn_overlay_src": attn_overlay_path,
        "mask_dst": save_mask_path,
        "overlay_dst": save_overlay_path,
        "copied": copied
    }

def parse_args():
    p = argparse.ArgumentParser(description="Copy attention heatmap files into a Visualization folder.")
    p.add_argument("--attn-root", "-i", required=True, help="注意力图根目录（包含每个样本子目录）")
    p.add_argument("--save-root", "-o", required=True, help="输出根目录（如 ./Visualization）")
    p.add_argument("--name", "-n", required=True, help="样本名称（子目录名与文件前缀）")

    # 可选：自定义输入/输出文件后缀
    p.add_argument("--mask-suffix", default="_attenblock.png", help="输入 mask 文件后缀（默认 _attenblock.png）")
    p.add_argument("--overlay-suffix", default="_blockmap.png", help="输入 overlay 文件后缀（默认 _blockmap.png）")
    p.add_argument("--mask-out-suffix", default="_attn_mask.png", help="输出 mask 文件后缀（默认 _attn_mask.png）")
    p.add_argument("--overlay-out-suffix", default="_attn_overlay.png", help="输出 overlay 文件后缀（默认 _attn_overlay.png）")

    p.add_argument("--quiet", action="store_true", help="减少日志输出")
    return p.parse_args()

def main():
    args = parse_args()

    if not os.path.isdir(args.attn_root):
        print(f"[error] 输入根目录不存在: {args.attn_root}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.save_root, exist_ok=True)

    info = copy_attention_maps(
        attn_root=args.attn_root,
        save_root=args.save_root,
        name=args.name,
        mask_suffix=args.mask_suffix,
        overlay_suffix=args.overlay_suffix,
        mask_out_suffix=args.mask_out_suffix,
        overlay_out_suffix=args.overlay_out_suffix,
    )

    if info["copied"]["mask"] or info["copied"]["overlay"]:
        if not args.quiet:
            print("[done]")
            for k, v in info.items():
                print(f"  {k}: {v}")
        sys.exit(0)
    else:
        print("[warn] 未复制任何文件（请检查路径与后缀设置）")
        sys.exit(2)

if __name__ == "__main__":
    main()