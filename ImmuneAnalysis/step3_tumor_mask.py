#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import cv2
import openslide
from PIL import Image

def tumor_mask(mask_path, slide_path, save_path, im_show=False, level=2, poly_thickness=5):
    """
    - 从 XML 读取肿瘤多边形（level0 坐标）
    - 在指定金字塔层（默认 level=2）生成二值 mask PNG（前景=255、背景=0）
    - 保存该层的 HE 缩略图（不带多边形）与带多边形的可视化 PNG
    """
    slide_id = os.path.splitext(os.path.basename(slide_path))[0]
    os.makedirs(save_path, exist_ok=True)

    # ---------- Step 1: 打开WSI ----------
    slide = openslide.OpenSlide(slide_path)

    # 兜底：如果传入 level 超出范围，改为最大 level
    max_level = slide.level_count - 1
    level = min(max(0, int(level)), max_level)

    # 各层尺寸注意顺序是 (width, height)
    w_lv, h_lv = slide.level_dimensions[level]
    w0, h0 = slide.level_dimensions[0]

    # ---------- Step 2: 解析XML ----------
    tree = ET.parse(mask_path)
    root = tree.getroot()

    # 支持常见的格式：<Annotation><Coordinates><Coordinate X=".." Y=".."/></Coordinates></Annotation>
    coords = []
    for ann in root.findall(".//Annotation/Coordinates/Coordinate"):
        x = float(ann.attrib['X'])
        y = float(ann.attrib['Y'])
        coords.append([x, y])

    if len(coords) == 0:
        # 兼容 Aperio XML: <Annotation><Regions><Region><Vertices><Vertex X=".." Y=".."/></Vertices></Region></Regions></Annotation>
        for v in root.findall(".//Annotation/Regions/Region/Vertices/Vertex"):
            x = float(v.attrib['X'])
            y = float(v.attrib['Y'])
            coords.append([x, y])

    if len(coords) == 0:
        raise ValueError("XML 未解析到任何多边形坐标（未找到 Coordinates/Coordinate 或 Vertices/Vertex 节点）。")

    coords = np.array(coords, dtype=np.float64)

    # ---------- Step 3: 坐标映射到目标 level ----------
    # XML 为 level0 坐标，映射到指定 level
    # 注意：OpenSlide 的 level_dimensions 是整数，计算缩放时用浮点更稳健
    scale_x = w_lv / float(w0)
    scale_y = h_lv / float(h0)
    coords_level = np.stack([coords[:, 0] * scale_x, coords[:, 1] * scale_y], axis=1).astype(np.int32)

    # ---------- Step 4: 生成目标 level 的二值 mask ----------
    # mask 尺寸与该层缩略图一致，单通道 uint8：前景255，背景0
    mask_level = np.zeros((h_lv, w_lv), dtype=np.uint8)
    cv2.fillPoly(mask_level, [coords_level], 255)

    # ---------- Step 5: HE 缩略图与多边形可视化 ----------
    he_thumb = np.array(slide.read_region((0, 0), level, (w_lv, h_lv)).convert("RGB"))
    overlay = he_thumb.copy()
    cv2.polylines(overlay, [coords_level], isClosed=True, color=(255, 0, 0), thickness=int(poly_thickness))

    if im_show:
        # 延迟导入以避免无显示环境报错
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.axis("off")
        plt.show()

    # ---------- Step 6: 保存 ----------
    he_png = os.path.join(save_path, f"{slide_id}_HE.png")
    overlay_png = os.path.join(save_path, f"{slide_id}_tumor_polylines.png")
    mask_png = os.path.join(save_path, f"{slide_id}_tumor_mask.png")

    Image.fromarray(he_thumb).save(he_png)
    Image.fromarray(overlay).save(overlay_png)
    Image.fromarray(mask_level).save(mask_png)

    return {
        "slide_id": slide_id,
        "level_used": level,
        "level_size_wh": (int(w_lv), int(h_lv)),
        "level0_size_wh": (int(w0), int(h0)),
        "he_thumb_png": os.path.abspath(he_png),
        "overlay_png": os.path.abspath(overlay_png),
        "mask_png": os.path.abspath(mask_png),
    }

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate tumor mask and visualization from XML annotations on whole-slide images."
    )
    p.add_argument("--xml", "-m", required=True, help="肿瘤多边形 XML 路径")
    p.add_argument("--wsi", "-i", required=True, help="WSI 路径（如 .svs/.tif 等）")
    p.add_argument("--outdir", "-o", required=True, help="输出目录")
    p.add_argument("--level", "-l", type=int, default=2, help="目标金字塔层（默认 2）")
    p.add_argument("--show", action="store_true", help="显示可视化（需要图形界面）")
    p.add_argument("--thickness", type=int, default=5, help="多边形轮廓绘制厚度（像素）")
    p.add_argument("--quiet", action="store_true", help="减少日志输出")
    return p.parse_args()

def main():
    args = parse_args()

    if not os.path.isfile(args.xml):
        print(f"[error] XML 不存在: {args.xml}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.wsi):
        print(f"[error] WSI 不存在: {args.wsi}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.outdir, exist_ok=True)

    try:
        info = tumor_mask(
            mask_path=args.xml,
            slide_path=args.wsi,
            save_path=args.outdir,
            im_show=args.show,
            level=args.level,
            poly_thickness=args.thickness
        )
    except Exception as e:
        print(f"[error] 处理失败: {e}", file=sys.stderr)
        sys.exit(2)

    if not args.quiet:
        print("[done]")
        for k, v in info.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()