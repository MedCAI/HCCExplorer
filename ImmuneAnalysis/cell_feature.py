#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================
# 默认列配置（可通过命令行覆盖）
# =========================
DEFAULT_MARKER_COLS = [
    "DAPI", "Foxp3", "CD19", "CD68", "CD4", "CD3", "CD8", "SampleAF"
]
DEFAULT_MARKER_VAR_COLS = [
    "DAPI_var", "Foxp3_var", "CD19_var", "CD68_var", "CD4_var", "CD3_var", "CD8_var", "SampleAF_var"
]
DEFAULT_BASE_COLS = [
    "cell_id", "x", "y", "nucleus_area"
]

# patch 识别：从 cell_id 提取 patch_id（例如 "1024_5120_1" -> "1024_5120"）
patch_regex = re.compile(r"^(\d+_\d+)_\d+$")

# =========================
# 工具函数
# =========================

def mad(a: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan
    med = np.median(a)
    return np.median(np.abs(a - med))

def iqr(a: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan
    return np.percentile(a, 75) - np.percentile(a, 25)

def coefvar(a: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan
    mu = np.mean(a)
    sd = np.std(a, ddof=1) if a.size > 1 else 0.0
    return sd / mu if mu != 0 else np.nan

def safe_skew(a: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    return skew(a) if a.size > 2 else np.nan

def safe_kurtosis(a: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    return kurtosis(a) if a.size > 3 else np.nan

def get_patch_id(cell_id: str) -> str:
    m = patch_regex.match(str(cell_id))
    if m:
        return m.group(1)
    s = str(cell_id)
    if "_" in s:
        return "_".join(s.split("_")[:-1])
    return s

def compute_thresholds(df: pd.DataFrame, cols: List[str], q_low: float, q_high: float, fixed_thresholds: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    t = {}
    for c in cols:
        vals = df[c].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            t[c] = {"weak": np.nan, "strong": np.nan}
            continue
        qlow = np.quantile(vals, q_low)
        qhigh = np.quantile(vals, q_high)
        weak = fixed_thresholds.get(c, {}).get("weak", qlow)
        strong = fixed_thresholds.get(c, {}).get("strong", qhigh)
        if strong < weak:
            strong = weak
        t[c] = {"weak": float(weak), "strong": float(strong)}
    return t

def positivity_flags(df: pd.DataFrame, col: str, thr: Dict[str, float]) -> Tuple[pd.Series, pd.Series]:
    weak = thr["weak"]
    strong = thr["strong"]
    s = df[col]
    pos = s >= weak
    strong_pos = s >= strong
    return pos.astype(int), strong_pos.astype(int)

def retype_cells(df: pd.DataFrame, thresholds: Dict[str, Dict[str, float]], marker_cols: List[str], min_dapi_quantile: float) -> pd.DataFrame:
    d = df.copy()
    # 生成阳性标志
    for m in marker_cols:
        pos, strong = positivity_flags(d, m, thresholds[m])
        d[f"{m}_pos"] = pos
        d[f"{m}_strong"] = strong

    # 初始类型与置信度
    cell_type = []
    confidence = []

    # 参考 DAPI 水平用于置信度下调
    dapi_vals = d["DAPI"].astype(float)
    dapi_low_thr = np.quantile(dapi_vals.dropna(), min_dapi_quantile) if dapi_vals.notna().any() else -np.inf

    for _, row in d.iterrows():
        cd3 = row.get("CD3_pos", 0) == 1
        cd4 = row.get("CD4_pos", 0) == 1
        cd8 = row.get("CD8_pos", 0) == 1
        foxp3 = row.get("Foxp3_pos", 0) == 1
        cd19 = row.get("CD19_pos", 0) == 1
        cd68 = row.get("CD68_pos", 0) == 1

        cd3s = row.get("CD3_strong", 0) == 1
        cd4s = row.get("CD4_strong", 0) == 1
        cd8s = row.get("CD8_strong", 0) == 1
        foxs = row.get("Foxp3_strong", 0) == 1
        cd19s = row.get("CD19_strong", 0) == 1
        cd68s = row.get("CD68_strong", 0) == 1

        this_type = "Other"
        conf = 0.2

        # B cell
        if cd19 or cd19s:
            this_type = "B cell"
            conf = 0.7 if cd19s else 0.55

        # Macrophage / myeloid
        if cd68 or cd68s:
            this_type = "Macrophage"
            conf = max(conf, 0.7 if cd68s else 0.55)

        # T lineage
        if cd3 or cd3s:
            this_type = "T cell"
            conf = max(conf, 0.6 if cd3 else 0.75)
            if foxp3 or foxs:
                this_type = "Treg"
                conf = max(conf, 0.75 if foxp3 else 0.85)
            elif cd4 or cd4s:
                this_type = "CD4 T"
                conf = max(conf, 0.7 if cd4 else 0.8)
            elif cd8 or cd8s:
                this_type = "CD8 T"
                conf = max(conf, 0.7 if cd8 else 0.8)

        # 冲突：B 与 T 强阳冲突时按强度覆盖
        if (cd19 or cd19s) and (cd3 or cd3s):
            b_score = int(cd19s)
            t_score = int(cd3s) + int(cd4s) + int(cd8s) + int(foxs)
            if b_score > t_score:
                this_type = "B cell"
                conf = max(conf, 0.7)
            else:
                conf = max(conf, 0.7)

        # DAPI 太低则降低置信度
        if not np.isnan(row["DAPI"]) and row["DAPI"] < dapi_low_thr:
            conf *= 0.7

        cell_type.append(this_type)
        confidence.append(min(1.0, round(conf, 3)))

    d["cell_type_retyped"] = cell_type
    d["cell_type_confidence"] = confidence
    return d

def aggregate_patch_features(df: pd.DataFrame, marker_cols: List[str], marker_var_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    d = df.copy()
    d["patch_id"] = d["cell_id"].apply(get_patch_id)

    feature_desc: Dict[str, str] = {}

    g = d.groupby("patch_id")
    feats = pd.DataFrame(index=g.size().index)
    feats["n_cells"] = g.size()
    feature_desc["n_cells"] = "Number of cells in patch"

    # 使用核面积合计作为 proxy
    feats["nucleus_area_sum"] = g["nucleus_area"].sum()
    feats["nucleus_area_mean"] = g["nucleus_area"].mean()
    feats["nucleus_area_median"] = g["nucleus_area"].median()
    feats["nucleus_area_sd"] = g["nucleus_area"].std()
    feats["nucleus_area_mad"] = g["nucleus_area"].apply(lambda x: mad(x.values))
    feats["nucleus_area_iqr"] = g["nucleus_area"].apply(lambda x: iqr(x.values))
    feature_desc.update({
        "nucleus_area_sum": "Sum of nucleus_area (proxy for patch tissue area)",
        "nucleus_area_mean": "Mean nucleus area",
        "nucleus_area_median": "Median nucleus area",
        "nucleus_area_sd": "SD of nucleus area",
        "nucleus_area_mad": "MAD of nucleus area",
        "nucleus_area_iqr": "IQR of nucleus area",
    })
    feats["cell_density_proxy"] = feats["n_cells"] / (feats["nucleus_area_sum"] + 1e-6)
    feature_desc["cell_density_proxy"] = "Cell count divided by sum of nucleus area (proxy density)"

    # 细胞类型组成与熵
    type_counts = g["cell_type_retyped"].value_counts().unstack(fill_value=0)
    for c in type_counts.columns:
        feats[f"type_count__{c}"] = type_counts[c]
        feature_desc[f"type_count__{c}"] = f"Count of {c} cells"
        feats[f"type_frac__{c}"] = type_counts[c] / feats["n_cells"]
        feature_desc[f"type_frac__{c}"] = f"Fraction of {c} cells"

    def shannon_entropy(counts: np.ndarray) -> float:
        total = counts.sum()
        if total == 0:
            return np.nan
        p = counts / total
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    feats["type_entropy"] = type_counts.apply(lambda row: shannon_entropy(row.values), axis=1)
    feature_desc["type_entropy"] = "Shannon entropy of cell-type composition"

    # Marker 统计
    for m in marker_cols:
        col = m
        feats[f"{m}__mean"] = g[col].mean()
        feats[f"{m}__median"] = g[col].median()
        feats[f"{m}__sd"] = g[col].std()
        feats[f"{m}__cv"] = g[col].apply(lambda x: coefvar(x.values))
        feats[f"{m}__mad"] = g[col].apply(lambda x: mad(x.values))
        feats[f"{m}__iqr"] = g[col].apply(lambda x: iqr(x.values))
        feats[f"{m}__skew"] = g[col].apply(lambda x: safe_skew(x.values))
        feats[f"{m}__kurt"] = g[col].apply(lambda x: safe_kurtosis(x.values))
        feats[f"{m}__p10"] = g[col].quantile(0.10)
        feats[f"{m}__p25"] = g[col].quantile(0.25)
        feats[f"{m}__p75"] = g[col].quantile(0.75)
        feats[f"{m}__p90"] = g[col].quantile(0.90)
        feats[f"{m}_pos_rate"] = g[f"{m}_pos"].mean()
        feats[f"{m}_strong_rate"] = g[f"{m}_strong"].mean()

        feature_desc.update({
            f"{m}__mean": f"Mean intensity of {m}",
            f"{m}__median": f"Median intensity of {m}",
            f"{m}__sd": f"Std of {m}",
            f"{m}__cv": f"Coeff of variation of {m}",
            f"{m}__mad": f"MAD of {m}",
            f"{m}__iqr": f"IQR of {m}",
            f"{m}__skew": f"Skewness of {m}",
            f"{m}__kurt": f"Kurtosis of {m}",
            f"{m}__p10": f"10th percentile of {m}",
            f"{m}__p25": f"25th percentile of {m}",
            f"{m}__p75": f"75th percentile of {m}",
            f"{m}__p90": f"90th percentile of {m}",
            f"{m}_pos_rate": f"Positivity rate of {m} (>= weak threshold)",
            f"{m}_strong_rate": f"Strong positivity rate of {m} (>= strong threshold)",
        })

    # 共表达特征
    co_pairs = [
        ("CD3", "CD4"), ("CD3", "CD8"), ("CD3", "Foxp3"),
        ("CD19", "CD68"), ("CD68", "Foxp3"), ("CD4", "Foxp3")
    ]
    for a, b in co_pairs:
        both = (d[f"{a}_pos"] == 1) & (d[f"{b}_pos"] == 1)
        rate = both.groupby(d["patch_id"]).mean()
        feats[f"coexp__{a}_{b}__rate"] = rate
        feature_desc[f"coexp__{a}_{b}__rate"] = f"Co-positivity rate of {a} and {b}"

    # 变异度/异质性（若存在方差列）
    for v in marker_var_cols:
        if v in d.columns:
            feats[f"{v}__mean"] = g[v].mean()
            feats[f"{v}__median"] = g[v].median()
            feats[f"{v}__sd"] = g[v].std()
            feature_desc[f"{v}__mean"] = f"Mean of per-cell variance {v}"
            feature_desc[f"{v}__median"] = f"Median of per-cell variance {v}"
            feature_desc[f"{v}__sd"] = f"SD of per-cell variance {v}"

    # 简易 Immunoscore 代理
    cd3_rate = feats["CD3_pos_rate"]
    cd8_rate = feats["CD8_pos_rate"]
    cd3_z = (cd3_rate - cd3_rate.mean()) / (cd3_rate.std() + 1e-6)
    cd8_z = (cd8_rate - cd8_rate.mean()) / (cd8_rate.std() + 1e-6)
    feats["immunoscore_proxy"] = cd3_z + cd8_z
    feature_desc["immunoscore_proxy"] = "Z-score sum of CD3_pos_rate and CD8_pos_rate"

    # 细胞比例与比值扩展
    eps = 1e-6
    # 基础计数
    tcell_cnt = feats.get("type_count__T cell", 0)
    cd4_cnt = feats.get("type_count__CD4 T", 0)
    cd8_cnt = feats.get("type_count__CD8 T", 0)
    treg_cnt = feats.get("type_count__Treg", 0)
    b_cnt = feats.get("type_count__B cell", 0)
    macro_cnt = feats.get("type_count__Macrophage", 0)

    # 汇总比例（分母用 n_cells，更稳定）
    feats["frac_T_lineage"] = (tcell_cnt + cd4_cnt + cd8_cnt + treg_cnt) / (feats["n_cells"] + eps)
    feature_desc["frac_T_lineage"] = "Fraction of T-lineage cells (T cell + CD4 T + CD8 T + Treg) among all cells"

    feats["frac_CD4T"] = cd4_cnt / (feats["n_cells"] + eps)
    feature_desc["frac_CD4T"] = "Fraction of CD4 T among all cells"

    feats["frac_CD8T"] = cd8_cnt / (feats["n_cells"] + eps)
    feature_desc["frac_CD8T"] = "Fraction of CD8 T among all cells"

    feats["frac_Treg"] = treg_cnt / (feats["n_cells"] + eps)
    feature_desc["frac_Treg"] = "Fraction of Treg among all cells"

    feats["frac_Bcell"] = b_cnt / (feats["n_cells"] + eps)
    feature_desc["frac_Bcell"] = "Fraction of B cell among all cells"

    feats["frac_Myeloid"] = macro_cnt / (feats["n_cells"] + eps)
    feature_desc["frac_Myeloid"] = "Fraction of Macrophage among all cells"

    # 现有比值保留
    feats["T_B_ratio"] = (cd4_cnt + cd8_cnt + treg_cnt + tcell_cnt) / (b_cnt + eps)
    feature_desc["T_B_ratio"] = "T lineage to B cell ratio (counts)"

    feats["T_myeloid_ratio"] = (cd4_cnt + cd8_cnt + treg_cnt + tcell_cnt) / (macro_cnt + eps)
    feature_desc["T_myeloid_ratio"] = "T lineage to Macrophage ratio (counts)"

    # 新增：CD4/CD8 比（多口径）
    feats["CD4_CD8_count_ratio"] = (cd4_cnt + eps) / (cd8_cnt + eps)
    feature_desc["CD4_CD8_count_ratio"] = "Ratio of CD4 T count to CD8 T count"

    cd4_pos = feats.get("CD4_pos_rate", 0.0)
    cd8_pos = feats.get("CD8_pos_rate", 0.0)
    feats["CD4_CD8_posratio"] = (cd4_pos + eps) / (cd8_pos + eps)
    feature_desc["CD4_CD8_posratio"] = "Ratio of CD4_pos_rate to CD8_pos_rate"

    cd4_str = feats.get("CD4_strong_rate", 0.0)
    cd8_str = feats.get("CD8_strong_rate", 0.0)
    feats["CD4_CD8_strongratio"] = (cd4_str + eps) / (cd8_str + eps)
    feature_desc["CD4_CD8_strongratio"] = "Ratio of CD4_strong_rate to CD8_strong_rate"

    # 可选：更综合的免疫平衡指标
    feats["CD8_to_suppressive_ratio"] = (cd8_cnt + eps) / (treg_cnt + macro_cnt + eps)
    feature_desc["CD8_to_suppressive_ratio"] = "CD8 T count divided by (Treg + Macrophage) count"

    feats = feats.reset_index().rename(columns={"index": "patch_id"})
    return feats, feature_desc

# =========================
# 命令行
# =========================

def parse_args():
    p = argparse.ArgumentParser(description="Extract patch-level features from per-cell mIF CSV with retyped cell labels.")
    p.add_argument("--csv", required=True, help="输入每细胞 CSV 路径")
    p.add_argument("--outdir", required=True, help="输出目录")

    # 阈值策略
    p.add_argument("--q-low", type=float, default=0.6, help="弱阳性分位数（默认 0.6）")
    p.add_argument("--q-high", type=float, default=0.85, help="强阳性分位数（默认 0.85）")
    p.add_argument("--fixed-thresholds", default=None, help="固定阈值 JSON 或 JSON 文件路径，如 '{\"CD3\": {\"weak\": 8.0, \"strong\": 12.0}}'")

    # 缩放
    p.add_argument("--robust-scale", action="store_true", help="对 marker 列使用 RobustScaler（中位数/四分位）")
    p.add_argument("--scaled-suffix", default="_scaled", help="缩放后列名后缀（默认 _scaled）")

    # DAPI 置信度分位
    p.add_argument("--min-dapi-quantile", type=float, default=0.2, help="低于该分位的 DAPI 下调置信度（默认 0.2）")

    # 列设置（通常不需要改）
    p.add_argument("--marker-cols", nargs="+", default=DEFAULT_MARKER_COLS, help="marker 列名列表")
    p.add_argument("--marker-var-cols", nargs="+", default=DEFAULT_MARKER_VAR_COLS, help="marker 方差列名列表")
    p.add_argument("--base-cols", nargs="+", default=DEFAULT_BASE_COLS, help="基本列名列表")

    return p.parse_args()

def load_fixed_thresholds(arg: str) -> Dict[str, Dict[str, float]]:
    if arg is None:
        return {}
    if os.path.isfile(arg):
        with open(arg, "r", encoding="utf-8") as f:
            return json.load(f)
    try:
        return json.loads(arg)
    except Exception as e:
        raise ValueError(f"无法解析 fixed-thresholds：{e}")

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 读取
    df = pd.read_csv(args.csv, sep=None, engine="python")

    marker_cols = list(args.marker_cols)
    marker_var_cols = list(args.marker_var_cols)
    base_cols = list(args.base_cols)

    # 基本列检查
    required = set(base_cols + marker_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 清洗
    df = df.copy()
    for c in required.union(set(marker_var_cols)):
        if c in df.columns:
            if c == "cell_id":
                df[c] = df[c].astype(str)
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").values
    df = df.dropna(subset=["x", "y", "nucleus_area"], how="any")

    fixed_thresholds = load_fixed_thresholds(args.fixed_thresholds)

    # 稳健缩放（可选）
    if args.robust_scale:
        scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
        vals = df[marker_cols].astype(float).values
        vals_scaled = scaler.fit_transform(vals)
        for i, m in enumerate(marker_cols):
            df[m + args.scaled_suffix] = vals_scaled[:, i]
        use_cols = [m + args.scaled_suffix for m in marker_cols]
        thresholds = compute_thresholds(df, use_cols, q_low=args.q_low, q_high=args.q_high, fixed_thresholds=fixed_thresholds)
        # 将缩放后的列复制回原名，原值保存在 _orig
        for m, u in zip(marker_cols, use_cols):
            df[m + "_orig"] = df[m]
            df[m] = df[u]
    else:
        thresholds = compute_thresholds(df, marker_cols, q_low=args.q_low, q_high=args.q_high, fixed_thresholds=fixed_thresholds)

    # 重标注
    df2 = retype_cells(df, thresholds, marker_cols=marker_cols, min_dapi_quantile=args.min_dapi_quantile)

    # 聚合 patch 特征
    patch_feats, feat_desc = aggregate_patch_features(df2, marker_cols=marker_cols, marker_var_cols=marker_var_cols)

    # 导出
    cells_out = os.path.join(args.outdir, "cells_retyped.csv")
    patch_out = os.path.join(args.outdir, "patch_features.csv")
    dict_out = os.path.join(args.outdir, "patch_feature_dict.json")

    df2.to_csv(cells_out, index=False)
    patch_feats.to_csv(patch_out, index=False)
    with open(dict_out, "w", encoding="utf-8") as f:
        json.dump(feat_desc, f, ensure_ascii=False, indent=2)

    print(f"Saved cell-level with retyped labels: {cells_out}")
    print(f"Saved patch-level features: {patch_out}")
    print(f"Saved feature dictionary: {dict_out}")

if __name__ == "__main__":
    main()