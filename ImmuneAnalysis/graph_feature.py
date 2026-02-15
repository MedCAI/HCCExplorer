#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import json
import argparse
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from community import community_louvain  # python-louvain

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================
# 默认配置（可被命令行参数覆盖）
# =========================
DEFAULT_MARKER_POS_COLS = ["CD3_pos", "CD4_pos", "CD8_pos", "Foxp3_pos", "CD19_pos", "CD68_pos"]
DEFAULT_TYPE_GROUP_MAP = {
    "CD4 T": "T",
    "CD8 T": "T",
    "Treg": "T",
    "T cell": "T",
    "B cell": "B",
    "Macrophage": "Myeloid",
    "Other": "Other",
}

# 这些全局变量会在 main() 中由参数赋值，供函数读取
marker_pos_cols: List[str] = []
type_group_map: Dict[str, str] = {}
graph_mode: str = "radius"
radius_r: float = 40.0
include_self: bool = False
knn_k: int = 6
use_distance_weight: bool = True
clip_min_distance: float = 1e-3
export_edgelist: bool = False


# =========================
# 工具
# =========================

def ensure_dir(d: str):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def summary_stats(arr: np.ndarray) -> Dict[str, float]:
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": np.nan, "median": np.nan, "sd": np.nan, "iqr": np.nan, "max": np.nan}
    q75, q25 = np.percentile(arr, 75), np.percentile(arr, 25)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "iqr": float(q75 - q25),
        "max": float(np.max(arr)),
    }

def shannon_entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    return float(-(p * np.log2(p)).sum())

def build_graph_for_patch(df_patch: pd.DataFrame, mode: str = "radius") -> Tuple[nx.Graph, pd.DataFrame]:
    # 节点属性
    coords = df_patch[["x", "y"]].values.astype(float)
    ids = df_patch["cell_id"].astype(str).values
    n = len(df_patch)

    G = nx.Graph()
    for i in range(n):
        attrs = {
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "cell_type": str(df_patch.iloc[i]["cell_type_retyped"]),
        }
        # 附加 marker 阳性位
        for m in marker_pos_cols:
            if m in df_patch.columns:
                attrs[m] = int(df_patch.iloc[i][m])
        G.add_node(ids[i], **attrs)

    # 边构建
    if n <= 1:
        return G, pd.DataFrame(columns=["src", "dst", "dist", "weight"])

    if mode == "radius":
        # 使用 sklearn 的 radius_neighbors_graph 获取邻接（距离矩阵）
        A = radius_neighbors_graph(coords, radius=radius_r, mode="distance", include_self=include_self)
        A = A.tocoo()
        edges = []
        for i, j, d in zip(A.row, A.col, A.data):
            if i >= j:  # 无向图避免双计数
                continue
            w = 1.0 / max(d, clip_min_distance) if use_distance_weight else 1.0
            edges.append((ids[i], ids[j], float(d), float(w)))
    elif mode == "knn":
        nbrs = NearestNeighbors(n_neighbors=min(knn_k + 1, n), algorithm="auto").fit(coords)
        dists, idx = nbrs.kneighbors(coords)
        edges = []
        for i in range(n):
            for k in range(1, idx.shape[1]):  # 跳过自身
                j = idx[i, k]
                if i == j:
                    continue
                ii, jj = sorted([i, j])
                d = float(np.linalg.norm(coords[ii] - coords[jj]))
                w = 1.0 / max(d, clip_min_distance) if use_distance_weight else 1.0
                edges.append((ids[ii], ids[jj], d, w))
        # 去重
        edges = list({(a, b): (a, b, d, w) for a, b, d, w in edges}.values())
    else:
        raise ValueError("graph_mode must be 'radius' or 'knn'")

    # 加边
    for a, b, d, w in edges:
        G.add_edge(a, b, dist=d, weight=w)

    edf = pd.DataFrame(edges, columns=["src", "dst", "dist", "weight"])
    return G, edf

def lcc_subgraph(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G
    comps = list(nx.connected_components(G))
    if not comps:
        return G
    largest = max(comps, key=len)
    return G.subgraph(largest).copy()

def compute_graph_features(G: nx.Graph, df_patch: pd.DataFrame, patch_id: str) -> Dict[str, float]:
    feats: Dict[str, float] = {"patch_id": patch_id}
    n = G.number_of_nodes()
    m = G.number_of_edges()
    feats["g_nodes"] = n
    feats["g_edges"] = m
    feats["g_edge_density"] = (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0

    degrees = np.array([d for _, d in G.degree()])
    feats["deg_mean"] = float(np.mean(degrees)) if n > 0 else np.nan
    feats["deg_var"] = float(np.var(degrees, ddof=1)) if n > 1 else np.nan
    sdeg = summary_stats(degrees)
    feats["deg_median"] = sdeg["median"]
    feats["deg_iqr"] = sdeg["iqr"]
    feats["deg_max"] = sdeg["max"]
    if n > 0:
        k = max(1, int(math.ceil(0.01 * n)))
        topk = np.sort(degrees)[-k:]
        feats["deg_top1pct_mean"] = float(np.mean(topk))
    else:
        feats["deg_top1pct_mean"] = np.nan

    # 聚类系数
    try:
        feats["clustering_avg"] = float(nx.average_clustering(G, weight="weight"))
    except Exception:
        feats["clustering_avg"] = np.nan

    # 连通分量
    comps = list(nx.connected_components(G))
    feats["n_components"] = len(comps) if n > 0 else 0
    if len(comps) > 0:
        sizes = np.array([len(c) for c in comps], dtype=float)
        feats["lcc_frac"] = float(np.max(sizes) / n)
    else:
        feats["lcc_frac"] = np.nan

    # 仅在最大连通分量上计算路径与中心性
    Gl = lcc_subgraph(G)
    nl = Gl.number_of_nodes()
    if nl >= 2:
        try:
            feats["diameter_lcc"] = float(nx.diameter(Gl))
        except Exception:
            feats["diameter_lcc"] = np.nan
        try:
            spl = dict(nx.all_pairs_dijkstra_path_length(Gl, weight="dist"))
            dists = []
            for s in spl:
                for t in spl[s]:
                    if s == t:
                        continue
                    dists.append(spl[s][t])
            feats["avg_shortest_path_lcc"] = float(np.mean(dists)) if len(dists) else np.nan
        except Exception:
            feats["avg_shortest_path_lcc"] = np.nan

        # 中心性
        try:
            bet = nx.betweenness_centrality(Gl, weight="dist", normalized=True)
            clo = nx.closeness_centrality(Gl, distance="dist")
            try:
                eig = nx.eigenvector_centrality_numpy(Gl, weight="weight")
            except Exception:
                eig = {n: np.nan for n in Gl.nodes()}
            for name, vec in [("bet", bet), ("clo", clo), ("eig", eig)]:
                arr = np.array([vec.get(v, np.nan) for v in Gl.nodes()], dtype=float)
                s = summary_stats(arr)
                feats[f"{name}_mean"] = s["mean"]
                feats[f"{name}_median"] = s["median"]
                feats[f"{name}_sd"] = s["sd"]
                feats[f"{name}_iqr"] = s["iqr"]
        except Exception:
            for name in ["bet", "clo", "eig"]:
                feats[f"{name}_mean"] = np.nan
                feats[f"{name}_median"] = np.nan
                feats[f"{name}_sd"] = np.nan
                feats[f"{name}_iqr"] = np.nan
    else:
        feats["diameter_lcc"] = np.nan
        feats["avg_shortest_path_lcc"] = np.nan
        for name in ["bet", "clo", "eig"]:
            feats[f"{name}_mean"] = np.nan
            feats[f"{name}_median"] = np.nan
            feats[f"{name}_sd"] = np.nan
            feats[f"{name}_iqr"] = np.nan

    # 社区结构（Louvain）
    if n >= 5 and m >= 4:
        try:
            part = community_louvain.best_partition(G, weight="weight", resolution=1.0, random_state=42)
            comm_ids = np.array(list(part.values()))
            n_comm = len(set(comm_ids))
            feats["comm_n"] = float(n_comm)
            feats["modularity"] = float(community_louvain.modularity(part, G, weight="weight"))
            counts = np.bincount(comm_ids)
            p = counts / counts.sum()
            feats["comm_max_frac"] = float(p.max())
            feats["comm_entropy"] = shannon_entropy(p)
        except Exception:
            feats["comm_n"] = np.nan
            feats["modularity"] = np.nan
            feats["comm_max_frac"] = np.nan
            feats["comm_entropy"] = np.nan
    else:
        feats["comm_n"] = np.nan
        feats["modularity"] = np.nan
        feats["comm_max_frac"] = np.nan
        feats["comm_entropy"] = np.nan

    # 类型同质性与跨类边
    node_types = {}
    for n_id, data in G.nodes(data=True):
        t = data.get("cell_type", "Other")
        node_types[n_id] = type_group_map.get(t, t)

    # assortativity（离散标签）
    try:
        uniq = sorted(set(node_types.values()))
        t2i = {t: i for i, t in enumerate(uniq)}
        attrs = {nid: t2i[node_types[nid]] for nid in G.nodes()}
        nx.set_node_attributes(G, attrs, "type_int")
        feats["assort_type"] = float(nx.attribute_assortativity_coefficient(G, "type_int"))
    except Exception:
        feats["assort_type"] = np.nan

    # 边上类型对分布
    pair_counts: Dict[Tuple[str, str], int] = {}
    total_edges = max(1, G.number_of_edges())
    for u, v in G.edges():
        a = node_types.get(u, "Other")
        b = node_types.get(v, "Other")
        key = (a, b) if a <= b else (b, a)
        pair_counts[key] = pair_counts.get(key, 0) + 1

    def frac_pair(a: str, b: str) -> float:
        key = (a, b) if a <= b else (b, a)
        return pair_counts.get(key, 0) / total_edges

    feats["edge_frac_T_T"] = frac_pair("T", "T")
    feats["edge_frac_T_B"] = frac_pair("T", "B")
    feats["edge_frac_T_Myeloid"] = frac_pair("T", "Myeloid")
    feats["edge_frac_B_B"] = frac_pair("B", "B")
    feats["edge_frac_B_Myeloid"] = frac_pair("B", "Myeloid")
    feats["edge_frac_Myeloid_Myeloid"] = frac_pair("Myeloid", "Myeloid")

    # 标志物共邻居富集（边两端同时阳性）
    for a in marker_pos_cols:
        for b in marker_pos_cols:
            if b < a:
                continue
            both_cnt = 0
            for u, v in G.edges():
                au = G.nodes[u].get(a, 0)
                av = G.nodes[v].get(a, 0)
                bu = G.nodes[u].get(b, 0)
                bv = G.nodes[v].get(b, 0)
                if int(au and bu and av and bv) == 1:
                    both_cnt += 1
            feats[f"edge_frac_{a}_{b}"] = both_cnt / total_edges if total_edges > 0 else np.nan

    return feats

def get_patch_id_series(df: pd.DataFrame) -> pd.Series:
    # 用 cell_id 去掉最后一段，得到 patch_id
    return df["cell_id"].astype(str).apply(lambda s: "_".join(s.split("_")[:-1]) if "_" in s else s)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-patch spatial graphs and compute graph features from single-cell coordinates."
    )
    parser.add_argument("--input", "-i", required=True, help="Input CSV path (must contain cell_id, x, y, cell_type_retyped).")
    parser.add_argument("--outdir", "-o", required=True, help="Output directory.")
    parser.add_argument("--mode", choices=["radius", "knn"], default="radius", help="Graph building mode.")
    parser.add_argument("--radius", type=float, default=40.0, help="Radius threshold for radius mode.")
    parser.add_argument("--include-self", dest="include_self", action="store_true", help="Include self-loops in radius graph.")
    parser.add_argument("--no-self", dest="include_self", action="store_false", help="Exclude self-loops in radius graph (default).")
    parser.set_defaults(include_self=False)

    parser.add_argument("--k", type=int, default=6, help="k for KNN mode.")
    parser.add_argument("--weight", dest="use_weight", action="store_true", help="Use 1/d distance weight (default).")
    parser.add_argument("--no-weight", dest="use_weight", action="store_false", help="Use unweighted edges (weight=1).")
    parser.set_defaults(use_weight=True)
    parser.add_argument("--clip-min-distance", type=float, default=1e-3, help="Min distance to avoid division by zero.")
    parser.add_argument("--export-edges", action="store_true", help="Export per-patch edge list CSVs.")
    parser.add_argument("--build-global", action="store_true", help="Also compute features on merged global graph.")

    parser.add_argument(
        "--marker-cols",
        type=str,
        default=",".join(DEFAULT_MARKER_POS_COLS),
        help="Comma-separated marker positive columns present in input CSV."
    )
    parser.add_argument(
        "--type-map",
        type=str,
        default=json.dumps(DEFAULT_TYPE_GROUP_MAP),
        help='JSON string mapping fine cell types to coarse groups, e.g. {"CD4 T":"T","B cell":"B"}'
    )
    return parser.parse_args()

def main():
    global marker_pos_cols, type_group_map, graph_mode, radius_r, include_self
    global knn_k, use_distance_weight, clip_min_distance, export_edgelist

    args = parse_args()

    input_cells_csv = args.input
    output_dir = args.outdir
    ensure_dir(output_dir)

    graph_mode = args.mode
    radius_r = float(args.radius)
    include_self = bool(args.include_self)
    knn_k = int(args.k)
    use_distance_weight = bool(args.use_weight)
    clip_min_distance = float(args.clip_min_distance)
    export_edgelist = bool(args.export_edges)

    # 解析 marker 列
    marker_pos_cols = [c.strip() for c in args.marker_cols.split(",") if c.strip()]
    # 解析类型映射
    try:
        type_group_map = json.loads(args.type_map)
        if not isinstance(type_group_map, dict):
            raise ValueError
    except Exception:
        raise ValueError("Invalid --type-map. Provide a JSON string, e.g. '{\"CD4 T\":\"T\",\"B cell\":\"B\"}'")

    # 读取数据
    df = pd.read_csv(input_cells_csv)
    needed = {"cell_id", "x", "y", "cell_type_retyped"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {input_cells_csv}: {missing}")

    use_cols = ["cell_id", "x", "y", "cell_type_retyped"] + [c for c in marker_pos_cols if c in df.columns]
    df = df[use_cols].copy()

    if "patch_id" not in df.columns:
        df["patch_id"] = get_patch_id_series(df)

    patch_ids = df["patch_id"].unique().tolist()
    patch_features: List[Dict[str, float]] = []

    # 每个 patch 分别建图并提取特征
    for pid in patch_ids:
        dp = df[df["patch_id"] == pid].reset_index(drop=True)
        G, edf = build_graph_for_patch(dp, mode=graph_mode)
        feats = compute_graph_features(G, dp, pid)
        patch_features.append(feats)

        if export_edgelist:
            ed_out = os.path.join(output_dir, f"edges_{pid}.csv")
            edf.to_csv(ed_out, index=False)

    # 汇总保存
    pf = pd.DataFrame(patch_features).sort_values("patch_id")
    pf.to_csv(os.path.join(output_dir, "patch_graph_features.csv"), index=False)

    # 可选：全局合图
    if args.build_global:
        Gg = nx.Graph()
        for pid in patch_ids:
            dp = df[df["patch_id"] == pid].reset_index(drop=True)
            Gp, _ = build_graph_for_patch(dp, mode=graph_mode)
            mapping = {n: f"{pid}::{n}" for n in Gp.nodes()}
            Gp = nx.relabel_nodes(Gp, mapping)
            Gg = nx.compose(Gg, Gp)
        gf = compute_graph_features(Gg, df, patch_id="GLOBAL")
        pd.DataFrame([gf]).to_csv(os.path.join(output_dir, "global_graph_features.csv"), index=False)

    # 特征字典
    desc = {
        "g_nodes": "Number of nodes (cells) in patch graph",
        "g_edges": "Number of edges",
        "g_edge_density": "Graph edge density",
        "deg_mean": "Mean node degree",
        "deg_var": "Variance of node degree",
        "deg_median": "Median node degree",
        "deg_iqr": "IQR of node degree",
        "deg_max": "Maximum node degree",
        "deg_top1pct_mean": "Mean degree among top 1% high-degree nodes",
        "clustering_avg": "Average clustering coefficient (weighted)",
        "n_components": "Number of connected components",
        "lcc_frac": "Fraction of nodes in the largest connected component",
        "diameter_lcc": "Diameter of largest connected component",
        "avg_shortest_path_lcc": "Average shortest path length in LCC (distance-weighted)",
        "bet_mean": "Betweenness centrality mean (LCC)",
        "bet_median": "Betweenness centrality median (LCC)",
        "bet_sd": "Betweenness centrality SD (LCC)",
        "bet_iqr": "Betweenness centrality IQR (LCC)",
        "clo_mean": "Closeness centrality mean (LCC)",
        "clo_median": "Closeness centrality median (LCC)",
        "clo_sd": "Closeness centrality SD (LCC)",
        "clo_iqr": "Closeness centrality IQR (LCC)",
        "eig_mean": "Eigenvector centrality mean (LCC)",
        "eig_median": "Eigenvector centrality median (LCC)",
        "eig_sd": "Eigenvector centrality SD (LCC)",
        "eig_iqr": "Eigenvector centrality IQR (LCC)",
        "comm_n": "Number of Louvain communities",
        "modularity": "Louvain modularity",
        "comm_max_frac": "Largest community fraction",
        "comm_entropy": "Shannon entropy of community size distribution",
        "assort_type": "Attribute assortativity by coarse type",
        "edge_frac_T_T": "Fraction of edges between T-T",
        "edge_frac_T_B": "Fraction of edges between T-B",
        "edge_frac_T_Myeloid": "Fraction of edges between T-Myeloid",
        "edge_frac_B_B": "Fraction of edges between B-B",
        "edge_frac_B_Myeloid": "Fraction of edges between B-Myeloid",
        "edge_frac_Myeloid_Myeloid": "Fraction of edges between Myeloid-Myeloid",
    }
    for a in marker_pos_cols:
        for b in marker_pos_cols:
            if b < a:
                continue
            desc[f"edge_frac_{a}_{b}"] = f"Fraction of edges with both endpoints positive for {a} and {b}"

    with open(os.path.join(output_dir, "patch_graph_feature_dict.json"), "w", encoding="utf-8") as f:
        json.dump(desc, f, ensure_ascii=False, indent=2)

    print(f"Saved patch graph features to: {os.path.join(output_dir, 'patch_graph_features.csv')}")
    print(f"Saved feature dictionary to: {os.path.join(output_dir, 'patch_graph_feature_dict.json')}")
    if export_edgelist:
        print("Per-patch edge lists also exported.")
    if args.build_global:
        print(f"Saved global graph features to: {os.path.join(output_dir, 'global_graph_features.csv')}")

if __name__ == "__main__":
    main()