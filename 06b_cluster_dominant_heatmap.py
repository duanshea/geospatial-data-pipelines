# -*- coding: utf-8 -*-
"""
Cluster vs dominant LSU class
- 基于 wbv_per_pixel_median.csv 计算：
  1) 每簇主导类别比例 + Wilson 置信区间（CSV）
  2) 列联热图（行归一），格子中标注百分比和计数（PNG/PDF）
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============ 路径 ============

BASE = Path(r"C:\Users\Duans\Desktop\bfastlibfastlite_project\lsu_abundance\wbv_vs_clusters_final")

CSV_IN  = BASE / "wbv_per_pixel_median.csv"              # 需包含: cluster, dominant(0=W,1=B,2=V)
CSV_OUT = BASE / "dominant_share_wilson_by_cluster.csv"  # 输出：各簇主导类别比例 + Wilson CI
FIG_OUT_PNG = BASE / "heatmap_cluster_vs_dominant_WBV.png"
FIG_OUT_PDF = BASE / "heatmap_cluster_vs_dominant_WBV.pdf"


# ============ 工具函数 ============

def wilson_ci(k, n, z=1.96):
    """
    Wilson score interval for binomial proportion.
    k: 成功次数
    n: 试验总数
    """
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


# ============ 主流程 ============

def main():
    # 读入像元级中位数结果
    df = pd.read_csv(CSV_IN)  # 需有 columns: cluster, dominant(0=W,1=B,2=V)

    # --- 1) 每簇主导类别比例 + Wilson CI ---
    rows = []
    for c, g in df.groupby("cluster"):
        n = len(g)
        for lab, name in enumerate(["Water", "Bare", "Veg"]):
            k = (g["dominant"] == lab).sum()
            lo, hi = wilson_ci(k, n)
            share = k / n if n > 0 else np.nan
            rows.append({
                "cluster": f"C{c}",
                "class": name,
                "share": share,
                "lo": lo,
                "hi": hi,
                "n": n,
                "k": k,
            })

    tab = pd.DataFrame(rows).sort_values(["cluster", "class"])
    tab.to_csv(CSV_OUT, index=False)
    print("Saved:", CSV_OUT)

    # --- 2) 列联热图（行归一 + 注释百分比与计数） ---
    cont = pd.crosstab(df["cluster"], df["dominant"]).sort_index()
    cont_prop = cont.div(cont.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    im = ax.imshow(cont_prop.values, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Water", "Bare", "Veg"])
    ax.set_yticks(range(cont_prop.shape[0]))
    ax.set_yticklabels([f"C{i}" for i in cont_prop.index])

    # 在格子中标注 “xx%\n(n)”
    for i in range(cont_prop.shape[0]):
        for j in range(cont_prop.shape[1]):
            pct = cont_prop.iloc[i, j] * 100
            n_ij = cont.iloc[i, j]
            ax.text(
                j,
                i,
                f"{pct:.0f}%\n({n_ij})",
                ha="center",
                va="center",
                fontsize=9,
            )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Row share")

    ax.set_xlabel("Dominant LSU class")
    ax.set_ylabel("Cluster")

    fig.tight_layout()
    fig.savefig(FIG_OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_OUT_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", FIG_OUT_PNG)


if __name__ == "__main__":
    main()
