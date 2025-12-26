# -*- coding: utf-8 -*-
"""
Compare clusters vs LSU abundance (Water/Bare/Veg) — polished figures
- 读取多期 *_water/_bare/_veg_abun_LSU.tif，像元层面取时间“中位数”
- 与聚类标签 (row,col,cluster) join（可选：只用采样像元）
- 生成：列联表、Purity、NMI、堆叠条（带标签）、箱+小提琴（5–95%）、三元图（含质心）
"""

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score as NMI

# ============ 路径（按需修改） ============
BASE = Path(r"C:\Users\Duans\Desktop\bfastlibfastlite_project\lsu_abundance")

W_DIR = Path(r"C:\Users\Duans\Desktop\bfastlibfastlite_project\lsu_abundance")          # *_water_abun_LSU.tif
B_DIR = Path(r"C:\Users\Duans\Desktop\bfastlibfastlite_project\lsu_abundance_bare")     # *_bare_abun_LSU.tif
V_DIR = Path(r"C:\Users\Duans\Desktop\bfastlibfastlite_project\lsu_abundance_veg")      # *_veg_abun_LSU.tif

LABELS_CSV = BASE / r"riparian_out\tslearn_clusters_BIC_h=0.08_0.10_0.12_20251005_204826\tslearn_labels_BIC_h=0.08_0.10_0.12.csv"
SAMPLE_CSV = BASE / r"stratified_sampling_20251005_171853\sample_pixels.csv"  # 若不想限于样本，设为 None

OUT_DIR = BASE / "wbv_vs_clusters_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 统一样式
plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120
})
COL_W, COL_B, COL_V = "#3B82F6", "#F59E0B", "#10B981"
CLUST_COLS = ["#2563EB", "#F97316", "#059669", "#7C3AED", "#0EA5E9", "#EF4444"]


# ============ 辅助函数 ============
def list_tifs(folder: Path, pattern: str):
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到 {pattern} 于 {folder}")
    return files


def read_labels(p_csv: Path) -> pd.DataFrame:
    lab = pd.read_csv(p_csv)
    if {"row", "col", "cluster"}.issubset(lab.columns):
        pass
    elif "series" in lab.columns and "cluster" in lab.columns:
        rc = lab["series"].astype(str).str.extract(r"pix_(\d+)_(\d+)")
        lab["row"] = rc[0].astype(int)
        lab["col"] = rc[1].astype(int)
    else:
        raise ValueError("labels 需要 row/col/cluster 或 series+cluster")
    lab["cluster"] = lab["cluster"].astype(int)
    return lab[["row", "col", "cluster"]]


def read_samples(p_csv: Path | None) -> pd.DataFrame | None:
    if p_csv is None:
        return None
    df = pd.read_csv(p_csv)
    if {"row", "col"}.issubset(df.columns):
        return df[["row", "col"]].drop_duplicates()
    elif "series" in df.columns:
        rc = df["series"].astype(str).str.extract(r"pix_(\d+)_(\d+)")
        df["row"] = rc[0].astype(int)
        df["col"] = rc[1].astype(int)
        return df[["row", "col"]].drop_duplicates()
    else:
        raise ValueError("sample_pixels.csv 需要 row/col 或 series")


def sample_median_at_indices(tif_list, rows, cols):
    vals = []
    for p in tif_list:
        with rasterio.open(p) as src:
            arr = src.read(1).astype("float32")
            vals.append(arr[rows, cols])
    arr = np.vstack(vals)  # (T, N)
    return np.nanmedian(arr, axis=0)  # (N,)


def add_value_labels(ax, xs, ys, fmt="{:.0f}%"):
    for x, y in zip(xs, ys):
        ax.text(x, y + 2, fmt.format(y), ha="center", va="bottom", fontsize=9)


def barycentric_to_xy(w, b, v):
    # 顶点：Water(0,0), Bare(1,0), Veg(0.5, sqrt(3)/2)
    x = b + 0.5 * v
    y = (np.sqrt(3) / 2) * v
    return x, y


def plot_ternary_matplotlib(df_in: pd.DataFrame, out_png: Path, out_pdf: Path):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, np.sqrt(3) / 2, 0]
    ax.plot(tri_x, tri_y, color="0.25", lw=1.2)

    for k, c in enumerate(sorted(df_in["cluster"].unique())):
        sub = df_in[df_in["cluster"] == c][["W", "B", "V"]].to_numpy()
        x, y = barycentric_to_xy(sub[:, 0], sub[:, 1], sub[:, 2])
        ax.scatter(
            x,
            y,
            s=9,
            alpha=0.45,
            color=CLUST_COLS[k % len(CLUST_COLS)],
            label=f"C{c}",
        )
        # 质心
        mW, mB, mV = sub.mean(axis=0)
        cx, cy = barycentric_to_xy(mW, mB, mV)
        ax.scatter(
            [cx],
            [cy],
            s=80,
            marker="x",
            color=CLUST_COLS[k % len(CLUST_COLS)],
        )

    ax.text(0.0, -0.06, "Water", ha="left", va="top")
    ax.text(1.0, -0.06, "Bare", ha="right", va="top")
    ax.text(0.5, np.sqrt(3) / 2 + 0.04, "Veg", ha="center", va="bottom")
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.08, np.sqrt(3) / 2 + 0.08)
    ax.axis("off")
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper right")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============ 主流程 ============

def main():
    labels = read_labels(LABELS_CSV)
    samples = read_samples(SAMPLE_CSV)
    idx = labels if samples is None else samples.merge(labels, on=["row", "col"], how="inner")
    print("Pixels to compare:", len(idx))

    rows = idx["row"].to_numpy(int)
    cols = idx["col"].to_numpy(int)

    W_tifs = list_tifs(W_DIR, "*_water_abun_LSU.tif")
    B_tifs = list_tifs(B_DIR, "*_bare_abun_LSU.tif")
    V_tifs = list_tifs(V_DIR, "*_veg_abun_LSU.tif")

    W_med = sample_median_at_indices(W_tifs, rows, cols)
    B_med = sample_median_at_indices(B_tifs, rows, cols)
    V_med = sample_median_at_indices(V_tifs, rows, cols)

    # 归一（防止数值误差导致 W+B+V != 1）
    S = W_med + B_med + V_med
    ok = np.isfinite(S) & (S > 0)
    W = np.zeros_like(W_med)
    B = np.zeros_like(B_med)
    V = np.zeros_like(V_med)
    W[ok] = W_med[ok] / S[ok]
    B[ok] = B_med[ok] / S[ok]
    V[ok] = V_med[ok] / S[ok]

    df = idx.copy()
    df["W"] = np.clip(W, 0, 1)
    df["B"] = np.clip(B, 0, 1)
    df["V"] = np.clip(V, 0, 1)
    df["dominant"] = np.argmax(df[["W", "B", "V"]].to_numpy(), axis=1)  # 0=W,1=B,2=V
    df.to_csv(OUT_DIR / "wbv_per_pixel_median.csv", index=False)
    print("Saved:", OUT_DIR / "wbv_per_pixel_median.csv")

    # ----- 列联、Purity、NMI -----
    cont = pd.crosstab(df["cluster"], df["dominant"]).sort_index()
    cont_prop = cont.div(cont.sum(axis=1), axis=0)
    purity = cont_prop.max(axis=1).mean()
    nmi = NMI(df["cluster"], df["dominant"])
    cont.to_csv(OUT_DIR / "contingency_counts.csv")
    cont_prop.to_csv(OUT_DIR / "contingency_rowprop.csv")
    with open(OUT_DIR / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"purity={purity:.3f}\nNMI={nmi:.3f}\n")
    print(f"Purity={purity:.3f}  NMI={nmi:.3f}")

    # ============ 画图：堆叠均值（带标签，按W排序） ============
    means = df.groupby("cluster")[["W", "B", "V"]].mean()
    order = means.sort_values("W").index.tolist()  # 按水从低到高
    means = means.loc[order]
    ns = df.groupby("cluster").size().loc[order].to_numpy()
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    bottom = np.zeros(len(order))
    for comp, col in zip(["W", "B", "V"], [COL_W, COL_B, COL_V]):
        vals = means[comp].to_numpy() * 100
        ax.bar(x, vals, bottom=bottom, color=col, label=comp)
        bottom += vals
    # 百分比顶端标签（总和）
    total = means.sum(axis=1).to_numpy() * 100
    add_value_labels(ax, x, total, fmt="{:.0f}%")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}\n(n={n})" for c, n in zip(order, ns)])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Fraction (%)")
    ax.set_xlabel("Cluster")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stacked_means_WBV_by_cluster.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "stacked_means_WBV_by_cluster.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ============ 画图：箱 + 小提琴（5–95%，无飞点） ============
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), sharey=True)
    comp_cfg = [("W", COL_W), ("B", COL_B), ("V", COL_V)]
    for ax, (comp, col) in zip(axes, comp_cfg):
        order = df.groupby("cluster")[comp].mean().sort_values().index.tolist()
        data = [df.loc[df["cluster"] == c, comp].to_numpy() for c in order]
        pos = np.arange(len(order))
        # violin
        parts = ax.violinplot(
            data,
            positions=pos,
            widths=0.85,
            showmeans=False,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(col)
            pc.set_alpha(0.18)
            pc.set_edgecolor("none")
        # box (5–95%)
        ax.boxplot(
            data,
            positions=pos,
            widths=0.45,
            showfliers=False,
            whis=[5, 95],
            patch_artist=True,
            boxprops=dict(facecolor=col, alpha=0.35, edgecolor=col),
            medianprops=dict(color="0.2", lw=1.3),
        )
        ax.set_xticks(pos)
        ax.set_xticklabels([f"C{c}\n(n={len(d)})" for c, d in zip(order, data)])
        ax.set_title(comp)
        if comp == "W":
            ax.set_ylabel("Fraction")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "boxviolin_WBV_by_cluster.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "boxviolin_WBV_by_cluster.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ============ 三元图 ============
    try:
        import ternary
        scale = 100
        pts = (df[["W", "B", "V"]] * scale).round().astype(int)

        fig, tax = ternary.figure(scale=scale)
        fig.set_size_inches(6.4, 5.6)
        tax.boundary(linewidth=1.2)
        tax.gridlines(multiple=20, color="0.88", linewidth=0.6)

        for k, c in enumerate(sorted(df["cluster"].unique())):
            mask = df["cluster"] == c
            cloud = list(
                map(
                    tuple,
                    pts.loc[mask, ["B", "V", "W"]].values,
                )
            )  # (left=B, right=V, bottom=W)
            tax.scatter(
                cloud,
                marker="o",
                s=9,
                color=CLUST_COLS[k % len(CLUST_COLS)],
                alpha=0.5,
                label=f"C{c}",
            )
            # 质心
            mW, mB, mV = df.loc[mask, ["W", "B", "V"]].mean().to_list()
            cm = (
                int(round(mB * scale)),
                int(round(mV * scale)),
                int(round(mW * scale)),
            )
            tax.scatter(
                [cm],
                marker="x",
                s=40,
                color=CLUST_COLS[k % len(CLUST_COLS)],
            )

        tax.left_axis_label("Bare (%)", offset=0.12)
        tax.right_axis_label("Veg (%)", offset=0.12)
        tax.bottom_axis_label("Water (%)", offset=-0.08)
        tax.ticks(axis="lbr", multiple=20, linewidth=1, fontsize=9)
        tax.clear_matplotlib_ticks()
        tax._redraw_labels()
        tax.legend(
            frameon=False,
            ncol=3,
            fontsize=9,
            loc="upper right",
            bbox_to_anchor=(1.22, 1.02),
        )

        fig.savefig(OUT_DIR / "ternary_WBV_by_cluster.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUT_DIR / "ternary_WBV_by_cluster.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("🔺 Saved ternary plot (python-ternary).")
    except Exception as e:
        print("python-ternary 不可用，改用 matplotlib 投影：", e)
        plot_ternary_matplotlib(
            df,
            OUT_DIR / "ternary_WBV_by_cluster_matplotlib.png",
            OUT_DIR / "ternary_WBV_by_cluster_matplotlib.pdf",
        )

    print("✅ Figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
