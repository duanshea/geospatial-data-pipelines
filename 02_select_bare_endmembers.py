#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle, ConnectionPatch
import matplotlib.patheffects as pe

# ===================== 路径 & 参数 =====================
img_path = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\register_image_by_ospy\masked_input\20240810_102132_44_2416_3B_AnalyticMS_SR_8b_clip.tif"

# 你的已有端元与候选
water_csv = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\selected_pixels_water_top3_water.csv"
veg_csv   = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\selected_pixels_vegetation_top3_vegetation.csv"
land_csv  = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\filtered_land_candidates.csv"  # 含 is_land_candidate=True

# 输出（命名与水/植被一致）
TOPK     = 3
bare_out = rf"C:\Users\Duans\Desktop\bfastlibfastlite_project\selected_pixels_bare_top{TOPK}_bare.csv"
bare_png = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\viz_endmember_bare.png"

# ===================== 读取影像 & 预处理 =====================
with rasterio.open(img_path) as src:
    b = src.read(masked=True).astype("float32")
    nir   = b[7].filled(np.nan) / 10000.0   # band 8
    red   = b[5].filled(np.nan) / 10000.0   # band 6
    green = b[3].filled(np.nan) / 10000.0   # band 4
    blue  = b[1].filled(np.nan) / 10000.0   # band 2

def make_rgb(R, G, B, p_low=0.5, p_high=99.5, gamma=1/1.8):
    rgb = np.stack([R, G, B], axis=-1)
    out = np.zeros_like(rgb, dtype=np.float32)
    for i in range(3):
        ch = rgb[..., i]
        finite = np.isfinite(ch)
        lo, hi = np.percentile(ch[finite], (p_low, p_high))
        x = np.clip((ch - lo)/(hi - lo + 1e-6), 0, 1)
        out[..., i] = np.power(x, gamma)
    return out

rgb  = make_rgb(red, green, blue)
ndvi = (nir - red) / (nir + red + 1e-6)
ndwi = (green - nir) / (green + nir + 1e-6)

def add_ndvi_ndwi(df):
    """给任何 (row,col) 表补齐 NDVI/NDWI 列。"""
    if "NDVI" not in df.columns:
        df["NDVI"] = np.nan
    if "NDWI" not in df.columns:
        df["NDWI"] = np.nan
    for i, r in df.iterrows():
        rr, cc = int(r["row"]), int(r["col"])
        if np.isnan(df.at[i, "NDVI"]):
            df.at[i, "NDVI"] = float(ndvi[rr, cc])
        if np.isnan(df.at[i, "NDWI"]):
            df.at[i, "NDWI"] = float(ndwi[rr, cc])
    return df

# ===================== 准备端元中心 & 裸地候选 =====================
water_df = add_ndvi_ndwi(pd.read_csv(water_csv))
veg_df   = add_ndvi_ndwi(pd.read_csv(veg_csv))

mw_ndvi, mw_ndwi = float(water_df["NDVI"].mean()), float(water_df["NDWI"].mean())
mv_ndvi, mv_ndwi = float(veg_df["NDVI"].mean()),   float(veg_df["NDWI"].mean())

land_df = pd.read_csv(land_csv)
land_df = land_df[land_df.get("is_land_candidate", True) == True].copy()
land_df = add_ndvi_ndwi(land_df)

# ===================== 选取裸地端元（远离水体&植被） =====================
# === 1) 基于端元统计做硬性剔除（gate）===
mw_ndvi, mw_ndwi = water_df["NDVI"].mean(), water_df["NDWI"].mean()
sw_ndvi, sw_ndwi = water_df["NDVI"].std(ddof=0), water_df["NDWI"].std(ddof=0)
mv_ndvi, mv_ndwi = veg_df["NDVI"].mean(),   veg_df["NDWI"].mean()
sv_ndvi, sv_ndwi = veg_df["NDVI"].std(ddof=0), veg_df["NDWI"].std(ddof=0)

gate_not_veg   = land_df["NDVI"] < (mv_ndvi - 1.0 * sv_ndvi)
gate_not_water = land_df["NDWI"] > (mw_ndwi + 1.0 * sw_ndwi)

candidates = land_df[gate_not_veg & gate_not_water].copy()
if candidates.empty:
    print("⚠️ Gate 太严了，没有候选。降低阈值试试（把 1.0 改小一些）。")
    candidates = land_df.copy()

# === 2) max–min distance 得分 ===
dw = np.sqrt((candidates["NDVI"] - mw_ndvi)**2 + (candidates["NDWI"] - mw_ndwi)**2)
dv = np.sqrt((candidates["NDVI"] - mv_ndvi)**2 + (candidates["NDWI"] - mv_ndwi)**2)
candidates["score"] = np.minimum(dw, dv)   # 最小距离越大越好

# === 3) 取 Top-K 并保存 ===
bare_top = candidates.sort_values("score", ascending=False).head(TOPK)[["row", "col", "NDVI", "NDWI", "score"]]
bare_top.rename(columns={"score": "dist_score"}, inplace=True)
bare_top.to_csv(bare_out, index=False)
print("✅ 选出的裸地端元：")
print(bare_top)
print(f"[OK] CSV saved → {bare_out}")

# ===================== 可视化（左NDVI-中RGB-右NDWI） =====================
def visualize_bare(points_df, title_prefix="Bare land pixel",
                   bins=100, zoom_size=7, inset_pct=30, save_path=None):
    all_ndvi = ndvi[np.isfinite(ndvi)]
    all_ndwi = ndwi[np.isfinite(ndwi)]
    half = zoom_size // 2

    n = len(points_df)
    fig, axs = plt.subplots(n, 3, figsize=(15, 4*n))
    if n == 1:
        axs = [axs]

    H, W, _ = rgb.shape

    for i, row in points_df.reset_index(drop=True).iterrows():
        rr, cc = int(row["row"]), int(row["col"])

        # 左：NDVI直方图
        axs[i][0].hist(all_ndvi, bins=bins, color="lightcoral", alpha=0.7)
        axs[i][0].axvline(float(ndvi[rr, cc]), color="darkred", linewidth=2)
        axs[i][0].set_title(f"NDVI = {float(ndvi[rr, cc]):.3f}")

        # 中：RGB + 放大窗
        ax = axs[i][1]
        ax.imshow(rgb)
        ax.set_title(f"{title_prefix} {i+1} (row={rr}, col={cc})")
        ax.axis("off")

        main_rect = Rectangle((cc - 0.5, rr - 0.5), 1, 1,
                              linewidth=2.0, edgecolor=(1, 0, 0), facecolor='none')
        ax.add_patch(main_rect)

        axins = inset_axes(
            ax, width=f"{inset_pct}%", height=f"{inset_pct}%",
            loc='lower right', borderpad=0.8
        )
        r0, r1 = max(0, rr - half), min(H, rr + half + 1)
        c0, c1 = max(0, cc - half), min(W, cc + half + 1)
        patch = rgb[r0:r1, c0:c1, :]
        axins.imshow(patch, interpolation='nearest')
        axins.set_xticks([]); axins.set_yticks([])

        zr = (rr - r0)
        zc = (cc - c0)
        zoom_rect = Rectangle((zc - 0.5, zr - 0.5), 1, 1,
                              linewidth=1.8, edgecolor=(1, 0, 0), facecolor='none')
        axins.add_patch(zoom_rect)

        con = ConnectionPatch(
            xyA=(zc, zr), coordsA=axins.transData,
            xyB=(cc, rr), coordsB=ax.transData,
            color=(1, 0, 0), linewidth=2.2
        )
        con.set_path_effects([pe.withStroke(linewidth=3.2, foreground='black')])
        ax.add_artist(con)

        # 右：NDWI直方图
        axs[i][2].hist(all_ndwi, bins=bins, color="lightblue", alpha=0.7)
        axs[i][2].axvline(float(ndwi[rr, cc]), color="blue", linewidth=2)
        axs[i][2].set_title(f"NDWI = {float(ndwi[rr, cc]):.3f}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=220)
        print(f"[OK] Figure saved → {save_path}")
    plt.show()


# 直接调用一次
visualize_bare(bare_top, zoom_size=7, inset_pct=30, save_path=bare_png)
