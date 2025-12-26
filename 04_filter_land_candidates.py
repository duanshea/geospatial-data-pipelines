#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from skimage.filters import threshold_multiotsu

# === 文件路径 ===
img_path = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\register_image_by_ospy\masked_input\20240810_102132_44_2416_3B_AnalyticMS_SR_8b_clip.tif"
land_path = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\selected_pixels_land.csv"
out_path = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\filtered_land_candidates.csv"

# === 1. 读取影像并计算全图 NDVI/NDWI ===
with rasterio.open(img_path) as src:
    bands = src.read(masked=True)
    nir   = bands[7].astype("float32").filled(np.nan) / 10000.0  # Band 8
    red   = bands[5].astype("float32").filled(np.nan) / 10000.0  # Band 6
    green = bands[3].astype("float32").filled(np.nan) / 10000.0  # Band 4

ndvi = (nir - red) / (nir + red + 1e-6)
ndwi = (green - nir) / (green + nir + 1e-6)

# === 2. 计算 Otsu 阈值（可用作参考） ===
ndvi_valid = ndvi[~np.isnan(ndvi)]
ndwi_valid = ndwi[~np.isnan(ndwi)]
ndvi_thresholds = threshold_multiotsu(ndvi_valid, classes=3)
ndwi_thresholds = threshold_multiotsu(ndwi_valid, classes=3)

print("Otsu NDVI 阈值:", ndvi_thresholds)
print("Otsu NDWI 阈值:", ndwi_thresholds)

# === 3. 手动设定水体/植被范围（来自已知端元）===
water_ndwi_range = (-0.71780926, -0.4510991)  # min, max
veg_ndvi_range   = (0.42279804, 0.7410607)    # min, max

# === 4. 读取裸地候选 CSV 并计算 NDVI/NDWI ===
df = pd.read_csv(land_path)
df["NDVI"] = np.nan
df["NDWI"] = np.nan

for i, row in df.iterrows():
    r, c = int(row["row"]), int(row["col"])
    df.at[i, "NDVI"] = (nir[r, c] - red[r, c]) / (nir[r, c] + red[r, c] + 1e-6)
    df.at[i, "NDWI"] = (green[r, c] - nir[r, c]) / (green[r, c] + nir[r, c] + 1e-6)

# === 5. 筛选裸地候选像素 ===
# 条件：NDVI 低于植被阈值下限 && NDWI 高于水体阈值上限
df["is_land_candidate"] = (df["NDVI"] < veg_ndvi_range[0]) & (df["NDWI"] > water_ndwi_range[1])

# === 6. 保存结果 ===
df.to_csv(out_path, index=False)
print(f"保留 {df['is_land_candidate'].sum()} 个裸地候选像素 → {out_path}")

# === 7. 可视化 NDVI vs NDWI ===
plt.figure(figsize=(8, 6))
plt.scatter(
    df["NDWI"], df["NDVI"],
    c=df["is_land_candidate"].map({True: "orange", False: "gray"}),
    label="Land candidates"
)
plt.axhline(veg_ndvi_range[0],   color="green", linestyle="--", label="Vegetation NDVI threshold")
plt.axvline(water_ndwi_range[1], color="blue",  linestyle="--", label="Water NDWI threshold")
plt.xlabel("NDWI")
plt.ylabel("NDVI")
plt.title("NDVI vs NDWI for Land Candidate Pixels")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
