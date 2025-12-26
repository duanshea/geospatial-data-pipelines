#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import rasterio
import os

# === 路径 ===
img_path = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\register_image_by_ospy\masked_input\20240810_102132_44_2416_3B_AnalyticMS_SR_8b_clip.tif"

# 三类端元（与前面脚本保持一致的文件名）
water_csv = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\selected_pixels_water_top3_water.csv"
veg_csv   = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\selected_pixels_vegetation_top3_vegetation.csv"
bare_csv  = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\selected_pixels_bare_top3_bare.csv"

# 输出
out_dir = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\validation"
os.makedirs(out_dir, exist_ok=True)
out_csv = os.path.join(out_dir, "lsu_endmembers_3avg.csv")
out_npy = os.path.join(out_dir, "lsu_endmembers_3avg.npy")

# === 工具函数 ===
def read_pixels_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    rows = df["row"].astype(int).tolist()
    cols = df["col"].astype(int).tolist()
    return list(zip(rows, cols))

def extract_avg_spectrum(src, pixels, scale=10000.0):
    """从 (row,col) 列表提取每个波段的像素值并求均值（忽略 NaN/NoData）。"""
    bands = src.read(masked=True).astype("float32")  # (B,H,W)
    B = bands.shape[0]
    specs = []
    for r, c in pixels:
        sp = []
        for b in range(B):
            v = bands[b, r, c]
            v = np.nan if np.ma.is_masked(v) else float(v)
            sp.append(v)
        specs.append(sp)
    specs = np.array(specs, dtype="float32") / scale
    avg = np.nanmean(specs, axis=0)
    return avg  # (B,)

# === 读取端元像素 ===
water_pixels = read_pixels_from_csv(water_csv)
veg_pixels   = read_pixels_from_csv(veg_csv)
bare_pixels  = read_pixels_from_csv(bare_csv)

# === 提取三类端元的平均光谱 ===
with rasterio.open(img_path) as src:
    band_count = src.count
    water_avg = extract_avg_spectrum(src, water_pixels)
    veg_avg   = extract_avg_spectrum(src, veg_pixels)
    bare_avg  = extract_avg_spectrum(src, bare_pixels)

# === 组织与保存 ===
labels = ["Water", "Vegetation", "BareSoil"]
avg_spectra = np.vstack([water_avg, veg_avg, bare_avg])  # (3, B)

df = pd.DataFrame(avg_spectra, columns=[f"Band{i+1}" for i in range(band_count)])
df["label"] = labels
df.to_csv(out_csv, index=False)
np.save(out_npy, avg_spectra)

print("✅ 已保存 LSU 三类平均端元：")
print(df)
print(f"→ CSV: {out_csv}")
print(f"→ NPY: {out_npy}")
