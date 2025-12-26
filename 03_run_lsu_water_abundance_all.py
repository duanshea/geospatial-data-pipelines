#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对参考影像 + 所有配准影像执行 LSU（NNLS），生成水体丰度图。

包含两部分：
1. 对参考图像（masked_input 下的 20240810...clip.tif）生成 water_abun_LSU.tif
2. 对 INPUT_DIR 下所有 *_8b_clip_registered.tif 批量生成 water_abun_LSU.tif

端元：
- 使用 lsu_endmembers_3avg.npy，行顺序为 [Water, Vegetation, BareSoil]
"""

import os
import glob
import numpy as np
import rasterio
from scipy.optimize import nnls

# -------------------- 配置 --------------------

# 参考影像（masked_input 里的那一幅）
REF_IMG_PATH = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\register_image_by_ospy\masked_input\20240810_102132_44_2416_3B_AnalyticMS_SR_8b_clip.tif"

# 批处理输入目录（配准后的影像所在目录）
INPUT_DIR = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\register_image_by_ospy"

# 端元矩阵
ENDMEMBER_PATH = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\validation\lsu_endmembers_3avg.npy"

# 输出目录（参考影像和批处理结果都放这里）
OUTPUT_DIR = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\lsu_abundance"

# 固定使用 nnls
CLEAN_OUTPUT_DIR = True
SKIP_PREFIX = ""      # 例如: "20240810_102132_44_2416" （需要时可自己改）
OUT_NODATA = -9999.0
# ------------------------------------------------


def load_endmembers(path):
    """
    加载端元矩阵，自动适配 (3, bands) 或 (bands, 3)，
    返回 E 形状为 (bands, 3) 且 dtype=float32
    """
    em = np.load(path).astype("float32")
    if em.ndim != 2:
        raise ValueError(f"端元矩阵维度应为2，当前是 {em.shape}")

    r, c = em.shape
    if r == 3:
        E = em.T.astype("float32")
    elif c == 3:
        E = em.astype("float32")
    else:
        raise ValueError(f"端元矩阵形状应为 (3, bands) 或 (bands, 3)，当前是 {em.shape}")
    if E.shape[1] != 3:
        raise ValueError(f"端元个数必须为3，当前是 {E.shape[1]}")
    return E


def solve_abundance_nnls(E, X, valid_mask):
    """
    逐像元 NNLS（非负 + sum>0 时归一化）
    E: (bands, 3)
    X: (N, bands)
    """
    A = np.full((X.shape[0], 3), np.nan, dtype="float32")
    idx = np.where(valid_mask)[0]
    for i in idx:
        coeffs, _ = nnls(E, X[i])
        s = coeffs.sum()
        if s > 0:
            coeffs = coeffs / s
        A[i] = coeffs.astype("float32")
    return A


def run_reference_image(E):
    """
    对参考影像 REF_IMG_PATH 生成水体丰度图（单幅）。
    输出命名与批处理风格一致：*_water_abun_LSU.tif
    """
    if not os.path.isfile(REF_IMG_PATH):
        print(f"[Warn] 参考影像不存在，跳过：{REF_IMG_PATH}")
        return False

    base = os.path.basename(REF_IMG_PATH).replace(".tif", "_water_abun_LSU.tif")
    out_path = os.path.join(OUTPUT_DIR, base)

    with rasterio.open(REF_IMG_PATH) as src:
        prof = src.profile
        img_m = src.read(masked=True).astype("float32") / 10000.0  # (bands, H, W)
        H, W = src.height, src.width

    # --- 构建掩膜（与批处理逻辑保持一致）---
    mask_from_file = np.ma.getmaskarray(img_m)                 # 原始掩膜
    data = img_m.data                                          # 原始值（ndarray）
    mask_value = (data <= 0.0) | (data > 1.2) | ~np.isfinite(data)
    mask_all = mask_from_file | mask_value                     # 最终掩膜（bands, H, W）

    # 至少一个波段无效就排除该像元
    valid_hw = ~np.any(mask_all, axis=0)                       # (H, W)
    img = np.where(valid_hw[None, :, :], data, np.nan)         # 强制无效值为 NaN

    # --- LSU 解混 ---
    X = img.reshape(img.shape[0], -1).T  # (N, bands)
    valid = ~np.any(np.isnan(X), axis=1)

    A = solve_abundance_nnls(E, X, valid)
    water_abun = A[:, 0].reshape(H, W)  # 第一列是 Water

    out_prof = prof.copy()
    out_prof.update(count=1, dtype="float32", compress="lzw", nodata=OUT_NODATA)
    out_band = np.where(np.isfinite(water_abun), water_abun, OUT_NODATA).astype("float32")

    with rasterio.open(out_path, "w", **out_prof) as dst:
        dst.write(out_band, 1)

    print(f"[OK] Reference water abundance saved: {out_path}")
    return True


def process_one(in_path, E, out_dir):
    """
    对单幅 *_8b_clip_registered.tif 影像执行 LSU，并输出水体丰度图。
    """
    file = os.path.basename(in_path)
    if SKIP_PREFIX and file.startswith(SKIP_PREFIX):
        print(f"[Skip] reference: {file}")
        return False

    with rasterio.open(in_path) as src:
        prof = src.profile
        band_count = src.count
        H, W = src.height, src.width

        if band_count != E.shape[0]:
            print(f"[Skip] {file}: 影像波段数={band_count} 与端元波段数={E.shape[0]} 不一致。")
            return False

        img_m = src.read(masked=True).astype("float32") / 10000.0  # masked array

        # 组合掩膜判断：包括原始掩膜、数值非法
        mask_from_file = np.ma.getmaskarray(img_m)           # (bands,H,W)
        data = img_m.data                                    # 原始值
        mask_value = (data <= 0.0) | (data > 1.2) | ~np.isfinite(data)
        mask_all = mask_from_file | mask_value               # 最终掩膜（bands,H,W）

        # 至少一个波段无效，就排除该像元
        valid_hw = ~np.any(mask_all, axis=0)                 # (H,W)
        img = np.where(valid_hw[None, :, :], data, np.nan)   # 强制无效值为 NaN

    X = img.reshape(img.shape[0], -1).T
    valid = ~np.any(np.isnan(X), axis=1)

    A = solve_abundance_nnls(E, X, valid)
    water_abun = A[:, 0].reshape(H, W)  # 第一列是水体

    out_name = file.replace(".tif", "_water_abun_LSU.tif")
    out_path = os.path.join(out_dir, out_name)

    prof.update(count=1, dtype="float32", compress="lzw", nodata=OUT_NODATA)
    out_band = np.where(np.isfinite(water_abun), water_abun, OUT_NODATA).astype("float32")

    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(out_band, 1)

    print(f"[OK] Saved: {out_path}")
    return True


def main():
    # 清空旧文件
    if CLEAN_OUTPUT_DIR and os.path.isdir(OUTPUT_DIR):
        for name in os.listdir(OUTPUT_DIR):
            p = os.path.join(OUTPUT_DIR, name)
            if os.path.isfile(p):
                os.remove(p)
        print(f"[CLEAN] 已清空输出目录 {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载端元
    E = load_endmembers(ENDMEMBER_PATH)
    bands_in_E = E.shape[0]

    # 1) 先对参考图像跑一次 LSU（水丰度）
    print("\n====== Step 1: Reference image water abundance ======")
    run_reference_image(E)

    # 2) 批处理所有配准后的影像
    print("\n====== Step 2: Registered images water abundance (batch) ======")
    tif_paths = glob.glob(os.path.join(INPUT_DIR, "**", "*_8b_clip_registered.tif"), recursive=True)
    print(f"将处理的配准后影像数：{len(tif_paths)}（端元波段数={bands_in_E}）")

    n_ok = 0
    for in_path in sorted(tif_paths):
        try:
            ok = process_one(in_path, E, OUTPUT_DIR)
            n_ok += int(ok)
        except Exception as e:
            print(f"[Error] {os.path.basename(in_path)} -> {e}")

    final_tifs = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".tif")]
    print(f"\n✅ 处理完成：成功 {n_ok} / 共 {len(tif_paths)}（求解器: nnls）")
    print(f"📊 输出目录中实际 .tif 数量：{len(final_tifs)}")


if __name__ == "__main__":
    main()
