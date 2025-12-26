# -*- coding: utf-8 -*-
"""
批处理生成植被和裸地丰度图 (Vegetation & Bare soil abundance, LSU + NNLS)

功能：
1. 对参考影像（masked_input 下的 20240810..._8b_clip.tif）计算植被/裸地丰度；
2. 对所有 *_8b_clip_registered.tif 批量计算植被/裸地丰度；
3. 输出结果：
   - lsu_abundance_veg 下：*_veg_abun_LSU.tif
   - lsu_abundance_bare 下：*_bare_abun_LSU.tif
"""

import os
import glob
import numpy as np
import rasterio
from scipy.optimize import nnls

# -------------------- 配置 --------------------

# 参考影像（和你水丰度/端元用的是同一幅）
REF_IMG_PATH = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\register_image_by_ospy\masked_input\20240810_102132_44_2416_3B_AnalyticMS_SR_8b_clip.tif"

# 批处理输入目录（配准后的影像所在目录）
INPUT_DIR = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\register_image_by_ospy"

# 端元矩阵（行顺序为 [Water, Vegetation, BareSoil]）
ENDMEMBER_PATH = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\validation\lsu_endmembers_3avg.npy"

# 输出目录（分开存放）
OUTPUT_DIR_VEG  = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\lsu_abundance_veg"
OUTPUT_DIR_BARE = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\lsu_abundance_bare"

CLEAN_OUTPUT_DIR = True
SKIP_PREFIX = ""   # 例如: "20240810_102132_44_2416" （如果你只想跳过某几景）
OUT_NODATA = -9999.0
# ------------------------------------------------


def load_endmembers(path):
    """
    加载端元矩阵，返回 (bands, 3)，三列对应 [water, vegetation, bare soil]
    """
    em = np.load(path).astype("float32")
    if em.ndim != 2:
        raise ValueError(f"端元矩阵维度应为2，当前是 {em.shape}")

    r, c = em.shape
    if r == 3:
        E = em.T.astype("float32")   # (bands, 3)
    elif c == 3:
        E = em.astype("float32")     # (bands, 3)
    else:
        raise ValueError(f"端元矩阵形状应为 (3, bands) 或 (bands, 3)，当前是 {em.shape}")

    if E.shape[1] != 3:
        raise ValueError(f"端元个数必须为3，当前是 {E.shape[1]}")
    return E


def solve_abundance_nnls(E, X, valid_mask):
    """
    逐像元 NNLS 求解丰度，并在 sum>0 时归一化到和为1

    E: (bands, 3)
    X: (N, bands)
    valid_mask: (N,) bool，True 表示该像元参与求解
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


def process_one_image(in_path, E):
    """
    对单幅影像执行 LSU，并输出植被 & 裸地丰度图。

    - in_path: 任意一景 8-band SR 影像（可以是参考图像或 *_8b_clip_registered）
    - E: (bands, 3) 端元矩阵
    """
    file = os.path.basename(in_path)
    print(f"[Process] {file}")

    with rasterio.open(in_path) as src:
        prof = src.profile
        band_count = src.count
        H, W = src.height, src.width

        if band_count != E.shape[0]:
            print(f"[Skip] {file}: 影像波段数={band_count} 与端元波段数={E.shape[0]} 不一致。")
            return False

        img_m = src.read(masked=True).astype("float32") / 10000.0  # (bands,H,W) masked array

        # 掩膜处理：原始掩膜 + 数值过滤
        mask_from_file = np.ma.getmaskarray(img_m)           # (bands,H,W)
        data = img_m.data                                    # 原始值 ndarray
        mask_value = (data <= 0.0) | (data > 1.2) | ~np.isfinite(data)
        mask_all = mask_from_file | mask_value
        valid_hw = ~np.any(mask_all, axis=0)                 # (H,W)
        img = np.where(valid_hw[None, :, :], data, np.nan)   # 无效像元强制为 NaN

    # LSU 求解
    X = img.reshape(img.shape[0], -1).T                      # (N, bands)
    valid = ~np.any(np.isnan(X), axis=1)

    A = solve_abundance_nnls(E, X, valid)
    veg_abun  = A[:, 1].reshape(H, W)  # 第二列：植被
    bare_abun = A[:, 2].reshape(H, W)  # 第三列：裸地

    # --- 保存函数 ---
    def save_one(arr, out_dir, suffix):
        os.makedirs(out_dir, exist_ok=True)
        out_name = file.replace(".tif", f"_{suffix}_abun_LSU.tif")
        out_path = os.path.join(out_dir, out_name)
        out_prof = prof.copy()
        out_prof.update(count=1, dtype="float32", compress="lzw", nodata=OUT_NODATA)
        out_band = np.where(np.isfinite(arr), arr, OUT_NODATA).astype("float32")
        with rasterio.open(out_path, "w", **out_prof) as dst:
            dst.write(out_band, 1)
        print(f"  [OK] Saved: {out_path}")

    # 植被 → 存到 veg 文件夹
    save_one(veg_abun, OUTPUT_DIR_VEG, "veg")

    # 裸地 → 存到 bare 文件夹
    save_one(bare_abun, OUTPUT_DIR_BARE, "bare")

    return True


def main():
    # 1) 清空旧目录
    for out_dir in [OUTPUT_DIR_VEG, OUTPUT_DIR_BARE]:
        if CLEAN_OUTPUT_DIR and os.path.isdir(out_dir):
            for name in os.listdir(out_dir):
                p = os.path.join(out_dir, name)
                if os.path.isfile(p):
                    os.remove(p)
            print(f"[CLEAN] 已清空输出目录 {out_dir}")
        os.makedirs(out_dir, exist_ok=True)

    # 2) 加载端元
    E = load_endmembers(ENDMEMBER_PATH)
    bands_in_E = E.shape[0]
    print(f"[INFO] 端元波段数 = {bands_in_E}")

    # 3) 先处理参考影像（masked_input 下那一幅）
    if os.path.isfile(REF_IMG_PATH):
        print("\n====== Step 1: Reference image (vegetation & bare) ======")
        try:
            process_one_image(REF_IMG_PATH, E)
        except Exception as e:
            print(f"[Error] 参考影像处理失败: {os.path.basename(REF_IMG_PATH)} -> {e}")
    else:
        print(f"[WARN] 未找到参考影像：{REF_IMG_PATH}")

    # 4) 批处理所有配准后的影像
    print("\n====== Step 2: Registered images (vegetation & bare, batch) ======")
    tif_paths = glob.glob(
        os.path.join(INPUT_DIR, "**", "*_8b_clip_registered.tif"),
        recursive=True
    )
    print(f"将处理的配准后影像数：{len(tif_paths)}（端元波段数={bands_in_E}）")

    n_ok = 0
    for in_path in sorted(tif_paths):
        file = os.path.basename(in_path)
        if SKIP_PREFIX and file.startswith(SKIP_PREFIX):
            print(f"[Skip] {file} (by prefix)")
            continue
        try:
            ok = process_one_image(in_path, E)
            n_ok += int(ok)
        except Exception as e:
            print(f"[Error] {file} -> {e}")

    veg_tifs  = [f for f in os.listdir(OUTPUT_DIR_VEG)  if f.lower().endswith(".tif")]
    bare_tifs = [f for f in os.listdir(OUTPUT_DIR_BARE) if f.lower().endswith(".tif")]
    print(f"\n✅ 植被+裸地丰度提取完成：成功 {n_ok} / 共 {len(tif_paths)}")
    print(f"📊 输出文件数：植被 {len(veg_tifs)}，裸地 {len(bare_tifs)}")


if __name__ == "__main__":
    main()
