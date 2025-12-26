# -*- coding: utf-8 -*-
"""
构建全河床（Full river bed）掩膜

基于 water_abun 时间立方体：
1) 计算每个像元的“水体出现频率” occ = mean(water_abun > water_thr)
2) 使用极小阈值 occ_any_thr 提取所有曾为水体的像元并集 → riverbed_full
3) 输出：
   - occ.tif  : 出现频率栅格 [0,1]
   - riverbed_full.tif : 全河床二值掩膜 (uint8, 1=曾为水体)
   - run_meta.txt : 记录阈值和面积（ha）

注：这是从原先“三套掩膜一体化脚本”中精简出的“全河床”部分，
    只保留论文中实际使用的 riverbed_full。
"""

import os
from datetime import datetime
import warnings

import numpy as np
import xarray as xr
import rasterio

# ------------------- 路径区（按需修改） -------------------

BASE_DIR = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\lsu_abundance"

# NetCDF 立方体（由 08_build_water_abun_timeseries_and_samples.py 生成）
NC_PATH  = os.path.join(BASE_DIR, "water_abun_stack_stable.nc")

# 参考 GeoTIFF（提供 CRS、transform 与像元大小）
REF_TIF  = os.path.join(
    BASE_DIR,
    "20241015_101928_22_24ee_3B_AnalyticMS_SR_8b_clip_registered_water_abun_LSU.tif"
)

# 输出目录：自动加时间戳
RUN_TAG   = "riverbed_full"
RUN_TS    = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR   = os.path.join(BASE_DIR, f"{RUN_TAG}_{RUN_TS}")
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------- 参数区 -------------------

VAR_NAME     = "water_abun"  # NetCDF 中水体丰度变量名
water_thr    = 0.60          # 判定“当日为水体”的丰度阈值，用于计算 occ
occ_any_thr  = 0.01          # 全河床并集的极小阈值（去掉纯噪声）

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ------------------- 工具函数 -------------------

def _save_tif_like(ref_profile, path, arr, dtype, nodata=None):
    """
    使用 ref_profile 作为模板保存单波段 GeoTIFF。
    """
    prof = ref_profile.copy()
    prof.update(
        driver="GTiff",
        dtype=dtype,
        count=1,
        compress="lzw",
        nodata=nodata,
    )
    with rasterio.open(path, "w", **prof) as dst:
        if np.issubdtype(np.dtype(dtype), np.floating):
            data = np.where(
                np.isfinite(arr),
                arr,
                np.float32(nodata) if nodata is not None else arr,
            ).astype(dtype)
        else:
            data = arr.astype(dtype)
        dst.write(data, 1)


def _area_ha(mask_uint8, px_size_m):
    """
    计算二值掩膜中 ==1 像元的面积（单位：ha）
    """
    px = int(mask_uint8.sum())
    return px * px_size_m * px_size_m / 1e4


# ------------------- 主流程 -------------------

def main():
    # 1) 读取 NetCDF 立方体
    if not os.path.exists(NC_PATH):
        raise FileNotFoundError(f"NetCDF not found: {NC_PATH}")

    ds = xr.open_dataset(NC_PATH)
    if VAR_NAME not in ds:
        raise ValueError(f"变量 {VAR_NAME} 不在 {NC_PATH} 中，可 print(ds) 检查变量名。")

    da = ds[VAR_NAME]          # dims: time, y, x
    stack = da.values          # (T, H, W)
    T, H, W = stack.shape

    # 2) 参考 GeoTIFF（空间参考）
    if not os.path.exists(REF_TIF):
        raise FileNotFoundError(f"Reference TIFF not found: {REF_TIF}")

    with rasterio.open(REF_TIF) as src:
        ref_prof = src.profile.copy()
        px_size = float(abs(src.transform.a))  # 假设像元为正方形

    # 3) 计算出现频率 occ（0~1）
    #    当日水体定义：water_abun > water_thr
    occ = np.nanmean(stack > water_thr, axis=0).astype(np.float32)

    # 4) 全河床并集：出现频率超过极小阈值 occ_any_thr
    riverbed_full = (occ >= occ_any_thr).astype(np.uint8)

    # 5) 保存栅格
    occ_path = os.path.join(OUT_DIR, "occ.tif")
    riverbed_path = os.path.join(OUT_DIR, "riverbed_full.tif")

    _save_tif_like(ref_prof, occ_path, occ, dtype="float32", nodata=np.nan)
    _save_tif_like(ref_prof, riverbed_path, riverbed_full, dtype="uint8", nodata=0)

    # 6) 统计面积 & 写 meta
    area_full = _area_ha(riverbed_full, px_size)

    meta_lines = [
        f"RUN_TAG       : {RUN_TAG}",
        f"RUN_TS        : {RUN_TS}",
        f"OUT_DIR       : {OUT_DIR}",
        f"NC_PATH       : {NC_PATH}",
        f"REF_TIF       : {REF_TIF}",
        f"VAR_NAME      : {VAR_NAME}",
        f"px_size_m     : {px_size:.3f}",
        "--- thresholds ---",
        f"water_thr     : {water_thr}",
        f"occ_any_thr   : {occ_any_thr}",
        "--- areas (ha) ---",
        f"riverbed_full : {area_full:.2f}",
    ]

    meta_path = os.path.join(OUT_DIR, "run_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines))

    # 7) 打印摘要
    print("=== Build FULL RIVER BED mask ===")
    print(f"T (dates): {T}, HxW: {H}x{W}, pixel_size: {px_size:.2f} m")
    print(f"OUT_DIR  : {OUT_DIR}")
    print("Saved:")
    print(f"  - {os.path.basename(occ_path)}")
    print(f"  - {os.path.basename(riverbed_path)}")
    print(f"  - {os.path.basename(meta_path)}")
    print(f"Area (ha): riverbed_full = {area_full:.2f}")


if __name__ == "__main__":
    main()
