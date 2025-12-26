# -*- coding: utf-8 -*-
"""
构建水体丰度时间序列与纯水像元样本

功能：
1. 从多个 *_water_abun_LSU.tif 读取时间栈 (T,H,W)，可选叠加 riparian 掩膜；
2. 剔除指定的 hard_drop 日期；
3. 计算稳定像元 (valid_ratio >= MIN_COVERAGE) 的区域中位数时间序列；
4. 抽样“纯水像元”（在大部分时相上 water_abun >= PURE_THRESHOLD），导出像元级时间序列；
5. 将稳定像元的水体丰度时间栈保存为多波段 GeoTIFF，并写出 band 对应的时间戳 txt。
"""

import os
import re
import glob
import time
import math
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
import xarray as xr  # 当前脚本未使用，但保留以便扩展


# -------------------
# 配置参数
# -------------------
ABUN_DIR = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\lsu_abundance"  # water_abun_LSU.tif 所在目录
RIPARIAN_MASK_PATH = os.path.join(ABUN_DIR, "riparian_mask.tif")             # 可选河岸掩膜

RUN_TAG = "region_series_median"                     # 自定义本次运行标签（便于对比）
RUN_TS  = time.strftime("%Y%m%d_%H%M%S")             # 自动加时间戳
OUT_DIR = os.path.join(ABUN_DIR, "runs", f"{RUN_TAG}_{RUN_TS}")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

EXCLUDE_DATES = {
    "hard_drop": ["20240911", "20241020"],                  # 严格剔除
    "keep_but_note": ["20240830", "20240922", "20241105"],  # 仅标记
}

INTERP_REGION_MEDIAN = False      # 区域中位数是否时间插值（False=保留缺口）
MIN_COVERAGE = 0.60               # 稳定像元阈值
PURE_THRESHOLD = 0.85             # 纯水丰度阈值
PURE_TIME_RATIO = 0.60            # 纯水像元需在多少比例时相 >= PURE_THRESHOLD
PURE_SAMPLE_MAX = 200             # 纯水像元最大抽样数量

NETCDF_ENCODING = {
    "water_abun": {"zlib": True, "complevel": 4, "dtype": "float32"}
}


# -------------------
# 小工具
# -------------------
def parse_datetime_from_name(p: str) -> str | None:
    """
    从文件名中解析时间戳:
    - 先匹配 YYYYMMDD_HHMMSS → YYYYMMDDHHMMSS
    - 否则匹配 YYYYMMDD_ → YYYYMMDD000000
    """
    bn = os.path.basename(p)
    m = re.search(r"(\d{8})_(\d{6})", bn)
    if m:
        return m.group(1) + m.group(2)
    m2 = re.search(r"(\d{8})_", bn)
    return (m2.group(1) + "000000") if m2 else None


def _almost_equal_transform(a: Affine, b: Affine, tol: float = 1e-6) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def read_mask_if_exists(path: str, like_profile: dict):
    """
    若掩膜存在，则检查尺寸/CRS/transform 一致性后返回布尔阵列；否则返回 None。
    """
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        assert src.width == like_profile["width"] and src.height == like_profile["height"], \
            "Riparian mask size mismatch."
        assert src.crs == like_profile["crs"], "Riparian mask CRS mismatch."
        assert _almost_equal_transform(src.transform, like_profile["transform"]), \
            "Riparian mask transform mismatch."
        return src.read(1, masked=True).filled(0).astype(bool)


# -------------------
# 主流程
# -------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # -------------------
    # 1) 收集文件 & 排序
    # -------------------
    files_all = glob.glob(os.path.join(ABUN_DIR, "*_water_abun_LSU.tif"))
    files_all = [(p, parse_datetime_from_name(p)) for p in files_all]
    files_all = [(p, dt) for p, dt in files_all if dt is not None]
    assert files_all, "No valid abundance files found."

    files_all.sort(key=lambda x: x[1])  # 'YYYYMMDDHHMMSS'
    files, dt_raw = zip(*files_all)

    # -------------------
    # 2) 读栈 (T,H,W)，一致性检查 + nodata->NaN
    # -------------------
    stack_list = []
    meta0 = None
    crs0 = None
    tfm0 = None

    for p in files:
        with rasterio.open(p) as src:
            arr = src.read(1, masked=True).filled(np.nan).astype("float32")
            if meta0 is None:
                meta0 = src.profile
                H, W = src.height, src.width
                crs0, tfm0 = src.crs, src.transform
            else:
                assert src.width == W and src.height == H, f"Size mismatch: {p}"
                assert src.crs == crs0, f"CRS mismatch: {p}"
                assert _almost_equal_transform(src.transform, tfm0), f"Transform mismatch: {p}"
            stack_list.append(arr)

    stack = np.stack(stack_list, axis=0)  # (T,H,W)
    T, H, W = stack.shape

    # -------------------
    # 3) riparian 掩膜（可选）
    # -------------------
    riparian_mask = read_mask_if_exists(RIPARIAN_MASK_PATH, meta0)
    if riparian_mask is not None:
        stack = np.where(riparian_mask, stack, np.nan)

    # -------------------
    # 4) QA 剔除 hard_drop 日期
    # -------------------
    dates8 = [dt[:8] for dt in dt_raw]
    keep_idx = [i for i, d8 in enumerate(dates8) if d8 not in EXCLUDE_DATES["hard_drop"]]
    stack = stack[keep_idx]
    dt_raw = [dt_raw[i] for i in keep_idx]
    dates8 = [dates8[i] for i in keep_idx]

    # -------------------
    # 5) 稳定像元掩膜
    # -------------------
    valid_ratio = np.isfinite(stack).sum(0) / stack.shape[0]
    stable_mask = valid_ratio >= MIN_COVERAGE

    # -------------------
    # 6) 区域中位数时序
    # -------------------
    time_index = pd.to_datetime(dt_raw, format="%Y%m%d%H%M%S")
    vals = np.nanmedian(np.where(stable_mask, stack, np.nan), axis=(1, 2)).astype("float32")
    ts_region = pd.Series(vals, index=time_index, name="water_abun_median").sort_index()
    if INTERP_REGION_MEDIAN:
        ts_region = ts_region.interpolate("time", limit_direction="both")

    # 输出：区域中位数 + 标记日期
    out_ts_csv = os.path.join(OUT_DIR, "timeseries_region_median.csv")
    ts_region.to_csv(
        out_ts_csv,
        index_label="timestamp",
        date_format="%Y-%m-%d %H:%M:%S",
        na_rep="",
    )

    keep_but_note = set(EXCLUDE_DATES["keep_but_note"])
    flags = pd.Series(
        ["note" if d8 in keep_but_note else "" for d8 in dates8],
        index=time_index,
        name="flag",
    ).sort_index()
    flags.to_csv(
        os.path.join(OUT_DIR, "timeseries_region_flags.csv"),
        index_label="timestamp",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    # -------------------
    # 7) 纯水像元抽样导出
    # -------------------
    is_water_often = (stack >= PURE_THRESHOLD).sum(0) >= int(PURE_TIME_RATIO * stack.shape[0])
    pure_water_mask = stable_mask & is_water_often

    ys, xs = np.where(pure_water_mask)
    if len(ys) > 0:
        sel_n = min(PURE_SAMPLE_MAX, len(ys))
        sel_idx = np.random.choice(len(ys), size=sel_n, replace=False)
        sel_coords = list(zip(ys[sel_idx], xs[sel_idx]))

        df_px = pd.DataFrame(index=time_index)
        for (yy, xx) in sel_coords:
            df_px[f"pix_{yy}_{xx}"] = stack[:, yy, xx].astype("float32")
        df_px = df_px.sort_index()
        df_px.to_csv(
            os.path.join(OUT_DIR, "timeseries_pixels_pure_water_sample.csv"),
            index_label="timestamp",
            date_format="%Y-%m-%d %H:%M:%S",
            na_rep="",
        )
        pd.DataFrame(sel_coords, columns=["row", "col"]).to_csv(
            os.path.join(OUT_DIR, "pure_water_sample_coords.csv"),
            index=False,
        )
    else:
        print("Warning: no pixels pass pure_water_mask; skip pixel-level CSV export.")

    # -------------------
    # 8) 保存为多波段 GeoTIFF（每个时相 = 1 个 band）
    # -------------------
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    out_tif = os.path.join(OUT_DIR, "water_abun_stack_stable.tif")

    # 写之前先把非稳定像元设 NaN（与 NetCDF 逻辑一致）
    stack_out = stack.astype("float32").copy()
    mask_inv = ~stable_mask
    stack_out[:, mask_inv] = np.nan

    # NaN -> nodata
    nodata_val = -9999.0
    stack_out = np.where(np.isfinite(stack_out), stack_out, nodata_val).astype("float32")

    # 建议平铺 + LZW 压缩 + BigTIFF（防大文件）
    profile = meta0.copy()
    profile.update(
        driver="GTiff",
        count=stack_out.shape[0],
        dtype="float32",
        nodata=nodata_val,
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="YES",
    )

    with rasterio.open(out_tif, "w", **profile) as dst:
        for b in range(stack_out.shape[0]):
            dst.write(stack_out[b], b + 1)

    # 顺手把时间戳也保存一份 txt，方便对应 band->time
    with open(
        os.path.join(OUT_DIR, "water_abun_stack_band_timestamps.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        for i, t in enumerate(pd.to_datetime(dt_raw, format="%Y%m%d%H%M%S")):
            f.write(f"band {i+1}: {t.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # -------------------
    print("Done:")
    print(f"  stacked timestamps : {len(time_index)}")
    print(f"  OUT_DIR            : {OUT_DIR}")
    print("  saved files        :")
    print("    - timeseries_region_median.csv")
    print("    - timeseries_region_flags.csv")
    print("    - timeseries_pixels_pure_water_sample.csv (if any)")
    print("    - pure_water_sample_coords.csv (if any)")
    print("    - water_abun_stack_stable.tif (compressed)")
    print("    - water_abun_stack_band_timestamps.txt")


if __name__ == "__main__":
    main()
