# -*- coding: utf-8 -*-
"""
检查 LSU 水体丰度 vs NDWI 一致性

功能：
- 以基准日为参考，计算 ΔAbun 与 ΔNDWI
- 判定水体“可疑暴涨”区域
- 输出：
  * 可疑掩膜 GeoTIFF
  * 并排对比图（PNG）
  * 汇总 CSV（每景一行）
"""

import os
import re
import glob
import shutil
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import pandas as pd

# ================= 参数区 =================
IMG_DIR   = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\register_image_by_ospy\masked_input"
ABUN_DIR  = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\lsu_abundance"
OUT_DIR   = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\lsu_abundance\qa_ndwi_check"

# 若 DATES 为空，则对 ABUN_DIR 中所有 *_water_abun_LSU.tif 进行检查
DATES = []            # 例如 ["20240810", "20240901"]；留空则全部
BASE_HINT = "20240810"  # 基准日关键字（用于在文件名中匹配基准日）

DELTA_ABUN_MIN = 0.20   # 判定“水体丰度暴涨”的阈值（ΔAbun）
DELTA_NDWI_OK  = -0.05  # 暴涨时 NDWI 需低于该值，否则视为“可疑”
MIN_VALID_FRAC = 0.2    # 有效像元比例阈值（低于此值则跳过该景）


# ================= 工具函数 =================
def try_clear_dir(path: str) -> None:
    """安全清空目录（跳过被占用文件）"""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        return
    for f in os.listdir(path):
        fp = os.path.join(path, f)
        try:
            if os.path.isfile(fp) or os.path.islink(fp):
                os.unlink(fp)
            elif os.path.isdir(fp):
                shutil.rmtree(fp)
        except PermissionError:
            print(f"[WARN] 文件被占用，跳过: {fp}")


def find_abun(date_token: str):
    """根据日期 token 在 ABUN_DIR 中查找对应 water_abun_LSU.tif 文件"""
    patt = os.path.join(ABUN_DIR, f"*{date_token}*water_abun_LSU.tif")
    cand = sorted(glob.glob(patt))
    return cand[0] if cand else None


def find_img(abun_path: str):
    """
    根据丰度图文件名在 IMG_DIR 中寻找对应原始影像。
    优先按替换后同名 tif，若找不到，则做模糊匹配。
    """
    name = os.path.basename(abun_path)
    core = re.sub(r"_water_abun_LSU\.tif$", ".tif", name)
    p = os.path.join(IMG_DIR, core)
    if os.path.exists(p):
        return p
    pats = sorted(glob.glob(os.path.join(IMG_DIR, f"*{core[:25]}*.tif")))
    return pats[0] if pats else None


def read_ndwi(img_path: str):
    """从原始影像计算 NDWI（McFeeters，基于 Green 与 NIR）"""
    with rasterio.open(img_path) as src:
        D = src.read(masked=True).astype("float32") / 10000.0
        nir, green = D[7], D[3]  # band 8, band 4
    return (green - nir) / (green + nir + 1e-6)


def read_abun(abun_path: str):
    """
    读取 LSU 水体丰度图：
    - 若为单波段，则直接使用该波段；
    - 若为三波段，则使用第 3 个波段（假设为水体丰度）。
    返回：水丰度（MaskedArray）和 profile。
    """
    with rasterio.open(abun_path) as src:
        A = src.read(masked=True).astype("float32")
        if A.shape[0] == 1:
            water = A[0]
        else:
            water = A[2]
        profile = src.profile
    return water, profile


def save_tif(profile, out_path: str, arr: np.ndarray) -> None:
    """保存单波段 GeoTIFF（float32, nodata=np.nan）"""
    prof = profile.copy()
    prof.update(count=1, dtype="float32", nodata=np.nan)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr.astype("float32"), 1)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """在样本数量/方差过小情况下返回 NaN 的相关性计算"""
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    return np.corrcoef(a, b)[0, 1]


# ================= 主流程 =================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try_clear_dir(OUT_DIR)

    # 收集丰度文件
    if DATES:
        abun_files = [find_abun(t) for t in DATES if find_abun(t)]
    else:
        abun_files = sorted(glob.glob(os.path.join(ABUN_DIR, "*water_abun_LSU.tif")))

    if not abun_files:
        raise RuntimeError("未找到水体丰度图（*water_abun_LSU.tif）")

    # 选定基准日
    base_file = next(
        (f for f in abun_files if BASE_HINT in os.path.basename(f)),
        abun_files[0]
    )
    print("[INFO] 基准日:", os.path.basename(base_file))

    abun_base, prof_base = read_abun(base_file)
    img_base = find_img(base_file)
    if not img_base:
        raise RuntimeError("基准日对应的原始影像未找到")

    ndwi_base = read_ndwi(img_base)
    base_data_abun = np.ma.filled(abun_base, np.nan)
    base_data_ndwi = np.ma.filled(ndwi_base, np.nan)

    summary = []

    # 遍历所有日期的水丰度图
    for abun_path in abun_files:
        name = os.path.basename(abun_path)
        print("\n[Check]", name)

        abun_t, prof_t = read_abun(abun_path)
        img_t = find_img(abun_path)
        if not img_t:
            print("  -> 找不到对应影像，跳过")
            continue
        ndwi_t = read_ndwi(img_t)

        abun_t_data = np.ma.filled(abun_t, np.nan)
        ndwi_t_data = np.ma.filled(ndwi_t, np.nan)

        # 有效像元：基准 & 当日的 Abun / NDWI 均非 NaN
        valid_mask = (
            ~np.isnan(base_data_abun) &
            ~np.isnan(base_data_ndwi) &
            ~np.isnan(abun_t_data) &
            ~np.isnan(ndwi_t_data)
        )

        if valid_mask.sum() < MIN_VALID_FRAC * valid_mask.size:
            print("  -> 有效像元太少，跳过")
            continue

        # 全图相关性（当日 Abun vs NDWI）
        corr_day = safe_corr(
            abun_t_data[valid_mask],
            ndwi_t_data[valid_mask]
        )

        # 变化量
        d_abun = abun_t_data - base_data_abun
        d_ndwi = ndwi_t_data - base_data_ndwi

        # “可疑暴涨”：水丰度涨幅大，但 NDWI 没有同步变“更水”
        suspicious = (d_abun > DELTA_ABUN_MIN) & (d_ndwi > DELTA_NDWI_OK)
        sus_frac = float(np.mean(suspicious[valid_mask]))

        print(f"  Corr={corr_day:.3f} | suspicious={sus_frac*100:.1f}%")

        # 保存可疑掩膜
        out_mask = os.path.join(
            OUT_DIR,
            name.replace("_water_abun_LSU.tif", "_suspicious.tif")
        )
        save_tif(prof_t, out_mask, np.where(suspicious, 1.0, np.nan))

        # 并排图
        fig, axs = plt.subplots(2, 3, figsize=(13, 8))
        base_date = os.path.basename(base_file)[:8]
        day_date = name[:8]

        im0 = axs[0, 0].imshow(base_data_abun, vmin=0, vmax=1.2, cmap="Blues")
        axs[0, 0].set_title(f"Water Abundance\n(Base {base_date})")
        axs[0, 0].axis("off")
        plt.colorbar(im0, ax=axs[0, 0], shrink=0.7)

        im1 = axs[0, 1].imshow(abun_t_data, vmin=0, vmax=1.2, cmap="Blues")
        axs[0, 1].set_title(f"Water Abundance\n(Day {day_date})")
        axs[0, 1].axis("off")
        plt.colorbar(im1, ax=axs[0, 1], shrink=0.7)

        im2 = axs[0, 2].imshow(d_abun, cmap="bwr", vmin=-1, vmax=1)
        axs[0, 2].set_title("Δ Water Abundance")
        axs[0, 2].axis("off")
        plt.colorbar(im2, ax=axs[0, 2], shrink=0.7)

        im3 = axs[1, 0].imshow(base_data_ndwi, vmin=-1, vmax=1, cmap="BrBG")
        axs[1, 0].set_title(f"NDWI (Base {base_date})")
        axs[1, 0].axis("off")
        plt.colorbar(im3, ax=axs[1, 0], shrink=0.7)

        im4 = axs[1, 1].imshow(ndwi_t_data, vmin=-1, vmax=1, cmap="BrBG")
        axs[1, 1].set_title(f"NDWI (Day {day_date})")
        axs[1, 1].axis("off")
        plt.colorbar(im4, ax=axs[1, 1], shrink=0.7)

        im5 = axs[1, 2].imshow(np.where(suspicious, 1, np.nan), cmap="Reds")
        axs[1, 2].set_title("Suspicious Areas")
        axs[1, 2].axis("off")
        plt.colorbar(im5, ax=axs[1, 2], shrink=0.7)

        plt.suptitle(name, fontsize=11)
        plt.tight_layout()
        fig_path = os.path.join(
            OUT_DIR,
            name.replace(".tif", "_quickview.png")
        )
        plt.savefig(fig_path, dpi=180)
        plt.close()

        # 汇总信息
        summary.append({
            "file": name,
            "corr_abun_ndwi": float(corr_day),
            "suspicious_frac": sus_frac,
            "delta_abun_median": float(np.nanmedian(d_abun[valid_mask])),
            "delta_abun_p95": float(np.nanpercentile(d_abun[valid_mask], 95)),
            "delta_ndwi_median": float(np.nanmedian(d_ndwi[valid_mask])),
        })

    # 写出汇总表
    if summary:
        out_csv = os.path.join(OUT_DIR, "lsu_ndwi_consistency_summary.csv")
        pd.DataFrame(summary).sort_values("file").to_csv(out_csv, index=False)
        print("\n[OK] 汇总表已写出 →", out_csv)
    else:
        print("\n[WARN] 没有可写的汇总结果")


if __name__ == "__main__":
    main()
