#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PlanetScope 影像配准脚本（AROSICS COREG_LOCAL）

功能：
1. 使用 UDM2 掩膜遮掉无效像素（云、阴影等）
2. 以指定参考影像为基准，对整个时间序列影像做本地配准
3. 导出：
   - 配准后的影像（*_registered.tif）
   - 配准质量统计表 all_pairs_coreg_stats.csv
   - 所有有效 Tie-points 表 all_coreg_tiepoints.csv
   - 每景的 Tie-points 可视化图 和 Shift 分布图（用于质检）
"""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import rasterio
from arosics import COREG_LOCAL
import matplotlib.pyplot as plt


# ======== 0. 参数设置（可根据需要修改，对应论文 Table 4.4） ========
GRID_RES = 150          # COREG_LOCAL 的网格分辨率（m）
RELIABILITY_MIN = 10    # 最小 Tie-point 可靠性阈值
MASK_INVALID_VALUE = 0  # 无 nodata 时，掩膜处填充值


# ======== 1. 设置路径 ========
# 原始 PlanetScope 影像目录
IMG_DIR = Path(
    r"C:\Users\Duans\Desktop\Final_Thesis\data\Isar-AugNov2024-KA2_psscene_analytic_8b_sr_udm2\PSScene"
)

# 输出根目录
OUT_DIR = Path(
    r"C:\Users\Duans\Desktop\bfastlibfastlite_project\register_image_by_ospy"
)
MASKED_DIR = OUT_DIR / "masked_input"
DEBUG_DIR = OUT_DIR / "tiepoints_debug"

# 参考影像（时间和名称需要和你论文中保持一致）
REF_NAME = "20240810_102132_44_2416_3B_AnalyticMS_SR_8b_clip.tif"
REF_PATH = IMG_DIR / REF_NAME


def apply_udm2_mask(img_path: Path, udm_path: Path, out_path: Path) -> None:
    """
    使用 UDM2 掩膜遮掉无效像素（0=invalid, 1=valid），
    在所有波段上统一应用掩膜。
    """
    with rasterio.open(img_path) as src_img, rasterio.open(udm_path) as src_mask:
        img = src_img.read()          # shape: (bands, H, W)
        mask = src_mask.read(1)       # 0=invalid, 1=valid

        # 扩展为所有波段
        full_mask = mask.astype(bool)[np.newaxis, :, :]

        fill_value = src_img.nodata if src_img.nodata is not None else MASK_INVALID_VALUE
        masked_img = np.where(full_mask, img, fill_value)

        profile = src_img.profile
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(masked_img)


def main():
    # ======== 2. 清空输出文件夹并创建 ========
    for d in [OUT_DIR, MASKED_DIR, DEBUG_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # ======== 3. 构建 target 影像列表 ========
    target_paths = sorted(
        [
            p
            for p in IMG_DIR.glob("*_3B_AnalyticMS_SR_8b_clip.tif")
            if p.name != REF_NAME
        ]
    )

    # ======== 4. 开始批量配准 ========
    stats_list = []
    tiepoints_all = []

    for sen_path in target_paths:
        sen_name = sen_path.name
        udm_name = sen_name.replace("_AnalyticMS_SR_8b_clip.tif", "_udm2_clip.tif")
        udm_path = sen_path.with_name(udm_name)

        masked_sen_path = MASKED_DIR / sen_name

        # 检查对应的掩膜是否存在
        if not udm_path.exists():
            print(f"⚠️ 缺失掩膜文件: {udm_path.name}，跳过")
            continue

        # 应用掩膜
        apply_udm2_mask(sen_path, udm_path, masked_sen_path)

        try:
            print(f"\n📌 正在配准：{sen_name}")
            out_reg_path = OUT_DIR / sen_name.replace(".tif", "_registered.tif")

            crl = COREG_LOCAL(
                im_ref=str(REF_PATH),
                im_tgt=str(masked_sen_path),
                grid_res=GRID_RES,
                r_b4match=6,  # Red band index（你的数据中红色为第 6 波段）
                s_b4match=6,  # Red band index
                path_out=str(out_reg_path),
                fmt_out="GTIFF",
                footprint_poly_ref=None,
                footprint_poly_tgt=None,
                mask_baddata_ref=None,
                mask_baddata_tgt=None,
            )

            # 执行配准
            crl.correct_shifts()

            # 可视化 Tie-points 分布
            crl.view_CoRegPoints(
                figsize=(8, 8),
                savefigPath=str(
                    DEBUG_DIR / sen_name.replace(".tif", "_tiepoints_overlay.png")
                ),
            )

            tab = crl.CoRegPoints_table.copy()
            if tab is None or tab.empty:
                print("⚠️ 无 Tie-points 返回，跳过")
                continue

            # 过滤无效或低可靠性 Tie-points
            filtered = tab[
                (tab["X_SHIFT_M"] != -9999)
                & (tab["Y_SHIFT_M"] != -9999)
                & (tab["RELIABILITY"] >= RELIABILITY_MIN)
            ].copy()

            if filtered.empty:
                print("⚠️ 有效 Tie-points 为空，跳过")
                continue

            filtered["sen_image"] = sen_name
            tiepoints_all.append(filtered)

            # 计算统计指标（mean dx/dy、RMSE、N）
            mx = filtered["X_SHIFT_M"].mean()
            my = filtered["Y_SHIFT_M"].mean()
            rmse = np.sqrt(
                (filtered["X_SHIFT_M"] ** 2 + filtered["Y_SHIFT_M"] ** 2).mean()
            )
            npts = len(filtered)

            stats_list.append(
                {
                    "ref_image": REF_NAME,
                    "sen_image": sen_name,
                    "mean_dx": mx,
                    "mean_dy": my,
                    "rmse": rmse,
                    "num_points": npts,
                }
            )

            # 保存位移分布图
            plt.figure(figsize=(6, 4))
            plt.hist2d(filtered["X_SHIFT_M"], filtered["Y_SHIFT_M"], bins=20, cmap="viridis")
            plt.colorbar(label="Tie-points Count")
            plt.xlabel("X Shift (m)")
            plt.ylabel("Y Shift (m)")
            plt.title("Tie-points Shift Distribution")
            plt.tight_layout()
            plt.savefig(
                DEBUG_DIR / sen_name.replace(".tif", "_shift_dist.png"),
                dpi=150,
            )
            plt.close()

            print(f"✅ 成功: dx={mx:.2f}, dy={my:.2f}, RMSE={rmse:.2f}, N={npts}")

        except Exception as e:
            print(f"❌ 配准失败: {sen_name} | 错误：{e}")

    # ======== 5. 加上参考影像自身记录 ========
    stats_list.append(
        {
            "ref_image": REF_NAME,
            "sen_image": REF_NAME,
            "mean_dx": 0.0,
            "mean_dy": 0.0,
            "rmse": 0.0,
            "num_points": 0,
        }
    )

    # ======== 6. 保存汇总结果 ========
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(OUT_DIR / "all_pairs_coreg_stats.csv", index=False)
    print(f"\n📦 所有配准完成，共 {len(stats_list)} 条记录，统计表已保存：")
    print(OUT_DIR / "all_pairs_coreg_stats.csv")

    if tiepoints_all:
        tiepoints_df = pd.concat(tiepoints_all, ignore_index=True)
        tiepoints_df.to_csv(OUT_DIR / "all_coreg_tiepoints.csv", index=False)
        print("📄 所有有效 Tie-points 已保存：")
        print(OUT_DIR / "all_coreg_tiepoints.csv")
    else:
        print("⚠️ 无有效 Tie-points 结果")


if __name__ == "__main__":
    main()
