# -*- coding: utf-8 -*-
"""
从 riverbed_full + water_abun_stack_stable.nc 进行【分层抽样】生成像素集（row,col）。

分层维度：
  1) 出现频率 occ（三档）：(0,0.2], (0.2,0.6], (0.6,1.0]
  2) 空间网格 Ny×Nx（沿程均衡）

核心约束（论文可写）：
  - 仅在 riverbed_full==1 且 stable_mask（>=min_coverage）且 occ 有效的像元中抽样；
  - 采用“Round-Robin 轮转式全局最小间距”：跨所有 occ 档 & 跨所有 grid cell
    共同满足最小欧氏间距 r（像元），兼顾各 bin×cell 的均衡性。

产物：
  - sample_pixels.csv（row,col,y,x,Easting,Northing,occ,occ_bin,cell_id）
  - sampling_summary.csv（每个 bin×cell 的 候选/采样 数）
  - run_meta.txt（参数与路径）
"""

import os
import time
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.transform import xy

# ================== 路径（按需修改） ==================
BASE_DIR   = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\lsu_abundance"
NC_PATH    = os.path.join(BASE_DIR, "water_abun_stack_stable.nc")  # 变量名：water_abun
RIVERBED_TIF = os.path.join(BASE_DIR, "riparian_full_suite_20251001_123359", "riverbed_full.tif")
REF_TIF      = os.path.join(BASE_DIR, "20241015_101928_22_24ee_3B_AnalyticMS_SR_8b_clip_registered_water_abun_LSU.tif")

RUN_TAG = "stratified_sampling"
RUN_TS  = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(BASE_DIR, f"{RUN_TAG}_{RUN_TS}")
os.makedirs(OUT_DIR, exist_ok=True)

# ================== 参数 ==================
RANDOM_SEED   = 42
np.random.seed(RANDOM_SEED)

water_thr     = 0.60   # 判定“当日为水”的丰度阈值（用于 occ）
min_coverage  = 0.60   # 稳定像元（>=60% 有效观测）

# occ 分层（统一 (lo, hi]；occ==0 不纳入第一档）
OCC_BINS      = [(0.0, 0.2), (0.2, 0.6), (0.6, 1.0)]

# 空间网格分区（Ny×Nx）
GRID_NY, GRID_NX = 3, 5

# 每个 bin×cell 的目标数（最大尝试）
TARGET_PER_CELL = 120

# 最小间距（像元；欧氏）
MIN_DIST_PIX = 3

# ================== 小工具 ==================
def make_disk_kernel(r: int) -> np.ndarray:
    """返回半径 r 的布尔圆盘核"""
    yy, xx = np.ogrid[-r:r+1, -r:r+1]
    return (xx*xx + yy*yy) <= r*r

def mark_disk(forbidden: np.ndarray, y: int, x: int, kernel: np.ndarray, r: int):
    """在 forbidden 上以 (y,x) 为中心画半径 r 的圆盘禁区（边界安全）"""
    H, W = forbidden.shape
    r0, r1 = max(0, y-r), min(H, y+r+1)
    c0, c1 = max(0, x-r), min(W, x+r+1)
    ky0 = r - (y - r0); ky1 = ky0 + (r1 - r0)
    kx0 = r - (x - c0); kx1 = kx0 + (c1 - c0)
    forbidden[r0:r1, c0:c1] |= kernel[ky0:ky1, kx0:kx1]

# ================== 读取数据 ==================
# NetCDF：occ 与 stable_mask
ds = xr.open_dataset(NC_PATH)
if "water_abun" not in ds:
    raise ValueError("变量 'water_abun' 不在 NetCDF 中。")
stack = ds["water_abun"].values.astype("float32")  # (T,H,W); NaN=无效
T, H, W = stack.shape

valid_ratio = np.isfinite(stack).sum(0) / T
stable_mask = valid_ratio >= min_coverage
occ = np.nanmean(stack > water_thr, axis=0).astype("float32")

# riverbed_full 约束域
with rasterio.open(RIVERBED_TIF) as src_rb:
    riverbed = src_rb.read(1).astype(bool)

# 参考 TIF（坐标转换）
with rasterio.open(REF_TIF) as ref:
    transform = ref.transform
    crs = ref.crs

# 候选域
candidates = riverbed & stable_mask & np.isfinite(occ)

# ================== 构建空间网格 ==================
rows_edges = np.linspace(0, H, GRID_NY+1, dtype=int)
cols_edges = np.linspace(0, W, GRID_NX+1, dtype=int)
cell_masks, cell_ids = {}, []
for iy in range(GRID_NY):
    for ix in range(GRID_NX):
        r0, r1 = rows_edges[iy], rows_edges[iy+1]
        c0, c1 = cols_edges[ix], cols_edges[ix+1]
        m = np.zeros((H, W), dtype=bool); m[r0:r1, c0:c1] = True
        cid = f"y{iy+1}x{ix+1}"
        cell_masks[cid] = m
        cell_ids.append(cid)

# ================== Round-Robin 全局最小间距抽样 ==================
sel_rows, sel_cols, sel_bins, sel_cells, sel_occs = [], [], [], [], []
struct = make_disk_kernel(MIN_DIST_PIX)

# 1) 预计算每个 (bin, cell) 的候选（乱序）
cand_stats = []
candidates_dict = {}       # (bin_name, cid) -> list[(y,x)]
combo_order = []           # 轮转顺序

for (lo, hi) in OCC_BINS:
    bin_name = f"({lo:.1f},{hi:.1f}]"
    for cid in cell_ids:
        mask_cell = cell_masks[cid]
        mask_bin  = (occ > lo) & (occ <= hi)
        mask_all  = candidates & mask_bin & mask_cell

        ys, xs = np.where(mask_all)
        n_cand = int(len(ys))
        cand_stats.append({"occ_bin": bin_name, "cell_id": cid, "candidates": n_cand})

        if n_cand == 0:
            continue
        idx = np.arange(n_cand)
        np.random.shuffle(idx)
        ys, xs = ys[idx], xs[idx]
        candidates_dict[(bin_name, cid)] = list(zip(ys, xs))
        combo_order.append((bin_name, cid))

# 2) 各组合的计数器
picked_count = {k: 0 for k in candidates_dict.keys()}

# 3) 全局 forbidden（跨所有 bin & cell）
forbidden = np.zeros((H, W), dtype=bool)

# 4) Round-Robin 轮转取样：每轮每组合最多取 1 个，直到无进展
progress = True
while progress:
    progress = False
    for key in combo_order:
        if picked_count.get(key, 0) >= TARGET_PER_CELL:
            continue
        cand_list = candidates_dict.get(key, [])
        # 在该组合的候选里找第一个不冲突的
        accepted_idx = None
        for i, (y, x) in enumerate(cand_list):
            if not forbidden[y, x]:
                accepted_idx = i
                break
        if accepted_idx is None:
            continue  # 该组合当前都被“挡住”
        y, x = cand_list.pop(accepted_idx)

        # 记录
        bin_name, cid = key
        sel_rows.append(int(y)); sel_cols.append(int(x))
        sel_bins.append(bin_name); sel_cells.append(cid); sel_occs.append(float(occ[y, x]))
        picked_count[key] += 1

        # 更新全局禁区
        mark_disk(forbidden, y, x, struct, MIN_DIST_PIX)

        progress = True
    # 一轮下来无任何新增 -> 结束

# ================== 合成/导出 ==================
if len(sel_rows) == 0:
    raise RuntimeError("没有抽到任何像素，请放宽参数或检查候选域。")

xs_geo, ys_geo = xy(transform, sel_rows, sel_cols)

df = pd.DataFrame({
    "row": sel_rows,
    "col": sel_cols,
    "x": sel_cols,
    "y": sel_rows,
    "Easting": xs_geo,
    "Northing": ys_geo,
    "occ": sel_occs,
    "occ_bin": sel_bins,
    "cell_id": sel_cells,
})

# 统计：每个 bin×cell 的候选/采样
summary_rows = []
cand_df = pd.DataFrame(cand_stats)
for (lo, hi) in OCC_BINS:
    bin_name = f"({lo:.1f},{hi:.1f}]"
    for cid in cell_ids:
        in_bin_cell = (df["occ_bin"] == bin_name) & (df["cell_id"] == cid)
        sampled = int(in_bin_cell.sum())
        candidates_cnt = int(cand_df[(cand_df.occ_bin==bin_name) & (cand_df.cell_id==cid)]["candidates"].sum())
        summary_rows.append({
            "occ_bin": bin_name,
            "cell_id": cid,
            "candidates": candidates_cnt,
            "sampled": sampled
        })
summary = pd.DataFrame(summary_rows)

# 写文件
out_csv = os.path.join(OUT_DIR, "sample_pixels.csv")
sum_csv = os.path.join(OUT_DIR, "sampling_summary.csv")
df.to_csv(out_csv, index=False)
summary.to_csv(sum_csv, index=False)

# 元信息
with open(os.path.join(OUT_DIR, "run_meta.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join([
        f"RUN_TAG: {RUN_TAG}",
        f"RUN_TS:  {RUN_TS}",
        f"NC_PATH: {NC_PATH}",
        f"RIVERBED_TIF: {RIVERBED_TIF}",
        f"REF_TIF: {REF_TIF}",
        f"water_thr: {water_thr}",
        f"min_coverage: {min_coverage}",
        f"OCC_BINS: {OCC_BINS}",
        f"GRID_NY×GRID_NX: {GRID_NY}×{GRID_NX}",
        f"TARGET_PER_CELL: {TARGET_PER_CELL}",
        f"MIN_DIST_PIX: {MIN_DIST_PIX}",
        f"selected_total: {len(df)}",
        f"random_seed: {RANDOM_SEED}",
        "assume_clean_stack: true",
    ]))

print("=== Stratified sampling (Round-Robin, global spacing) done ===")
print(f"OUT_DIR: {OUT_DIR}")
print(f"selected pixels: {len(df)}")
print("saved: sample_pixels.csv, sampling_summary.csv, run_meta.txt")
