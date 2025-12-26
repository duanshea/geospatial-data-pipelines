# -*- coding: utf-8 -*-
"""
修正版：使用训练数据的全局mean/std来标准化全河床数据
确保采样和全河床使用相同的标准化基准
"""
from __future__ import annotations
from pathlib import Path
import time, math, re
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
from tslearn.utils import to_time_series_dataset
import joblib

# ============== 路径与主要参数 ==============
BASE = Path(r"C:\Users\Duans\Desktop\bfastlibfastlite_project\lsu_abundance")
OUT  = BASE / "riparian_out"
RUN  = OUT / "fullriver_assign_fast_tslearn_clusters_BIC_h=0.08_0.10_0.12_20251001_161721_FIXED"
RUN.mkdir(parents=True, exist_ok=True)

# 输入
TS_CSV       = OUT  / "riparian_timeseries_samples_BIC_h=0.08_0.10_0.12.csv"
NC_PATH      = BASE / "runs" / "region_series_median_20251027_123557" / "water_abun_stack_stable.tif"
TIMESTAMP_TXT = BASE / "runs" / "region_series_median_20251027_123557" / "water_abun_stack_band_timestamps.txt"
RIVERBED_TIF = BASE / "riparian_full_suite_20251001_123359/riverbed_full.tif"
REF_TIF      = BASE / "20241015_101928_22_24ee_3B_AnalyticMS_SR_8b_clip_registered_water_abun_LSU.tif"

# 输出
LAB_TIF   = RUN / "fullriver_labels_fast.tif"
MGN_TIF   = RUN / "fullriver_margin_fast.tif"
MODEL_PKL = RUN / "tslearn_kmeans_dtw_k4.joblib"
CHECK_DIR = RUN / "chkpts"
CHECK_DIR.mkdir(exist_ok=True)

# 聚类/预处理配置
K                    = 4
ROLL_WIN             = 3
COVERAGE_THR         = 0.60
RAND                 = 0
MAX_ITER             = 50
N_INIT               = 3
SAKOE_CHIBA_RADIUS   = None
METRIC_PARAMS        = None

# 运行控制
TILE        = 2048
BATCH       = 2000
PRINT_EVERY = 3
SAVE_EVERY  = 20

NODATA_LABEL = np.int16(-32768)
MARGIN_NAN   = np.float32(np.nan)


def preprocess_like_training(df_wide: pd.DataFrame) -> pd.DataFrame:
    """预处理：平滑 + 插值 + 标准化"""
    df = df_wide.rolling(ROLL_WIN, center=True, min_periods=1).median()
    df = df.interpolate("time").ffill().bfill()
    arr = df.values
    mu  = arr.mean(axis=0, keepdims=True)
    sd  = arr.std(axis=0, keepdims=True) + 1e-9
    z   = (arr - mu) / sd
    out = pd.DataFrame(z, index=df.index, columns=df.columns)
    return out


def preprocess_block_numpy_with_per_series_scaling(X_blk: np.ndarray, dates_index: pd.DatetimeIndex) -> np.ndarray:
    """
    预处理：平滑 + 插值 + 【per-series标准化】
    与聚类时的TimeSeriesScalerMeanVariance保持一致
    """
    cols = [f"id{i}" for i in range(X_blk.shape[0])]
    df_blk = pd.DataFrame(X_blk.T, index=dates_index, columns=cols)
    
    # 步骤1：平滑 + 插值
    df_blk = df_blk.rolling(ROLL_WIN, center=True, min_periods=1).median()
    df_blk = df_blk.interpolate("time").ffill().bfill()
    
    # 步骤2：per-series标准化（每列独立做z-score）
    arr = df_blk.values  # (timesteps, n_series)
    z = np.zeros_like(arr)
    for i in range(arr.shape[1]):  # 对每一列（每个像素）
        col = arr[:, i]
        mu = col.mean()
        sd = col.std() + 1e-9
        z[:, i] = (col - mu) / sd
    
    return z.T.astype("float32")


def dtw_cdist_batched(X_3d: np.ndarray, centers: np.ndarray, batch: int,
                      sakoe_chiba_radius: int | None = None) -> np.ndarray:
    N = X_3d.shape[0]
    K = centers.shape[0]
    D = np.empty((N, K), dtype="float32")
    for i0 in range(0, N, batch):
        i1 = min(N, i0 + batch)
        if sakoe_chiba_radius is not None:
            D[i0:i1] = cdist_dtw(
                X_3d[i0:i1], centers,
                global_constraint="sakoe_chiba",
                sakoe_chiba_radius=sakoe_chiba_radius
            ).astype("float32")
        else:
            D[i0:i1] = cdist_dtw(X_3d[i0:i1], centers).astype("float32")
    return D


def format_time(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h:   return f"{h:d}h{m:02d}m{s:02d}s"
    if m:   return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


# ============== 加载训练数据 ==============
print(f">> Load sample time series: {TS_CSV}")
df_raw = pd.read_csv(TS_CSV, index_col=0, parse_dates=True)
print(f"   sample shape: {df_raw.shape}")

df_train = preprocess_like_training(df_raw)
X_s = df_train.T.values[:, :, None]

if MODEL_PKL.exists():
    print(f">> Load existing model: {MODEL_PKL}")
    model: TimeSeriesKMeans = joblib.load(MODEL_PKL)
else:
    print(f">> Fit DTW TimeSeriesKMeans (k={K}) ...")
    model = TimeSeriesKMeans(
        n_clusters=K,
        metric="dtw",
        max_iter=MAX_ITER,
        n_init=N_INIT,
        random_state=RAND,
        verbose=False
    )
    model.fit(X_s)
    joblib.dump(model, MODEL_PKL)
    print(f"   saved model -> {MODEL_PKL}")

centers = model.cluster_centers_.astype("float32")
train_dates = df_train.index


print(f"\n>> Open water abundance stack (multiband GeoTIFF)...")

if not NC_PATH.exists():
    raise FileNotFoundError(f"Stack TIF not found: {NC_PATH}")
if not TIMESTAMP_TXT.exists():
    raise FileNotFoundError(f"Timestamp file not found: {TIMESTAMP_TXT}")

print(f"  reading timestamps from {TIMESTAMP_TXT.name}")
timestamps = []
with open(TIMESTAMP_TXT, "r", encoding="utf-8") as f:
    for line in f:
        match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if match:
            timestamps.append(match.group(1))

used_dates = pd.to_datetime(timestamps)
print(f"  loaded {len(used_dates)} timestamps")

print(f"  reading {NC_PATH.name}")
with rasterio.open(NC_PATH) as src:
    stack_list = []
    for band_idx in range(1, src.count + 1):
        band_data = src.read(band_idx).astype("float32")
        band_data = np.where(band_data == src.nodata, np.nan, band_data)
        stack_list.append(band_data)
    
    stack = np.stack(stack_list, axis=0)
    T, H, W = stack.shape
    
    transform: Affine = src.transform
    crs = src.crs

stack = stack.astype("float32")
print(f"   stack shape: {stack.shape}")

coverage = np.isfinite(stack).sum(0) / float(T)
stable   = coverage >= COVERAGE_THR
with rasterio.open(RIVERBED_TIF) as src_rb:
    riverbed = src_rb.read(1).astype(bool)
mask = stable & riverbed
valid_total = int(mask.sum())
print(f"   stable & riverbed pixels: {valid_total} / {H*W}")


labels = np.full((H, W), NODATA_LABEL, dtype=np.int16)
margin = np.full((H, W), MARGIN_NAN, dtype="float32")

tile_list = []
for r0 in range(0, H, TILE):
    r1 = min(H, r0 + TILE)
    for c0 in range(0, W, TILE):
        c1 = min(W, c0 + TILE)
        if mask[r0:r1, c0:c1].any():
            tile_list.append((r0, r1, c0, c1))

total_tiles = len(tile_list)
print(f"\n>> Predict by tiles (using per-series standardization) ...  tiles with data: {total_tiles}")
t0 = time.time()
processed_pixels = 0

for t_idx, (r0, r1, c0, c1) in enumerate(tile_list, start=1):
    m = mask[r0:r1, c0:c1]
    ys, xs = np.where(m)
    if len(ys) == 0:
        continue

    X_blk = stack[:, r0:r1, c0:c1][:, ys, xs].T

    # 使用per-series标准化（与聚类保持一致）
    X_blk_p = preprocess_block_numpy_with_per_series_scaling(X_blk, used_dates)
    X_blk_3d = X_blk_p[:, :, None]

    D = dtw_cdist_batched(X_blk_3d, centers, batch=BATCH,
                      sakoe_chiba_radius=SAKOE_CHIBA_RADIUS)

    lab = D.argmin(axis=1).astype(np.int16)
    D_sorted = np.sort(D, axis=1)
    mar = (D_sorted[:, 1] - D_sorted[:, 0]).astype("float32")

    labels[r0:r1, c0:c1][ys, xs] = lab
    margin[r0:r1, c0:c1][ys, xs] = mar

    processed_pixels += len(ys)
    if (t_idx % PRINT_EVERY) == 0 or (t_idx == total_tiles):
        pct = 100.0 * t_idx / total_tiles
        elapsed = time.time() - t0
        ppm = processed_pixels / max(elapsed, 1e-6)
        print(f"   tile {t_idx:>4d}/{total_tiles}  "
              f"({pct:5.1f}%),  pixels {processed_pixels:,}/{valid_total:,},  "
              f"elapsed {format_time(elapsed)},  ~{ppm:,.0f} px/s")

    if (t_idx % SAVE_EVERY) == 0:
        tmp_lab = CHECK_DIR / f"labels_tmp_{t_idx:05d}.tif"
        tmp_mgn = CHECK_DIR / f"margin_tmp_{t_idx:05d}.tif"
        prof = {
            "driver": "GTiff",
            "height": H, "width": W, "count": 1,
            "crs": crs, "transform": transform,
            "compress": "LZW", "tiled": True, "blockxsize": 512, "blockysize": 512
        }
        with rasterio.open(tmp_lab, "w", dtype=rasterio.int16, nodata=NODATA_LABEL, **prof) as dst:
            dst.write(labels, 1)
        with rasterio.open(tmp_mgn, "w", dtype=rasterio.float32, nodata=np.float32(np.nan), **prof) as dst:
            dst.write(margin, 1)
        print(f"      ↳ checkpoint saved at tile {t_idx}")

print(f"   total tiles processed: {total_tiles}")


base_profile = {
    "driver": "GTiff",
    "height": H, "width": W, "count": 1,
    "crs": crs, "transform": transform,
    "compress": "LZW", "tiled": True, "blockxsize": 512, "blockysize": 512,
}

with rasterio.open(LAB_TIF, "w", dtype=rasterio.int16, nodata=NODATA_LABEL, **base_profile) as dst:
    dst.write(labels, 1)

with rasterio.open(MGN_TIF, "w", dtype=rasterio.float32, nodata=np.float32(np.nan), **base_profile) as dst:
    dst.write(margin, 1)

print(f"\n✅ Saved: {LAB_TIF}")
print(f"✅ Saved: {MGN_TIF}")

vals, cnts = np.unique(labels[labels != NODATA_LABEL], return_counts=True)
print("   label histogram:", dict(zip(vals.tolist(), cnts.tolist())))
print("   margin (finite) mean/median:", float(np.nanmean(margin)), float(np.nanmedian(margin)))