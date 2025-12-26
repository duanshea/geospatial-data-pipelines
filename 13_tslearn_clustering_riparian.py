# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# tslearn
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset


# —— 根目录（不变即可）——
BASE    = Path(r"C:\Users\Duans\Desktop\bfastlibfastlite_project\lsu_abundance")
OUT_DIR = BASE / "riparian_out"

# —— 输入：指向这次要聚类的时序 CSV（469 像素那份）——
TS_CSV = OUT_DIR / "riparian_timeseries_samples_BIC_h=0.08_0.10_0.12.csv"
# 若以后要自动取最新一份，可以改成：
# TS_CSV = sorted(OUT_DIR.glob("riparian_timeseries_samples_*.csv"),
#                 key=lambda p: p.stat().st_mtime)[-1]

# —— 生成运行标识（从文件名抽取），并新建**独立输出目录** —— 
RUN_TAG = TS_CSV.stem.replace("riparian_timeseries_samples_", "")
ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR_TAG = OUT_DIR / f"tslearn_clusters_{RUN_TAG}_{ts}"
OUT_DIR_TAG.mkdir(parents=True, exist_ok=True)

print("Using TS CSV:", TS_CSV)
print("Outputs to :", OUT_DIR_TAG)

# —— 输出文件名：全部写到新子目录，绝不覆盖旧 run —— 
LABELS_CSV    = OUT_DIR_TAG / f"tslearn_labels_{RUN_TAG}.csv"
CENTROIDS_CSV = OUT_DIR_TAG / f"tslearn_centroids_{RUN_TAG}.csv"

# （可选）若要后面做地图合并，用同 RUN_TAG 找到 meta：
# META_CSV = OUT_DIR / f"riparian_samples_{RUN_TAG}.csv"


# 预处理参数
SMOOTH_WIN        = 3               # 3 点中位数平滑（与你之前保持一致）
BLACKLIST_DATES   = {"2024-08-13"}  # 黑名单日期（整行置 NaN）
INTERP_METHOD     = "time"          # 按时间插值
FILL_REMAIN_NAN   = "ffill_bfill"   # 插值后残余 NaN 的兜底：'ffill_bfill' 或 'mean'

# 聚类参数
N_CLUSTERS        = 4               # 聚类数：先 4 类，后续可改 3/5/6
METRIC            = "dtw"           # 'dtw' 或 'euclidean' 或 'softdtw'
MAX_ITER          = 50
RANDOM_STATE      = 0
N_INIT            = 3               # 多次随机初始化取最好


# =========================
# 读入与预处理
# =========================
df = pd.read_csv(TS_CSV, index_col=0, parse_dates=True)
print(f"Loaded: {df.shape[1]} pixels × {df.shape[0]} timesteps")

# 黑名单日期 -> NaN
if BLACKLIST_DATES:
    bad_days = pd.to_datetime(sorted(BLACKLIST_DATES)).date
    mask_bad = df.index.normalize().isin(pd.to_datetime(bad_days))
    df.loc[mask_bad] = np.nan

# 3 点中位数平滑
df_smooth = df.rolling(SMOOTH_WIN, center=True, min_periods=1).median()

# 按时间插值 + 兜底填充
df_interp = df_smooth.copy()
df_interp = df_interp.interpolate(method=INTERP_METHOD, axis=0, limit_direction="both")

if FILL_REMAIN_NAN == "ffill_bfill":
    df_interp = df_interp.ffill().bfill()
elif FILL_REMAIN_NAN == "mean":
    col_means = df_interp.mean(axis=0)
    df_interp = df_interp.fillna(col_means)

# 若仍有 NaN，最后再用列均值兜底（保证 tslearn 不会报错）
if df_interp.isna().any().any():
    df_interp = df_interp.fillna(df_interp.mean(axis=0))


# =========================
# 组装为 tslearn 所需的 3D 数组
# 形状: (n_series, n_timestamps, 1)
# =========================
X = df_interp.T.values  # (像元数, 时间点)
X = X[:, :, None]       # 加 channel 维
X = to_time_series_dataset(X)  # 校验形状

# 标准化（按每条序列做 z-score）
scaler = TimeSeriesScalerMeanVariance()
X_scaled = scaler.fit_transform(X)


# =========================
# 聚类
# =========================
if METRIC == "softdtw":
    model = TimeSeriesKMeans(
        n_clusters=N_CLUSTERS,
        metric="softdtw",
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )
else:
    model = TimeSeriesKMeans(
        n_clusters=N_CLUSTERS,
        metric=METRIC,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )

labels = model.fit_predict(X_scaled)         # 每个像元的簇标签
centroids = model.cluster_centers_[:, :, 0]  # (k, 时间点)

# 反标准化质心（近似，为了更好对比原数据量级）
overall_mean = df_interp.values.mean()
overall_std  = df_interp.values.std() if df_interp.values.std() > 0 else 1.0
centroids_rescaled = centroids * overall_std + overall_mean


# =========================
# 输出结果
# =========================
# 1) 标签：逐像元
labels_df = pd.DataFrame({
    "series": df.columns,
    "cluster": labels
})
# 解析 row/col（列名形如 pix_row_col）
labels_df[["row", "col"]] = labels_df["series"].str.extract(r"pix_(\d+)_(\d+)").astype(int)
labels_df.to_csv(LABELS_CSV, index=False)
print("Saved:", LABELS_CSV)

# 2) 质心：逐簇时间序列（行索引为时间）
centroids_df = pd.DataFrame(
    centroids_rescaled.T,
    index=df_interp.index,
    columns=[f"cluster_{i}" for i in range(N_CLUSTERS)]
)
centroids_df.to_csv(CENTROIDS_CSV, index=True, date_format="%Y-%m-%d %H:%M:%S")
print("Saved:", CENTROIDS_CSV)

# 3) 打印簇大小（文本信息即可）
sizes = labels_df["cluster"].value_counts().sort_index()
print("\nCluster sizes:")
print(sizes.to_string())
