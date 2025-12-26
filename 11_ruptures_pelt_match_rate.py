# -*- coding: utf-8 -*-
# === Ruptures-PELT（自动对齐目标检出率 / 黑名单+3点中位数 / beta用未平滑方差 / 防覆盖）===
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

# ---------------- 配置区 ----------------
ROOT = Path(r"C:\Users\Duans\Desktop\bfastlibfastlite_project\lsu_abundance\riparian_out")

TS_CSV = None  # 不填则自动取最新 riparian_timeseries_samples_*.csv
TARGET_RATE = 157/2476   # ≈ 0.0634
TOL = 0.005
MIN_SIZE = 5

# 粗扫范围拉高；细扫范围更宽
PEN_LIST_COARSE = [2.0, 3.0, 4.0, 4.5, 4.824, 5.0, 5.2, 5.4, 5.6, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 12.0]
FINE_STEP = 0.1
FINE_WIDTH = 2.0

# 预处理设置
BLACKLIST_DATES = {"2024-08-13"}
SMOOTH_WIN = 3

ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

def pick_latest_ts(root: Path) -> Path:
    cand = list(root.glob("riparian_timeseries_samples_*.csv"))
    assert cand, f"在 {root} 下找不到 riparian_timeseries_samples_*.csv"
    return max(cand, key=lambda p: p.stat().st_mtime)

if TS_CSV is None:
    TS_CSV = pick_latest_ts(ROOT)
RUN_TAG = TS_CSV.stem.replace("riparian_timeseries_samples_", "")
OUT_DIR = ROOT / f"rup_{RUN_TAG}_{ts_tag}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("Using TS CSV:", TS_CSV)
print("OUT_DIR:", OUT_DIR)

# ---------- 读取 + 预处理（拆成 raw_pp 与 smooth 两份） ----------
def load_ts_only_pixels(csv_path: Path) -> pd.DataFrame:
    df0 = pd.read_csv(csv_path)
    time_like = [c for c in df0.columns if str(c).lower() in ["timestamp","time","date","datetime"]]
    if time_like:
        tcol = time_like[0]
        df0[tcol] = pd.to_datetime(df0[tcol], errors="coerce")
        df = df0.set_index(tcol).sort_index()
    else:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    pix_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("pix_")]
    if not pix_cols:
        raise ValueError("未发现像素列（列名形如 'pix_y_x'）。")
    return df[pix_cols]

df_raw = load_ts_only_pixels(TS_CSV)

# raw_pp：黑名单置 NaN（不平滑）
df_ts_raw_pp = df_raw.copy()
if BLACKLIST_DATES:
    bad_days = pd.to_datetime(sorted(BLACKLIST_DATES)).date
    mask_bad = df_ts_raw_pp.index.normalize().isin(pd.to_datetime(bad_days))
    df_ts_raw_pp.loc[mask_bad] = np.nan

# smooth：在 raw_pp 基础上 3 点居中中位数
df_ts_smooth = df_ts_raw_pp.rolling(SMOOTH_WIN, center=True, min_periods=1).median()

# 另存两份时序（便于复现）
(df_ts_raw_pp).to_csv(OUT_DIR / f"ts_blacklist_only_{RUN_TAG}.csv", date_format="%Y-%m-%d %H:%M:%S")
(df_ts_smooth).to_csv(OUT_DIR / f"ts_preprocessed_{RUN_TAG}.csv",   date_format="%Y-%m-%d %H:%M:%S")
print("Saved preprocessed TS:", OUT_DIR / f"ts_preprocessed_{RUN_TAG}.csv")

# ---------- 用“平滑序列拟合 + 未平滑方差定penalty”的 PELT ----------
def run_pelt_with_refvar(y_smooth, y_ref_for_var, min_size=5, penalty_scale=2.0):
    """拟合用 y_smooth；beta 用 y_ref_for_var 的方差计算。"""
    ys = np.asarray(y_smooth, float)
    yr = np.asarray(y_ref_for_var, float)
    n = len(ys)
    if n < min_size:
        return []
    algo = rpt.Pelt(model="l2", min_size=min_size).fit(ys)
    var_ref = float(np.nanvar(yr))  # 未平滑（但已黑名单）的方差
    if not np.isfinite(var_ref) or var_ref == 0:
        var_ref = float(np.nanvar(ys))  # 兜底
    beta = penalty_scale * var_ref * np.log(n)
    bkps = algo.predict(pen=beta)   # 包含 n
    return bkps[:-1]

def scan_penalties(df_smooth: pd.DataFrame, df_ref: pd.DataFrame, pen_list: list[float], min_size: int) -> pd.DataFrame:
    pixels = df_smooth.columns.tolist()
    T = df_smooth.shape[0]
    print(f"像素: {len(pixels)}，时间步: {T}，min_size={min_size}")

    rows = []
    for ps in pen_list:
        n_with = 0
        for col in pixels:
            s_s = df_smooth[col].dropna()
            s_r = df_ref[col].reindex(df_smooth.index).dropna()
            if len(s_s) < min_size:
                continue
            # 为了对齐方差与时间轴，这里简单按相同索引切片
            common = s_s.index.intersection(s_r.index)
            ys = s_s.loc[common].values
            yr = s_r.loc[common].values
            bkps = run_pelt_with_refvar(ys, yr, min_size=min_size, penalty_scale=ps)
            if len(bkps) > 0:
                n_with += 1
        rate = n_with / len(pixels) if len(pixels) else 0.0
        rows.append({"penalty_scale": ps, "n_total": len(pixels), "n_with_breaks": n_with, "rate": rate})
        print(f"ps={ps:<5} | 有断点像素: {n_with}/{len(pixels)} ({rate:.2%})")
    return pd.DataFrame(rows)

def run_and_collect_all(df_smooth: pd.DataFrame, df_ref: pd.DataFrame, pen_list: list[float], min_size: int, out_dir: Path, run_tag: str):
    pixels = df_smooth.columns.tolist()
    all_rows, summary_rows = [], []
    for ps in pen_list:
        n_with = 0
        for col in pixels:
            s_s = df_smooth[col].dropna()
            s_r = df_ref[col].reindex(df_smooth.index).dropna()
            if len(s_s) < min_size:
                all_rows.append({"pixel": col, "y": np.nan, "x": np.nan,
                                 "penalty_scale": ps, "break_index": np.nan,
                                 "break_time": pd.NaT, "num_breaks": 0})
                continue
            common = s_s.index.intersection(s_r.index)
            ys = s_s.loc[common].values
            yr = s_r.loc[common].values
            bkps = run_pelt_with_refvar(ys, yr, min_size=min_size, penalty_scale=ps)
            if len(bkps) > 0:
                n_with += 1
            try:
                yy, xx = col.replace("pix_", "").split("_")
                yy, xx = int(yy), int(xx)
            except Exception:
                yy, xx = np.nan, np.nan

            if bkps:
                for b in bkps:
                    bi = int(np.clip(int(b), 0, len(common)-1))
                    t_break = common[bi]
                    all_rows.append({
                        "pixel": col, "y": yy, "x": xx,
                        "penalty_scale": ps,
                        "break_index": bi,
                        "break_time": pd.to_datetime(t_break),
                        "num_breaks": len(bkps)
                    })
            else:
                all_rows.append({
                    "pixel": col, "y": yy, "x": xx,
                    "penalty_scale": ps,
                    "break_index": np.nan,
                    "break_time": pd.NaT,
                    "num_breaks": 0
                })
        rate = n_with / len(pixels) if len(pixels) else 0.0
        summary_rows.append({"run_tag": run_tag, "penalty_scale": ps, "n_total": len(pixels),
                             "n_with_breaks": n_with, "rate": rate})
        print(f"ps={ps:<5} | 有断点像素: {n_with}/{len(pixels)} ({rate:.2%})")

    df_all = pd.DataFrame(all_rows)
    df_sum = pd.DataFrame(summary_rows)
    all_csv = out_dir / f"ruptures_pelt_all_{run_tag}.csv"
    sum_csv = out_dir / f"ruptures_pelt_summary_{run_tag}.csv"
    df_all.to_csv(all_csv, index=False, date_format="%Y-%m-%d %H:%M:%S")
    df_sum.to_csv(sum_csv, index=False)
    print("Saved:\n -", all_csv, "\n -", sum_csv)
    return df_all, df_sum

def choose_penalty_by_target(df_sum: pd.DataFrame, target: float) -> float:
    i = (df_sum["rate"] - target).abs().idxmin()
    return float(df_sum.loc[i, "penalty_scale"])

def fine_grid_around(ps_center: float, step=0.1, width=2.0) -> list[float]:
    lo = max(0.1, ps_center - width)
    hi = ps_center + width
    grid = np.arange(lo, hi + 1e-9, step)
    return sorted({float(np.round(x, 4)) for x in grid})

# ---------- 主流程 ----------
print("\n[Coarse] penalty 粗扫：", PEN_LIST_COARSE)
df_sum_coarse = scan_penalties(df_ts_smooth, df_ts_raw_pp, PEN_LIST_COARSE, MIN_SIZE)
coarse_csv = OUT_DIR / f"ruptures_pelt_summary_coarse_{RUN_TAG}.csv"
df_sum_coarse.to_csv(coarse_csv, index=False)
print("Saved:", coarse_csv)

ps0 = choose_penalty_by_target(df_sum_coarse, TARGET_RATE)
gap0 = float((df_sum_coarse.set_index("penalty_scale").loc[ps0, "rate"] - TARGET_RATE))
print(f"粗扫最接近 TARGET={TARGET_RATE:.2%} 的 penalty≈{ps0:.3f}（差值 {gap0:.2%}）")

if abs(gap0) > TOL:
    PEN_LIST_FINE = fine_grid_around(ps0, step=FINE_STEP, width=FINE_WIDTH)
    pen_union = sorted({float(np.round(x,4)) for x in PEN_LIST_COARSE + PEN_LIST_FINE})
    print("\n[Fine] penalty 细扫：", pen_union)
    df_all, df_sum_all = run_and_collect_all(df_ts_smooth, df_ts_raw_pp, pen_union, MIN_SIZE, OUT_DIR, RUN_TAG)
    i_best = (df_sum_all["rate"] - TARGET_RATE).abs().idxmin()
    ps_chosen = float(df_sum_all.loc[i_best, "penalty_scale"])
    rate_chosen = float(df_sum_all.loc[i_best, "rate"])
    print(f"\n✅ 选中 penalty={ps_chosen:.3f}，检测率={rate_chosen:.2%}（目标={TARGET_RATE:.2%}）")
else:
    df_all, df_sum_all = run_and_collect_all(df_ts_smooth, df_ts_raw_pp, PEN_LIST_COARSE, MIN_SIZE, OUT_DIR, RUN_TAG)
    ps_chosen = ps0
    rate_chosen = float(df_sum_coarse.set_index("penalty_scale").loc[ps0, "rate"])
    print(f"\n✅ 选中 penalty={ps_chosen:.3f}（已在容忍范围内），检测率={rate_chosen:.2%}（目标={TARGET_RATE:.2%}）")

# 导出“选中 penalty”的明细与两张小图
df_pick = df_all[(df_all["penalty_scale"] == ps_chosen) & (df_all["break_time"].notna())].copy()
pick_csv = OUT_DIR / f"ruptures_pelt_breaks_chosen_pen{str(ps_chosen).replace('.','p')}_{RUN_TAG}.csv"
df_pick.to_csv(pick_csv, index=False, date_format="%Y-%m-%d %H:%M:%S")
print("Chosen penalty detail saved ->", pick_csv)

plt.figure(figsize=(5, 3.2))
df_plot = df_sum_all.sort_values("penalty_scale")
plt.plot(df_plot["penalty_scale"], df_plot["rate"], marker="o")
plt.axhline(TARGET_RATE, linestyle="--", linewidth=1)
plt.axvline(ps_chosen, linestyle=":", linewidth=1)
plt.xlabel("penalty_scale"); plt.ylabel("Share of pixels with ≥1 break")
plt.title(f"PELT(L2) detection rate vs penalty (chosen={ps_chosen:.3f})")
plt.tight_layout()
plt.savefig(OUT_DIR / f"ruptures_rate_vs_penalty_{RUN_TAG}.png", dpi=220)
plt.close()

daily = (df_pick["break_time"].dt.floor("D").value_counts().sort_index())
plt.figure(figsize=(7.2, 3.2))
daily.plot(kind="bar")
plt.title(f"Daily breaks @ penalty={ps_chosen:.3f} (rate={rate_chosen:.2%})")
plt.xlabel("Date"); plt.ylabel("Break count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / f"ruptures_daily_hist_pen{str(ps_chosen).replace('.','p')}_{RUN_TAG}.png", dpi=220)
plt.close()

print("\n✅ Done. 所有输出已集中到：", OUT_DIR)
print(f"   - 选中 penalty = {ps_chosen:.3f}，检出率 ≈ {rate_chosen:.2%}（目标 {TARGET_RATE:.2%}）")
print(f"   - 预处理 blacklisted（未平滑）与平滑版时序均已导出，便于复现。")
