# -*- coding: utf-8 -*-
"""
Ruptures Binseg + linear（斜率断点）——自动对齐目标检出率（6.34%）
- 输入：riparian_out 下“最新”的 riparian_timeseries_samples_*.csv（2476 像素）
- 预处理：黑名单日期 + 3点中位数
- 搜索：Binseg + linear，MAX_M 段，MIN_SIZE 最小段长
- 选择：LWZ/BIC × scale 网格；若未命中目标 → 自动外扩；命中后再细化
- QC：TWSV 改善 + (|Δ均值| 或 |Δ斜率|) + 边界保护
- 输出：每档明细/汇总、最佳档标注、（可选）与 PELT 并表、小图（检出率 vs scale、最佳档日直方图）
"""

from pathlib import Path
from datetime import datetime
import re, time, json
import numpy as np
import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
from pandas import Timestamp

# =========================
# 路径 & 目标检出率（按需改 BASE）
# =========================
BASE    = Path(r"C:\Users\Duans\Desktop\bfastlibfastlite_project\lsu_abundance")
OUT_DIR = BASE / "riparian_out"

# 自动取“最新”的时序 CSV（2476像素那份）
TS_CSV  = max(OUT_DIR.glob("riparian_timeseries_samples_*.csv"),
              key=lambda p: p.stat().st_mtime)
RUN_TAG = TS_CSV.stem.replace("riparian_timeseries_samples_", "")

# —— 目标检出率（像素层面“≥1断点”的占比）——
TARGET_RATE = 0.0634  # 6.34%
TOL         = 0.005   # ±0.5 个百分点容忍

print("Using TS CSV:", TS_CSV)
print("RUN_TAG     :", RUN_TAG)

# 独立输出目录
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUT_DIR / f"rup_binseg_linear_{RUN_TAG}_{ts}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
print("OUT_DIR     :", RUN_DIR)

# =========================
# 与 BFAST 设置配套
# =========================
SEARCH_METHOD = "binseg"    # 或 "bottomup"
COST_MODEL    = "linear"    # 关键：linear（斜率突变）
MAX_M         = 4           # 最大段数（断点数 = MAX_M-1）
MIN_SIZE      = 5           # 与 T≈49、h≈0.08 对齐

# 初始 IC×scale 网格（不够时会自动外扩）
CRITERIA      = ["LWZ", "BIC"]
CRIT_SCALES   = [0.6, 0.8, 1.0, 1.2]   # 先小范围；后面自动外扩

# 预处理 & QC（linear 下适度放宽斜率阈值）
SMOOTH_WIN        = 3
BLACKLIST_DATES   = {"2024-08-13"}
EDGE_GUARD        = 4
DELTA_THRESH      = 0.12
SLOPE_DELTA_MIN   = 0.007
TWSV_IMPROVE_MIN  = 0.20

# =========================
# 可选：找最近的 PELT 汇总（用于对照）
# =========================
def find_latest_pelt_summary(root: Path) -> Path | None:
    cands = sorted(list(root.glob("rup_*/*summary*.csv")) + list(root.glob("ruptures_pelt_summary_*.csv")),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for p in cands:
        if "pelt" in p.name.lower():
            return p
    return None
PELT_SUMMARY_CSV = find_latest_pelt_summary(OUT_DIR)

# =========================
# 信息准则
# =========================
def lwz_penalty(rss, m, n, params_per_seg, scale=1.0):
    p = (m + 1) * params_per_seg
    rss_term = n * np.log(max(rss, 1e-12) / n)
    pen_term = 0.299 * (np.log(n) ** 2.1) * p
    return rss_term + scale * pen_term

def bic_penalty(rss, m, n, params_per_seg, scale=1.0):
    p = (m + 1) * params_per_seg
    rss_term = n * np.log(max(rss, 1e-12) / n)
    pen_term = p * np.log(n)
    return rss_term + scale * pen_term

# =========================
# 工具（NaN 处理、拟合、QC）
# =========================
def segment_rss_linear_or_l2(signal, bkps, cost_model="linear"):
    """对“压紧后”的 y 计算分段 RSS；bkps 为右端索引（含 n）。"""
    y = np.asarray(signal, float)
    starts = [0] + list(bkps[:-1]); ends = list(bkps)
    rss = 0.0
    for s, e in zip(starts, ends):
        seg = y[s:e]
        msk = np.isfinite(seg)
        if msk.sum() < 2:
            continue
        if cost_model.lower() == "l2":
            mu = np.nanmean(seg[msk])
            rss += np.nansum((seg[msk] - mu) ** 2)
        else:  # linear
            t = np.arange(e - s, dtype=float)[msk]
            A = np.vstack([t, np.ones_like(t)]).T
            coeff, *_ = np.linalg.lstsq(A, seg[msk], rcond=None)  # [a,b]
            yhat = A @ coeff
            rss += float(np.sum((seg[msk] - yhat) ** 2))
    return float(rss)

def pick_best_bkps(signal, search_method="binseg", max_m=4, min_size=5,
                   criterion="LWZ", scale=1.0, cost_model="linear"):
    """
    Binseg/BottomUp + linear：
      - 对 NaN 做掩膜，在“压紧”的子序列上拟合；
      - linear 成本传二维 [y, t]；l2 传 y.reshape(-1,1)
      - bkps（子序列右端索引）映回原时序右端索引
      - 返回：orig_inner + [n_orig]
    """
    y_orig = np.asarray(signal, float)
    n_orig = len(y_orig)
    idx_all = np.arange(n_orig)

    mask = np.isfinite(y_orig)             # ruptures 不支持 NaN
    y = y_orig[mask]
    idx_map = idx_all[mask]                # 子序列索引 → 原索引
    n = len(y)
    if n < min_size:
        return [n_orig], np.nan

    nbkps_max = max(0, n // min_size - 1)
    nbkps_try = list(range(0, min(max_m - 1, nbkps_max) + 1))
    if not nbkps_try:
        return [n_orig], np.nan

    if cost_model.lower() == "linear":
        t = np.arange(n, dtype=float)
        sig_for_fit = np.column_stack([y, t])  # 关键：二维
    else:
        sig_for_fit = y.reshape(-1, 1)

    if search_method.lower() == "binseg":
        model = rpt.Binseg(model=cost_model, min_size=min_size).fit(sig_for_fit)
    elif search_method.lower() == "bottomup":
        model = rpt.BottomUp(model=cost_model, min_size=min_size).fit(sig_for_fit)
    else:
        raise ValueError("search_method must be 'binseg' or 'bottomup'")

    penalty_fn = lwz_penalty if criterion.upper()=="LWZ" else bic_penalty
    params_per_seg = 1 if cost_model.lower()=="l2" else 2

    # baseline：0 断点（子序列）
    best_bkps_red = [n]
    base_rss = segment_rss_linear_or_l2(y, [n], cost_model=cost_model)
    best_score = penalty_fn(base_rss, 0, n, params_per_seg, scale=scale)

    for k in nbkps_try[1:]:
        try:
            bkps_red = model.predict(n_bkps=k)  # 包含 n
        except Exception:
            continue
        rss   = segment_rss_linear_or_l2(y, bkps_red, cost_model=cost_model)
        score = penalty_fn(rss, k, n, params_per_seg, scale=scale)
        if score < best_score:
            best_bkps_red, best_score = bkps_red, score

    # 映回原时序右端索引
    red_inner = [b for b in best_bkps_red[:-1] if 1 <= b <= n-1]
    if red_inner:
        orig_inner = [int(idx_map[b-1]) + 1 for b in red_inner]
    else:
        orig_inner = []
    return orig_inner + [n_orig], best_score

def ols_slope(y):
    y = np.asarray(y, float)
    t = np.arange(len(y), dtype=float)
    msk = np.isfinite(y)
    if msk.sum() < 2: return np.nan
    A = np.vstack([t[msk], np.ones_like(t[msk])]).T
    m, _ = np.linalg.lstsq(A, y[msk], rcond=None)[0]
    return float(m)

def twsv_ratio(y, cuts_idx):
    """Total Within-Segment Variance ratio."""
    y = np.asarray(y, float)
    N = len(y)
    if N < 3 or not np.any(np.isfinite(y)):
        return np.nan
    cuts = [0] + list(cuts_idx) + [N]
    num = 0.0
    for s0, s1 in zip(cuts[:-1], cuts[1:]):
        seg = y[s0:s1]
        m = np.isfinite(seg)
        if m.sum() >= 2:
            num += (m.sum()-1) * np.nanvar(seg[m], ddof=1)
    den = (N-1) * np.nanvar(y[np.isfinite(y)], ddof=1)
    return float(num/den) if den > 0 else np.nan

def filter_edge_indices(bidx, T, guard=EDGE_GUARD):
    return [int(b) for b in bidx if guard <= int(b) <= T-1-guard]

def summarize_breaks(idx_time, y, bidx, delta_thr=DELTA_THRESH):
    out=[]
    if not bidx: return out
    cuts = [0] + list(bidx) + [len(y)]
    for k,b in enumerate(bidx):
        pre  = y[cuts[k]:cuts[k+1]]
        post = y[cuts[k+1]:cuts[k+2]]
        mu_pre  = float(np.nanmean(pre))  if np.any(np.isfinite(pre))  else np.nan
        mu_post = float(np.nanmean(post)) if np.any(np.isfinite(post)) else np.nan
        dy = mu_post - mu_pre if (np.isfinite(mu_pre) and np.isfinite(mu_post)) else np.nan
        lab = "L→W" if (np.isfinite(dy) and dy >= delta_thr) else ("W→L" if (np.isfinite(dy) and dy <= -delta_thr) else "weak/unclear")
        out.append({
            "break_k": k+1,
            "break_index": int(b),
            "break_time": Timestamp(idx_time[int(b)]),
            "mu_pre": mu_pre, "mu_post": mu_post, "delta": dy,
            "slope_pre": ols_slope(pre), "slope_post": ols_slope(post),
            "label_raw": lab
        })
    return out

# =========================
# 数据读取 + 预处理（保存两份，用于复现）
# =========================
df_raw = pd.read_csv(TS_CSV, index_col=0, parse_dates=True)

def preprocess_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    # 黑名单
    if BLACKLIST_DATES:
        bad_days = pd.to_datetime(sorted(BLACKLIST_DATES)).date
        mask_bad = df.index.normalize().isin(pd.to_datetime(bad_days))
        df.loc[mask_bad] = np.nan
    # 3点中位数
    if SMOOTH_WIN and SMOOTH_WIN > 1:
        df = df.rolling(SMOOTH_WIN, center=True, min_periods=1).median()
    return df

df_black = df_raw.copy()
if BLACKLIST_DATES:
    bad_days = pd.to_datetime(sorted(BLACKLIST_DATES)).date
    mask_bad = df_black.index.normalize().isin(pd.to_datetime(bad_days))
    df_black.loc[mask_bad] = np.nan
df_smooth = preprocess_df(df_raw)

# 保存两份（黑名单后、平滑后）
pre_black_csv  = RUN_DIR / f"ts_blacklisted_{RUN_TAG}.csv"
pre_smooth_csv = RUN_DIR / f"ts_preprocessed_{RUN_TAG}.csv"
df_black.to_csv(pre_black_csv,  date_format="%Y-%m-%d %H:%M:%S")
df_smooth.to_csv(pre_smooth_csv, date_format="%Y-%m-%d %H:%M:%S")
print("Saved preprocessed TS:")
print(" -", pre_black_csv)
print(" -", pre_smooth_csv)

# =========================
# 单次跑一个(IC, scale)
# =========================
def run_binseg_linear_ic_once(criterion: str, crit_scale: float, max_m: int, min_size: int) -> dict:
    t0 = time.time()
    df = df_smooth  # 用平滑+黑名单后的
    series_rows, break_rows = [], []
    n_pix = df.shape[1]

    for j, col in enumerate(df.columns, 1):
        y = df[col].astype("float32").to_numpy()
        dates_arr = df.index.to_numpy()

        bkps_all, score = pick_best_bkps(
            signal=y, search_method=SEARCH_METHOD, max_m=max_m, min_size=min_size,
            criterion=criterion, scale=crit_scale, cost_model=COST_MODEL
        )
        bidx = filter_edge_indices(bkps_all[:-1], T=len(y), guard=EDGE_GUARD)

        base_ratio = twsv_ratio(y, [])
        final_idx = []
        br_summ   = summarize_breaks(dates_arr, y, bidx, delta_thr=DELTA_THRESH)
        for rec in br_summ:
            slope_jump = abs(rec["slope_post"] - rec["slope_pre"])
            seg_ratio  = twsv_ratio(y, [rec["break_index"]])
            improve    = (base_ratio - seg_ratio) if (np.isfinite(base_ratio) and np.isfinite(seg_ratio)) else np.nan

            ok_twsv  = (np.isfinite(improve) and improve >= TWSV_IMPROVE_MIN)
            ok_delta = (np.isfinite(rec["delta"]) and abs(rec["delta"]) >= DELTA_THRESH)
            ok_slope = (np.isfinite(slope_jump) and slope_jump >= SLOPE_DELTA_MIN)
            pass_qc  = ok_twsv and (ok_delta or ok_slope)

            if pass_qc:
                final_idx.append(rec["break_index"])
                rr, cc = (int(col.split("_")[1]), int(col.split("_")[2])) if re.match(r"pix_\d+_\d+$", col) else (np.nan, np.nan)
                break_rows.append({
                    "series": col, "row": rr, "col": cc,
                    "break_k": rec["break_k"], "break_index": rec["break_index"], "break_time": rec["break_time"],
                    "mu_pre": rec["mu_pre"], "mu_post": rec["mu_post"], "delta": rec["delta"],
                    "slope_pre": rec["slope_pre"], "slope_post": rec["slope_post"],
                    "label": ("L→W" if (np.isfinite(rec['delta']) and rec['delta'] >= 0) else "W→L"),
                    "twsv_improve": improve,
                    "search_method": SEARCH_METHOD, "cost_model": COST_MODEL,
                    "criterion": criterion, "crit_scale": crit_scale,
                    "min_size": min_size, "max_m": max_m,
                    "score_selected": score
                })

        rr, cc = (int(col.split("_")[1]), int(col.split("_")[2])) if re.match(r"pix_\d+_\d+$", col) else (np.nan, np.nan)
        series_rows.append({
            "series": col, "row": rr, "col": cc,
            "n_time": len(y), "break_count": len(final_idx),
            "break_dates": ";".join([Timestamp(dates_arr[i]).isoformat() for i in final_idx]) if final_idx else "",
            "search_method": SEARCH_METHOD, "cost_model": COST_MODEL,
            "criterion": criterion, "crit_scale": crit_scale,
            "min_size": min_size, "max_m": max_m,
            "score_selected": score
        })

        if (j % 200 == 0) or (j == n_pix):
            print(f"[{j}/{n_pix}] {criterion} x {crit_scale}")

    # 保存本档结果
    tag_cs  = str(crit_scale).replace(".", "p")
    tag     = f"binseg_linear_{criterion.lower()}_cs{tag_cs}_m{max_m}_ms{min_size}"
    out_series = RUN_DIR / f"rup_series_summary_{tag}.csv"
    out_breaks = RUN_DIR / f"rup_breaks_detail_{tag}.csv"
    pd.DataFrame(series_rows).to_csv(out_series, index=False, date_format="%Y-%m-%d %H:%M:%S")
    pd.DataFrame(break_rows).to_csv(out_breaks, index=False, date_format="%Y-%m-%d %H:%M:%S")

    df_series = pd.DataFrame(series_rows)
    n_total = len(df_series)
    n_with  = int((df_series["break_count"] > 0).sum())
    rate    = (n_with / n_total) if n_total else 0.0

    print(f"  → saved [{tag}]  n_total={n_total}, n_with={n_with} ({rate*100:.2f}%)")
    return {
        "method": "Binseg-linear", "criterion": criterion, "crit_scale": crit_scale,
        "max_m": max_m, "min_size": min_size,
        "n_total": n_total, "n_with_breaks": n_with, "rate": rate,
        "series_csv": str(out_series), "breaks_csv": str(out_breaks)
    }

# 跑一批 scale
def run_grid(scales: list[float]) -> pd.DataFrame:
    rows = []
    for crit in CRITERIA:
        for s in scales:
            rows.append(run_binseg_linear_ic_once(criterion=crit, crit_scale=float(s),
                                                  max_m=MAX_M, min_size=MIN_SIZE))
    return pd.DataFrame(rows)

# 自动外扩：直到某个 criterion 的“最大 scale 的检出率” ≤ 目标+容忍
def auto_expand_until_cover(df_sum: pd.DataFrame, scales: list[float],
                            expand_factor=1.5, max_scale_cap=6.0) -> tuple[pd.DataFrame, list[float]]:
    cur_scales = sorted(set(scales))
    cur_df = df_sum.copy()
    for crit in CRITERIA:
        while True:
            sub = cur_df[cur_df["criterion"]==crit]
            if sub.empty:
                break
            s_max = max(cur_scales)
            rate_at_smax = float(sub[sub["crit_scale"]==s_max]["rate"].mean()) if (sub["crit_scale"]==s_max).any() else np.inf
            if rate_at_smax <= TARGET_RATE + TOL:
                break  # 已经覆盖到目标附近
            # 还太高 → 外扩
            new_scale = min(s_max*expand_factor, max_scale_cap)
            if new_scale <= s_max + 1e-9:
                break
            cur_scales.append(round(new_scale, 3))
            new_df = run_grid([new_scale])
            cur_df = pd.concat([cur_df, new_df], ignore_index=True)
    return cur_df, sorted(set(cur_scales))

# 细化：在“全体最接近目标”的附近 ±0.5 做 0.05 步长微扫
def refine_around_best(df_sum: pd.DataFrame) -> pd.DataFrame:
    df_sum["abs_diff"] = (df_sum["rate"] - TARGET_RATE).abs()
    best = df_sum.sort_values(["abs_diff","criterion","crit_scale"]).iloc[0]
    crit = best["criterion"]; s0 = float(best["crit_scale"])
    fine = np.round(np.arange(max(0.1, s0-0.5), s0+0.5+1e-9, 0.05), 2).tolist()
    # 去掉已跑过的
    done = set([(r["criterion"], round(float(r["crit_scale"]),2)) for _, r in df_sum.iterrows()])
    need = [s for s in fine if (crit, round(float(s),2)) not in done]
    if not need:
        return df_sum
    new_rows = []
    for s in need:
        new_rows.append(run_binseg_linear_ic_once(criterion=crit, crit_scale=float(s),
                                                  max_m=MAX_M, min_size=MIN_SIZE))
    df_new = pd.DataFrame(new_rows)
    out = pd.concat([df_sum, df_new], ignore_index=True)
    return out

# =========================
# 主流程：初扫 → 外扩 → 细化 → 选最佳
# =========================
# 初扫
df_summary = run_grid(CRIT_SCALES)

# 外扩直到覆盖
df_summary, final_scales = auto_expand_until_cover(df_summary, CRIT_SCALES,
                                                   expand_factor=1.5, max_scale_cap=6.0)

# 细化一次
df_summary = refine_around_best(df_summary)

# 保存网格汇总
summary_csv = RUN_DIR / "rup_results_binseg_linear_ic_grid_summary.csv"
df_summary.to_csv(summary_csv, index=False)
print("\nGrid summary saved ->", summary_csv)

# 选最接近 TARGET 的组合
df_summary["abs_diff"] = (df_summary["rate"] - TARGET_RATE).abs()
best = df_summary.sort_values(["abs_diff","criterion","crit_scale"]).iloc[0].to_dict()
best_json = RUN_DIR / "best_choice.json"
with open(best_json, "w", encoding="utf-8") as f:
    json.dump(best, f, ensure_ascii=False, indent=2)
print("Best choice to match TARGET:", best)
print("Saved ->", best_json)

# =========================
# 小图：检出率 vs scale；最佳档日直方图
# =========================
try:
    # rate vs scale（分 criterion）
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for crit in df_summary["criterion"].unique():
        sub = (df_summary[df_summary["criterion"]==crit]
               .sort_values("crit_scale"))
        ax.plot(sub["crit_scale"], sub["rate"]*100, marker="o", label=crit)
    ax.axhline(TARGET_RATE*100, linestyle="--", linewidth=1)
    ax.set_xlabel("scale"); ax.set_ylabel("Share with ≥1 break (%)")
    ax.set_title(f"Binseg-linear detection rate vs scale  ({RUN_TAG})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(RUN_DIR / "rate_vs_scale.png", dpi=220)
    plt.close(fig)

    # 最佳档：按日直方图
    breaks_csv = Path(best["breaks_csv"])
    df_breaks = pd.read_csv(breaks_csv, parse_dates=["break_time"])
    daily = df_breaks["break_time"].dt.floor("D").value_counts().sort_index()
    plt.figure(figsize=(8,3.0))
    daily.plot(kind="bar")
    plt.title(f"Daily breaks  [{best['criterion']}, scale={best['crit_scale']}]  (target={TARGET_RATE*100:.2f}%)")
    plt.xlabel("Date"); plt.ylabel("Break count")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(RUN_DIR / "daily_breaks_best.png", dpi=220)
    plt.close()
except Exception as e:
    print("Plotting skipped:", e)

# =========================
# 可选：并表最近一份 PELT 汇总
# =========================
if PELT_SUMMARY_CSV and Path(PELT_SUMMARY_CSV).exists():
    try:
        df_pelt = pd.read_csv(PELT_SUMMARY_CSV)
        if "penalty_scale" in df_pelt.columns:
            df_pelt = df_pelt.rename(columns={
                "penalty_scale":"crit_scale",
                "n_total":"n_total",
                "n_with_breaks":"n_with_breaks",
                "rate":"rate"
            })
            df_pelt["method"]    = "PELT-l2"
            df_pelt["criterion"] = "penalty"
            df_pelt["max_m"]     = np.nan
            df_pelt["min_size"]  = np.nan
            df_pelt = df_pelt[["method","criterion","crit_scale","max_m","min_size","n_total","n_with_breaks","rate"]]
        df_all = pd.concat([df_pelt, df_summary[["method","criterion","crit_scale","max_m","min_size","n_total","n_with_breaks","rate"]]], ignore_index=True)
        compare_csv = RUN_DIR / "rup_results_pelt_vs_binseg_linear_summary.csv"
        df_all.to_csv(compare_csv, index=False)
        print("\nPELT summary merged ->", compare_csv)
    except Exception as e:
        print("⚠️ Merge PELT summary failed:", e)
else:
    print("\nℹ️ No PELT summary found; skip merge.")
