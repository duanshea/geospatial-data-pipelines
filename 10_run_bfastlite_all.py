# -*- coding: utf-8 -*-
"""
批量对所有候选像元运行 BFAST-Lite，并输出断点结果与汇总统计。

使用方式（命令行）：
    python 11_run_bfastlite_all.py            # 默认使用 LWZ
    python 11_run_bfastlite_all.py LWZ        # 指定 LWZ
    python 11_run_bfastlite_all.py BIC        # 指定 BIC

说明：
- 通过命令行参数控制模型选择准则（LWZ 或 BIC），会自动写入 RUN_TAG，
  并体现在输出文件名和 run_info JSON/TXT 中，便于区分不同实验。
"""

from pathlib import Path
import sys, re, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # 目前未绘图，但保留以备扩展

# ========== 从命令行读取准则（LWZ / BIC） ==========
# 用法:
#   python 11_run_bfastlite_all.py          -> 默认 LWZ
#   python 11_run_bfastlite_all.py LWZ      -> LWZ
#   python 11_run_bfastlite_all.py BIC      -> BIC

if len(sys.argv) >= 2:
    crit_arg = sys.argv[1].upper()
    if crit_arg not in ("LWZ", "BIC"):
        raise ValueError(f"未知 criterion: {crit_arg}，请用 LWZ 或 BIC")
else:
    crit_arg = "LWZ"   # 默认

CRIT_CAND = [crit_arg]

# ========== 运行标签 ==========
# 这里把准则名写进 RUN_TAG，方便区分不同实验
RUN_TAG = f"{crit_arg}_h=0.08_0.10_0.12"     # 例：BIC_h=0.08_0.10_0.12 或 LWZ_h=0.08_0.10_0.12


# ========== 尝试导入 bfastlite ==========
try:
    from bfastlite import bfastlite
except Exception:
    PKG_DIR = r"C:\Users\Duans\Desktop\bfastlibfastlite_project\bfastlite"
    if PKG_DIR not in sys.path:
        sys.path.append(PKG_DIR)
    from bfastlite import bfastlite

# ========== 路径 ==========

BASE       = Path(r"C:\Users\Duans\Desktop\bfastlibfastlite_project\lsu_abundance")
OUT_DIR    = BASE / "riparian_out"
TS_CSV     = OUT_DIR / "riparian_timeseries_samples.csv"     # 全体候选的时序
META_CSV   = OUT_DIR / "riparian_samples.csv"                # 全体候选的元数据

# —— 本次运行的独立输出目录与文件（不会覆盖别的实验） ——
OUT_DIR_TAG = OUT_DIR / f"results_{RUN_TAG}"
OUT_DIR_TAG.mkdir(parents=True, exist_ok=True)
OUT_CSV     = OUT_DIR_TAG / f"bfastlite_breaks_all_{RUN_TAG}.csv"
SUM_CSV     = OUT_DIR_TAG / f"summary_labels_{RUN_TAG}.csv"
WEEK_CSV    = OUT_DIR_TAG / f"summary_breaks_by_week_{RUN_TAG}.csv"
RUNINFO_TXT = OUT_DIR_TAG / f"run_info_{RUN_TAG}.txt"
RUNINFO_JSON= OUT_DIR_TAG / f"run_info_{RUN_TAG}.json"

assert TS_CSV.exists() and META_CSV.exists(), "先运行导出候选像元的脚本生成 TS_CSV 和 META_CSV！"

# ========== 检测设置 ==========

FORMULA   = "response ~ trend"   # 分段线性（识别斜率变化）
ORDER     = 1
STL       = "none"
DECOMP    = "none"

H_CAND     = [0.08, 0.10, 0.12]  # 最小段长占比
MAX_BREAKS = 2
LEVEL      = 0.0                 # 结构检验（此处关闭）

# ========== 预处理与把关 ==========

SMOOTH_WIN         = 3     # 3点滚动中位数
EDGE_GUARD         = 5     # 断点离首尾至少 5 帧
BLACKLIST_DATES    = {"2024-08-13"}  # 不可信日期（整天置 NaN）
DELTA_THRESH       = 0.15  # |Δ均值| 判显著
SLOPE_DELTA_MIN    = 0.01  # 斜率跳变阈值
TWSV_IMPROVE_MIN   = 0.20  # 段内方差比改善阈值（越大越好）

# ========== 工具函数 ==========

def ols_slope(y: np.ndarray) -> float:
    """对给定序列做简单线性回归，返回斜率。"""
    y = np.asarray(y, "float32")
    t = np.arange(len(y), dtype="float32")
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    A = np.vstack([t[mask], np.ones_like(t[mask])]).T
    m, _ = np.linalg.lstsq(A, y[mask], rcond=None)[0]
    return float(m)

def twsv_ratio(y: np.ndarray, cuts_idx: list[int]) -> float:
    """
    Total Within-Segment Variance ratio:
    sum_k((nk-1)*var_k) / ((N-1)*var_all)
    """
    y = np.asarray(y, "float32")
    N = len(y)
    if N < 3 or np.nanstd(y) == 0:
        return np.nan
    cuts = [0] + list(cuts_idx) + [N]
    num = 0.0
    for s0, s1 in zip(cuts[:-1], cuts[1:]):
        seg = y[s0:s1]
        if len(seg) >= 2:
            num += (len(seg)-1) * np.nanvar(seg, ddof=1)
    den = (N-1) * np.nanvar(y, ddof=1)
    return float(num/den) if den > 0 else np.nan

def filter_edge_breaks(bidx, T, guard=EDGE_GUARD):
    """过滤掉距离两端太近的断点。"""
    return [int(b) for b in bidx if guard <= int(b) <= T-1-guard]

def summarize_breaks(res: dict, delta_thr=DELTA_THRESH):
    """
    把 bfastlite 结果转成多条记录（每个断点一条），不做 QC。
    返回 list[dict]，每个 dict 包含：
      break_k, break_index, break_time, mu_pre, mu_post, delta, slope_pre, slope_post, label_raw
    """
    idx_time = res["data_pp"].index
    y = res["data_pp"]["response"].to_numpy("float32")
    bidx = res.get("breakpoints") or []
    bidx = [int(b) for b in bidx if 1 <= int(b) < len(y)-1]
    if not bidx:
        return []
    cuts = [0] + bidx + [len(y)]
    out = []
    for k, b in enumerate(bidx):
        pre  = y[cuts[k]:cuts[k+1]]
        post = y[cuts[k+1]:cuts[k+2]]
        mu_pre, mu_post = float(np.nanmean(pre)), float(np.nanmean(post))
        dy = mu_post - mu_pre
        if dy >= delta_thr:
            lab = "L→W"   # 从低到高（更湿/更水）
        elif dy <= -delta_thr:
            lab = "W→L"   # 从高到低（更干/更陆）
        else:
            lab = "weak/unclear"
        out.append({
            "break_k": k+1,
            "break_index": int(b),
            "break_time": idx_time[int(b)],
            "mu_pre": mu_pre,
            "mu_post": mu_post,
            "delta": dy,
            "slope_pre": ols_slope(pre),
            "slope_post": ols_slope(post),
            "label_raw": lab
        })
    return out


# ========== 读数据 ==========

ts = pd.read_csv(TS_CSV, index_col=0, parse_dates=True)
meta = pd.read_csv(META_CSV)

# 黑名单日期置 NaN（按“日期”匹配，不含时分秒）
if BLACKLIST_DATES:
    bad_days = pd.to_datetime(sorted(BLACKLIST_DATES)).date
    bad_mask = ts.index.normalize().isin(pd.to_datetime(bad_days))
    ts.loc[bad_mask] = np.nan

print(f"Loaded time series: {ts.shape[1]} pixels × {ts.shape[0]} timesteps")
print(f"RUN_TAG   = {RUN_TAG}")
print(f"Criterion = {CRIT_CAND}, h ∈ {H_CAND}")

# ========== 计时 ==========

t0 = time.time()
rows = []

# ========== 主循环 ==========

for j, col in enumerate(ts.columns, 1):
    # 1) 平滑
    y = ts[col].astype("float32").rolling(
        SMOOTH_WIN, center=True, min_periods=1
    ).median()
    ydf = pd.DataFrame({"response": y})
    start_ts = ydf.index[0]

    picked = {"h": None, "crit": None, "model": "trend"}
    res_ok = None
    bidx_ok = []

    # 2) 小网格兜底搜索（h × criterion）
    for h in H_CAND:
        for crit in CRIT_CAND:
            try:
                res = bfastlite(
                    data=ydf, formula=FORMULA, order=ORDER,
                    stl=STL, decomp=DECOMP, h=h, level=LEVEL,
                    max_breaks=MAX_BREAKS, freq=None, start=start_ts,
                    criterion=crit
                )
            except ValueError:
                continue

            bidx = res.get("breakpoints") or []
            bidx = filter_edge_breaks(bidx, T=len(ydf), guard=EDGE_GUARD)
            if len(bidx) == 0:
                continue

            # 3) 质量把关（TWSV 改善 + 斜率变化 + Δ幅度）
            base_ratio = twsv_ratio(ydf["response"].values, [])
            seg_ratio  = twsv_ratio(ydf["response"].values, bidx)
            improve    = base_ratio - seg_ratio

            br_summ = summarize_breaks({"data_pp": ydf, "breakpoints": bidx})
            ok_any = False
            for rec in br_summ:
                slope_jump = abs(rec["slope_post"] - rec["slope_pre"])
                if (
                    abs(rec["delta"]) >= DELTA_THRESH
                    and improve >= TWSV_IMPROVE_MIN
                    and slope_jump >= SLOPE_DELTA_MIN
                ):
                    ok_any = True
                    break

            if ok_any:
                res_ok  = res
                bidx_ok = bidx
                picked.update({"h": h, "crit": crit})
                break
        if res_ok is not None:
            break

    # 4) 汇总写行
    if res_ok is None or len(bidx_ok) == 0:
        rows.append({
            "series": col,
            "break_count": 0,
            "break_k": np.nan,
            "break_time": pd.NaT,
            "break_index": np.nan,
            "mu_pre": np.nan,
            "mu_post": np.nan,
            "delta": np.nan,
            "slope_pre": np.nan,
            "slope_post": np.nan,
            "label": "no-break",
            "twsv_improve": np.nan,
            "used_model": picked["model"],
            "used_h": picked["h"],
            "used_criterion": picked["crit"]
        })
    else:
        br_summ = summarize_breaks({"data_pp": ydf, "breakpoints": bidx_ok})
        base_ratio_all = twsv_ratio(ydf["response"].values, [])
        for rec in br_summ:
            slope_jump = abs(rec["slope_post"] - rec["slope_pre"])
            seg_ratio_single = twsv_ratio(ydf["response"].values, [rec["break_index"]])
            improve_single   = base_ratio_all - seg_ratio_single
            # 终版标签
            if (
                abs(rec["delta"]) >= DELTA_THRESH
                and improve_single >= TWSV_IMPROVE_MIN
                and slope_jump >= SLOPE_DELTA_MIN
            ):
                final_label = "L→W" if rec["delta"] >= 0 else "W→L"
            else:
                final_label = "weak/unclear"
            rows.append({
                "series": col,
                "break_count": len(bidx_ok),
                **rec,
                "label": final_label,
                "twsv_improve": improve_single,
                "used_model": picked["model"],
                "used_h": picked["h"],
                "used_criterion": picked["crit"]
            })

    if j % 100 == 0 or j == ts.shape[1]:
        print(f"[{j}/{ts.shape[1]}] processed")

# ========== 合并 meta & 保存主结果 ==========

df = pd.DataFrame(rows)
df[["y","x"]] = df["series"].str.extract(r"pix_(\d+)_(\d+)").astype("float32")

if {"y","x"}.issubset(meta.columns):
    df = df.merge(meta, on=["y","x"], how="left", suffixes=("","_meta"))
elif "series" in meta.columns:
    meta = meta.copy()
    meta[["y","x"]] = meta["series"].str.extract(r"pix_(\d+)_(\d+)").astype("float32")
    df = df.merge(meta, on=["y","x"], how="left", suffixes=("","_meta"))

front = [
    "series","y","x",
    "break_count","break_k","break_time","break_index",
    "mu_pre","mu_post","delta","slope_pre","slope_post",
    "label","twsv_improve",
    "used_model","used_h","used_criterion"
]
df = df[front + [c for c in df.columns if c not in front]]

df.to_csv(OUT_CSV, index=False, date_format="%Y-%m-%d %H:%M:%S")
print("\nSaved:", OUT_CSV)

# ========== 统计汇总（打印 + 另存 CSV） ==========

summary = df.groupby("label", dropna=False).size().rename("count").to_frame()
summary["pct"] = (summary["count"] / len(df) * 100).round(2)
print("\nLabel summary:")
print(summary)
summary.to_csv(SUM_CSV)

bt = df.loc[df["break_time"].notna(), "break_time"]
if not bt.empty:
    by_week = bt.dt.to_period("W-SUN").value_counts().sort_index().rename("count").to_frame()
    print("\nBreaks by week:")
    print(by_week)
    by_week.to_csv(WEEK_CSV)

# ========== 计时收尾 ==========

t1 = time.time()
elapsed_sec = t1 - t0
print(f"\nRun finished with {RUN_TAG}. Time elapsed: {elapsed_sec/60:.1f} minutes")

# 保存运行信息
run_info = {
    "RUN_TAG": RUN_TAG,
    "CRIT_CAND": CRIT_CAND,
    "H_CAND": H_CAND,
    "MAX_BREAKS": MAX_BREAKS,
    "LEVEL": LEVEL,
    "SMOOTH_WIN": SMOOTH_WIN,
    "EDGE_GUARD": EDGE_GUARD,
    "BLACKLIST_DATES": sorted(list(BLACKLIST_DATES)),
    "DELTA_THRESH": DELTA_THRESH,
    "SLOPE_DELTA_MIN": SLOPE_DELTA_MIN,
    "TWSV_IMPROVE_MIN": TWSV_IMPROVE_MIN,
    "n_total_rows": int(len(df)),
    "n_series": int(df["series"].nunique()),
    "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
    "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1)),
    "elapsed_seconds": elapsed_sec,
    "out_csv": str(OUT_CSV),
    "summary_csv": str(SUM_CSV),
    "week_csv": str(WEEK_CSV),
    "ts_csv": str(TS_CSV),
    "meta_csv": str(META_CSV),
}
with open(RUNINFO_JSON, "w", encoding="utf-8") as f:
    json.dump(run_info, f, ensure_ascii=False, indent=2)
with open(RUNINFO_TXT, "w", encoding="utf-8") as f:
    f.write(json.dumps(run_info, ensure_ascii=False, indent=2))
print(f"Run info saved to:\n- {RUNINFO_JSON}\n- {RUNINFO_TXT}")
