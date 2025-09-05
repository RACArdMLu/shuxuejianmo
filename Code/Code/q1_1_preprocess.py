# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd

from config import (
    DATA_FILE, SHEET_MALE, SHEET_FEMALE,
    OUTDIR,
    GW_MIN, GW_MAX, BMI_MIN, BMI_MAX, GC_MIN, GC_MAX,
    Y_TRIM_Q_LOW, Y_TRIM_Q_HIGH
)

os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# 工具：孕周字符串 -> 小数周
# -----------------------------
def parse_gestation(s: str) -> float:
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower()
    # 直接数值
    try:
        return float(s)
    except ValueError:
        pass
    # 形如 13w+5 或 13w
    m = re.match(r"(\d+)\s*w(?:\s*\+\s*(\d+))?", s)
    if not m:
        return np.nan
    w = int(m.group(1))
    d = int(m.group(2)) if m.group(2) else 0
    return w + d/7.0

# -----------------------------
# 工具：列名鲁棒匹配
# -----------------------------
def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # 降维再试：大小写与空格
    low_map = {c: str(c).strip().lower() for c in df.columns}
    for c in candidates:
        target = str(c).strip().lower()
        for col, low in low_map.items():
            if target == low:
                return col
    raise KeyError(f"列未找到，候选：{candidates}\n现有列：{list(df.columns)[:30]} ...")

# -----------------------------
# 读取数据
# -----------------------------
print(f"[INFO] 读取：{DATA_FILE} | sheet={SHEET_MALE}/{SHEET_FEMALE}")
df_male_raw = pd.read_excel(DATA_FILE, sheet_name=SHEET_MALE)
df_fem_raw  = pd.read_excel(DATA_FILE, sheet_name=SHEET_FEMALE)

# -----------------------------
# 映射关键列
# -----------------------------
# 男/女共用映射逻辑

def map_and_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    bmi_col   = pick_col(df_raw, ["孕妇BMI","孕妇 BMI 指标","孕妇BMI指标","BMI","孕妇 BMI","孕妇BMI 指标"])
    gw_col    = pick_col(df_raw, ["检测孕周","孕妇本次检测时的孕周（周数+天数）","检测孕周（周数+天数）"])
    y_col     = pick_col(df_raw, ["Y染色体浓度","Y 染色体浓度"]) if any(k in df_raw.columns for k in ["Y染色体浓度","Y 染色体浓度"]) else None
    gc_col    = pick_col(df_raw, ["GC含量","GC 含量"]) if any(k in df_raw.columns for k in ["GC含量","GC 含量"]) else None
    map_col   = pick_col(df_raw, ["在参考基因组上比对的比例","总读段数中在参考基因组上比对的比例"]) if any(k in df_raw.columns for k in ["在参考基因组上比对的比例","总读段数中在参考基因组上比对的比例"]) else None
    filt_col  = pick_col(df_raw, ["被过滤掉读段数的比例","被过滤掉的读段数占总读段数的比例"]) if any(k in df_raw.columns for k in ["被过滤掉读段数的比例","被过滤掉的读段数占总读段数的比例"]) else None
    reads_col = pick_col(df_raw, ["原始读段数","原始测序数据的总读段数（个）"]) if any(k in df_raw.columns for k in ["原始读段数","原始测序数据的总读段数（个）"]) else None
    pid_col   = pick_col(df_raw, ["孕妇代码","孕妇编号","样本序号"]) if any(k in df_raw.columns for k in ["孕妇代码","孕妇编号","样本序号"]) else None

    df = df_raw.copy()
    df["gest_week"] = df[gw_col].apply(parse_gestation)
    df["bmi"]       = pd.to_numeric(df[bmi_col], errors='coerce')
    if y_col is not None:
        df["y"]         = pd.to_numeric(df[y_col], errors='coerce')
    if gc_col is not None:
        df["gc"]        = pd.to_numeric(df[gc_col], errors='coerce')
    if map_col is not None:
        df["map_rate"]  = pd.to_numeric(df[map_col], errors='coerce')
    if filt_col is not None:
        df["filt_rate"] = pd.to_numeric(df[filt_col], errors='coerce')
    if reads_col is not None:
        df["reads"]     = pd.to_numeric(df[reads_col], errors='coerce')
    if pid_col is not None:
        df["pid"]       = df[pid_col].astype(str)

    return df

male = map_and_clean(df_male_raw)
fem  = map_and_clean(df_fem_raw)

# -----------------------------
# 质量控制与极端值处理（男胎）
# -----------------------------
# 基本范围
male = male[(male["gest_week"]>=GW_MIN) & (male["gest_week"]<=GW_MAX)]
male = male[(male["bmi"]>=BMI_MIN) & (male["bmi"]<=BMI_MAX)]
if "gc" in male.columns:
    male = male[(male["gc"]>=GC_MIN) & (male["gc"]<=GC_MAX)]

# y 合理与分位裁剪
if "y" in male.columns:
    male = male[male["y"].notna() & (male["y"]>=0)]
    if len(male) > 10:
        ql, qh = male["y"].quantile([Y_TRIM_Q_LOW, Y_TRIM_Q_HIGH])
        male = male[(male["y"]>=ql) & (male["y"]<=qh)]

male = male.reset_index(drop=True)

# 女胎仅保留清洗后的共性字段
fem = fem[(fem["bmi"]>=BMI_MIN) & (fem["bmi"]<=BMI_MAX)]
if "gc" in fem.columns:
    fem = fem[(fem["gc"]>=GC_MIN) & (fem["gc"]<=GC_MAX)]
fem = fem.reset_index(drop=True)

# -----------------------------
# 导出
# -----------------------------
male_out = os.path.join(OUTDIR, "clean_male.csv")
fem_out  = os.path.join(OUTDIR, "clean_female.csv")

male.to_csv(male_out, index=False)
fem.to_csv(fem_out, index=False)

print(f"[DONE] 预处理完成：\n  male -> {male_out}（{len(male)} 行）\n  female -> {fem_out}（{len(fem)} 行）")
