# -*- coding: utf-8 -*-
"""
项目配置（直接 import 使用，无需命令行参数）
"""
from pathlib import Path

# 数据文件（固定为脚本所在目录下的附件.xlsx，避免工作目录差异）
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = str(BASE_DIR / "附件.xlsx")

# 工作表名称（如为 csv 可忽略）
SHEET_MALE = "男胎检测数据"
SHEET_FEMALE = "女胎检测数据"

# 输出目录（相对当前目录）
OUTDIR = "outputs"
OUTDIR_Q1 = f"{OUTDIR}/q1"
OUTDIR_Q1_FIGS = f"{OUTDIR_Q1}/figs"

# 关键参数
Y_THRESHOLD = 0.04  # Y 浓度达标阈值（4%）
RANDOM_SEED = 2025

# 质量控制与范围（可按需调整）
GW_MIN, GW_MAX = 8.0, 30.0          # 孕周范围（周）
BMI_MIN, BMI_MAX = 15.0, 50.0       # BMI 合理范围
GC_MIN, GC_MAX = 0.35, 0.65         # GC 合理范围

# 极端值处理（对 y 的分位裁剪）
Y_TRIM_Q_LOW, Y_TRIM_Q_HIGH = 0.01, 0.99
