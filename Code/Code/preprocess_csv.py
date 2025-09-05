# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd

from config_modified import (
    DATA_FILE_MALE, DATA_FILE_FEMALE,
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
# 主函数：读取和预处理
# -----------------------------
def main():
    print(f"[INFO] 读取：{DATA_FILE_MALE} 和 {DATA_FILE_FEMALE}")
    
    # 读取男胎数据
    df_male_raw = pd.read_csv(DATA_FILE_MALE, encoding='gbk')
    print(f"[INFO] 男胎原始数据形状：{df_male_raw.shape}")
    
    # 读取女胎数据  
    df_female_raw = pd.read_csv(DATA_FILE_FEMALE, encoding='gbk')
    print(f"[INFO] 女胎原始数据形状：{df_female_raw.shape}")
    
    print(f"[INFO] 男胎数据列名：{list(df_male_raw.columns)}")
    
    # 为了简化，这里只处理男胎数据用于第一问分析
    df = df_male_raw.copy()
    
    # 标准化列名映射
    col_map = {
        '检测孕周': 'gest_week',
        '孕妇BMI': 'bmi', 
        'Y染色体浓度': 'y',
        'GC含量': 'gc_content',
        '总读段数中在参考基因组上比对的比例': 'map_rate',
        '被过滤掉的读段数占总读段数的比例': 'filter_rate',
        '原始测序数据的总读段数（个）': 'total_reads'
    }
    
    # 重命名列
    for old_name, new_name in col_map.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # 解析孕周
    if 'gest_week' in df.columns:
        df['gest_week'] = df['gest_week'].apply(parse_gestation)
    
    # 基本质量控制
    print(f"[INFO] 开始质量控制...")
    initial_count = len(df)
    
    # 去除缺失值和异常值
    df = df.dropna(subset=['gest_week', 'bmi', 'y'])
    df = df[(df['gest_week'] >= GW_MIN) & (df['gest_week'] <= GW_MAX)]
    df = df[(df['bmi'] >= BMI_MIN) & (df['bmi'] <= BMI_MAX)]
    df = df[df['y'] > 0]  # Y染色体浓度必须为正
    
    print(f"[INFO] 质量控制后：{initial_count} -> {len(df)} 样本")
    
    # 极端值裁剪
    y_low = df['y'].quantile(Y_TRIM_Q_LOW)
    y_high = df['y'].quantile(Y_TRIM_Q_HIGH)
    df = df[(df['y'] >= y_low) & (df['y'] <= y_high)]
    
    print(f"[INFO] 极端值裁剪后：{len(df)} 样本")
    
    # 保存清洗后的数据
    output_file = os.path.join(OUTDIR, "clean_male.csv")
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"[INFO] 清洗后数据保存至：{output_file}")
    
    # 基本统计
    print(f"\n[INFO] 基本统计：")
    print(f"Y染色体浓度：均值={df['y'].mean():.6f}, 标准差={df['y'].std():.6f}")
    print(f"孕周：均值={df['gest_week'].mean():.2f}, 标准差={df['gest_week'].std():.2f}")
    print(f"BMI：均值={df['bmi'].mean():.2f}, 标准差={df['bmi'].std():.2f}")
    
    return df

if __name__ == "__main__":
    df = main()