# -*- coding: utf-8 -*-
"""
NIPT问题1分析脚本：胎儿Y染色体浓度与孕周、BMI关系建模
主要功能：
1. 相关性分析（Pearson、Spearman、偏相关）
2. 可视化分析（散点图、热力图、箱线图、3D曲面图）
3. 统计建模（Gamma-GLM、分位数回归）
4. 模型诊断（残差分析、Cook距离、异方差检验）
5. 特殊图表（雷达图）

作者：数学建模助手
日期：2025年
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm, colors as mcolors

import statsmodels.api as sm
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import het_breuschpagan

from config import OUTDIR, OUTDIR_Q1, OUTDIR_Q1_FIGS

# ================================
# 1. 环境配置与字体设置
# ================================
# 统一中文字体与负号显示（按可用字体自动回退）
rcParams["font.sans-serif"] = [
    "PingFang SC",          # macOS 系统中文
    "Hiragino Sans GB",     # macOS 常见
    "Heiti SC", "STHeiti", # 黑体
    "Microsoft YaHei",      # Windows雅黑
    "SimHei",               # 黑体
    "Arial Unicode MS",     # 覆盖面广
    "DejaVu Sans"
]
rcParams["font.family"] = "sans-serif"
rcParams["axes.unicode_minus"] = False

# 创建输出目录
os.makedirs(OUTDIR_Q1_FIGS, exist_ok=True)

# ================================
# 2. 数据加载与预处理
# ================================
# 读取预处理后的男胎数据
male_csv = os.path.join(OUTDIR, "clean_male.csv")
if not os.path.exists(male_csv):
    raise FileNotFoundError(f"未找到 {male_csv}，请先运行 q1_1_preprocess.py")

df = pd.read_csv(male_csv)

# 检查必要列是否存在
needed = ["gest_week","bmi","y"]  # 孕周、BMI、Y染色体浓度
for c in needed:
    if c not in df.columns:
        raise KeyError(f"缺少必要列：{c}")

# 测序质量列可选（如果不存在则填充NaN）
for opt in ["gc","map_rate","filt_rate","reads"]:
    if opt not in df.columns:
        df[opt] = np.nan

# 删除关键变量缺失的样本
df = df.dropna(subset=["gest_week","bmi","y"]).reset_index(drop=True)

# ================================
# 3. 相关性分析
# ================================
# 存储分析结果的列表
metrics_lines = []

def corr_report(x, y, namex, namey, method="pearson"):
    """
    计算并格式化相关性报告
    
    参数:
    - x, y: 待分析的两个变量
    - namex, namey: 变量名称
    - method: 相关性方法（"pearson"或"spearman"）
    
    返回:
    - 格式化的相关性报告字符串
    """
    if method == "pearson":
        r, p = stats.pearsonr(x, y)
        return f"{namey} ~ {namex}  Pearson r={r:.3f}, p={p:.3e}"
    else:
        r, p = stats.spearmanr(x, y)
        return f"{namey} ~ {namex}  Spearman  ={r:.3f}, p={p:.3e}"

# 分析变量对
pairs = [("gest_week","孕周"),("bmi","BMI"),("gc","GC"),("map_rate","map_rate"),("filt_rate","filt_rate"),("reads","reads")]

# Pearson相关性分析
metrics_lines.append("[相关性：Pearson]")
for col, cname in pairs:
    s = corr_report(df[col].values, df["y"].values, col, "y", "pearson")
    metrics_lines.append(s)

# Spearman相关性分析（非参数，对异常值更稳健）
metrics_lines.append("\n[相关性：Spearman]")
for col, cname in pairs:
    s = corr_report(df[col].values, df["y"].values, col, "y", "spearman")
    metrics_lines.append(s)

# ================================
# 4. 偏相关分析（控制测序质量变量）
# ================================
# 偏相关分析：在控制测序质量变量的情况下，分析Y染色体浓度与孕周、BMI的关系
# 这可以排除测序质量对相关性的干扰，得到更纯净的生物学关系

# 构建控制变量矩阵（测序质量指标）
controls = np.column_stack([df["gc"].values, df["map_rate"].values, df["filt_rate"].values, df["reads"].values])
Xc = sm.add_constant(controls, has_constant='add')

# 计算各变量在控制测序质量后的残差
res_y   = sm.OLS(df["y"].values, Xc).fit().resid      # Y染色体浓度残差
res_gw  = sm.OLS(df["gest_week"].values, Xc).fit().resid  # 孕周残差
res_bmi = sm.OLS(df["bmi"].values, Xc).fit().resid    # BMI残差

# 计算偏相关系数
r_gw, p_gw = stats.pearsonr(res_gw, res_y)
r_bmi, p_bmi = stats.pearsonr(res_bmi, res_y)

# 记录偏相关结果
metrics_lines.append("\n[偏相关：控制GC/map/过滤/读段]")
metrics_lines.append(f"y ⟂ gest_week | controls  r={r_gw:.3f}, p={p_gw:.3e}")
metrics_lines.append(f"y ⟂ BMI | controls        r={r_bmi:.3f}, p={p_bmi:.3e}")

# ================================
# 5. 偏相关可视化：气泡图
# ================================
# 生成偏相关气泡图，展示控制测序质量后的变量关系
# 气泡大小和颜色反映数据点密度，更直观地显示数据分布

# 孕周 vs Y染色体浓度偏相关气泡图
plt.figure()
# 计算点密度用于气泡大小与颜色映射
xy = np.vstack([res_gw, res_y])
dens = stats.gaussian_kde(xy)(xy)
# 归一化密度到合适的气泡大小范围
sizes = 10 + 90 * (dens - dens.min()) / (dens.max() - dens.min() + 1e-12)
scatter = plt.scatter(res_gw, res_y, s=sizes, c=dens, cmap=plt.colormaps.get_cmap('turbo'), 
                     alpha=0.35, linewidths=0.3, edgecolors='white')
plt.xlabel("孕周残差（控制质量项后）")
plt.ylabel("Y 浓度残差")
plt.title("偏相关气泡图：孕周 vs Y（控制质量项后）")
cb = plt.colorbar(scatter, fraction=0.046, pad=0.04)
cb.set_label("点密度", rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR_Q1_FIGS, "partialcorr_gw_y.png"), dpi=160)
plt.close()

# BMI vs Y染色体浓度偏相关气泡图
plt.figure()
xy = np.vstack([res_bmi, res_y])
dens = stats.gaussian_kde(xy)(xy)
sizes = 10 + 90 * (dens - dens.min()) / (dens.max() - dens.min() + 1e-12)
scatter = plt.scatter(res_bmi, res_y, s=sizes, c=dens, cmap=plt.colormaps.get_cmap('turbo'), 
                     alpha=0.35, linewidths=0.3, edgecolors='white')
plt.xlabel("BMI 残差（控制质量项后）")
plt.ylabel("Y 浓度残差")
plt.title("偏相关气泡图：BMI vs Y（控制质量项后）")
cb = plt.colorbar(scatter, fraction=0.046, pad=0.04)
cb.set_label("点密度", rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR_Q1_FIGS, "partialcorr_bmi_y.png"), dpi=160)
plt.close()

# ================================
# 6. 可视化辅助函数
# ================================
def savefig_tidy(path):
    """
    保存图片的辅助函数，统一设置布局和分辨率
    """
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

# KDE 等高线辅助函数（极浅色）
_DEF_LEVELS = [0.2, 0.4, 0.6, 0.8]

def overlay_kde_contours(x, y, ax, levels=None, cmap="Greys", alpha=0.28, gridsize=120):
    """
    在散点图上叠加KDE密度等高线，营造密度氛围
    
    参数:
    - x, y: 数据点坐标
    - ax: matplotlib轴对象
    - levels: 等高线水平，默认使用_DEF_LEVELS
    - cmap: 等高线颜色映射
    - alpha: 透明度
    - gridsize: 网格分辨率
    """
    x = np.asarray(x); y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size < 50:  # 点太少不画等高线
        return
    # 为稳健与速度，必要时下采样
    if x.size > 10000:
        idx = np.random.choice(x.size, 10000, replace=False)
        x = x[idx]; y = y[idx]
    try:
        kde = stats.gaussian_kde(np.vstack([x, y]))
    except Exception:
        return
    x_min, x_max = np.percentile(x, [1, 99])
    y_min, y_max = np.percentile(y, [1, 99])
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, gridsize),
                         np.linspace(y_min, y_max, gridsize))
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    # 归一化后按分位确定等高线
    zq = np.quantile(zz, _DEF_LEVELS if levels is None else levels)
    ax.contour(xx, yy, zz, levels=zq, cmap=cmap, alpha=alpha, linewidths=1.0, zorder=1)

# 点密度计算（用于散点着色）
def point_density_colors(x, y, cmap_name="turbo"):
    """
    计算散点图中每个点的密度，用于颜色映射
    
    参数:
    - x, y: 数据点坐标
    - cmap_name: 颜色映射名称
    
    返回:
    - mask: 有效数据点的掩码
    - colors: 颜色数组
    - norm: 归一化对象
    """
    x = np.asarray(x); y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size == 0:
        return mask, None, None
    kde = stats.gaussian_kde(np.vstack([x, y]))
    z = kde(np.vstack([x, y]))
    norm = mcolors.Normalize(vmin=np.percentile(z, 5), vmax=np.percentile(z, 95))
    cmap = plt.colormaps.get_cmap(cmap_name)
    colors = cmap(norm(z))
    return mask, colors, norm

# ================================
# 7. 散点图+LOWESS+KDE等高线可视化
# ================================
# 生成Y染色体浓度与各变量的散点图，包含：
# - 密度着色散点（turbo颜色映射）
# - KDE密度等高线（营造密度氛围）
# - LOWESS平滑趋势线（展示非线性关系）
# - 轻微抖动（减少点重叠）

for col, xlabel in [("gest_week","孕周（周）"),("bmi","BMI"),("gc","GC"),("map_rate","比对比例"),("filt_rate","被过滤比例"),("reads","总读段数")]:
    x = df[col].values
    y = df["y"].values
    fig, ax = plt.subplots()
    
    # 轻微抖动减少遮挡（按x范围的0.3%）
    xr = np.nanmax(x) - np.nanmin(x)
    jitter = np.random.uniform(-0.003, 0.003, size=len(x)) * (xr if np.isfinite(xr) and xr>0 else 1.0)
    x_jit = x + jitter
    
    # 密度着色 + 白色描边
    mask, colors_arr, norm = point_density_colors(x_jit, y, cmap_name="turbo")
    if colors_arr is None:
        sc = ax.scatter(x_jit, y, s=16, alpha=0.35, color="#4C78A8", linewidths=0.3, edgecolors="white", zorder=2)
        cbar = None
    else:
        sc = ax.scatter(x_jit[mask], y[mask], s=16, alpha=0.35, c=colors_arr, linewidths=0.3, edgecolors="white", zorder=2)
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.colormaps.get_cmap("turbo")), ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("密度", rotation=90)
    
    # 叠加浅色KDE等高线
    overlay_kde_contours(x_jit, y, ax, cmap="plasma", alpha=0.22)
    
    # LOWESS平滑趋势线
    idx = np.isfinite(x) & np.isfinite(y)
    if idx.sum() > 10:
        lo = lowess(y[idx], x[idx], frac=0.3, return_sorted=True)
        ax.plot(lo[:,0], lo[:,1], color="#1f1f1f", linewidth=2.2, alpha=0.95, zorder=3)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Y 染色体浓度")
    ax.set_title(f"Y vs {xlabel}（LOWESS + KDE 等高线）")
    savefig_tidy(os.path.join(OUTDIR_Q1_FIGS, f"scatter_lowess_{col}.png"))

# ================================
# 8. 相关性热力图可视化
# ================================
# 生成Pearson和Spearman相关系数热力图，展示变量间的线性关系强度
# 使用viridis颜色映射，白色网格，粗体数字标注

cols = ["gest_week","bmi","y","gc","map_rate","filt_rate","reads"]

def plot_corr_heatmap(C: pd.DataFrame, title: str, outname: str):
    """
    绘制相关性热力图
    
    参数:
    - C: 相关系数矩阵
    - title: 图表标题
    - outname: 输出文件名
    """
    # 对称色轴（围绕0）
    vmax = np.nanmax(np.abs(C.values))
    vmax = 1.0 if not np.isfinite(vmax) or vmax <= 0 else float(vmax)
    vlim = max(0.3, min(1.0, vmax))
    fig, ax = plt.subplots(figsize=(8.5, 6.6))
    im = ax.imshow(C, cmap="viridis", vmin=-vlim, vmax=vlim)
    
    # 细白色网格
    ax.set_xticks(np.arange(len(C.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(C.index)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.0, alpha=0.8)
    ax.tick_params(which='minor', bottom=False, left=False)
    
    # 轴与刻度
    ax.set_xticks(range(len(C.columns)))
    ax.set_yticks(range(len(C.index)))
    ax.set_xticklabels(C.columns, rotation=45, ha='right')
    ax.set_yticklabels(C.index)
    
    # 去除上右脊柱
    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)
    
    # 注释值（粗体数字）
    for i in range(len(C.index)):
        for j in range(len(C.columns)):
            val = C.iloc[i, j]
            if not np.isfinite(val):
                txt = ""
            else:
                txt = f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=10, fontweight='bold')
    
    # 颜色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("相关系数", rotation=90)
    
    # 标题
    ax.set_title(title, fontsize=14, pad=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR_Q1_FIGS, outname), dpi=180)
    plt.close()

# 生成Pearson和Spearman相关系数热力图
C_pearson = df[cols].corr(method="pearson")
plot_corr_heatmap(C_pearson, "Pearson 相关系数热力图", "heatmap_pearson.png")

C_spear = df[cols].corr(method="spearman")
plot_corr_heatmap(C_spear, "Spearman 相关系数热力图", "heatmap_spearman.png")

# ================================
# 9. 分箱箱线图可视化
# ================================
# 按孕周和BMI区间分组，展示Y染色体浓度的分布特征
# 包含中位数、四分位数、异常值等统计信息

# 按孕周区间分组的箱线图
bins_gw = np.arange(np.floor(df["gest_week"].min()), np.ceil(df["gest_week"].max())+1, 2)
df["gest_bin"] = pd.cut(df["gest_week"], bins=bins_gw)
ax = df.boxplot(column="y", by="gest_bin", patch_artist=True,
                boxprops=dict(edgecolor="#555", linewidth=1.2),
                medianprops=dict(color="#d62728", linewidth=2.0),
                whiskerprops=dict(color="#555", linewidth=1.0),
                capprops=dict(color="#555", linewidth=1.0),
                flierprops=dict(marker='o', markersize=3, markerfacecolor="#555", alpha=0.25, markeredgecolor='none'))
# 填充柔和配色（viridis渐变）
try:
    cmap = plt.colormaps.get_cmap('viridis')
    for i, b in enumerate(ax.artists):
        b.set_facecolor(cmap(i/max(1, len(ax.artists)-1)))
        b.set_alpha(0.4)
except Exception:
    pass
plt.xticks(rotation=45)
plt.xlabel("孕周区间"); plt.ylabel("Y 浓度")
plt.title("按孕周区间分组的 Y 浓度分布"); plt.suptitle("")
ax.grid(axis='y', linestyle='--', alpha=0.25)
for spine in ['top','right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR_Q1_FIGS, "box_y_by_gwbin.png"), dpi=160)
plt.close()

# 按BMI区间分组的箱线图
bins_bmi = [15,20,25,30,35,40,50]
df["bmi_bin"]  = pd.cut(df["bmi"], bins=bins_bmi)
ax = df.boxplot(column="y", by="bmi_bin", patch_artist=True,
                boxprops=dict(edgecolor="#555", linewidth=1.2),
                medianprops=dict(color="#d62728", linewidth=2.0),
                whiskerprops=dict(color="#555", linewidth=1.0),
                capprops=dict(color="#555", linewidth=1.0),
                flierprops=dict(marker='o', markersize=3, markerfacecolor="#555", alpha=0.25, markeredgecolor='none'))
# 填充柔和配色（plasma渐变）
try:
    cmap = plt.colormaps.get_cmap('plasma')
    for i, b in enumerate(ax.artists):
        b.set_facecolor(cmap(i/max(1, len(ax.artists)-1)))
        b.set_alpha(0.4)
except Exception:
    pass
plt.xticks(rotation=45)
plt.xlabel("BMI 区间"); plt.ylabel("Y 浓度")
plt.title("按 BMI 区间分组的 Y 浓度分布"); plt.suptitle("")
ax.grid(axis='y', linestyle='--', alpha=0.25)
for spine in ['top','right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR_Q1_FIGS, "box_y_by_bmibin.png"), dpi=160)
plt.close()

# ================================
# 10. 3D曲面图可视化
# ================================
# 展示Y染色体浓度在(孕周,BMI)二维网格上的均值分布
# 3D曲面图直观显示两个主要影响因素对Y浓度的联合效应

from mpl_toolkits.mplot3d import Axes3D

# 创建孕周和BMI的网格
gw_grid = np.linspace(df["gest_week"].min(), df["gest_week"].max(), 12)
bmi_grid = np.linspace(df["bmi"].min(), df["bmi"].max(), 12)
H = np.zeros((len(bmi_grid)-1, len(gw_grid)-1))

# 计算每个网格单元的Y浓度均值
for i in range(len(bmi_grid)-1):
    for j in range(len(gw_grid)-1):
        mask = (df["bmi"].values>=bmi_grid[i])&(df["bmi"].values<bmi_grid[i+1])&\
               (df["gest_week"].values>=gw_grid[j])&(df["gest_week"].values<gw_grid[j+1])
        yy = df["y"].values[mask]
        H[i, j] = np.nan if len(yy)==0 else yy.mean()

# 创建3D曲面图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 创建网格坐标
X, Y = np.meshgrid(gw_grid[:-1], bmi_grid[:-1])
Z = H

# 绘制曲面
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)

# 添加等高线投影（底部）
ax.contour(X, Y, Z, zdir='z', offset=Z.min()-0.01, cmap='viridis', alpha=0.5)

# 设置标签和标题
ax.set_xlabel('孕周（周）')
ax.set_ylabel('BMI')
ax.set_zlabel('Y 染色体浓度')
ax.set_title('3D曲面图：Y 浓度在 (孕周,BMI) 网格上的均值')

# 添加颜色条
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.1)

# 设置视角（俯仰角25度，方位角45度）
ax.view_init(elev=25, azim=45)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR_Q1_FIGS, "heat_mean_y_on_gw_bmi.png"), dpi=180)
plt.close()

# ================================
# 11. 统计建模：Gamma-GLM + 分位数回归
# ================================
# 使用Gamma-GLM建模Y染色体浓度，考虑其正值特性和可能的异方差性
# 同时使用分位数回归分析不同分位数的效应

def build_cubic_truncated_power_basis(x_raw, n_knots=4):
    """
    构建三次截断幂样条基函数，用于捕捉孕周的非线性效应
    
    参数:
    - x_raw: 原始孕周数据
    - n_knots: 内部结点数量
    
    返回:
    - X_basis: 样条基矩阵
    - xc_mean: 中心化均值
    - knots: 结点位置
    """
    x = np.asarray(x_raw).astype(float)
    xc_mean = x.mean()
    xc = x - xc_mean
    # 内部结点（分位点）
    quantiles = np.linspace(0.2, 0.8, n_knots)
    knots = np.quantile(xc, quantiles)
    knots = np.unique(np.round(knots, 6))
    Xg = np.column_stack([xc, xc**2, xc**3])
    for k in knots:
        Xg = np.column_stack([Xg, np.maximum(0.0, xc - k)**3])
    X_basis = np.column_stack([np.ones(len(x)), Xg])
    return X_basis, xc_mean, knots

# 构建孕周的样条基
X_gw, gw_center, gw_knots = build_cubic_truncated_power_basis(df["gest_week"].values, n_knots=4)

# BMI的二次项（标准化）
bmi_c = (df["bmi"].values - df["bmi"].values.mean())/df["bmi"].values.std()
bmi2_c = bmi_c**2
X_full = np.column_stack([X_gw, bmi_c, bmi2_c])

# 处理Y染色体浓度的正值约束
y_pos = df["y"].values.copy()
y_pos[y_pos <= 0] = 1e-6  # 将非正值替换为极小正值

# 拟合Gamma-GLM模型（对数链接）
glm_gamma = sm.GLM(y_pos, X_full, family=sm.families.Gamma(sm.families.links.log()))
res_glm = glm_gamma.fit(maxiter=200)

# ================================
# 12. 模型诊断：残差分析
# ================================
# 生成GLM模型的诊断图表，评估模型假设的合理性

def qq_with_envelope(resid, sims=300, outpath=None, title="Gamma-GLM 偏差残差 QQ 图（含95%分位带）"):
    """
    绘制带模拟分位带的QQ图，用于检验残差的正态性
    
    参数:
    - resid: 残差向量
    - sims: 模拟次数
    - outpath: 输出路径
    - title: 图表标题
    """
    resid = np.asarray(resid)
    resid = resid[np.isfinite(resid)]
    n = resid.size
    # 生成模拟正态分布数据
    theo = np.sort(np.random.normal(size=(sims, n)), axis=1)
    lo = np.percentile(theo, 2.5, axis=0)
    hi = np.percentile(theo, 97.5, axis=0)
    theo_mean = np.mean(theo, axis=0)
    obs_sorted = np.sort(resid)
    plt.figure()
    # 分位带（95%置信区间）
    plt.fill_between(theo_mean, lo, hi, color="#C5CAE9", alpha=0.6, label="95%分位带")
    # 样本点
    plt.scatter(theo_mean, obs_sorted, s=10, color="#1f1f1f", alpha=0.8, label="残差分位")
    # 对角参考线
    lims = [min(theo_mean.min(), obs_sorted.min()), max(theo_mean.max(), obs_sorted.max())]
    plt.plot(lims, lims, color="#D32F2F", linewidth=1.2)
    plt.xlabel("理论正态分位")
    plt.ylabel("样本残差分位")
    plt.title(title)
    plt.legend(frameon=False)
    if outpath:
        savefig_tidy(outpath)
    else:
        plt.tight_layout()
        plt.show()

# 生成GLM偏差残差QQ图
qq_with_envelope(res_glm.resid_deviance, sims=300, outpath=os.path.join(OUTDIR_Q1_FIGS, "glm_resid_qq.png"))

# 异方差检验（Breusch-Pagan检验）
ols_ref = sm.OLS(np.log(y_pos), X_full).fit()
bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(ols_ref.resid, ols_ref.model.exog)
metrics_lines.append("\n[参考] OLS(log y) 的 Breusch–Pagan 异方差检验：")
metrics_lines.append(f"LM={bp_lm:.3f}, LM_p={bp_lm_p:.3e}, F={bp_f:.3f}, F_p={bp_f_p:.3e}")

# 拟合值 vs 标准化偏差残差图（叠加LOWESS与密度等高线）
infl = res_glm.get_influence(observed=True)
lev = infl.hat_matrix_diag
resid_dev = np.asarray(res_glm.resid_deviance)
std_resid = resid_dev / np.sqrt(np.maximum(1e-8, 1.0 - lev))
fitted = np.asarray(res_glm.fittedvalues)

fig, ax = plt.subplots()
ax.scatter(fitted, std_resid, s=10, alpha=0.35, color="#4C78A8", edgecolors="white", linewidths=0.2)
try:
    overlay_kde_contours(fitted, std_resid, ax, cmap="Greys", alpha=0.25)
except Exception:
    pass
idx = np.isfinite(fitted) & np.isfinite(std_resid)
if idx.sum() > 10:
    lo = lowess(std_resid[idx], fitted[idx], frac=0.3, return_sorted=True)
    ax.plot(lo[:,0], lo[:,1], color="#D32F2F", linewidth=2.0, alpha=0.9)
ax.axhline(0, color="#333", linewidth=1.0)
ax.set_xlabel("拟合值")
ax.set_ylabel("标准化偏差残差")
ax.set_title("Gamma-GLM 拟合值 vs 标准化偏差残差（含LOWESS与密度等高线）")
savefig_tidy(os.path.join(OUTDIR_Q1_FIGS, "glm_fitted_vs_resid.png"))

# Cook距离图（识别高影响点）
cooks = infl.cooks_distance[0]
fig, ax = plt.subplots(figsize=(10, 4))
ax.stem(range(len(cooks)), cooks, basefmt=" ")
thr = 4.0 / max(1, len(cooks))
ax.axhline(thr, color="#D32F2F", linestyle="--", linewidth=1.2, label=f"阈值≈{thr:.4f}")
ax.set_xlabel("样本索引")
ax.set_ylabel("Cook 距离")
ax.set_title("Cook 距离图（高杠杆/高影响点识别）")
ax.legend(frameon=False)
savefig_tidy(os.path.join(OUTDIR_Q1_FIGS, "glm_cooks_distance.png"))

# 记录Top影响点
top_k = 5
top_idx = np.argsort(cooks)[-top_k:][::-1]
metrics_lines.append("\n[Cook 距离 Top 样本]")
for rank, i in enumerate(top_idx, 1):
    metrics_lines.append(f"Top{rank}: idx={i}, Cook={cooks[i]:.6f}")

# ================================
# 13. 分位数回归分析
# ================================
# 使用分位数回归分析不同分位数下的效应，提供更全面的关系描述

taus = [0.25, 0.5, 0.75]  # 25%、50%、75%分位数
qr_summ = []
for t in taus:
    qr = sm.QuantReg(y_pos, X_full).fit(q=t)
    qr_summ.append((t, qr))

# ================================
# 14. 预测曲线可视化
# ================================
# 展示不同BMI水平下，孕周对Y染色体浓度的预测效应

def predict_curve_glm(weeks, bmi_value, res, gw_center, gw_knots):
    """
    基于GLM模型预测不同孕周和BMI下的Y染色体浓度
    
    参数:
    - weeks: 孕周数组
    - bmi_value: BMI值
    - res: 拟合的GLM模型
    - gw_center: 孕周中心化均值
    - gw_knots: 样条结点
    
    返回:
    - mu: 预测的Y染色体浓度
    """
    weeks = np.asarray(weeks).astype(float)
    xc = weeks - gw_center
    Xg = np.column_stack([xc, xc**2, xc**3])
    for k in gw_knots:
        Xg = np.column_stack([Xg, np.maximum(0.0, xc - k)**3])
    const = np.ones((len(weeks), 1))
    bmi_c0 = (bmi_value - df["bmi"].values.mean())/df["bmi"].values.std()
    bmi2_c0 = bmi_c0**2
    bmi_block = np.column_stack([np.full(len(weeks), bmi_c0), np.full(len(weeks), bmi2_c0)])
    X_new = np.column_stack([const, Xg, bmi_block])
    mu = res.predict(X_new)
    return mu

# 生成不同BMI水平的预测曲线
w_grid = np.linspace(df["gest_week"].min(), df["gest_week"].max(), 100)
bmi_levels = [22, 28, 34]  # 低、中、高BMI水平
plt.figure()
for b in bmi_levels:
    y_hat = predict_curve_glm(w_grid, b, res_glm, gw_center, gw_knots)
    plt.plot(w_grid, y_hat, label=f"BMI={b}")
plt.xlabel("孕周（周）"); plt.ylabel("预测的 Y 浓度")
plt.title("Gamma-GLM：不同 BMI 的孕周-预测曲线")
# 可选择加图例
# plt.legend()
savefig_tidy(os.path.join(OUTDIR_Q1_FIGS, "glm_pred_curves.png"))

# ================================
# 15. 特殊美观图表：雷达图
# ================================
# 极坐标雷达图展示不同BMI组在各孕周区间的达标率模式
# 直观显示BMI对达标时间的影响

def create_radar_chart():
    """
    创建极坐标雷达图，展示不同BMI组的孕周达标率模式
    
    图表含义：
    - 每个"花朵"代表一个BMI组
    - 花瓣长度表示该孕周区间的达标率
    - 颜色区分不同BMI组
    - 半透明填充增强视觉效果
    """
    # 按BMI分组计算各孕周的达标率
    bmi_groups = [(15, 25), (25, 30), (30, 35), (35, 50)]
    gw_bins = np.arange(8, 31, 3)  # 孕周区间
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    angles = np.linspace(0, 2 * np.pi, len(gw_bins)-1, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    for i, (bmi_min, bmi_max) in enumerate(bmi_groups):
        mask = (df['bmi'] >= bmi_min) & (df['bmi'] < bmi_max)
        if mask.sum() == 0:
            continue
            
        group_data = df[mask]
        success_rates = []
        
        for j in range(len(gw_bins)-1):
            gw_mask = (group_data['gest_week'] >= gw_bins[j]) & (group_data['gest_week'] < gw_bins[j+1])
            if gw_mask.sum() > 0:
                success_rate = (group_data[gw_mask]['y'] >= 0.04).mean()
            else:
                success_rate = 0
            success_rates.append(success_rate)
        
        success_rates += success_rates[:1]  # 闭合
        ax.plot(angles, success_rates, 'o-', linewidth=2, label=f'BMI {bmi_min}-{bmi_max}', color=colors[i])
        ax.fill(angles, success_rates, alpha=0.25, color=colors[i])
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f'{gw_bins[i]}-{gw_bins[i+1]}周' for i in range(len(gw_bins)-1)])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.grid(True)
    ax.set_title('不同BMI组的孕周达标率雷达图', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR_Q1_FIGS, "radar_chart_bmi_gw.png"), dpi=180, bbox_inches='tight')
    plt.close()

# 生成特殊图表
create_radar_chart()

# ================================
# 16. 结果输出与摘要
# ================================
# 将所有分析结果汇总到文本文件中，便于后续查看和报告撰写

summary_txt = os.path.join(OUTDIR_Q1, "q1_model_summary.txt")
os.makedirs(OUTDIR_Q1, exist_ok=True)
with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("================ [相关性结果] ================\n")
    f.write("\n".join(metrics_lines))
    f.write("\n\n================ [Gamma-GLM 摘要] ================\n")
    f.write(str(res_glm.summary()))
    f.write("\n\n================ [分位数回归 tau=0.25/0.5/0.75] ================\n")
    for t, qr in qr_summ:
        f.write(f"\n--- tau={t} ---\n")
        f.write(str(qr.summary()))

print(f"[DONE] 第一问分析完成（图表与诊断已补齐）：图表->{OUTDIR_Q1_FIGS}，摘要->{summary_txt}")
