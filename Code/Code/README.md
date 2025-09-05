# C题— 运行说明

## 环境配置
- **Python**：建议 3.9+（3.10/3.11 亦可）
- **依赖**：pandas、numpy、scipy、statsmodels、matplotlib、openpyxl

安装示例（任选其一）：

- 使用 venv + pip：

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install pandas numpy scipy statsmodels matplotlib openpyxl
```

- 使用 conda：

```bash
conda create -n c python=3.10 -y
conda activate c
pip install pandas numpy scipy statsmodels matplotlib openpyxl


如果网络超时使用清华源安装：
pip install -U pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.26.4 scipy==1.11.4 pandas==2.2.2 statsmodels==0.14.2 matplotlib==3.8.4 openpyxl==3.1.5
```

## 目录与文件
建议将以下文件放在同一目录（当前目录示例）：
- `config.py`：统一配置文件（数据路径、sheet 名、输出目录、阈值等）
- `q1_1_preprocess.py`：第一问·数据预处理与清洗
- `q1_2_analysis.py`：第一问·相关性与关系模型分析
- `outputs/`：脚本运行的输出目录（自动创建）

## 配置 `config.py`
在 `config.py` 中按需修改以下关键项（示例）：

```python
# 数据路径与工作表名
DATA_FILE = "附件.xlsx"                # 原始数据文件（xlsx/csv）

# 输出目录
OUTDIR = "outputs"                    # 所有结果输出根目录

# 关键参数
Y_THRESHOLD = 0.04                     # Y 浓度达标阈值（4%）
RANDOM_SEED = 2025
```


## 第一问脚本说明与运行顺序
1) `q1_1_preprocess.py`（数据预处理与清洗）
- 功能：
  - 读取原始数据；对关键列做鲁棒匹配（BMI、检测孕周、Y 浓度、GC、比对率、过滤率、总读段、孕妇代码等）。
  - 将孕周格式（如 `13w+5`、`23w`）转换为小数周；校核/复算 BMI；质量控制（合理范围过滤）与极端值裁剪（如 1%–99% 分位）。
  - 拆分并导出清洗后的数据：
    - `outputs/clean_male.csv`
    - `outputs/clean_female.csv`
- 如何运行：
  - 在 IDE 中直接运行该脚本（无需命令行参数）。
  - 成功后将在 `outputs/` 下看到清洗后的 CSV。

2) `q1_2_analysis.py`（相关性与关系模型）
- 功能：
  - 读取 `outputs/clean_male.csv`。
  - 相关性分析：Pearson / Spearman / 偏相关（控制 GC、比对率、过滤率、读段数）。
  - 可视化：
    - 散点 + LOWESS（Y vs 孕周 / BMI / 质量项），每图单独画布。
    - 相关矩阵热力图、孕周/BMI 分箱箱线图、(孕周,BMI) 网格均值图。
  - 建模与显著性：
    - Gamma-GLM（log 链接）：孕周三次截断幂样条 + BMI 二次项，输出摘要与诊断图。
    - 分位数回归（τ=0.25/0.5/0.75）：展示不同分位下效应差异。
  - 输出：
    - 图表：`outputs/q1/figs/*.png`
    - 文本：`outputs/q1/q1_model_summary.txt`
- 如何运行：
  - 在 IDE 中直接运行该脚本（确保 `q1_1_preprocess.py` 已先运行）。

## 结果位置
- 预处理结果：`outputs/clean_male.csv`、`outputs/clean_female.csv`
- 第一问图表与模型摘要：`outputs/q1/`

## 常见问题
- 中文字体乱码：可在脚本中设置 Matplotlib 字体或使用英文标签。
- 无法找到列：检查原表列名是否与附录含义匹配，或在脚本的候选列名列表中补充你的表头名称。
- 读取 Excel 失败：请确认已安装 `openpyxl`，或先将数据另存为 `.xlsx`。

##图片含义
- 不用担心看不懂图片含义，代码中有很详细的注释解释每个图的含义，直接稍微修改下写在论文里即可