# -*- coding: utf-8 -*-
"""
提取并运行Jupyter notebook中的方法
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data():
    """数据加载和预处理（模拟Jupyter notebook方法）"""
    print("=" * 60)
    print("数据加载和预处理（Jupyter方法）")
    print("=" * 60)
    
    # 读取男胎数据，使用不同的方法模拟notebook中的处理
    df = pd.read_csv('../../题目材料/boy.csv', encoding='gbk')
    print(f"原始数据形状: {df.shape}")
    
    # 先解析孕周，再进行聚合
    def parse_gestation(s):
        if pd.isna(s):
            return np.nan
        s = str(s).strip().lower()
        try:
            return float(s)
        except ValueError:
            import re
            m = re.match(r"(\d+)\s*w(?:\s*\+\s*(\d+))?", s)
            if not m:
                return np.nan
            w = int(m.group(1))
            d = int(m.group(2)) if m.group(2) else 0
            return w + d/7.0
    
    df['weeks_parsed'] = df['检测孕周'].apply(parse_gestation)
    
    # 处理多次检测数据（聚合）
    # 按孕妇代码分组，取均值
    group_cols = ['孕妇代码']
    agg_dict = {
        'weeks_parsed': 'mean',
        '孕妇BMI': 'mean', 
        'Y染色体浓度': 'mean',
        '年龄': 'mean',
        '身高': 'mean',
        '体重': 'mean'
    }
    
    df_agg = df.groupby(group_cols).agg(agg_dict).reset_index()
    print(f"聚合后数据形状: {df_agg.shape}")
    
    # 重命名列
    df_agg = df_agg.rename(columns={'weeks_parsed': 'weeks'})
    
    # 清洗数据
    df_clean = df_agg.dropna(subset=['weeks', '孕妇BMI', 'Y染色体浓度']).copy()
    df_clean = df_clean[(df_clean['weeks'] >= 8) & (df_clean['weeks'] <= 30)]
    df_clean = df_clean[(df_clean['孕妇BMI'] >= 15) & (df_clean['孕妇BMI'] <= 50)]
    df_clean = df_clean[df_clean['Y染色体浓度'] > 0]
    
    # 极端值处理
    y_low = df_clean['Y染色体浓度'].quantile(0.01)
    y_high = df_clean['Y染色体浓度'].quantile(0.99)
    df_clean = df_clean[(df_clean['Y染色体浓度'] >= y_low) & (df_clean['Y染色体浓度'] <= y_high)]
    
    print(f"最终有效样本数: {len(df_clean)}")
    print(f"Y染色体浓度范围: {df_clean['Y染色体浓度'].min():.6f} - {df_clean['Y染色体浓度'].max():.6f}")
    print(f"Y染色体浓度均值: {df_clean['Y染色体浓度'].mean():.6f} ± {df_clean['Y染色体浓度'].std():.6f}")
    
    return df_clean

def optimal_feature_engineering(df):
    """特征工程（模拟notebook方法）"""
    print("\n" + "=" * 60)
    print("特征工程")
    print("=" * 60)
    
    # 基础特征
    features = {
        'weeks': df['weeks'],
        '孕妇BMI': df['孕妇BMI'],
        '年龄': df['年龄'],
        '身高': df['身高'],
        '体重': df['体重']
    }
    
    # 计算BMI（如果需要）
    if '身高' in df.columns and '体重' in df.columns:
        height_m = df['身高'] / 100  # 转换为米
        features['BMI_calculated'] = df['体重'] / (height_m ** 2)
    
    # 医学相关的特征工程
    features['BMI_squared'] = df['孕妇BMI'] ** 2
    features['weeks_squared'] = df['weeks'] ** 2
    features['BMI_weeks_interaction'] = df['孕妇BMI'] * df['weeks']
    
    # 如果有身高体重数据，添加更多特征
    if '身高' in df.columns and '体重' in df.columns:
        features['weight_height_ratio'] = df['体重'] / df['身高']
        features['age_BMI_interaction'] = df['年龄'] * df['孕妇BMI']
        features['age_weeks_interaction'] = df['年龄'] * df['weeks']
    
    # 组装特征矩阵
    X_engineered = pd.DataFrame(features)
    y_target = df['Y染色体浓度'].values
    
    # 计算特征相关性
    corr_with_target = {}
    for feature in X_engineered.columns:
        if X_engineered[feature].notna().sum() > 0:
            corr, p_val = stats.pearsonr(X_engineered[feature].dropna(), 
                                       y_target[X_engineered[feature].notna()])
            corr_with_target[feature] = {'correlation': corr, 'p_value': p_val}
    
    print("特征与目标变量的相关性:")
    for feature, stats_dict in corr_with_target.items():
        corr = stats_dict['correlation']
        p_val = stats_dict['p_value']
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{feature:20s}: r={corr:7.4f}, p={p_val:.3e} {significance}")
    
    return X_engineered.fillna(0), y_target, corr_with_target

def train_models(X, y):
    """训练多种模型（模拟notebook方法）"""
    print("\n" + "=" * 60)
    print("模型训练和选择")
    print("=" * 60)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练多个模型
    models = {}
    results = {}
    
    # 1. OLS回归
    print("1. 训练OLS回归...")
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)
    y_pred_ols = ols.predict(X_test_scaled)
    r2_ols = r2_score(y_test, y_pred_ols)
    models['OLS'] = ols
    results['OLS'] = r2_ols
    print(f"   OLS R² = {r2_ols:.6f}")
    
    # 2. Ridge回归（多个alpha）
    print("2. 训练Ridge回归...")
    alphas = [0.1, 1.0, 10.0, 100.0]
    best_ridge_alpha = None
    best_ridge_score = -np.inf
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        y_pred_ridge = ridge.predict(X_test_scaled)
        r2_ridge = r2_score(y_test, y_pred_ridge)
        print(f"   Ridge (α={alpha}) R² = {r2_ridge:.6f}")
        
        if r2_ridge > best_ridge_score:
            best_ridge_score = r2_ridge
            best_ridge_alpha = alpha
            models['Ridge'] = ridge
            results['Ridge'] = r2_ridge
    
    # 3. Lasso回归
    print("3. 训练Lasso回归...")
    lasso_alphas = [0.001, 0.01, 0.1, 1.0]
    best_lasso_alpha = None
    best_lasso_score = -np.inf
    
    for alpha in lasso_alphas:
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train_scaled, y_train)
        y_pred_lasso = lasso.predict(X_test_scaled)
        r2_lasso = r2_score(y_test, y_pred_lasso)
        print(f"   Lasso (α={alpha}) R² = {r2_lasso:.6f}")
        
        if r2_lasso > best_lasso_score:
            best_lasso_score = r2_lasso
            best_lasso_alpha = alpha
            models['Lasso'] = lasso
            results['Lasso'] = r2_lasso
    
    # 选择最佳模型
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    best_score = results[best_model_name]
    
    print(f"\n最佳模型: {best_model_name} (R² = {best_score:.6f})")
    
    return {
        'models': models,
        'results': results,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'best_score': best_score,
        'scaler': scaler,
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test
    }

def comprehensive_statistical_tests(model_data):
    """统计假设检验和显著性分析（模拟notebook方法）"""
    print("\n" + "=" * 60)
    print("统计假设检验和显著性分析")
    print("=" * 60)
    
    best_model = model_data['best_model']
    X_test = model_data['X_test']
    y_test = model_data['y_test']
    
    # 预测
    y_pred = best_model.predict(X_test)
    residuals = y_test - y_pred
    
    # 1. 正态性检验
    print("1. 正态性检验")
    print("-" * 40)
    
    # Shapiro-Wilk检验
    if len(residuals) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        print(f"Shapiro-Wilk检验:")
        print(f"  统计量: {shapiro_stat:.6f}")
        print(f"  p值: {shapiro_p:.6f}")
        print(f"  结论: {'拒绝正态分布假设' if shapiro_p < 0.05 else '接受正态分布假设'} (α=0.05)")
    
    # Kolmogorov-Smirnov检验
    ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))
    print(f"\nKolmogorov-Smirnov检验:")
    print(f"  统计量: {ks_stat:.6f}")
    print(f"  p值: {ks_p:.6f}")
    print(f"  结论: {'拒绝正态分布假设' if ks_p < 0.05 else '接受正态分布假设'} (α=0.05)")
    
    # Jarque-Bera检验
    jb_stat, jb_p = stats.jarque_bera(residuals)
    print(f"\nJarque-Bera检验:")
    print(f"  统计量: {jb_stat:.6f}")
    print(f"  p值: {jb_p:.6f}")
    print(f"  结论: {'拒绝正态分布假设' if jb_p < 0.05 else '接受正态分布假设'} (α=0.05)")
    
    # 2. 残差分析
    print(f"\n2. 残差分析")
    print("-" * 40)
    print(f"残差均值: {residuals.mean():.6f}")
    print(f"残差标准差: {residuals.std():.6f}")
    print(f"残差最小值: {residuals.min():.6f}")
    print(f"残差最大值: {residuals.max():.6f}")
    
    # 3. 模型性能评估
    print(f"\n3. 模型性能评估")
    print("-" * 40)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # MAPE
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # 预测准确率
    accuracy_10 = np.mean(np.abs((y_test - y_pred) / y_test) <= 0.1) * 100
    accuracy_20 = np.mean(np.abs((y_test - y_pred) / y_test) <= 0.2) * 100
    
    print(f"R² (决定系数): {r2:.4f}")
    print(f"RMSE (均方根误差): {rmse:.6f}")
    print(f"MAE (平均绝对误差): {mae:.6f}")
    print(f"MAPE (平均绝对百分比误差): {mape:.2f}%")
    print(f"预测准确率 (±10%): {accuracy_10:.1f}%")
    print(f"预测准确率 (±20%): {accuracy_20:.1f}%")
    
    # 4. 效应大小评估
    print(f"\n4. 效应大小评估")
    print("-" * 40)
    
    # Cohen's效应大小
    if r2 < 0.01:
        effect_size = "无效应"
    elif r2 < 0.09:
        effect_size = "小效应"
    elif r2 < 0.25:
        effect_size = "中等效应"
    else:
        effect_size = "大效应"
    
    print(f"Cohen's 效应大小分类: {effect_size}")
    print(f"解释方差百分比: {r2*100:.2f}%")
    print(f"未解释方差百分比: {(1-r2)*100:.2f}%")
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'accuracy_10': accuracy_10,
        'accuracy_20': accuracy_20,
        'effect_size': effect_size,
        'shapiro_p': shapiro_p if len(residuals) <= 5000 else None,
        'ks_p': ks_p,
        'jb_p': jb_p,
        'residuals': residuals,
        'y_pred': y_pred
    }

def main():
    """主函数：运行完整的Jupyter notebook方法"""
    
    # 1. 数据加载和预处理
    df = load_and_preprocess_data()
    
    # 2. 特征工程
    X_engineered, y_target, feature_correlations = optimal_feature_engineering(df)
    
    # 3. 模型训练
    model_data = train_models(X_engineered, y_target)
    
    # 4. 统计检验
    statistical_results = comprehensive_statistical_tests(model_data)
    
    # 总结
    print("\n" + "=" * 60)
    print("Jupyter方法总结")
    print("=" * 60)
    print(f"最佳模型: {model_data['best_model_name']}")
    print(f"测试集R²: {statistical_results['r2']:.4f}")
    print(f"RMSE: {statistical_results['rmse']:.6f}")
    print(f"效应大小: {statistical_results['effect_size']}")
    
    return {
        'model_data': model_data,
        'statistical_results': statistical_results,
        'feature_correlations': feature_correlations
    }

if __name__ == "__main__":
    results = main()