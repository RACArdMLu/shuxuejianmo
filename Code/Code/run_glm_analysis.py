# -*- coding: utf-8 -*-
"""
运行GLM分析适配到清洗后的数据
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def run_glm_analysis():
    """运行Gamma-GLM分析"""
    print("=" * 60)
    print("开始运行Gamma-GLM分析")
    print("=" * 60)
    
    # 读取清洗后的数据
    df = pd.read_csv('outputs/clean_male.csv')
    print(f"数据形状: {df.shape}")
    
    # 准备变量
    y = df['y'].values  # Y染色体浓度
    gest_week = df['gest_week'].values  # 孕周
    bmi = df['bmi'].values  # BMI
    
    # 确保没有负值或零值（Gamma分布要求正值）
    y_pos = np.maximum(y, 1e-6)
    
    # 构建设计矩阵
    X = np.column_stack([
        np.ones(len(df)),  # 截距
        gest_week,         # 孕周
        bmi,               # BMI
        gest_week**2,      # 孕周平方项
        bmi**2,            # BMI平方项
        gest_week * bmi    # 交互项
    ])
    
    feature_names = ['截距', '孕周', 'BMI', '孕周²', 'BMI²', '孕周×BMI']
    
    print(f"设计矩阵形状: {X.shape}")
    print(f"特征名称: {feature_names}")
    
    # 拟合Gamma-GLM模型（对数链接）
    print("\n拟合Gamma-GLM模型...")
    try:
        glm_gamma = sm.GLM(y_pos, X, family=sm.families.Gamma(sm.families.links.log()))
        glm_result = glm_gamma.fit(maxiter=200)
        
        print("GLM模型拟合成功!")
        print("\n模型摘要:")
        print(glm_result.summary())
        
        # 计算模型指标
        aic = glm_result.aic
        bic = glm_result.bic
        deviance = glm_result.deviance
        null_deviance = glm_result.null_deviance
        pseudo_r2 = 1 - deviance / null_deviance
        
        print(f"\n模型评估指标:")
        print(f"AIC: {aic:.3f}")
        print(f"BIC: {bic:.3f}")
        print(f"偏差: {deviance:.3f}")
        print(f"零模型偏差: {null_deviance:.3f}")
        print(f"伪R²: {pseudo_r2:.3f}")
        
        # 回归系数分析
        print(f"\n回归系数及显著性:")
        for i, name in enumerate(feature_names):
            coef = glm_result.params[i]
            pval = glm_result.pvalues[i]
            significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"{name:8s}: {coef:8.4f} (p={pval:.3e}) {significance}")
        
        # 残差分析
        print(f"\n残差分析:")
        residuals = glm_result.resid_deviance
        fitted = glm_result.fittedvalues
        
        # 正态性检验
        if len(residuals) <= 5000:
            _, p_norm = stats.shapiro(residuals)
            print(f"Shapiro-Wilk正态性检验: p={p_norm:.6f}")
        
        # 异方差检验（使用OLS进行参考）
        ols_ref = sm.OLS(np.log(y_pos), X).fit()
        bp_lm, bp_p, _, _ = het_breuschpagan(ols_ref.resid, ols_ref.model.exog)
        print(f"Breusch-Pagan异方差检验: LM={bp_lm:.3f}, p={bp_p:.6f}")
        
        return {
            'model': glm_result,
            'pseudo_r2': pseudo_r2,
            'aic': aic,
            'bic': bic,
            'coefficients': dict(zip(feature_names, glm_result.params)),
            'p_values': dict(zip(feature_names, glm_result.pvalues)),
            'residuals': residuals,
            'fitted': fitted
        }
        
    except Exception as e:
        print(f"GLM拟合失败: {e}")
        return None

def run_ridge_analysis():
    """运行Ridge回归分析用于比较"""
    print("=" * 60)
    print("开始运行Ridge回归分析")
    print("=" * 60)
    
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    
    # 读取数据
    df = pd.read_csv('outputs/clean_male.csv')
    
    # 准备特征
    X_features = df[['gest_week', 'bmi']].values
    y = df['y'].values
    
    # 添加多项式特征
    X_poly = np.column_stack([
        X_features,                              # 原始特征
        X_features[:, 0]**2,                     # 孕周²
        X_features[:, 1]**2,                     # BMI²
        X_features[:, 0] * X_features[:, 1]      # 交互项
    ])
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 拟合Ridge回归
    alphas = [0.1, 1.0, 10.0, 100.0]
    best_alpha = None
    best_score = -np.inf
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        score = ridge.score(X_test_scaled, y_test)
        print(f"Ridge (α={alpha}): R² = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_alpha = alpha
    
    # 使用最佳alpha重新拟合
    ridge_best = Ridge(alpha=best_alpha)
    ridge_best.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred_train = ridge_best.predict(X_train_scaled)
    y_pred_test = ridge_best.predict(X_test_scaled)
    
    # 评估
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"\n最佳Ridge回归结果 (α={best_alpha}):")
    print(f"训练集 R²: {r2_train:.4f}")
    print(f"测试集 R²: {r2_test:.4f}")
    print(f"训练集 RMSE: {rmse_train:.6f}")
    print(f"测试集 RMSE: {rmse_test:.6f}")
    
    return {
        'best_alpha': best_alpha,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'model': ridge_best,
        'scaler': scaler
    }

def main():
    """主函数：运行两种方法并比较"""
    
    # 运行GLM分析
    glm_results = run_glm_analysis()
    
    print("\n" + "="*60)
    
    # 运行Ridge分析
    ridge_results = run_ridge_analysis()
    
    print("\n" + "="*60)
    print("方法比较总结")
    print("="*60)
    
    if glm_results:
        print(f"Gamma-GLM 伪R²: {glm_results['pseudo_r2']:.4f}")
        print(f"Gamma-GLM AIC: {glm_results['aic']:.3f}")
        print(f"Gamma-GLM BIC: {glm_results['bic']:.3f}")
    
    if ridge_results:
        print(f"Ridge回归 测试R²: {ridge_results['r2_test']:.4f}")
        print(f"Ridge回归 最佳α: {ridge_results['best_alpha']}")
    
    return glm_results, ridge_results

if __name__ == "__main__":
    main()