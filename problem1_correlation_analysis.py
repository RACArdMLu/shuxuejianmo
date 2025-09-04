#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 1: Analyze the correlation between fetal Y chromosome concentration 
and maternal gestational age, BMI and other indicators.
Establish a correlation model and test its significance.

Author: Generated for NIPT Mathematical Modeling Competition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data():
    """Load and preprocess the NIPT data"""
    print("Loading NIPT data...")
    df = pd.read_excel('附件.xlsx')
    
    # Convert gestational age to numeric (weeks)
    def convert_gestational_age(age_str):
        if pd.isna(age_str):
            return np.nan
        if isinstance(age_str, (int, float)):
            return float(age_str)
        
        age_str = str(age_str).strip().lower()
        
        # Handle format like "12w+3" (12 weeks + 3 days)
        if 'w' in age_str:
            # Remove 'w' and split by '+'
            age_str = age_str.replace('w', '')
            if '+' in age_str:
                parts = age_str.split('+')
                weeks = float(parts[0])
                days = float(parts[1]) if len(parts) > 1 else 0
                return weeks + days/7
            else:
                return float(age_str)
        
        # Handle format like "12+3" (12 weeks + 3 days)
        if '+' in age_str:
            parts = age_str.split('+')
            weeks = float(parts[0])
            days = float(parts[1]) if len(parts) > 1 else 0
            return weeks + days/7
        
        return float(age_str)
    
    df['孕周_数值'] = df['检测孕周'].apply(convert_gestational_age)
    
    # Filter for male fetuses only (Y chromosome concentration available)
    male_df = df[df['Y染色体浓度'].notna()].copy()
    
    print(f"Total samples: {len(df)}")
    print(f"Male fetus samples: {len(male_df)}")
    
    return male_df

def exploratory_data_analysis(df):
    """Perform exploratory data analysis"""
    print("\n=== Exploratory Data Analysis ===")
    
    # Basic statistics
    key_columns = ['孕周_数值', '孕妇BMI', 'Y染色体浓度', '年龄', '身高', '体重']
    print("\nBasic Statistics:")
    print(df[key_columns].describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df[key_columns].isnull().sum())
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[key_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f')
    plt.title('Correlation Matrix of Key Variables')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlation_matrix

def analyze_y_chromosome_correlation(df):
    """Analyze correlation between Y chromosome concentration and other factors"""
    print("\n=== Y Chromosome Concentration Correlation Analysis ===")
    
    # Remove rows with missing Y chromosome concentration or key variables
    analysis_df = df[['孕周_数值', '孕妇BMI', 'Y染色体浓度', '年龄', '身高', '体重']].dropna()
    
    target = 'Y染色体浓度'
    features = ['孕周_数值', '孕妇BMI', '年龄', '身高', '体重']
    
    # Calculate correlation coefficients
    correlations = {}
    for feature in features:
        corr_coef, p_value = stats.pearsonr(analysis_df[feature], analysis_df[target])
        correlations[feature] = {'correlation': corr_coef, 'p_value': p_value}
        print(f"{feature} vs {target}:")
        print(f"  Pearson correlation: {corr_coef:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significance: {'Significant' if p_value < 0.05 else 'Not significant'}")
        print()
    
    return correlations, analysis_df

def create_scatter_plots(df):
    """Create scatter plots for key relationships"""
    print("Creating scatter plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    features = ['孕周_数值', '孕妇BMI', '年龄', '身高', '体重']
    target = 'Y染色体浓度'
    
    for i, feature in enumerate(features):
        if i < len(axes):
            # Remove missing values for this pair
            plot_data = df[[feature, target]].dropna()
            
            axes[i].scatter(plot_data[feature], plot_data[target], alpha=0.6, s=30)
            
            # Add trend line
            if len(plot_data) > 1:
                z = np.polyfit(plot_data[feature], plot_data[target], 1)
                p = np.poly1d(z)
                axes[i].plot(plot_data[feature], p(plot_data[feature]), "r--", alpha=0.8)
                
                # Calculate R²
                y_pred = p(plot_data[feature])
                r2 = r2_score(plot_data[target], y_pred)
                axes[i].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[i].transAxes,
                           bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
            
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel(target)
            axes[i].set_title(f'{target} vs {feature}')
            axes[i].grid(True, alpha=0.3)
    
    # Remove unused subplot
    if len(features) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('scatter_plots_y_chromosome.png', dpi=300, bbox_inches='tight')
    plt.show()

def build_regression_models(df):
    """Build regression models for Y chromosome concentration"""
    print("\n=== Regression Model Analysis ===")
    
    # Prepare data
    analysis_df = df[['孕周_数值', '孕妇BMI', 'Y染色体浓度', '年龄', '身高', '体重']].dropna()
    
    X = analysis_df[['孕周_数值', '孕妇BMI', '年龄', '身高', '体重']]
    y = analysis_df['Y染色体浓度']
    
    print(f"Analysis dataset size: {len(analysis_df)} samples")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Multiple linear regression
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    
    print(f"\nMultiple Linear Regression Results:")
    print(f"R² Score: {r2:.4f}")
    print(f"Adjusted R²: {1 - (1-r2)*(len(y)-1)/(len(y)-X.shape[1]-1):.4f}")
    
    # Feature importance (standardized coefficients)
    feature_names = ['孕周_数值', '孕妇BMI', '年龄', '身高', '体重']
    coefficients = model.coef_
    
    print(f"\nStandardized Coefficients:")
    for name, coef in zip(feature_names, coefficients):
        print(f"{name}: {coef:.4f}")
    
    print(f"Intercept: {model.intercept_:.4f}")
    
    # Statistical significance test
    from scipy.stats import f
    
    # Calculate F-statistic
    n = len(y)
    k = X.shape[1]
    mse_model = np.mean((y - y_pred) ** 2)
    mse_residual = np.mean((y - np.mean(y)) ** 2)
    f_stat = (mse_residual - mse_model) / mse_model * (n - k - 1) / k
    p_value_f = 1 - f.cdf(f_stat, k, n - k - 1)
    
    print(f"\nModel Significance Test:")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value_f:.4f}")
    print(f"Model significance: {'Significant' if p_value_f < 0.05 else 'Not significant'}")
    
    # Residual analysis
    residuals = y - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Residuals vs Fitted
    axes[0].scatter(y_pred, residuals, alpha=0.6)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Fitted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Fitted Values')
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Normal Q-Q Plot of Residuals')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, scaler, analysis_df

def analyze_gestational_age_effect(df):
    """Special analysis for gestational age effect"""
    print("\n=== Gestational Age Effect Analysis ===")
    
    # Filter data with valid gestational age and Y chromosome concentration
    ga_df = df[['孕周_数值', 'Y染色体浓度', '孕妇BMI']].dropna()
    
    # Analyze Y chromosome concentration reaching 4% threshold
    ga_df['达标'] = ga_df['Y染色体浓度'] >= 0.04
    
    # Group by gestational age
    ga_groups = ga_df.groupby(pd.cut(ga_df['孕周_数值'], bins=np.arange(10, 26, 1))).agg({
        'Y染色体浓度': ['mean', 'std', 'count'],
        '达标': 'mean'
    }).round(4)
    
    print("Y Chromosome Concentration by Gestational Age Groups:")
    print(ga_groups)
    
    # Plot Y chromosome concentration trend with gestational age
    plt.figure(figsize=(12, 8))
    
    # Create age bins for better visualization
    ga_df['孕周_分组'] = pd.cut(ga_df['孕周_数值'], bins=np.arange(10, 26, 1))
    grouped_stats = ga_df.groupby('孕周_分组').agg({
        'Y染色体浓度': ['mean', 'std'],
        '达标': 'mean'
    })
    
    # Plot mean Y chromosome concentration
    weeks = [interval.mid for interval in grouped_stats.index if not pd.isna(interval)]
    means = grouped_stats['Y染色体浓度']['mean'].dropna()
    stds = grouped_stats['Y染色体浓度']['std'].dropna()
    
    plt.subplot(2, 1, 1)
    plt.errorbar(weeks, means, yerr=stds, marker='o', capsize=5, capthick=2)
    plt.axhline(y=0.04, color='r', linestyle='--', label='4% Threshold')
    plt.xlabel('Gestational Age (weeks)')
    plt.ylabel('Y Chromosome Concentration')
    plt.title('Y Chromosome Concentration vs Gestational Age')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot proportion reaching threshold
    plt.subplot(2, 1, 2)
    proportion_达标 = grouped_stats['达标']['mean'].dropna()
    plt.plot(weeks[:len(proportion_达标)], proportion_达标, marker='s', color='green')
    plt.xlabel('Gestational Age (weeks)')
    plt.ylabel('Proportion Reaching 4% Threshold')
    plt.title('Proportion of Samples Reaching 4% Threshold vs Gestational Age')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gestational_age_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return ga_groups

def main():
    """Main analysis function"""
    print("=== NIPT Problem 1: Y Chromosome Concentration Correlation Analysis ===")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Exploratory data analysis
    correlation_matrix = exploratory_data_analysis(df)
    
    # Analyze correlations
    correlations, analysis_df = analyze_y_chromosome_correlation(df)
    
    # Create visualizations
    create_scatter_plots(df)
    
    # Build regression models
    model, scaler, model_df = build_regression_models(df)
    
    # Special analysis for gestational age
    ga_analysis = analyze_gestational_age_effect(df)
    
    # Summary
    print("\n=== SUMMARY ===")
    print("1. Correlation Analysis:")
    for feature, results in correlations.items():
        significance = "significant" if results['p_value'] < 0.05 else "not significant"
        print(f"   {feature}: r = {results['correlation']:.4f}, p = {results['p_value']:.4f} ({significance})")
    
    print(f"\n2. Multiple Regression Model:")
    print(f"   R² = {r2_score(model_df['Y染色体浓度'], model.predict(scaler.transform(model_df[['孕周_数值', '孕妇BMI', '年龄', '身高', '体重']]))):.4f}")
    
    print(f"\n3. Key Findings:")
    print(f"   - Gestational age shows {'strong' if abs(correlations['孕周_数值']['correlation']) > 0.5 else 'moderate' if abs(correlations['孕周_数值']['correlation']) > 0.3 else 'weak'} correlation with Y chromosome concentration")
    print(f"   - BMI shows {'strong' if abs(correlations['孕妇BMI']['correlation']) > 0.5 else 'moderate' if abs(correlations['孕妇BMI']['correlation']) > 0.3 else 'weak'} correlation with Y chromosome concentration")
    
    return correlations, model, scaler

if __name__ == "__main__":
    main()