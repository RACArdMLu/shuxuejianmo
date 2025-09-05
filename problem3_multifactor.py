#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 3: Multi-factor analysis for Y chromosome concentration timing
Comprehensive analysis considering height, weight, age, detection errors,
and Y chromosome concentration达标比例

Author: Generated for NIPT Mathematical Modeling Competition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data():
    """Load and preprocess the NIPT data for Problem 3"""
    print("Loading NIPT data for Problem 3...")
    df = pd.read_excel('附件.xlsx')
    
    # Convert gestational age to numeric (weeks)
    def convert_gestational_age(age_str):
        if pd.isna(age_str):
            return np.nan
        if isinstance(age_str, (int, float)):
            return float(age_str)
        
        age_str = str(age_str).strip().lower()
        
        if 'w' in age_str:
            age_str = age_str.replace('w', '')
            if '+' in age_str:
                parts = age_str.split('+')
                weeks = float(parts[0])
                days = float(parts[1]) if len(parts) > 1 else 0
                return weeks + days/7
            else:
                return float(age_str)
        
        if '+' in age_str:
            parts = age_str.split('+')
            weeks = float(parts[0])
            days = float(parts[1]) if len(parts) > 1 else 0
            return weeks + days/7
        
        return float(age_str)
    
    df['孕周_数值'] = df['检测孕周'].apply(convert_gestational_age)
    
    # Filter for male fetuses only
    male_df = df[df['Y染色体浓度'].notna()].copy()
    
    # Add threshold achievement flag
    male_df['Y染色体达标'] = male_df['Y染色体浓度'] >= 0.04
    
    # Create additional features
    male_df['身体状况指数'] = male_df['体重'] / (male_df['身高'] / 100) ** 2  # Same as BMI but calculated differently
    male_df['年龄分组'] = pd.cut(male_df['年龄'], bins=[20, 25, 30, 35, 45], labels=['20-25', '25-30', '30-35', '35-45'])
    
    print(f"Male fetus samples: {len(male_df)}")
    
    return male_df

def comprehensive_feature_analysis(df):
    """Perform comprehensive analysis of all factors affecting Y chromosome concentration"""
    print("\n=== Comprehensive Multi-Factor Analysis ===")
    
    # All potential features
    feature_columns = ['孕周_数值', '孕妇BMI', '年龄', '身高', '体重', 
                      'GC含量', '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']
    
    # Remove rows with missing values in key columns
    analysis_df = df[feature_columns + ['Y染色体浓度', 'Y染色体达标']].dropna()
    
    print(f"Analysis dataset size: {len(analysis_df)} samples")
    
    # Correlation analysis
    print("\nCorrelation with Y chromosome concentration:")
    correlations = {}
    for feature in feature_columns:
        corr_coef, p_value = stats.pearsonr(analysis_df[feature], analysis_df['Y染色体浓度'])
        correlations[feature] = {'correlation': corr_coef, 'p_value': p_value}
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"  {feature}: r = {corr_coef:.4f}, p = {p_value:.4f} {significance}")
    
    # Feature importance using Random Forest
    X = analysis_df[feature_columns]
    y = analysis_df['Y染色体浓度']
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nRandom Forest Feature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    return analysis_df, correlations, feature_importance

def analyze_threshold_achievement_by_factors(df):
    """Analyze factors affecting threshold achievement timing"""
    print("\n=== Threshold Achievement Analysis by Multiple Factors ===")
    
    # Group by pregnant woman and find earliest threshold achievement
    threshold_data = []
    
    for woman_code in df['孕妇代码'].unique():
        woman_data = df[df['孕妇代码'] == woman_code].sort_values('孕周_数值')
        
        # Get woman's characteristics (assuming they don't change much)
        woman_info = woman_data.iloc[0]
        
        # Find all达标 timepoints
        达标_times = woman_data[woman_data['Y染色体达标'] == True]['孕周_数值'].tolist()
        
        if len(达标_times) > 0:
            earliest_time = min(达标_times)
            latest_time = max(达标_times)
            
            # Calculate达标比例 for this woman across all her tests
            达标_proportion = woman_data['Y染色体达标'].mean()
            
            threshold_data.append({
                '孕妇代码': woman_code,
                '最早达标时间': earliest_time,
                '最晚达标时间': latest_time,
                '达标比例': 达标_proportion,
                '测试次数': len(woman_data),
                '孕妇BMI': woman_info['孕妇BMI'],
                '年龄': woman_info['年龄'],
                '身高': woman_info['身高'],
                '体重': woman_info['体重'],
                'GC含量_平均': woman_data['GC含量'].mean(),
                'Y染色体浓度_最大': woman_data['Y染色体浓度'].max(),
                'Y染色体浓度_平均': woman_data['Y染色体浓度'].mean()
            })
    
    threshold_df = pd.DataFrame(threshold_data)
    
    print(f"Women with threshold achievement: {len(threshold_df)}")
    print(f"Average tests per woman: {threshold_df['测试次数'].mean():.1f}")
    print(f"Average达标比例: {threshold_df['达标比例'].mean():.2%}")
    
    return threshold_df

def build_predictive_model(threshold_df):
    """Build predictive model for threshold achievement timing"""
    print("\n=== Predictive Model for Threshold Timing ===")
    
    # Features for prediction
    features = ['孕妇BMI', '年龄', '身高', '体重', 'GC含量_平均', 'Y染色体浓度_最大']
    target = '最早达标时间'
    
    # Prepare data
    model_df = threshold_df[features + [target]].dropna()
    X = model_df[features]
    y = model_df[target]
    
    print(f"Model dataset size: {len(model_df)} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred_train = rf_model.predict(X_train_scaled)
    y_pred_test = rf_model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"Model Performance:")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Testing R²: {test_r2:.4f}")
    print(f"  Training RMSE: {train_rmse:.4f} weeks")
    print(f"  Testing RMSE: {test_rmse:.4f} weeks")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\nFeature Importance for Threshold Timing Prediction:")
    for _, row in feature_importance.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    return rf_model, scaler, feature_importance

def advanced_bmi_grouping(threshold_df):
    """Advanced BMI grouping considering multiple factors"""
    print("\n=== Advanced BMI Grouping with Multiple Factors ===")
    
    # Use multiple features for clustering
    clustering_features = ['孕妇BMI', '年龄', '身高', '体重', '达标比例', '最早达标时间']
    
    # Prepare data for clustering
    cluster_data = threshold_df[clustering_features].dropna()
    
    # Standardize features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # Try different numbers of clusters and find optimal
    inertias = []
    silhouette_scores = []
    K_range = range(2, 8)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        inertias.append(kmeans.inertia_)
        
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(scaled_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Choose optimal number of clusters (elbow method + silhouette)
    optimal_k = 4  # Can be adjusted based on elbow plot
    print(f"Using {optimal_k} clusters for grouping")
    
    # Final clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Add cluster labels to dataframe
    cluster_df = cluster_data.copy()
    cluster_df['Cluster'] = cluster_labels
    
    # Analyze clusters
    cluster_analysis = cluster_df.groupby('Cluster').agg({
        '孕妇BMI': ['min', 'max', 'mean'],
        '年龄': ['min', 'max', 'mean'],
        '最早达标时间': ['mean', 'std'],
        '达标比例': ['mean', 'std']
    }).round(4)
    
    print("\nAdvanced Cluster Analysis:")
    print(cluster_analysis)
    
    return cluster_df, cluster_labels

def calculate_comprehensive_optimal_timing(cluster_df):
    """Calculate optimal timing considering multiple factors and error scenarios"""
    print("\n=== Comprehensive Optimal Timing Calculation ===")
    
    optimal_results = {}
    
    for cluster_id in cluster_df['Cluster'].unique():
        cluster_data = cluster_df[cluster_df['Cluster'] == cluster_id]
        
        # Basic statistics
        bmi_range = f"[{cluster_data['孕妇BMI'].min():.1f}, {cluster_data['孕妇BMI'].max():.1f}]"
        age_range = f"[{cluster_data['年龄'].min():.0f}, {cluster_data['年龄'].max():.0f}]"
        mean_threshold_time = cluster_data['最早达标时间'].mean()
        mean_达标_ratio = cluster_data['达标比例'].mean()
        
        # Multi-objective optimization for timing
        def comprehensive_risk_function(test_time):
            """Comprehensive risk function considering multiple factors"""
            
            # Factor 1: Detection rate (proportion who达标 by test time)
            detection_rate = (cluster_data['最早达标时间'] <= test_time).mean()
            
            # Factor 2: Risk based on timing windows
            if test_time <= 12:
                timing_risk = 1.0  # Low risk
            elif test_time <= 27:
                timing_risk = 2.0 + (test_time - 12) * 0.1  # Increasing risk
            else:
                timing_risk = 5.0  # High risk
            
            # Factor 3: False negative risk (weighted by达标比例)
            false_negative_risk = (1 - detection_rate) * 10 * (1 - mean_达标_ratio)
            
            # Factor 4: Population heterogeneity penalty
            time_std = cluster_data['最早达标时间'].std()
            heterogeneity_penalty = time_std * 0.5
            
            # Combined risk score
            total_risk = timing_risk + false_negative_risk + heterogeneity_penalty
            
            return total_risk
        
        # Find optimal timing
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(comprehensive_risk_function, bounds=(10, 24), method='bounded')
        optimal_time = result.x
        
        # Calculate metrics at optimal timing
        detection_rate_optimal = (cluster_data['最早达标时间'] <= optimal_time).mean()
        risk_score_optimal = comprehensive_risk_function(optimal_time)
        
        # Error sensitivity analysis
        error_impact = []
        error_rates = [0.0, 0.05, 0.10, 0.15, 0.20]
        
        for error_rate in error_rates:
            # Simulate detection with error
            np.random.seed(42)
            simulated_detection_rate = detection_rate_optimal * (1 - error_rate * 0.5)  # Simplified error model
            error_impact.append(simulated_detection_rate)
        
        optimal_results[cluster_id] = {
            'cluster_size': len(cluster_data),
            'bmi_range': bmi_range,
            'age_range': age_range,
            'mean_threshold_time': mean_threshold_time,
            'mean_达标_ratio': mean_达标_ratio,
            'optimal_timing': optimal_time,
            'detection_rate': detection_rate_optimal,
            'risk_score': risk_score_optimal,
            'error_sensitivity': error_impact
        }
        
        print(f"\nCluster {cluster_id}:")
        print(f"  BMI Range: {bmi_range}")
        print(f"  Age Range: {age_range}")
        print(f"  Sample Size: {len(cluster_data)}")
        print(f"  Mean Threshold Time: {mean_threshold_time:.1f} weeks")
        print(f"  Mean达标比例: {mean_达标_ratio:.1%}")
        print(f"  Optimal NIPT Timing: {optimal_time:.1f} weeks")
        print(f"  Detection Rate: {detection_rate_optimal:.1%}")
        print(f"  Risk Score: {risk_score_optimal:.2f}")
    
    return optimal_results

def visualize_comprehensive_results(cluster_df, optimal_results):
    """Create comprehensive visualizations"""
    print("\n=== Creating Comprehensive Visualizations ===")
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # 1. Multi-dimensional scatter plot (BMI vs Age, colored by cluster)
    ax1 = axes[0, 0]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, cluster_id in enumerate(cluster_df['Cluster'].unique()):
        cluster_data = cluster_df[cluster_df['Cluster'] == cluster_id]
        ax1.scatter(cluster_data['孕妇BMI'], cluster_data['年龄'], 
                   c=colors[i % len(colors)], alpha=0.7, label=f'Cluster {cluster_id}')
    ax1.set_xlabel('BMI')
    ax1.set_ylabel('Age')
    ax1.set_title('BMI vs Age by Clusters')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Threshold time vs达标比例
    ax2 = axes[0, 1]
    for i, cluster_id in enumerate(cluster_df['Cluster'].unique()):
        cluster_data = cluster_df[cluster_df['Cluster'] == cluster_id]
        ax2.scatter(cluster_data['最早达标时间'], cluster_data['达标比例'], 
                   c=colors[i % len(colors)], alpha=0.7, label=f'Cluster {cluster_id}')
    ax2.set_xlabel('Earliest Threshold Time (weeks)')
    ax2.set_ylabel('达标比例')
    ax2.set_title('Threshold Time vs达标比例')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Optimal timing comparison
    ax3 = axes[1, 0]
    clusters = list(optimal_results.keys())
    timings = [optimal_results[c]['optimal_timing'] for c in clusters]
    detection_rates = [optimal_results[c]['detection_rate'] for c in clusters]
    
    bars = ax3.bar(clusters, timings, alpha=0.7)
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Optimal Timing (weeks)')
    ax3.set_title('Optimal NIPT Timing by Cluster')
    
    # Add detection rate labels
    for bar, rate in zip(bars, detection_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{rate:.1%}', ha='center', va='bottom')
    ax3.grid(True, alpha=0.3)
    
    # 4. Risk scores comparison
    ax4 = axes[1, 1]
    risk_scores = [optimal_results[c]['risk_score'] for c in clusters]
    ax4.bar(clusters, risk_scores, alpha=0.7, color='red')
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Risk Score')
    ax4.set_title('Risk Score by Cluster')
    ax4.grid(True, alpha=0.3)
    
    # 5. Error sensitivity analysis
    ax5 = axes[2, 0]
    error_rates = [0.0, 0.05, 0.10, 0.15, 0.20]
    for i, cluster_id in enumerate(clusters):
        error_impact = optimal_results[cluster_id]['error_sensitivity']
        ax5.plot(error_rates, error_impact, 'o-', label=f'Cluster {cluster_id}', 
                color=colors[i % len(colors)])
    ax5.set_xlabel('Error Rate')
    ax5.set_ylabel('Detection Rate')
    ax5.set_title('Error Sensitivity Analysis')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Comprehensive comparison matrix
    ax6 = axes[2, 1]
    
    # Create comparison matrix
    metrics = ['Optimal Timing', 'Detection Rate', 'Risk Score', 'Sample Size']
    comparison_data = []
    
    for cluster_id in clusters:
        row = [
            optimal_results[cluster_id]['optimal_timing'],
            optimal_results[cluster_id]['detection_rate'] * 100,  # Convert to percentage
            optimal_results[cluster_id]['risk_score'],
            optimal_results[cluster_id]['cluster_size']
        ]
        comparison_data.append(row)
    
    # Normalize data for heatmap
    comparison_array = np.array(comparison_data).T
    normalized_data = (comparison_array - comparison_array.min(axis=1, keepdims=True)) / \
                     (comparison_array.max(axis=1, keepdims=True) - comparison_array.min(axis=1, keepdims=True))
    
    im = ax6.imshow(normalized_data, aspect='auto', cmap='RdYlBu_r')
    ax6.set_xticks(range(len(clusters)))
    ax6.set_xticklabels([f'Cluster {c}' for c in clusters])
    ax6.set_yticks(range(len(metrics)))
    ax6.set_yticklabels(metrics)
    ax6.set_title('Normalized Comparison Matrix')
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(clusters)):
            ax6.text(j, i, f'{comparison_data[j][i]:.1f}', 
                    ha="center", va="center", color="white" if normalized_data[i, j] > 0.5 else "black")
    
    plt.colorbar(im, ax=ax6)
    
    plt.tight_layout()
    plt.savefig('problem3_comprehensive_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function for Problem 3"""
    print("=== NIPT Problem 3: Comprehensive Multi-Factor Analysis ===")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Comprehensive feature analysis
    analysis_df, correlations, feature_importance = comprehensive_feature_analysis(df)
    
    # Analyze threshold achievement by multiple factors
    threshold_df = analyze_threshold_achievement_by_factors(df)
    
    # Build predictive model
    model, scaler, model_importance = build_predictive_model(threshold_df)
    
    # Advanced BMI grouping
    cluster_df, cluster_labels = advanced_bmi_grouping(threshold_df)
    
    # Calculate comprehensive optimal timing
    optimal_results = calculate_comprehensive_optimal_timing(cluster_df)
    
    # Create visualizations
    visualize_comprehensive_results(cluster_df, optimal_results)
    
    # Final Summary
    print("\n=== PROBLEM 3 COMPREHENSIVE SUMMARY ===")
    print("\nOptimal BMI Groups and NIPT Timing (Multi-Factor Analysis):")
    
    for cluster_id in sorted(optimal_results.keys()):
        result = optimal_results[cluster_id]
        print(f"\nCluster {cluster_id}:")
        print(f"  BMI Range: {result['bmi_range']}")
        print(f"  Age Range: {result['age_range']}")
        print(f"  Sample Size: {result['cluster_size']}")
        print(f"  Optimal NIPT Timing: {result['optimal_timing']:.1f} weeks")
        print(f"  Expected Detection Rate: {result['detection_rate']:.1%}")
        print(f"  Risk Score: {result['risk_score']:.2f}")
        print(f"  Error Sensitivity: Detection rate drops by {(1-result['error_sensitivity'][-1]/result['detection_rate']):.1%} at 20% error")
    
    print(f"\nTop 3 Most Important Features for Threshold Timing:")
    for i, (_, row) in enumerate(model_importance.head(3).iterrows()):
        print(f"  {i+1}. {row['Feature']}: {row['Importance']:.3f}")
    
    return optimal_results, cluster_df, model

if __name__ == "__main__":
    main()