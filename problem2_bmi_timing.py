#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 2: BMI grouping and optimal NIPT timing for male fetuses
Determine optimal BMI groups and NIPT timing to minimize potential risks

Author: Generated for NIPT Mathematical Modeling Competition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data():
    """Load and preprocess the NIPT data for Problem 2"""
    print("Loading NIPT data for Problem 2...")
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
    
    # Add threshold achievement flag
    male_df['Y染色体达标'] = male_df['Y染色体浓度'] >= 0.04
    
    print(f"Male fetus samples: {len(male_df)}")
    
    return male_df

def analyze_time_to_threshold(df):
    """Analyze the time when Y chromosome concentration first reaches 4%"""
    print("\n=== Time to Threshold Analysis ===")
    
    # Group by pregnant woman to find earliest threshold achievement time
    threshold_times = []
    
    for woman_code in df['孕妇代码'].unique():
        woman_data = df[df['孕妇代码'] == woman_code].sort_values('孕周_数值')
        
        # Find first time reaching threshold
        达标_data = woman_data[woman_data['Y染色体达标'] == True]
        if len(达标_data) > 0:
            earliest_time = 达标_data['孕周_数值'].min()
            bmi = woman_data['孕妇BMI'].iloc[0]
            
            threshold_times.append({
                '孕妇代码': woman_code,
                '最早达标时间': earliest_time,
                '孕妇BMI': bmi,
                '年龄': woman_data['年龄'].iloc[0],
                '身高': woman_data['身高'].iloc[0],
                '体重': woman_data['体重'].iloc[0]
            })
    
    threshold_df = pd.DataFrame(threshold_times)
    
    print(f"Number of women with threshold achievement: {len(threshold_df)}")
    print("\nBasic statistics for earliest threshold time:")
    print(threshold_df['最早达标时间'].describe())
    
    return threshold_df

def determine_bmi_groups(threshold_df, n_groups=4):
    """Determine optimal BMI groups using clustering and risk analysis"""
    print(f"\n=== BMI Grouping Analysis (n_groups={n_groups}) ===")
    
    # Method 1: Equal-frequency grouping (quantile-based)
    quantile_groups = pd.qcut(threshold_df['孕妇BMI'], q=n_groups, labels=False)
    threshold_df['Quantile_Group'] = quantile_groups
    
    # Method 2: K-means clustering on BMI
    kmeans = KMeans(n_clusters=n_groups, random_state=42)
    bmi_array = threshold_df['孕妇BMI'].values.reshape(-1, 1)
    cluster_groups = kmeans.fit_predict(bmi_array)
    threshold_df['Cluster_Group'] = cluster_groups
    
    # Method 3: Custom grouping based on BMI distribution and threshold times
    # Analyze relationship between BMI and threshold time
    correlation = threshold_df['孕妇BMI'].corr(threshold_df['最早达标时间'])
    print(f"Correlation between BMI and earliest threshold time: {correlation:.4f}")
    
    # Create custom groups based on BMI ranges and optimization
    def calculate_risk_score(group_data):
        """Calculate risk score for a group based on timing"""
        mean_time = group_data['最早达标时间'].mean()
        
        # Risk scoring based on timing windows
        early_risk = (mean_time <= 12).sum() * 1  # Low risk
        medium_risk = ((mean_time > 12) & (mean_time <= 27)).sum() * 2  # Medium risk  
        late_risk = (mean_time > 27).sum() * 5  # High risk
        
        return (early_risk + medium_risk + late_risk) / len(group_data)
    
    # Try different grouping strategies and select the one with minimum total risk
    best_groups = None
    best_risk = float('inf')
    best_method = None
    
    for method in ['Quantile_Group', 'Cluster_Group']:
        grouped = threshold_df.groupby(method)
        total_risk = 0
        
        for group_id, group_data in grouped:
            risk = calculate_risk_score(group_data)
            total_risk += risk * len(group_data)
        
        avg_risk = total_risk / len(threshold_df)
        
        if avg_risk < best_risk:
            best_risk = avg_risk
            best_groups = method
            best_method = method
    
    print(f"Best grouping method: {best_method} with average risk score: {best_risk:.4f}")
    
    # Analyze each group
    grouped_stats = threshold_df.groupby(best_groups).agg({
        '孕妇BMI': ['min', 'max', 'mean', 'count'],
        '最早达标时间': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print("\nBMI Group Statistics:")
    print(grouped_stats)
    
    return threshold_df, best_groups, grouped_stats

def calculate_optimal_timing(threshold_df, group_column):
    """Calculate optimal NIPT timing for each BMI group"""
    print("\n=== Optimal NIPT Timing Calculation ===")
    
    optimal_timings = {}
    
    for group_id in threshold_df[group_column].unique():
        group_data = threshold_df[threshold_df[group_column] == group_id]
        
        # Get BMI range for this group
        bmi_min = group_data['孕妇BMI'].min()
        bmi_max = group_data['孕妇BMI'].max()
        bmi_mean = group_data['孕妇BMI'].mean()
        
        # Calculate statistics for threshold times
        times = group_data['最早达标时间']
        mean_time = times.mean()
        std_time = times.std()
        min_time = times.min()
        percentile_10 = times.quantile(0.1)
        percentile_25 = times.quantile(0.25)
        
        # Risk-based optimal timing calculation
        # We want to minimize risk while ensuring reasonable detection rate
        
        def risk_function(test_time):
            """Calculate total risk for a given test time"""
            # Early testing before threshold may lead to false negatives
            early_tests = (times < test_time).sum()
            detection_rate = early_tests / len(times)
            
            # Risk components:
            # 1. Risk of early detection failure (false negative)
            false_negative_risk = (1 - detection_rate) * 10
            
            # 2. Risk based on timing windows
            if test_time <= 12:
                timing_risk = 1  # Low risk - early detection
            elif test_time <= 27:
                timing_risk = 2  # Medium risk - middle period
            else:
                timing_risk = 5  # High risk - late detection
            
            # 3. Balance between detection rate and timing risk
            total_risk = false_negative_risk + timing_risk
            
            return total_risk
        
        # Find optimal timing that minimizes risk
        result = minimize_scalar(risk_function, bounds=(10, 25), method='bounded')
        optimal_time = result.x
        
        # Alternative: Use statistical approach (mean - 0.5*std, but ensure it's reasonable)
        statistical_optimal = max(min_time, mean_time - 0.5 * std_time)
        statistical_optimal = min(statistical_optimal, 24)  # Don't go too late
        
        # Use the more conservative (earlier) of the two approaches
        final_optimal = min(optimal_time, statistical_optimal)
        
        optimal_timings[group_id] = {
            'BMI_range': f"[{bmi_min:.2f}, {bmi_max:.2f}]",
            'BMI_mean': bmi_mean,
            'sample_size': len(group_data),
            'mean_threshold_time': mean_time,
            'std_threshold_time': std_time,
            'optimal_timing': final_optimal,
            'detection_rate_at_optimal': (times <= final_optimal).mean(),
            'risk_score': risk_function(final_optimal)
        }
        
        print(f"\nGroup {group_id} (BMI {bmi_min:.2f}-{bmi_max:.2f}):")
        print(f"  Sample size: {len(group_data)}")
        print(f"  Mean threshold time: {mean_time:.2f} weeks")
        print(f"  Optimal NIPT timing: {final_optimal:.2f} weeks")
        print(f"  Expected detection rate: {(times <= final_optimal).mean():.2%}")
    
    return optimal_timings

def analyze_detection_errors(threshold_df, optimal_timings, group_column):
    """Analyze the impact of detection errors on results"""
    print("\n=== Detection Error Analysis ===")
    
    # Simulate different error scenarios
    error_scenarios = [0.0, 0.05, 0.10, 0.15, 0.20]  # 0% to 20% error
    
    results = []
    
    for error_rate in error_scenarios:
        for group_id in threshold_df[group_column].unique():
            group_data = threshold_df[threshold_df[group_column] == group_id]
            optimal_time = optimal_timings[group_id]['optimal_timing']
            
            # Simulate detection errors
            np.random.seed(42)  # For reproducibility
            
            # Add random error to Y chromosome concentration measurements
            simulated_concentrations = []
            for _, row in group_data.iterrows():
                # Get original concentration at optimal time (simulated)
                base_concentration = 0.04  # Threshold
                
                # Add random error
                error = np.random.normal(0, error_rate * base_concentration)
                new_concentration = base_concentration + error
                
                # Check if it would still be detected as达标
                detected = new_concentration >= 0.04
                simulated_concentrations.append(detected)
            
            # Calculate metrics with errors
            detection_rate_with_error = np.mean(simulated_concentrations)
            original_detection_rate = optimal_timings[group_id]['detection_rate_at_optimal']
            
            results.append({
                'group': group_id,
                'error_rate': error_rate,
                'original_detection_rate': original_detection_rate,
                'detection_rate_with_error': detection_rate_with_error,
                'detection_loss': original_detection_rate - detection_rate_with_error
            })
    
    error_df = pd.DataFrame(results)
    
    # Plot error impact
    plt.figure(figsize=(15, 10))
    
    for i, group_id in enumerate(threshold_df[group_column].unique()):
        group_error_data = error_df[error_df['group'] == group_id]
        
        plt.subplot(2, 2, i+1)
        plt.plot(group_error_data['error_rate'], group_error_data['detection_rate_with_error'], 
                'o-', label=f'With Error')
        plt.axhline(y=group_error_data['original_detection_rate'].iloc[0], 
                   color='r', linestyle='--', label='Original Rate')
        
        bmi_range = optimal_timings[group_id]['BMI_range']
        plt.title(f'Group {group_id} - BMI {bmi_range}')
        plt.xlabel('Error Rate')
        plt.ylabel('Detection Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return error_df

def visualize_results(threshold_df, optimal_timings, group_column):
    """Create visualizations for the results"""
    print("\n=== Creating Visualizations ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. BMI distribution by groups
    ax1 = axes[0, 0]
    for group_id in threshold_df[group_column].unique():
        group_data = threshold_df[threshold_df[group_column] == group_id]
        ax1.hist(group_data['孕妇BMI'], alpha=0.7, label=f'Group {group_id}', bins=15)
    ax1.set_xlabel('BMI')
    ax1.set_ylabel('Frequency')
    ax1.set_title('BMI Distribution by Groups')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Threshold time vs BMI
    ax2 = axes[0, 1]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, group_id in enumerate(threshold_df[group_column].unique()):
        group_data = threshold_df[threshold_df[group_column] == group_id]
        ax2.scatter(group_data['孕妇BMI'], group_data['最早达标时间'], 
                   alpha=0.6, c=colors[i % len(colors)], label=f'Group {group_id}')
        
        # Add optimal timing line
        bmi_range = [group_data['孕妇BMI'].min(), group_data['孕妇BMI'].max()]
        optimal_time = optimal_timings[group_id]['optimal_timing']
        ax2.plot(bmi_range, [optimal_time, optimal_time], 
                color=colors[i % len(colors)], linestyle='--', linewidth=2)
    
    ax2.set_xlabel('BMI')
    ax2.set_ylabel('Earliest Threshold Time (weeks)')
    ax2.set_title('Threshold Time vs BMI with Optimal Timing')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Optimal timing by group
    ax3 = axes[1, 0]
    groups = list(optimal_timings.keys())
    timings = [optimal_timings[g]['optimal_timing'] for g in groups]
    detection_rates = [optimal_timings[g]['detection_rate_at_optimal'] for g in groups]
    
    bars = ax3.bar(groups, timings, alpha=0.7)
    ax3.set_xlabel('BMI Group')
    ax3.set_ylabel('Optimal NIPT Timing (weeks)')
    ax3.set_title('Optimal NIPT Timing by BMI Group')
    
    # Add detection rate labels on bars
    for bar, rate in zip(bars, detection_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{rate:.1%}', ha='center', va='bottom')
    
    ax3.grid(True, alpha=0.3)
    
    # 4. Risk analysis
    ax4 = axes[1, 1]
    risk_scores = [optimal_timings[g]['risk_score'] for g in groups]
    ax4.bar(groups, risk_scores, alpha=0.7, color='red')
    ax4.set_xlabel('BMI Group')
    ax4.set_ylabel('Risk Score')
    ax4.set_title('Risk Score by BMI Group')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('problem2_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function for Problem 2"""
    print("=== NIPT Problem 2: BMI Grouping and Optimal NIPT Timing ===")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Analyze time to threshold
    threshold_df = analyze_time_to_threshold(df)
    
    # Determine optimal BMI groups
    threshold_df, best_group_method, group_stats = determine_bmi_groups(threshold_df)
    
    # Calculate optimal timing for each group
    optimal_timings = calculate_optimal_timing(threshold_df, best_group_method)
    
    # Analyze detection errors
    error_analysis = analyze_detection_errors(threshold_df, optimal_timings, best_group_method)
    
    # Create visualizations
    visualize_results(threshold_df, optimal_timings, best_group_method)
    
    # Summary
    print("\n=== PROBLEM 2 SUMMARY ===")
    print("Optimal BMI Groups and NIPT Timing:")
    
    for group_id in sorted(optimal_timings.keys()):
        info = optimal_timings[group_id]
        print(f"\nGroup {group_id}:")
        print(f"  BMI Range: {info['BMI_range']}")
        print(f"  Sample Size: {info['sample_size']}")
        print(f"  Optimal NIPT Timing: {info['optimal_timing']:.1f} weeks")
        print(f"  Expected Detection Rate: {info['detection_rate_at_optimal']:.1%}")
        print(f"  Risk Score: {info['risk_score']:.2f}")
    
    return threshold_df, optimal_timings, error_analysis

if __name__ == "__main__":
    main()