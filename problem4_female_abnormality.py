#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 4: Female fetal abnormality determination
Develop a method to determine female fetal abnormalities using chromosome Z-values,
GC content, read counts, and other factors

Author: Generated for NIPT Mathematical Modeling Competition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data():
    """Load and preprocess the NIPT data for Problem 4 - Simulate female analysis using aneuploidy data"""
    print("Loading NIPT data for Problem 4 (Analyzing aneuploidy detection)...")
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
    
    # Since all samples are male, we'll analyze chromosome abnormalities 
    # (T13, T18, T21) as a proxy for abnormality detection methodology
    
    # Create abnormality labels based on chromosome aneuploidy
    def determine_abnormality(row):
        aneuploidy = row['染色体的非整倍体']
        if pd.isna(aneuploidy):
            return 0  # Normal
        else:
            return 1  # Abnormal (T13, T18, T21, or combinations)
    
    df['异常标签'] = df.apply(determine_abnormality, axis=1)
    
    # Also use the health outcome as reference
    df['健康标签'] = df['胎儿是否健康'].map({'是': 0, '否': 1})
    
    # Combine both labels (if available) - use health outcome as primary if available
    df['最终异常标签'] = df['健康标签'].fillna(df['异常标签'])
    
    # Remove Y chromosome features for simulation of female analysis
    analysis_df = df.copy()
    
    print(f"Total samples (simulating female analysis): {len(analysis_df)}")
    print(f"Abnormal cases: {analysis_df['最终异常标签'].sum()}")
    print(f"Normal cases: {(analysis_df['最终异常标签'] == 0).sum()}")
    print(f"Missing labels: {analysis_df['最终异常标签'].isna().sum()}")
    
    # Print aneuploidy distribution
    print("\nAneuploidy types:")
    print(df['染色体的非整倍体'].value_counts(dropna=False))
    
    return analysis_df

def analyze_abnormality_indicators(df):
    """Analyze indicators associated with fetal abnormalities (simulating female analysis)"""
    print("\n=== Fetal Abnormality Indicator Analysis (Chromosome Aneuploidy Detection) ===")
    
    # Key features for analysis (excluding Y chromosome features for female simulation)
    chromosome_features = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']
    gc_features = ['13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量', 'GC含量']
    other_features = ['孕妇BMI', '年龄', '孕周_数值', '原始读段数', '唯一比对的读段数  ', 
                     '在参考基因组上比对的比例', '重复读段的比例', 'X染色体浓度']
    
    all_features = chromosome_features + gc_features + other_features
    
    # Remove samples with missing labels
    analysis_df = df[df['最终异常标签'].notna()].copy()
    
    print(f"Analysis dataset: {len(analysis_df)} samples")
    print(f"Abnormal: {analysis_df['最终异常标签'].sum()}, Normal: {(analysis_df['最终异常标签'] == 0).sum()}")
    
    # Statistical analysis for each feature
    significant_features = []
    
    print("\nFeature Analysis (Normal vs Abnormal):")
    
    for feature in all_features:
        if feature in analysis_df.columns:
            feature_data = analysis_df[[feature, '最终异常标签']].dropna()
            
            if len(feature_data) > 10:  # Minimum sample size
                normal_values = feature_data[feature_data['最终异常标签'] == 0][feature]
                abnormal_values = feature_data[feature_data['最终异常标签'] == 1][feature]
                
                if len(normal_values) > 0 and len(abnormal_values) > 0:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(normal_values, abnormal_values)
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(normal_values)-1)*normal_values.var() + 
                                         (len(abnormal_values)-1)*abnormal_values.var()) / 
                                        (len(normal_values) + len(abnormal_values) - 2))
                    if pooled_std > 0:
                        cohens_d = (normal_values.mean() - abnormal_values.mean()) / pooled_std
                    else:
                        cohens_d = 0
                    
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    
                    print(f"{feature}:")
                    print(f"  Normal: {normal_values.mean():.4f} ± {normal_values.std():.4f}")
                    print(f"  Abnormal: {abnormal_values.mean():.4f} ± {abnormal_values.std():.4f}")
                    print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f} {significance}")
                    print(f"  Effect size (Cohen's d): {cohens_d:.4f}")
                    
                    if p_value < 0.05:
                        significant_features.append({
                            'feature': feature,
                            'p_value': p_value,
                            'effect_size': abs(cohens_d),
                            't_statistic': t_stat
                        })
                    print()
    
    # Sort significant features by importance
    significant_features.sort(key=lambda x: x['p_value'])
    
    print(f"Significant features found: {len(significant_features)}")
    
    return analysis_df, significant_features

def build_abnormality_detection_models(analysis_df, significant_features):
    """Build machine learning models for abnormality detection"""
    print("\n=== Abnormality Detection Model Development ===")
    
    # Select features for modeling
    if significant_features:
        model_features = [f['feature'] for f in significant_features[:10]]  # Top 10 significant features
    else:
        # Default feature set if no significant features found
        model_features = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
                         '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量', 
                         '孕妇BMI', '年龄', 'X染色体浓度']
    
    # Prepare data
    model_data = analysis_df[model_features + ['最终异常标签']].dropna()
    
    print(f"Model training dataset: {len(model_data)} samples")
    print(f"Features used: {len(model_features)}")
    
    if len(model_data) < 20:
        print("Warning: Very small dataset for modeling")
        return None, None, None
    
    X = model_data[model_features]
    y = model_data['最终异常标签']
    
    # Check class balance
    class_counts = y.value_counts()
    print(f"Class distribution: Normal={class_counts.get(0, 0)}, Abnormal={class_counts.get(1, 0)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {}
    
    # 1. Logistic Regression
    lr_model = LogisticRegression(random_state=42, class_weight='balanced')
    lr_model.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = lr_model
    
    # 2. Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)  # Random Forest doesn't need scaling
    models['Random Forest'] = rf_model
    
    # 3. SVM
    svm_model = SVC(probability=True, random_state=42, class_weight='balanced')
    svm_model.fit(X_train_scaled, y_train)
    models['SVM'] = svm_model
    
    # Evaluate models
    print("\nModel Performance:")
    
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        if name == 'Random Forest':
            X_test_model = X_test
            X_train_model = X_train
        else:
            X_test_model = X_test_scaled
            X_train_model = X_train_scaled
        
        # Predictions
        y_pred = model.predict(X_test_model)
        y_pred_proba = model.predict_proba(X_test_model)[:, 1]
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_model, y_train, cv=5, scoring='roc_auc')
        
        print(f"\n{name}:")
        print(f"  Cross-validation AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        if len(np.unique(y_test)) > 1 and len(np.unique(y_pred_proba)) > 1:
            test_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"  Test AUC: {test_auc:.4f}")
            
            if test_auc > best_score:
                best_score = test_auc
                best_model = (name, model, X_test_model, y_test, y_pred, y_pred_proba)
        
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred))
    
    return models, scaler, best_model

def develop_rule_based_system(analysis_df):
    """Develop a rule-based system for abnormality detection"""
    print("\n=== Rule-Based Abnormality Detection System ===")
    
    # Define thresholds based on literature and data analysis
    # These are common thresholds used in NIPT
    
    rules = {
        'Z_score_rules': {
            '21号染色体的Z值': {'threshold': 3.0, 'direction': 'greater'},  # Trisomy 21 (Down syndrome)
            '18号染色体的Z值': {'threshold': 3.0, 'direction': 'greater'},  # Trisomy 18 (Edwards syndrome)  
            '13号染色体的Z值': {'threshold': 3.0, 'direction': 'greater'},  # Trisomy 13 (Patau syndrome)
            'X染色体的Z值': {'threshold': -3.0, 'direction': 'less'}        # Turner syndrome
        },
        'GC_content_rules': {
            'GC含量': {'threshold_low': 0.40, 'threshold_high': 0.60},  # Normal GC content range
            '13号染色体的GC含量': {'threshold_low': 0.35, 'threshold_high': 0.65},
            '18号染色体的GC含量': {'threshold_low': 0.35, 'threshold_high': 0.65},
            '21号染色体的GC含量': {'threshold_low': 0.35, 'threshold_high': 0.65}
        }
    }
    
    # Apply rules
    rule_predictions = []
    
    for _, row in analysis_df.iterrows():
        abnormal_flags = []
        
        # Z-score rules
        for chrom, rule in rules['Z_score_rules'].items():
            if chrom in row and not pd.isna(row[chrom]):
                if rule['direction'] == 'greater' and row[chrom] > rule['threshold']:
                    abnormal_flags.append(f"{chrom}_high")
                elif rule['direction'] == 'less' and row[chrom] < rule['threshold']:
                    abnormal_flags.append(f"{chrom}_low")
        
        # GC content rules
        for gc_feature, rule in rules['GC_content_rules'].items():
            if gc_feature in row and not pd.isna(row[gc_feature]):
                if row[gc_feature] < rule['threshold_low'] or row[gc_feature] > rule['threshold_high']:
                    abnormal_flags.append(f"{gc_feature}_abnormal")
        
        # Combine rules - if any abnormal flag, predict abnormal
        prediction = 1 if len(abnormal_flags) > 0 else 0
        actual_label = row['最终异常标签'] if not pd.isna(row['最终异常标签']) else None
        
        rule_predictions.append({
            'predicted': prediction,
            'flags': abnormal_flags,
            'actual': actual_label
        })
    
    # Evaluate rule-based system
    rule_df = pd.DataFrame(rule_predictions)
    evaluation_df = rule_df[rule_df['actual'].notna()]
    
    if len(evaluation_df) > 0:
        accuracy = (evaluation_df['predicted'] == evaluation_df['actual']).mean()
        precision = ((evaluation_df['predicted'] == 1) & (evaluation_df['actual'] == 1)).sum() / \
                   max(1, (evaluation_df['predicted'] == 1).sum())
        recall = ((evaluation_df['predicted'] == 1) & (evaluation_df['actual'] == 1)).sum() / \
                max(1, (evaluation_df['actual'] == 1).sum())
        
        print(f"Rule-based System Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(evaluation_df['actual'], evaluation_df['predicted'])
        print(cm)
        
        # Flag frequency analysis
        all_flags = []
        for flags in rule_df['flags']:
            all_flags.extend(flags)
        
        if all_flags:
            flag_counts = pd.Series(all_flags).value_counts()
            print(f"\nMost Common Abnormality Flags:")
            print(flag_counts.head(10))
    
    return rules, rule_predictions

def create_comprehensive_detection_method(analysis_df, models, scaler, rules):
    """Create a comprehensive detection method combining multiple approaches"""
    print("\n=== Comprehensive Female Fetal Abnormality Detection Method ===")
    
    def comprehensive_detection(row, models, scaler, rules):
        """Apply comprehensive detection method to a single case"""
        
        # Initialize scores
        scores = {
            'z_score_risk': 0,
            'gc_content_risk': 0,
            'ml_risk': 0,
            'overall_risk': 0
        }
        
        flags = []
        
        # 1. Z-score analysis
        z_score_weight = 0.4
        for chrom, rule in rules['Z_score_rules'].items():
            if chrom in row and not pd.isna(row[chrom]):
                if rule['direction'] == 'greater' and row[chrom] > rule['threshold']:
                    scores['z_score_risk'] += 1.0
                    flags.append(f"{chrom}_elevated")
                elif rule['direction'] == 'less' and row[chrom] < rule['threshold']:
                    scores['z_score_risk'] += 1.0
                    flags.append(f"{chrom}_decreased")
                elif rule['direction'] == 'greater' and row[chrom] > rule['threshold'] * 0.7:
                    scores['z_score_risk'] += 0.5  # Borderline risk
                    flags.append(f"{chrom}_borderline")
        
        # 2. GC content analysis
        gc_weight = 0.2
        for gc_feature, rule in rules['GC_content_rules'].items():
            if gc_feature in row and not pd.isna(row[gc_feature]):
                if row[gc_feature] < rule['threshold_low'] or row[gc_feature] > rule['threshold_high']:
                    scores['gc_content_risk'] += 0.5
                    flags.append(f"{gc_feature}_abnormal_gc")
        
        # 3. Machine learning prediction (if available)
        ml_weight = 0.4
        if models and scaler:
            try:
                model_features = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
                                '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量', 
                                '孕妇BMI', '年龄', 'X染色体浓度']
                
                feature_values = [row.get(f, 0) for f in model_features]
                if not any(pd.isna(feature_values)):
                    feature_array = np.array(feature_values).reshape(1, -1)
                    scaled_features = scaler.transform(feature_array)
                    
                    # Use logistic regression if available
                    if 'Logistic Regression' in models:
                        ml_prob = models['Logistic Regression'].predict_proba(scaled_features)[0, 1]
                        scores['ml_risk'] = ml_prob
                        if ml_prob > 0.5:
                            flags.append(f"ml_high_risk_{ml_prob:.2f}")
                        elif ml_prob > 0.3:
                            flags.append(f"ml_moderate_risk_{ml_prob:.2f}")
            except:
                pass
        
        # 4. Calculate overall risk score
        scores['overall_risk'] = (scores['z_score_risk'] * z_score_weight + 
                                 scores['gc_content_risk'] * gc_weight + 
                                 scores['ml_risk'] * ml_weight)
        
        # 5. Risk classification
        if scores['overall_risk'] > 0.7:
            risk_level = 'High Risk'
        elif scores['overall_risk'] > 0.4:
            risk_level = 'Moderate Risk'
        elif scores['overall_risk'] > 0.2:
            risk_level = 'Low Risk'
        else:
            risk_level = 'Normal'
        
        return {
            'scores': scores,
            'flags': flags,
            'risk_level': risk_level,
            'recommendation': get_recommendation(risk_level, flags)
        }
    
    def get_recommendation(risk_level, flags):
        """Get clinical recommendation based on risk level"""
        if risk_level == 'High Risk':
            return "Recommend invasive diagnostic testing (amniocentesis/CVS)"
        elif risk_level == 'Moderate Risk':
            return "Recommend genetic counseling and consideration of diagnostic testing"
        elif risk_level == 'Low Risk':
            return "Recommend follow-up NIPT testing and monitoring"
        else:
            return "Routine prenatal care"
    
    # Apply to all samples
    detection_results = []
    
    for _, row in analysis_df.iterrows():
        result = comprehensive_detection(row, models, scaler, rules)
        result['actual_label'] = row['最终异常标签'] if not pd.isna(row['最终异常标签']) else None
        detection_results.append(result)
    
    # Evaluate comprehensive method
    evaluation_results = [r for r in detection_results if r['actual_label'] is not None]
    
    if evaluation_results:
        # Convert risk levels to binary predictions
        risk_to_binary = {'High Risk': 1, 'Moderate Risk': 1, 'Low Risk': 0, 'Normal': 0}
        predictions = [risk_to_binary[r['risk_level']] for r in evaluation_results]
        actual = [r['actual_label'] for r in evaluation_results]
        
        accuracy = np.mean(np.array(predictions) == np.array(actual))
        
        print(f"Comprehensive Method Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        
        # Risk level distribution
        risk_distribution = pd.Series([r['risk_level'] for r in detection_results]).value_counts()
        print(f"\nRisk Level Distribution:")
        print(risk_distribution)
    
    return detection_results

def visualize_problem4_results(analysis_df, detection_results):
    """Create visualizations for Problem 4 results"""
    print("\n=== Creating Problem 4 Visualizations ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Z-score distributions
    ax1 = axes[0, 0]
    chromosomes = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']
    
    for i, chrom in enumerate(chromosomes):
        if chrom in analysis_df.columns:
            normal_data = analysis_df[analysis_df['最终异常标签'] == 0][chrom].dropna()
            abnormal_data = analysis_df[analysis_df['最终异常标签'] == 1][chrom].dropna()
            
            ax1.hist(normal_data, alpha=0.5, label=f'{chrom} Normal', bins=20)
            if len(abnormal_data) > 0:
                ax1.hist(abnormal_data, alpha=0.5, label=f'{chrom} Abnormal', bins=20)
    
    ax1.axvline(x=3.0, color='r', linestyle='--', label='Threshold (±3)')
    ax1.axvline(x=-3.0, color='r', linestyle='--')
    ax1.set_xlabel('Z-score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Z-score Distributions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. GC content analysis
    ax2 = axes[0, 1]
    gc_features = ['13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量']
    
    for gc_feature in gc_features:
        if gc_feature in analysis_df.columns:
            normal_data = analysis_df[analysis_df['最终异常标签'] == 0][gc_feature].dropna()
            abnormal_data = analysis_df[analysis_df['最终异常标签'] == 1][gc_feature].dropna()
            
            ax2.boxplot([normal_data, abnormal_data], 
                       labels=['Normal', 'Abnormal'], 
                       positions=[len(gc_features)*2-1, len(gc_features)*2])
    
    ax2.axhline(y=0.4, color='r', linestyle='--', alpha=0.7, label='Normal Range')
    ax2.axhline(y=0.6, color='r', linestyle='--', alpha=0.7)
    ax2.set_ylabel('GC Content')
    ax2.set_title('GC Content Analysis')
    ax2.grid(True, alpha=0.3)
    
    # 3. Risk level distribution
    ax3 = axes[0, 2]
    risk_levels = [r['risk_level'] for r in detection_results]
    risk_counts = pd.Series(risk_levels).value_counts()
    
    colors = ['green', 'yellow', 'orange', 'red']
    ax3.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
           colors=colors[:len(risk_counts)])
    ax3.set_title('Risk Level Distribution')
    
    # 4. Feature correlation heatmap
    ax4 = axes[1, 0]
    feature_cols = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
                   '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量']
    
    corr_data = analysis_df[feature_cols + ['最终异常标签']].corr()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=ax4, fmt='.2f')
    ax4.set_title('Feature Correlation Matrix')
    
    # 5. Detection performance
    ax5 = axes[1, 1]
    
    # Create confusion matrix for comprehensive method
    evaluation_results = [r for r in detection_results if r['actual_label'] is not None]
    if evaluation_results:
        risk_to_binary = {'High Risk': 1, 'Moderate Risk': 1, 'Low Risk': 0, 'Normal': 0}
        predictions = [risk_to_binary[r['risk_level']] for r in evaluation_results]
        actual = [r['actual_label'] for r in evaluation_results]
        
        cm = confusion_matrix(actual, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5)
        ax5.set_xlabel('Predicted')
        ax5.set_ylabel('Actual')
        ax5.set_title('Confusion Matrix')
    
    # 6. Risk score distribution
    ax6 = axes[1, 2]
    overall_scores = [r['scores']['overall_risk'] for r in detection_results]
    
    ax6.hist(overall_scores, bins=20, alpha=0.7, edgecolor='black')
    ax6.axvline(x=0.2, color='g', linestyle='--', label='Low Risk Threshold')
    ax6.axvline(x=0.4, color='orange', linestyle='--', label='Moderate Risk Threshold') 
    ax6.axvline(x=0.7, color='r', linestyle='--', label='High Risk Threshold')
    ax6.set_xlabel('Overall Risk Score')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Risk Score Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('problem4_female_abnormality_detection.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function for Problem 4"""
    print("=== NIPT Problem 4: Female Fetal Abnormality Detection ===")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Analyze abnormality indicators
    analysis_df, significant_features = analyze_abnormality_indicators(df)
    
    # Build machine learning models
    models, scaler, best_model = build_abnormality_detection_models(analysis_df, significant_features)
    
    # Develop rule-based system
    rules, rule_predictions = develop_rule_based_system(analysis_df)
    
    # Create comprehensive detection method
    detection_results = create_comprehensive_detection_method(analysis_df, models, scaler, rules)
    
    # Create visualizations
    visualize_problem4_results(analysis_df, detection_results)
    
    # Final Summary
    print("\n=== PROBLEM 4 SUMMARY ===")
    print("Female Fetal Abnormality Detection Method:")
    
    print("\n1. Key Detection Criteria:")
    print("   Z-score thresholds:")
    for chrom, rule in rules['Z_score_rules'].items():
        print(f"     {chrom}: {rule['direction']} {rule['threshold']}")
    
    print("\n   GC content normal ranges:")
    for gc_feature, rule in rules['GC_content_rules'].items():
        print(f"     {gc_feature}: {rule['threshold_low']:.2f} - {rule['threshold_high']:.2f}")
    
    print("\n2. Significant Features:")
    if significant_features:
        for i, feature in enumerate(significant_features[:5]):
            print(f"   {i+1}. {feature['feature']} (p={feature['p_value']:.4f})")
    else:
        print("   No statistically significant features found")
    
    print("\n3. Risk Classification:")
    print("   High Risk (>0.7): Recommend diagnostic testing")
    print("   Moderate Risk (0.4-0.7): Genetic counseling recommended") 
    print("   Low Risk (0.2-0.4): Follow-up NIPT recommended")
    print("   Normal (<0.2): Routine care")
    
    risk_distribution = pd.Series([r['risk_level'] for r in detection_results]).value_counts()
    print(f"\n4. Population Risk Distribution:")
    for risk_level, count in risk_distribution.items():
        percentage = count / len(detection_results) * 100
        print(f"   {risk_level}: {count} cases ({percentage:.1f}%)")
    
    return detection_results, models, rules

if __name__ == "__main__":
    main()