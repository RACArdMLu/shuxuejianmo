import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据预处理
def preprocess_data(df):
    # 清洗数据
    df_clean = df.copy()
    
    # 处理多次检测：对同一孕妇同一次采血的多次检测取平均值
    df_agg = df_clean.groupby(['孕妇代码', '检测抽血次数']).agg({
        '年龄': 'first',  # 年龄
        '身高': 'first',  # 身高
        '体重': 'mean',   # 体重（可能有变化）
        '检测孕周': 'first',  # 孕周
        '孕妇BMI': 'mean',   # BMI
        'Y染色体浓度': 'mean',   # Y染色体浓度
        'Y染色体的Z值': 'mean'    # Y染色体Z值
    }).reset_index()
    
    # 筛选有效数据：Y染色体浓度非空且大于0
    df_valid = df_agg[(df_agg['Y染色体浓度'] > 0) & (df_agg['Y染色体浓度'].notna())]
    
    # 提取孕周数值（假设格式为"XXw+Y"）
    def extract_weeks(week_str):
        try:
            if 'w' in str(week_str):
                week_part = str(week_str).split('w')[0]
                day_part = str(week_str).split('+')[1] if '+' in str(week_str) else '0'
                return float(week_part) + float(day_part)/7
            return float(week_str)
        except:
            return np.nan
    
    df_valid['weeks'] = df_valid['检测孕周'].apply(extract_weeks)
    df_valid = df_valid.dropna(subset=['weeks', '孕妇BMI', 'Y染色体浓度'])
    
    return df_valid

# 2. 多元回归分析
def advanced_regression_analysis(df):
    X_features = ['weeks', '孕妇BMI', '年龄', '身高', '体重']  # 孕周、BMI、年龄、身高、体重
    y_target = 'Y染色体浓度'  # Y染色体浓度
    
    # 准备数据
    X = df[X_features].values
    y = df[y_target].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 多元线性回归
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # 预测和评估
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    
    print(f"多元线性回归 R² = {r2:.4f}")
    print("特征重要性（系数）：")
    for i, feature in enumerate(X_features):
        print(f"{feature}: {model.coef_[i]:.6f}")
    
    return model, scaler, r2

# 3. 非线性模型
def nonlinear_models(df):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.model_selection import train_test_split
    
    X_features = ['weeks', '孕妇BMI', '年龄', '身高', '体重']
    X = df[X_features].values
    y = df['Y染色体浓度'].values
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 多项式特征
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # 多项式回归
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)
    r2_poly = r2_score(y_test, y_pred_poly)
    
    # 随机森林
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    
    print(f"多项式回归 R² (测试集) = {r2_poly:.4f}")
    print(f"随机森林 R² (测试集) = {r2_rf:.4f}")
    return poly_model, rf_model, r2_poly, r2_rf

# 3.5 尝试更多模型
def try_more_models(df):
    from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X_features = ['weeks', '孕妇BMI', '年龄', '身高', '体重']
    X = df[X_features].values
    y = df['Y染色体浓度'].values
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        '梯度提升': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
        'SVM回归': SVR(kernel='rbf'),
        'K近邻': KNeighborsRegressor(n_neighbors=5),
        '神经网络': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            if name in ['SVM回归', '神经网络']:
                # 使用标准化数据
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                # 使用原始数据
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            r2 = r2_score(y_test, y_pred)
            results[name] = r2
            print(f"{name} R² (测试集) = {r2:.4f}")
        except Exception as e:
            print(f"{name} 模型训练失败: {e}")
            results[name] = None
    
    return results

# 3.6 广义线性回归
def generalized_linear_models(df):
    import statsmodels.api as sm
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np
    
    X_features = ['weeks', '孕妇BMI', '年龄', '身高', '体重']
    X = df[X_features].values
    y = df['Y染色体浓度'].values
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 添加常数项
    X_train_const = sm.add_constant(X_train_scaled)
    X_test_const = sm.add_constant(X_test_scaled)
    
    results = {}
    
    try:
        # 1. Gamma分布的GLM（适用于正值连续数据）
        # 由于Y染色体浓度都是正值，Gamma分布可能更合适
        if np.all(y_train > 0):  # 检查是否所有值都为正
            gamma_model = sm.GLM(y_train, X_train_const, family=sm.families.Gamma(link=sm.families.links.log()))
            gamma_fitted = gamma_model.fit()
            y_pred_gamma = gamma_fitted.predict(X_test_const)
            r2_gamma = r2_score(y_test, y_pred_gamma)
            results['Gamma GLM'] = r2_gamma
            print(f"Gamma GLM R² (测试集) = {r2_gamma:.4f}")
        else:
            print("数据包含非正值，跳过Gamma GLM")
    except Exception as e:
        print(f"Gamma GLM 失败: {e}")
    
    try:
        # 2. 高斯分布的GLM（等同于普通线性回归，但可以用不同链接函数）
        gaussian_model = sm.GLM(y_train, X_train_const, family=sm.families.Gaussian())
        gaussian_fitted = gaussian_model.fit()
        y_pred_gaussian = gaussian_fitted.predict(X_test_const)
        r2_gaussian = r2_score(y_test, y_pred_gaussian)
        results['Gaussian GLM'] = r2_gaussian
        print(f"Gaussian GLM R² (测试集) = {r2_gaussian:.4f}")
    except Exception as e:
        print(f"Gaussian GLM 失败: {e}")
    
    try:
        # 3. 逆高斯分布的GLM
        if np.all(y_train > 0):
            inv_gaussian_model = sm.GLM(y_train, X_train_const, family=sm.families.InverseGaussian())
            inv_gaussian_fitted = inv_gaussian_model.fit()
            y_pred_inv_gaussian = inv_gaussian_fitted.predict(X_test_const)
            r2_inv_gaussian = r2_score(y_test, y_pred_inv_gaussian)
            results['InverseGaussian GLM'] = r2_inv_gaussian
            print(f"InverseGaussian GLM R² (测试集) = {r2_inv_gaussian:.4f}")
    except Exception as e:
        print(f"InverseGaussian GLM 失败: {e}")
    
    try:
        # 4. Tweedie分布的GLM（适用于有很多零值或正偏斜的数据）
        from sklearn.linear_model import TweedieRegressor
        tweedie_model = TweedieRegressor(power=1.5, alpha=0.1)  # power=1.5对应复合泊松-Gamma分布
        tweedie_model.fit(X_train_scaled, y_train)
        y_pred_tweedie = tweedie_model.predict(X_test_scaled)
        r2_tweedie = r2_score(y_test, y_pred_tweedie)
        results['Tweedie GLM'] = r2_tweedie
        print(f"Tweedie GLM R² (测试集) = {r2_tweedie:.4f}")
    except Exception as e:
        print(f"Tweedie GLM 失败: {e}")
    
    return results

# 4. 可视化
def visualize_results(df, model, scaler):
    # 设置中文字体显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    X_features = ['weeks', '孕妇BMI', '年龄', '身高', '体重']
    X = df[X_features].values
    y = df['Y染色体浓度'].values
    
    # 标准化
    X_scaled = scaler.transform(X)
    
    # 预测
    y_pred = model.predict(X_scaled)
    
    # 散点图和回归线
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y, y=y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('实际Y染色体浓度')
    plt.ylabel('预测Y染色体浓度')
    plt.title('实际值 vs 预测值')
    plt.show()

# 主函数
def main():
    # 加载数据
    df = load_and_preprocess_data()
    
    # 数据预处理
    df_clean = preprocess_data(df)
    
    print("="*50)
    print("1. 多元回归分析")
    # 多元回归分析
    model, scaler, r2 = advanced_regression_analysis(df_clean)
    
    print("="*50)
    print("2. 非线性模型")
    # 非线性模型
    poly_model, rf_model, r2_poly, r2_rf = nonlinear_models(df_clean)
    
    print("="*50)
    print("3. 尝试更多模型")
    # 尝试更多模型
    more_results = try_more_models(df_clean)
    
    print("="*50)
    print("4. 广义线性回归模型")
    # 广义线性回归
    glm_results = generalized_linear_models(df_clean)
    
    print("="*50)
    print("5. 所有模型结果总结")
    print(f"多元线性回归: {r2:.4f}")
    print(f"多项式回归: {r2_poly:.4f}")
    print(f"随机森林: {r2_rf:.4f}")
    for name, result in more_results.items():
        if result is not None:
            print(f"{name}: {result:.4f}")
    for name, result in glm_results.items():
        if result is not None:
            print(f"{name}: {result:.4f}")
    
    # 可视化结果
    # visualize_results(df_clean, model, scaler)
def load_and_preprocess_data():
    """Load and preprocess the NIPT data"""
    print("Loading NIPT data...")
    df = pd.read_csv('CUMCM2025Problems/C题/boy.csv', encoding='gbk')
    # df = pd.read_excel('附件.xlsx')
    
    # Convert gestational age to numeric (weeks)

    print("Data loaded and preprocessed.")
    return df

if __name__ == "__main__":
    main()