import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
import time

# Sklearn & XGBoost
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KernelDensity
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

# Stats
from statsmodels.tsa.ar_model import AutoReg
from arch import arch_model
from scipy.stats import wasserstein_distance, kurtosis

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

# ================= Configuration =================
CSV_FILE_PATH = "./data/data4train.csv"
MODEL_DIR = 'models'
RESULT_FILE = 'model_metrics_comparison.csv'  # 保存对比结果的CSV

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# =========================================================================
# !!! 核心定义：11个特征的严格顺序，必须与 dispatch.py 完全一致 !!!
# =========================================================================
FEATURE_ORDER = [
    'Upstream_Lvl',  # 上游水位
    'Total_Gate_Opening',  # 总开度
    'Total_Load',  # 总负荷
    'Delta_Gate',  # 开度变化
    'Delta_Load',  # 负荷变化
    'Error_t-1',  # 误差 t-1
    'Error_t-2',  # 误差 t-2
    'Delta_Load_t-1',  # 负荷变化 t-1
    'Delta_Load_t-2',  # 负荷变化 t-2
    'Delta_Gate_t-1',  # 开度变化 t-1
    'Delta_Gate_t-2'  # 开度变化 t-2
]


# ================= 1. 数据处理 =================

def load_and_analyze_data(csv_path):
    print(f"Loading data from: {csv_path} ...")
    try:
        df = pd.read_csv(csv_path)
    except:
        df = pd.read_csv(csv_path, encoding='gbk')

    df.columns = df.columns.str.strip()

    # 映射字典：兼容中文或不规范的英文表头
    col_map = {
        '沙坪上游水位': 'Upstream_Lvl',
        '总开度': 'Total_Gate_Opening', 'Gate_Opening': 'Total_Gate_Opening',
        '总负荷': 'Total_Load',
        '开度变化': 'Delta_Gate',
        '负荷变化': 'Delta_Load',
        '误差_t-1': 'Error_t-1',
        '误差_t-2': 'Error_t-2',
        '负荷变化_t-1': 'Delta_Load_t-1',
        '负荷变化_t-2': 'Delta_Load_t-2',
        '开度变化_t-1': 'Delta_Gate_t-1',
        '开度变化_t-2': 'Delta_Gate_t-2',
        '本时刻误差': 'Target_Error',
        'Target_Error': 'Target_Error',
        'datetime': 'Datetime'
    }

    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
        df.sort_index(inplace=True)

    target_col = 'Target_Error'

    # 检查特征完整性
    missing_cols = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    df.dropna(inplace=True)

    # 单位处理
    if df[target_col].abs().mean() > 5.0:
        print("[Info] Converting large units to meters (/100)...")
        df[target_col] = df[target_col] / 100.0
        for col in ['Error_t-1', 'Error_t-2']:
            if col in df.columns: df[col] = df[col] / 100.0

    # 3-Sigma 清洗
    mean, std = df[target_col].mean(), df[target_col].std()
    df = df[(df[target_col] >= mean - 4 * std) & (df[target_col] <= mean + 4 * std)]

    X = df[FEATURE_ORDER]
    y = df[target_col]

    split = int(len(X) * 0.8)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


# ================= 2. 基础模型 =================
class BaseMLEngine:
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
        self.train_residuals_real = None

    def fit(self, X_train, y_train):
        print(f"  Training Base Model: [{self.model_type}]...")
        X_s = self.scaler_X.fit_transform(X_train[FEATURE_ORDER])
        y_s = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        if self.model_type == 'XGB':
            self.model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.03, n_jobs=-1, random_state=42)
            self.model.fit(X_s, y_s)

        elif self.model_type == 'MLP':
            self.model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='tanh', max_iter=500, random_state=42)
            self.model.fit(X_s, y_s)

        elif self.model_type == 'LSTM':
            X_lstm = X_s.reshape((X_s.shape[0], 1, X_s.shape[1]))
            model = Sequential([
                Input(shape=(1, len(FEATURE_ORDER))),
                LSTM(64, return_sequences=True),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            model.fit(X_lstm, y_s, epochs=40, batch_size=32, verbose=0, callbacks=[early_stop])
            self.model = model

        pred_train_real = self.predict(X_train)
        self.train_residuals_real = y_train.values - pred_train_real

    def predict(self, X):
        X_s = self.scaler_X.transform(X[FEATURE_ORDER])
        if self.model_type == 'LSTM':
            X_s = X_s.reshape((X_s.shape[0], 1, X_s.shape[1]))
            pred_s = self.model.predict(X_s, verbose=0).ravel()
        else:
            pred_s = self.model.predict(X_s)
        return self.scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()


# ================= 3. 统计生成模型 =================
class GenerativeModel:
    def __init__(self, base_engine, stat_type='KDE'):
        self.base_engine = base_engine
        self.stat_type = stat_type
        self.model_name = f"{base_engine.model_type} + {stat_type}"
        self.stat_model_def = None
        self.stat_model_res = None
        self.kde_model = None

    def fit_residuals(self, residuals_real):
        if self.stat_type == 'KDE':
            std_res = np.std(residuals_real)
            n = len(residuals_real)
            base_bw = 1.06 * std_res * (n ** (-1 / 5))
            # LSTM 微调
            bw = base_bw * 0.6 if self.base_engine.model_type == 'LSTM' else base_bw
            self.kde_model = KernelDensity(kernel='gaussian', bandwidth=bw)
            self.kde_model.fit(residuals_real.reshape(-1, 1))

        elif self.stat_type == 'AR':
            self.stat_model_res = AutoReg(residuals_real, lags=2).fit()

        elif self.stat_type == 'GARCH':
            try:
                scale = 100.0  # 缩放以避免数值问题
                self.stat_model_def = arch_model(residuals_real * scale, vol='Garch', p=1, q=1, dist='normal')
                self.stat_model_res = self.stat_model_def.fit(disp='off')
            except:
                self.stat_model_def = None

    def generate(self, X_test, n_scenarios=100):
        # 批量评估用 Generate
        base_pred = self.base_engine.predict(X_test)
        n_samples = len(base_pred)
        sim_residuals = np.zeros((n_samples, n_scenarios))

        if self.stat_type == 'KDE':
            sim_residuals = self.kde_model.sample(n_samples * n_scenarios).reshape(n_samples, n_scenarios)

        elif self.stat_type == 'AR':
            try:
                # 评估阶段使用静态预测
                start = len(self.stat_model_res.model.endog)
                mu = self.stat_model_res.predict(start=start, end=start + n_samples - 1)
                sigma = np.sqrt(self.stat_model_res.sigma2)
                sim_residuals = mu[:, None] + np.random.normal(0, sigma, (n_samples, n_scenarios))
            except:
                sim_residuals = np.random.normal(0, np.std(self.base_engine.train_residuals_real),
                                                 (n_samples, n_scenarios))

        elif self.stat_type == 'GARCH':
            if self.stat_model_res is not None:
                # 简化模拟
                std = np.std(self.base_engine.train_residuals_real)
                sim_residuals = np.random.normal(0, std, (n_samples, n_scenarios))
            else:
                sim_residuals = np.random.normal(0, np.std(self.base_engine.train_residuals_real),
                                                 (n_samples, n_scenarios))

        elif self.stat_type == 'Pure ML':
            std = np.std(self.base_engine.train_residuals_real)
            sim_residuals = np.random.normal(0, std, (n_samples, n_scenarios))

        return base_pred[:, None] + sim_residuals

    def save(self, directory):
        # 替换空格，统一文件名格式
        safe_name = f"{self.base_engine.model_type}_{self.stat_type.replace(' ', '')}"
        filename = os.path.join(directory, f"{safe_name}.pkl")

        if self.base_engine.model_type == 'LSTM':
            keras_model = self.base_engine.model
            self.base_engine.model = None
            keras_path = filename.replace('.pkl', '.keras')
            keras_model.save(keras_path)
            joblib.dump(self, filename)
            self.base_engine.model = keras_model
        else:
            joblib.dump(self, filename)


# ================= 4. 指标计算 =================
def calculate_metrics(y_true, ensemble):
    pred_mean = np.mean(ensemble, axis=1)
    rmse = np.sqrt(mean_squared_error(y_true, pred_mean))

    # CRPS (近似)
    subset = ensemble[:, :20]
    term1 = np.mean(np.abs(ensemble - y_true[:, None]), axis=1)
    diff_matrix = np.abs(subset[:, :, None] - subset[:, None, :])
    term2 = np.sum(diff_matrix, axis=(1, 2)) / (2 * 20 * 20)
    crps = np.mean(term1 - term2)

    # PICP & MPIW
    lower = np.percentile(ensemble, 2.5, axis=1)
    upper = np.percentile(ensemble, 97.5, axis=1)
    within = np.sum((y_true >= lower) & (y_true <= upper))
    picp = within / len(y_true)
    mpiw = np.mean(upper - lower)

    # Wasserstein Distance & Kurtosis Diff
    y_ens_sample = np.random.choice(ensemble.flatten(), size=len(y_true), replace=False)
    wd = wasserstein_distance(y_true, y_ens_sample)
    k_diff = abs(kurtosis(y_true) - kurtosis(y_ens_sample))

    return rmse, crps, picp, mpiw, wd, k_diff


# ================= 5. 主程序 =================
def main():
    X_train, X_test, y_train, y_test = load_and_analyze_data(CSV_FILE_PATH)

    ml_types = ['XGB', 'MLP', 'LSTM']
    stat_types = ['Pure ML', 'AR', 'GARCH', 'KDE']

    all_results = []

    print(f"\n{'Model':<20} | {'RMSE':<8} | {'CRPS':<8} | {'PICP':<8} | {'MPIW':<8} | {'W-Dist':<8} | {'Kurt-Diff':<9}")
    print("-" * 105)

    for ml in ml_types:
        # 1. 训练 Base Model
        engine = BaseMLEngine(ml)
        engine.fit(X_train, y_train)

        # 2. 组合不同的统计模型
        for stat in stat_types:
            model = GenerativeModel(engine, stat_type=stat)
            model.fit_residuals(engine.train_residuals_real)

            # 评估
            ensemble = model.generate(X_test, n_scenarios=100)
            rmse, crps, picp, mpiw, wd, k_diff = calculate_metrics(y_test.values, ensemble)

            name = f"{ml} + {stat}"
            print(f"{name:<20} | {rmse:.4f}   | {crps:.4f}   | {picp:.4f}   | {mpiw:.4f}   | {wd:.4f}   | {k_diff:.4f}")

            all_results.append({
                'Base': ml, 'Stat': stat, 'Model': name,
                'RMSE': rmse, 'CRPS': crps, 'PICP': picp,
                'MPIW': mpiw, 'W-Dist': wd, 'Kurt-Diff': k_diff
            })

            # 保存每一个模型
            model.save(MODEL_DIR)

    # 保存结果表
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(RESULT_FILE, index=False)
    print(f"\nResults saved to {RESULT_FILE}")
    print(f"All models saved to {MODEL_DIR}/")

    # 绘图
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    sns.barplot(data=res_df, x='Base', y='RMSE', hue='Stat', ax=axes[0], palette='Blues')
    axes[0].set_title('RMSE (Trend)')
    sns.barplot(data=res_df, x='Base', y='CRPS', hue='Stat', ax=axes[1], palette='Greens')
    axes[1].set_title('CRPS (Probabilistic)')
    sns.barplot(data=res_df, x='Base', y='W-Dist', hue='Stat', ax=axes[2], palette='Purples')
    axes[2].set_title('W-Dist (Distribution)')
    sns.barplot(data=res_df, x='Base', y='Kurt-Diff', hue='Stat', ax=axes[3], palette='Oranges')
    axes[3].set_title('Kurtosis Diff (Extreme)')
    plt.tight_layout()
    plt.savefig('final_metrics_comparison.png')
    plt.show()


if __name__ == "__main__":
    main()
