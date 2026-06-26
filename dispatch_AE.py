import pandas as pd
import numpy as np
import joblib
import os
import warnings
from tqdm import tqdm
import random

# === 必需的库导入 ===
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import load_model

# 引用物理和调度逻辑
from sphdrostation import HydroStationSimulator
from dispatch_strategy import DispatchController

warnings.filterwarnings('ignore')

# ================= Configuration =================
DEBUG_MODE = False

MODEL_PATH = 'models/LSTM_AR.pkl'
XGB_AR_MODEL_PATH = 'models/XGB_AR.pkl'  # 替换 ARMAX 为 XGB+AR
DATA_PATH = './data/data.csv'
OUTPUT_DIR = './results_case_study'
RANDOM_SEED = 42
STEPS_PER_DAY = 288  # 24h * 12 steps/h

# 将模拟天数从14天修改为28天
NUM_EPISODES = 1 if DEBUG_MODE else 28

# 物理-数据融合参数
STEADY_STATE_NOISE_STD = 0.02
DYNAMIC_NOISE_SCALE = 3.0
ACTION_THRESHOLD_GATE = 0.01
ACTION_THRESHOLD_LOAD = 0.5
ERROR_CLIP_MIN = -0.8
ERROR_CLIP_MAX = 0.8

CUSTOM_GAUSS_MEAN = 0.0148
CUSTOM_GAUSS_STD = 0.4946

Z_DEAD = 550.0
Z_NORMAL = 554.0
Z_TARGET_MIN = 553.5
Z_TARGET_MAX = 553.8

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

FEATURE_ORDER = [
    'Upstream_Lvl', 'Total_Gate_Opening', 'Total_Load',
    'Delta_Gate', 'Delta_Load',
    'Error_t-1', 'Error_t-2',
    'Delta_Load_t-1', 'Delta_Load_t-2',
    'Delta_Gate_t-1', 'Delta_Gate_t-2'
]


# ================= 解决 Joblib 加载报错：直接植入类定义 =================
class PureStatisticalGenerator:
    def __init__(self, stat_type):
        self.model_type = 'Baseline'
        self.stat_type = stat_type
        self.model = None
        self.kde = None
        self.std_res = 0

    def generate(self, X_test, n_scenarios=100):
        n_samples = len(X_test)
        if self.stat_type == 'KDE':
            return self.kde.sample(n_samples * n_scenarios).reshape(n_samples, n_scenarios)
        if self.stat_type in ['AR', 'ARMAX']:
            exog_test = X_test.values if self.stat_type == 'ARMAX' else None
            try:
                start_idx = len(self.model.data.endog)
                mu = self.model.predict(start=start_idx, end=start_idx + n_samples - 1, exog=exog_test)
                noise = np.random.normal(0, self.std_res, (n_samples, n_scenarios))
                return mu[:, None] + noise
            except:
                return np.random.normal(0, self.std_res, (n_samples, n_scenarios))
        return np.random.normal(0, self.std_res, (n_samples, n_scenarios))


class MLStatGenerator:
    def __init__(self, model_type, stat_type):
        self.model_type = model_type
        self.stat_type = stat_type
        self.ml_model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
        self.stat_model_res = None
        self.kde_model = None
        self.train_residuals = None

    def _ml_predict(self, X):
        X_s = self.scaler_X.transform(X[FEATURE_ORDER])
        if self.model_type == 'LSTM':
            X_s = X_s.reshape((X_s.shape[0], 1, X_s.shape[1]))
            pred_s = self.ml_model.predict(X_s, verbose=0).ravel()
        else:
            pred_s = self.ml_model.predict(X_s)
        return self.scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()


# ================= 核心：动态误差生成器 =================
class ErrorGenerator:
    def __init__(self, model_path, xgbar_path):
        print(f"Loading Proposed Model from {model_path}...")
        self.wrapper = joblib.load(model_path)

        # 恢复 Keras LSTM 模型
        if hasattr(self.wrapper, 'model_type') and self.wrapper.model_type == 'LSTM':
            keras_path = model_path.replace('.pkl', '.keras')
            if os.path.exists(keras_path):
                self.wrapper.ml_model = load_model(keras_path)

        print(f"Loading XGB+AR Baseline from {xgbar_path}...")
        try:
            self.xgbar_wrapper = joblib.load(xgbar_path)
        except Exception as e:
            print(f"警告: XGB+AR模型加载失败 ({e})，将回退到随机噪声。")
            self.xgbar_wrapper = None

        self.gauss_mean = CUSTOM_GAUSS_MEAN
        self.gauss_std = CUSTOM_GAUSS_STD

    def generate_proposed(self, feat_dict):
        d_gate = abs(feat_dict.get('Delta_Gate', 0))
        d_load = abs(feat_dict.get('Delta_Load', 0))
        is_dynamic = (d_gate > ACTION_THRESHOLD_GATE) or (d_load > ACTION_THRESHOLD_LOAD)

        if is_dynamic:
            decay_factor = 0.85
            noise_std = STEADY_STATE_NOISE_STD * DYNAMIC_NOISE_SCALE
            ar_strength = 0.4
        else:
            decay_factor = 0.3
            noise_std = STEADY_STATE_NOISE_STD
            ar_strength = 0.1

        df = pd.DataFrame([feat_dict])[FEATURE_ORDER]
        base_pred = self.wrapper._ml_predict(df)[0]

        ar_component = 0.0
        if self.wrapper.stat_type == 'AR':
            params = self.wrapper.stat_model_res.params
            const = params.get('const', 0.0) if hasattr(params, 'get') else params[0]
            phi1 = params.get('y.L1', 0.0) if hasattr(params, 'get') else params[1]
            phi2 = params.get('y.L2', 0.0) if hasattr(params, 'get') else params[2]

            e_t1 = feat_dict.get('Error_t-1', 0)
            e_t2 = feat_dict.get('Error_t-2', 0)
            ar_component = const + phi1 * e_t1 + phi2 * e_t2

        raw_error = base_pred + ar_component * ar_strength
        final_error = raw_error * decay_factor + np.random.normal(0, noise_std)

        return np.clip(final_error, ERROR_CLIP_MIN, ERROR_CLIP_MAX)

    def generate_gaussian(self):
        return np.random.normal(self.gauss_mean, self.gauss_std)

    def generate_xgbar(self, feat_dict):
        # 针对 XGB+AR 的生成逻辑（同Proposed融合机制）
        if self.xgbar_wrapper is None:
            return np.random.normal(0, self.gauss_std)
        try:
            d_gate = abs(feat_dict.get('Delta_Gate', 0))
            d_load = abs(feat_dict.get('Delta_Load', 0))
            is_dynamic = (d_gate > ACTION_THRESHOLD_GATE) or (d_load > ACTION_THRESHOLD_LOAD)

            if is_dynamic:
                decay_factor = 0.85
                noise_std = STEADY_STATE_NOISE_STD * DYNAMIC_NOISE_SCALE
                ar_strength = 0.4
            else:
                decay_factor = 0.3
                noise_std = STEADY_STATE_NOISE_STD
                ar_strength = 0.1

            df = pd.DataFrame([feat_dict])[FEATURE_ORDER]
            base_pred = self.xgbar_wrapper._ml_predict(df)[0]

            ar_component = 0.0
            if self.xgbar_wrapper.stat_type == 'AR':
                params = self.xgbar_wrapper.stat_model_res.params
                const = params.get('const', 0.0) if hasattr(params, 'get') else params[0]
                phi1 = params.get('y.L1', 0.0) if hasattr(params, 'get') else params[1]
                phi2 = params.get('y.L2', 0.0) if hasattr(params, 'get') else params[2]

                e_t1 = feat_dict.get('Error_t-1', 0)
                e_t2 = feat_dict.get('Error_t-2', 0)
                ar_component = const + phi1 * e_t1 + phi2 * e_t2

            raw_error = base_pred + ar_component * ar_strength
            final_error = raw_error * decay_factor + np.random.normal(0, noise_std)

            return np.clip(final_error, ERROR_CLIP_MIN, ERROR_CLIP_MAX)
        except Exception:
            return np.random.normal(0, self.gauss_std)


# ================= 评价指标计算模块 =================
def calculate_metrics(df_scene, total_inflow_vol, simulator):
    metrics = {}
    z_values = df_scene['Z'].values
    load_values = df_scene['Load'].values

    for var, data in zip(['Gate', 'Load'], [df_scene['Gate'].values, load_values]):
        diff = np.diff(data)
        metrics[f'I_freq_{var}'] = np.sum(np.abs(diff) > 1e-4)
        if len(diff) > 1:
            metrics[f'I_osc_{var}'] = np.sum((diff[1:] * diff[:-1]) < -1e-6)
        else:
            metrics[f'I_osc_{var}'] = 0

    unsafe_steps = np.sum((z_values < Z_DEAD) | (z_values > Z_NORMAL))
    metrics['Time_Overshoot_Steps'] = unsafe_steps

    optimal_steps = np.sum((z_values >= Z_TARGET_MIN) & (z_values <= Z_TARGET_MAX))
    metrics['Time_Optimal_Steps'] = optimal_steps

    z_start, z_end = z_values[0], z_values[-1]
    vol_start = simulator.get_vol_by_level(z_start) * 10000
    vol_end = simulator.get_vol_by_level(z_end) * 10000
    delta_storage = vol_end - vol_start

    total_water_used = total_inflow_vol - delta_storage
    total_power_kwh = np.sum(load_values) * (5 / 60) * 1000

    if total_power_kwh > 1e-3:
        metrics['WCR'] = total_water_used / total_power_kwh
    else:
        metrics['WCR'] = np.nan

    return metrics


# ================= 特征工程 & 辅助函数 =================
def construct_features(row, prev_aux_info, is_continuous):
    g_cols = ['沙坪二级1#闸开度', '沙坪二级2#闸开度', '沙坪二级3#闸开度', '沙坪二级4#闸开度', '沙坪二级5#闸开度']
    p_cols = ['沙坪二级1#机组', '沙坪二级2#机组', '沙坪二级3#机组', '沙坪二级4#机组', '沙坪二级5#机组',
              '沙坪二级6#机组']

    if is_continuous and prev_aux_info is not None:
        upstream = prev_aux_info['Current_Z_Proposed']
        total_gate = prev_aux_info['Decision_Gate']
        total_load = prev_aux_info['Decision_Load']
        delta_gate = total_gate - prev_aux_info['Prev_Gate']
        delta_load = total_load - prev_aux_info['Prev_Load']
        error_t1 = prev_aux_info['Generated_Error']
        error_t2 = prev_aux_info.get('Error_t1', 0)
        delta_gate_t1 = prev_aux_info.get('Delta_Gate', 0)
        delta_gate_t2 = prev_aux_info.get('Delta_Gate_t1', 0)
        delta_load_t1 = prev_aux_info.get('Delta_Load', 0)
        delta_load_t2 = prev_aux_info.get('Delta_Load_t1', 0)
    else:
        upstream = row['沙坪上游水位']
        total_gate = row[g_cols].fillna(0).sum()
        total_load = row[p_cols].fillna(0).sum()
        delta_gate = delta_load = error_t1 = error_t2 = 0
        delta_gate_t1 = delta_gate_t2 = delta_load_t1 = delta_load_t2 = 0

    features = {
        'Upstream_Lvl': upstream, 'Total_Gate_Opening': total_gate, 'Total_Load': total_load,
        'Delta_Gate': delta_gate, 'Delta_Load': delta_load,
        'Error_t-1': error_t1, 'Error_t-2': error_t2,
        'Delta_Load_t-1': delta_load_t1, 'Delta_Load_t-2': delta_load_t2,
        'Delta_Gate_t-1': delta_gate_t1, 'Delta_Gate_t-2': delta_gate_t2
    }
    new_aux_info = {
        'Prev_Gate': total_gate, 'Prev_Load': total_load,
        'Delta_Gate': delta_gate, 'Delta_Load': delta_load,
        'Delta_Gate_t1': delta_gate_t1, 'Delta_Load_t1': delta_load_t1,
        'Error_t1': error_t1, 'Generated_Error': 0,
        'Current_Z_Proposed': upstream,
        'Decision_Gate': total_gate, 'Decision_Load': total_load
    }
    return features, new_aux_info


def get_forecast_inflow(df, current_idx, steps=12):
    inflows = []
    for i in range(steps):
        target_idx = current_idx + i
        if target_idx < len(df):
            inflows.append(df.iloc[target_idx]['沙坪入库流量'])
        else:
            inflows.append(inflows[-1] if inflows else 0.0)
    return inflows


# ================= 主程序 =================
def run_simulation():
    print("1. Reading Data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding='gbk')

    df.columns = df.columns.str.strip()
    time_col = [c for c in df.columns if 'time' in c.lower() or '时间' in c][0]
    df.rename(columns={time_col: 'datetime'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.sort_values(by='datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("2. Identifying Valid 24h Segments...")
    valid_starts = []
    for i in range(len(df) - STEPS_PER_DAY):
        t_start = df.iloc[i]['datetime']
        t_end = df.iloc[i + STEPS_PER_DAY - 1]['datetime']
        if abs(((t_end - t_start).total_seconds() / 60) - (STEPS_PER_DAY - 1) * 5) <= 10:
            valid_starts.append(i)

    if len(valid_starts) < NUM_EPISODES:
        print(f"Error: Not enough segments. Found {len(valid_starts)}.")
        return

    np.random.seed(RANDOM_SEED)
    selected_starts = np.random.choice(valid_starts, NUM_EPISODES, replace=False)
    selected_starts.sort()

    print(f"3. Running Simulation for {NUM_EPISODES} Episodes (Debug Mode: {DEBUG_MODE})...")
    generator = ErrorGenerator(MODEL_PATH, XGB_AR_MODEL_PATH)
    sim = HydroStationSimulator()
    controller = DispatchController(sim)

    all_step_results = []
    episode_metrics = []

    for ep_id, start_idx in enumerate(selected_starts):
        print(f"\nProcessing Episode {ep_id + 1}/{NUM_EPISODES}...")
        df_ep = df.iloc[start_idx: start_idx + STEPS_PER_DAY].copy()
        row_0 = df_ep.iloc[0]
        init_z = row_0['沙坪上游水位']

        # 替换场景名为 XGB_AR
        states = {
            'Perfect': {'z': init_z, 'gate': 0, 'load': 0},
            'Gaussian': {'z': init_z, 'gate': 0, 'load': 0},
            'XGB_AR': {'z': init_z, 'gate': 0, 'load': 0},
            'Proposed': {'z': init_z, 'gate': 0, 'load': 0}
        }

        g_cols = ['沙坪二级1#闸开度', '沙坪二级2#闸开度', '沙坪二级3#闸开度', '沙坪二级4#闸开度', '沙坪二级5#闸开度']
        p_cols = ['沙坪二级1#机组', '沙坪二级2#机组', '沙坪二级3#机组', '沙坪二级4#机组', '沙坪二级5#机组',
                  '沙坪二级6#机组']

        init_gate = row_0[g_cols].fillna(0).sum()
        init_load = row_0[p_cols].fillna(0).sum()

        for k in states:
            states[k]['gate'] = init_gate
            states[k]['load'] = init_load

        prev_aux_info = None
        episode_data = []

        for t_step in tqdm(range(STEPS_PER_DAY), leave=False):
            abs_idx = start_idx + t_step
            row = df.iloc[abs_idx]
            current_time = row['datetime']
            is_continuous = (t_step > 0)

            feat_dict, aux_info = construct_features(row, prev_aux_info, is_continuous)

            if is_continuous:
                err_prop = generator.generate_proposed(feat_dict)
                err_gauss = generator.generate_gaussian()
                err_xgbar = generator.generate_xgbar(feat_dict)
            else:
                err_prop = err_gauss = err_xgbar = 0.0

            aux_info['Generated_Error'] = err_prop

            future_inflows = get_forecast_inflow(df, abs_idx, steps=12)
            z_down = row['沙坪下游水位']

            # 更新结果字典
            step_res = {
                'Episode_ID': ep_id + 1, 'Datetime': current_time, 'Inflow': row['沙坪入库流量'],
                'Err_Prop': err_prop, 'Err_Gauss': err_gauss, 'Err_XGB_AR': err_xgbar
            }

            scenarios = [
                ('Perfect', 0.0), ('Gaussian', err_gauss),
                ('XGB_AR', err_xgbar), ('Proposed', err_prop)
            ]

            for scene_name, injected_error in scenarios:
                curr_state = states[scene_name]

                plan = controller.find_optimal_plan(
                    curr_state['z'], z_down, future_inflows,
                    curr_state['gate'], curr_state['load'], dt=300
                )

                phy_res = sim.calculate_step(
                    curr_state['z'], z_down, future_inflows[0],
                    plan['units'], plan['gates'], dt=300
                )

                real_next_z = phy_res['cal_next_z_up'] + injected_error

                states[scene_name]['z'] = real_next_z
                states[scene_name]['gate'] = plan['gate_total']
                states[scene_name]['load'] = plan['load_total']

                step_res[f'{scene_name}_Z'] = real_next_z
                step_res[f'{scene_name}_Gate'] = plan['gate_total']
                step_res[f'{scene_name}_Load'] = plan['load_total']

                if scene_name == 'Proposed':
                    aux_info['Current_Z_Proposed'] = real_next_z
                    aux_info['Decision_Gate'] = plan['gate_total']
                    aux_info['Decision_Load'] = plan['load_total']

            prev_aux_info = aux_info
            episode_data.append(step_res)
            all_step_results.append(step_res)

        ep_df = pd.DataFrame(episode_data)
        total_inflow_vol = ep_df['Inflow'].sum() * 300.0

        for scene in ['Perfect', 'Gaussian', 'XGB_AR', 'Proposed']:
            scene_data = ep_df[[f'{scene}_Z', f'{scene}_Gate', f'{scene}_Load']].rename(
                columns={f'{scene}_Z': 'Z', f'{scene}_Gate': 'Gate', f'{scene}_Load': 'Load'}
            )
            met = calculate_metrics(scene_data, total_inflow_vol, sim)
            met['Episode_ID'] = ep_id + 1
            # 兼容您绘图脚本所需的名称 XGB+AR
            met['Scenario'] = 'XGB+AR' if scene == 'XGB_AR' else scene
            episode_metrics.append(met)

    print("4. Saving Results...")
    pd.DataFrame(all_step_results).to_csv(os.path.join(OUTPUT_DIR, 'sim_detailed.csv'), index=False)
    pd.DataFrame(episode_metrics).to_csv(os.path.join(OUTPUT_DIR, 'sim_metrics_summary.csv'), index=False)

    print(f"Done. Check {OUTPUT_DIR} for results.")


if __name__ == "__main__":
    run_simulation()