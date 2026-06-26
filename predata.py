import pandas as pd


def process_hydropower_data_v2(file_path, output_path):
    print("正在读取数据...")
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return

    # 1. 预处理：转换时间格式并排序
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # 2. 构造历史特征 (Lag Features)
    # 需求：误差、负荷变化、开度变化 都要取 t-1 和 t-2

    # --- t-1 时刻 ---
    df['误差_t-1'] = df['本时刻误差'].shift(1)
    df['负荷变化_t-1'] = df['负荷变化'].shift(1)
    df['开度变化_t-1'] = df['开度变化'].shift(1)

    # --- t-2 时刻 ---
    df['误差_t-2'] = df['本时刻误差'].shift(2)
    df['负荷变化_t-2'] = df['负荷变化'].shift(2)
    df['开度变化_t-2'] = df['开度变化'].shift(2)

    # 3. 处理时间连续性 (剔除中断点)
    # 计算时间差
    df['time_diff'] = df['datetime'].diff()

    # 自动获取最常见的时间间隔 (如1小时)
    if len(df) > 2:
        freq = df['time_diff'].mode()[0]
        print(f"检测到数据标准时间间隔为: {freq}")
    else:
        print("数据量太少，无法计算。")
        return

    # 筛选条件：
    # 为了保证 t-1 和 t-2 都有意义，必须保证：
    # 1. t 与 t-1 连续 (当前行时间差正常)
    # 2. t-1 与 t-2 连续 (上一行时间差正常)
    condition_continuous_1 = (df['time_diff'] == freq)
    condition_continuous_2 = (df['time_diff'].shift(1) == freq)

    # 提取有效行
    valid_rows = condition_continuous_1 & condition_continuous_2
    df_final = df[valid_rows].copy()

    # 4. 整理最终列顺序 (共13列)
    columns_to_keep = [
        # --- 本时刻数据 (7列) ---
        'datetime',
        '本时刻误差',
        '沙坪上游水位',
        '总开度',
        '总负荷',
        '开度变化',
        '负荷变化',

        # --- 历史数据 (6列) ---
        '误差_t-1',
        '误差_t-2',

        '负荷变化_t-1',
        '负荷变化_t-2',

        '开度变化_t-1',
        '开度变化_t-2'
    ]

    # 校验列名是否存在
    missing = [c for c in columns_to_keep if c not in df_final.columns]
    if missing:
        print(f"警告：处理后的数据缺少以下列 (可能是原Excel表头不匹配): {missing}")
        return

    df_final = df_final[columns_to_keep]

    # 5. 保存
    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("-" * 30)
    print(f"处理完成！")
    print(f"原始行数: {len(df)}")
    print(f"有效行数: {len(df_final)}")
    print(f"生成的列数: {len(df_final.columns)} (包含datetime)")
    print(f"文件保存至: {output_path}")


# --- 运行配置 ---
input_file = './data/vlide.xlsx'  # 输入文件名
output_file = './data/data4train.csv'  # 输出文件名

if __name__ == '__main__':
    process_hydropower_data_v2(input_file, output_file)