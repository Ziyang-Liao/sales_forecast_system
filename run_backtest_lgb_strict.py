#!/usr/bin/env python3
"""LightGBM 严格回测：一次性预测60天，不使用任何未来真实值

与 run_backtest_lgb.py 的区别：
  1. 不做7天滚动回填真实值，一次性预测全部60天
  2. lag特征用预测值自回归更新
  3. 未来不可知的协变量(sessions/ppc_clicks/ppc_ad_order_quantity/conversion_rate)
     用训练期最后7天均值填充，不用测试期真实值
  4. 未来可知的协变量(ppc_fee/discount_rate/is_promo/时间特征/qty_yoy)保留真实值
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import time

DATA_DIR = '/home/ec2-user/sales_forecast_system/data'
OUT_DIR = '/home/ec2-user/sales_forecast_system/results'

FUTURE_KNOWN = ['is_promo', 'discount_rate', 'ppc_fee', 'day_of_week', 'is_weekend', 'month', 'qty_yoy']
FUTURE_UNKNOWN = ['sessions', 'ppc_clicks', 'ppc_ad_order_quantity', 'conversion_rate']
LAG_FEATURES = ['lag_1', 'lag_7', 'lag_14', 'lag_28', 'roll_mean_7', 'roll_mean_14', 'roll_mean_28', 'roll_std_7']
FEATURES = FUTURE_KNOWN + FUTURE_UNKNOWN + LAG_FEATURES

LGB_PARAMS = {
    'objective': 'regression', 'metric': 'mae', 'verbosity': -1,
    'n_estimators': 300, 'learning_rate': 0.05,
    'num_leaves': 31, 'min_child_samples': 10,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 0.1,
}


def add_time_features(df):
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['date'].dt.month
    return df


def add_yoy(train_df, target_dates):
    idx = train_df.set_index('date')['quantity']
    return [np.mean([idx.get(w, 0) for w in pd.date_range(d - pd.DateOffset(years=1) - pd.Timedelta(days=3),
            d - pd.DateOffset(years=1) + pd.Timedelta(days=3))]) for d in target_dates]


def add_lag_features(df):
    df = df.copy()
    for lag in [1, 7, 14, 28]:
        df[f'lag_{lag}'] = df['quantity'].shift(lag)
    for w in [7, 14, 28]:
        df[f'roll_mean_{w}'] = df['quantity'].shift(1).rolling(w, min_periods=1).mean()
    df['roll_std_7'] = df['quantity'].shift(1).rolling(7, min_periods=1).std()
    return df


def calc_acc(pred, actual):
    if actual > 0:
        return max(0, 1 - abs(pred - actual) / actual) * 100
    return 100.0 if pred == 0 else 0.0


def main():
    train_all = pd.read_csv(f'{DATA_DIR}/daily_train.csv', parse_dates=['date'])
    test_all = pd.read_csv(f'{DATA_DIR}/daily_test.csv', parse_dates=['date'])
    skus = pd.read_csv(f'{DATA_DIR}/sku_list.csv')['sku'].tolist()

    train_all = add_time_features(train_all)
    test_all = add_time_features(test_all)

    print(f"LightGBM 严格回测: {len(skus)} SKU, 一次性预测60天, 无真实值回填\n")

    results = []
    t_total = time.time()

    for sku in skus:
        t0 = time.time()
        train = train_all[train_all['sku'] == sku].sort_values('date').copy()
        test = test_all[test_all['sku'] == sku].sort_values('date').copy()
        if len(train) == 0 or len(test) == 0:
            continue

        train['qty_yoy'] = add_yoy(train, train['date'])
        test['qty_yoy'] = add_yoy(train, test['date'])

        # 未来不可知协变量：用训练期最后7天均值填充
        for col in FUTURE_UNKNOWN:
            fill_val = train[col].tail(7).mean()
            test[col] = fill_val

        # 合并训练+测试，计算lag特征
        combined = pd.concat([train, test], ignore_index=True)
        combined = add_lag_features(combined)

        # 训练集部分
        train_part = combined.iloc[:len(train)].dropna(subset=FEATURES)
        if len(train_part) < 30:
            continue

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(train_part[FEATURES], train_part['quantity'])

        # 一次性预测60天：逐天自回归，lag用预测值更新
        temp = combined.copy()
        all_preds = []
        for i in range(len(train), len(combined)):
            row_feats = temp.iloc[i:i+1][FEATURES].fillna(0)
            pred = max(0, model.predict(row_feats)[0])
            all_preds.append(pred)

            # 用预测值（不是真实值）更新后续行的lag
            temp.iloc[i, temp.columns.get_loc('quantity')] = pred
            for j in range(i + 1, len(combined)):
                for lag_val in [1, 7, 14, 28]:
                    if j - lag_val >= 0:
                        temp.iloc[j, temp.columns.get_loc(f'lag_{lag_val}')] = temp.iloc[j - lag_val]['quantity']
                for w in [7, 14, 28]:
                    s = max(0, j - w)
                    temp.iloc[j, temp.columns.get_loc(f'roll_mean_{w}')] = temp.iloc[s:j]['quantity'].mean()
                s7 = max(0, j - 7)
                temp.iloc[j, temp.columns.get_loc('roll_std_7')] = temp.iloc[s7:j]['quantity'].std()

        actuals = test['quantity'].values.astype(float)
        acc_list = []
        for i in range(len(test)):
            acc = calc_acc(all_preds[i], actuals[i])
            acc_list.append(acc)
            results.append({
                'sku': sku, 'date': test.iloc[i]['date'].strftime('%Y-%m-%d'),
                'actual': int(actuals[i]), 'predicted': round(all_preds[i], 1),
                'accuracy': round(acc, 1),
            })

        print(f"{sku}: 准确率={np.mean(acc_list):.1f}% ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(results)
    out_path = f'{OUT_DIR}/lgb_strict_backtest.csv'
    df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"严格回测完成: {df['sku'].nunique()} SKU, {len(df)} 条记录")
    print(f"整体准确率: {df['accuracy'].mean():.1f}%")
    print(f">=70%占比: {(df['accuracy'] >= 70).mean()*100:.1f}%")
    print(f"<1%占比: {(df['accuracy'] < 1).mean()*100:.1f}%")
    print(f"总耗时: {time.time()-t_total:.0f}s")
    print(f"结果: {out_path}")


if __name__ == '__main__':
    main()
