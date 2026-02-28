#!/usr/bin/env python3
"""LightGBM 回测：与 Chronos-2 生产版同条件对比"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import time

DATA_DIR = '/home/ec2-user/sales_forecast_system/data'
OUT_DIR = '/home/ec2-user/sales_forecast_system/results'
ROLL_DAYS = 7

FEATURES = [
    'is_promo', 'discount_rate', 'ppc_fee',
    'day_of_week', 'is_weekend', 'month', 'qty_yoy',
    'sessions', 'ppc_clicks', 'ppc_ad_order_quantity', 'conversion_rate',
    # lag features
    'lag_1', 'lag_7', 'lag_14', 'lag_28',
    'roll_mean_7', 'roll_mean_14', 'roll_mean_28',
    'roll_std_7',
]


def add_time_features(df):
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['date'].dt.month
    return df


def add_yoy(train_df, target_dates):
    idx = train_df.set_index('date')['quantity']
    yoy = []
    for d in target_dates:
        last_year = d - pd.DateOffset(years=1)
        window = pd.date_range(last_year - pd.Timedelta(days=3), last_year + pd.Timedelta(days=3))
        yoy.append(np.mean([idx.get(w, 0) for w in window]))
    return yoy


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

    print(f"LightGBM 回测: {len(skus)} SKU, 滚动窗口={ROLL_DAYS}天\n")

    lgb_params = {
        'objective': 'regression', 'metric': 'mae', 'verbosity': -1,
        'n_estimators': 300, 'learning_rate': 0.05,
        'num_leaves': 31, 'min_child_samples': 10,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 0.1,
    }

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

        # 滚动预测
        all_preds = []
        current_train = train.copy()

        for start in range(0, len(test), ROLL_DAYS):
            end = min(start + ROLL_DAYS, len(test))
            batch = test.iloc[start:end].copy()

            # 合并历史+待预测，计算lag特征
            combined = pd.concat([current_train, batch], ignore_index=True)
            combined = add_lag_features(combined)

            train_part = combined.iloc[:len(current_train)]
            test_part = combined.iloc[len(current_train):]

            # 训练（dropna因为lag导致前几行缺失）
            tr = train_part.dropna(subset=FEATURES)
            if len(tr) < 30:
                all_preds.extend([0] * len(batch))
                current_train = pd.concat([current_train, batch], ignore_index=True)
                continue

            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(tr[FEATURES], tr['quantity'])

            # 逐天预测（多步时用预测值更新lag）
            temp = combined.copy()
            for i in range(len(current_train), len(combined)):
                row_feats = temp.iloc[i:i+1][FEATURES]
                if row_feats.isna().any(axis=1).iloc[0]:
                    row_feats = row_feats.fillna(0)
                pred = max(0, model.predict(row_feats)[0])
                all_preds.append(pred)
                # 用预测值更新后续lag
                temp.iloc[i, temp.columns.get_loc('quantity')] = pred
                # 重算后续行的lag（只影响未来行）
                for j in range(i + 1, len(combined)):
                    for lag_val in [1, 7, 14, 28]:
                        if j - lag_val >= 0:
                            temp.iloc[j, temp.columns.get_loc(f'lag_{lag_val}')] = temp.iloc[j - lag_val]['quantity']
                    for w in [7, 14, 28]:
                        s = max(0, j - w)
                        temp.iloc[j, temp.columns.get_loc(f'roll_mean_{w}')] = temp.iloc[s:j]['quantity'].mean()
                    s7 = max(0, j - 7)
                    temp.iloc[j, temp.columns.get_loc('roll_std_7')] = temp.iloc[s7:j]['quantity'].std()

            # 滚动更新：用实际值
            for i in range(len(batch)):
                row = batch.iloc[i:i+1].copy()
                row['quantity'] = test.iloc[start + i]['quantity']
                row['sessions'] = test.iloc[start + i]['sessions']
                row['ppc_clicks'] = test.iloc[start + i]['ppc_clicks']
                row['ppc_ad_order_quantity'] = test.iloc[start + i]['ppc_ad_order_quantity']
                row['conversion_rate'] = test.iloc[start + i]['conversion_rate']
                current_train = pd.concat([current_train, row], ignore_index=True)

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
    out_path = f'{OUT_DIR}/lgb_backtest.csv'
    df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"LightGBM 回测完成: {df['sku'].nunique()} SKU, {len(df)} 条记录")
    print(f"整体准确率: {df['accuracy'].mean():.1f}%")
    print(f">=70%占比: {(df['accuracy'] >= 70).mean()*100:.1f}%")
    print(f"<1%占比: {(df['accuracy'] < 1).mean()*100:.1f}%")
    print(f"总耗时: {time.time()-t_total:.0f}s")
    print(f"结果: {out_path}")


if __name__ == '__main__':
    main()
