#!/usr/bin/env python3
"""
Chronos-2 回测 V2：时间特征 + 滚动预测
1. 时间特征：星期几、月份、是否月初月末、去年同期销量
2. 滚动预测：每7天更新context
"""
import pandas as pd
import numpy as np
import time

DATA_DIR = '/home/ec2-user/sales_forecast_system/data'
OUT_DIR = '/home/ec2-user/sales_forecast_system/results'
ROLL_DAYS = 7  # 滚动窗口


def add_time_features(df):
    """添加时间特征"""
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=周一
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
    df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
    return df


def add_yoy_feature(train_df, target_dates, sku):
    """计算去年同期销量"""
    yoy = []
    train_indexed = train_df.set_index('date')['quantity']
    for d in target_dates:
        last_year = d - pd.DateOffset(years=1)
        # 取去年同期前后3天均值
        window = pd.date_range(last_year - pd.Timedelta(days=3), last_year + pd.Timedelta(days=3))
        vals = [train_indexed.get(w, 0) for w in window]
        yoy.append(np.mean(vals))
    return yoy


def calc_acc(pred, actual):
    if actual > 0:
        return max(0, 1 - abs(pred - actual) / actual) * 100
    return 100.0 if pred == 0 else 0.0


def main():
    from chronos import Chronos2Pipeline

    train_all = pd.read_csv(f'{DATA_DIR}/daily_train.csv', parse_dates=['date'])
    test_all = pd.read_csv(f'{DATA_DIR}/daily_test.csv', parse_dates=['date'])
    skus = pd.read_csv(f'{DATA_DIR}/sku_list.csv')['sku'].tolist()

    # 添加时间特征
    train_all = add_time_features(train_all)
    test_all = add_time_features(test_all)

    print("加载 Chronos-2...")
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    print(f"V2回测: {len(skus)} SKU, 滚动窗口={ROLL_DAYS}天\n")

    results = []

    for sku in skus:
        t0 = time.time()
        train = train_all[train_all['sku'] == sku].sort_values('date').copy()
        test = test_all[test_all['sku'] == sku].sort_values('date').copy()

        if len(train) == 0 or len(test) == 0:
            continue

        # 添加去年同期特征
        train['qty_yoy'] = add_yoy_feature(train, train['date'], sku)
        test['qty_yoy'] = add_yoy_feature(train, test['date'], sku)

        # 滚动预测
        all_preds = []
        current_train = train.copy()

        for start in range(0, len(test), ROLL_DAYS):
            end = min(start + ROLL_DAYS, len(test))
            batch = test.iloc[start:end]
            pred_len = len(batch)

            # 构建context（历史全量特征）
            ctx = pd.DataFrame({
                'id': sku,
                'timestamp': current_train['date'].values,
                'target': current_train['quantity'].values.astype(float),
                'is_promo': current_train['is_promo'].values.astype(float),
                'discount_rate': current_train['discount_rate'].values.astype(float),
                'ppc_fee': current_train['ppc_fee'].values.astype(float),
                'day_of_week': current_train['day_of_week'].values.astype(float),
                'is_weekend': current_train['is_weekend'].values.astype(float),
                'month': current_train['month'].values.astype(float),
                'qty_yoy': current_train['qty_yoy'].values.astype(float),
            })

            # 构建future（只含可规划特征+时间特征）
            fut = pd.DataFrame({
                'id': sku,
                'timestamp': batch['date'].values,
                'is_promo': batch['is_promo'].values.astype(float),
                'discount_rate': batch['discount_rate'].values.astype(float),
                'ppc_fee': batch['ppc_fee'].values.astype(float),
                'day_of_week': batch['day_of_week'].values.astype(float),
                'is_weekend': batch['is_weekend'].values.astype(float),
                'month': batch['month'].values.astype(float),
                'qty_yoy': batch['qty_yoy'].values.astype(float),
            })

            try:
                pred_df = pipeline.predict_df(
                    ctx, future_df=fut, prediction_length=pred_len,
                    quantile_levels=[0.5],
                    id_column='id', timestamp_column='timestamp', target='target',
                )
                preds = np.maximum(pred_df['0.5'].values[:pred_len], 0)
            except Exception as e:
                print(f"{sku}: 预测失败 - {e}")
                preds = np.zeros(pred_len)

            all_preds.extend(preds)

            # 滚动更新：用预测值回填（不使用真实值）
            for i in range(pred_len):
                row = batch.iloc[i:i+1].copy()
                row['quantity'] = all_preds[start + i]
                current_train = pd.concat([current_train, row], ignore_index=True)

        # 记录结果
        actuals = test['quantity'].values.astype(float)
        acc_list = []
        for i in range(len(test)):
            acc = calc_acc(all_preds[i], actuals[i])
            acc_list.append(acc)
            results.append({
                'sku': sku,
                'date': test.iloc[i]['date'].strftime('%Y-%m-%d'),
                'actual': int(actuals[i]),
                'predicted': round(all_preds[i], 1),
                'accuracy': round(acc, 1),
                'is_promo': int(test.iloc[i]['is_promo']),
                'discount_rate': round(test.iloc[i]['discount_rate'], 4),
                'ppc_fee': round(test.iloc[i]['ppc_fee'], 2),
            })

        elapsed = time.time() - t0
        mean_acc = np.mean(acc_list)
        print(f"{sku}: 准确率={mean_acc:.1f}% ({elapsed:.1f}s)")

    # 保存
    df = pd.DataFrame(results)
    out_path = f'{OUT_DIR}/chronos2_backtest_v2.csv'
    df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"V2回测完成: {df['sku'].nunique()} SKU, {len(df)} 条记录")
    print(f"整体准确率: {df['accuracy'].mean():.1f}%")
    print(f">=70%占比: {(df['accuracy'] >= 70).mean()*100:.1f}%")
    print(f"<1%占比: {(df['accuracy'] < 1).mean()*100:.1f}%")
    print(f"结果: {out_path}")


if __name__ == '__main__':
    main()
