#!/usr/bin/env python3
"""
Chronos-2 回测脚本
使用协变量（促销标记、折扣率、广告费）预测每日销量
输入: data/daily_train.csv, data/daily_test.csv, data/sku_list.csv
输出: results/chronos2_backtest_full.csv

用法:
    python3.11 run_backtest.py                    # 跑全部SKU
    python3.11 run_backtest.py --sku H0002 H0007  # 跑指定SKU
"""
import argparse
import pandas as pd
import numpy as np
import time
import os

DATA_DIR = '/home/ec2-user/sales_forecast_system/data'
OUT_DIR = '/home/ec2-user/sales_forecast_system/results'


def run_backtest(sku_filter=None):
    from chronos import Chronos2Pipeline

    train_all = pd.read_csv(f'{DATA_DIR}/daily_train.csv', parse_dates=['date'])
    test_all = pd.read_csv(f'{DATA_DIR}/daily_test.csv', parse_dates=['date'])
    sku_list = pd.read_csv(f'{DATA_DIR}/sku_list.csv')

    skus = sku_filter if sku_filter else sku_list['sku'].tolist()

    print("加载 Chronos-2...")
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    print(f"回测 {len(skus)} 个SKU\n")

    results = []

    for sku in skus:
        t0 = time.time()
        train = train_all[train_all['sku'] == sku].sort_values('date')
        test = test_all[test_all['sku'] == sku].sort_values('date')

        if len(train) == 0 or len(test) == 0:
            print(f"{sku}: 无数据，跳过")
            continue

        pred_len = len(test)

        # Chronos-2 格式
        # 历史数据：包含所有协变量（完整信息）
        context_df = pd.DataFrame({
            'id': sku,
            'timestamp': train['date'].values,
            'target': train['quantity'].values.astype(float),
            'is_promo': train['is_promo'].values.astype(float),
            'discount_rate': train['discount_rate'].values.astype(float),
            'ppc_fee': train['ppc_fee'].values.astype(float),
            'sessions': train['sessions'].values.astype(float),
            'ppc_clicks': train['ppc_clicks'].values.astype(float),
            'ppc_ad_order_quantity': train['ppc_ad_order_quantity'].values.astype(float),
            'conversion_rate': train['conversion_rate'].values.astype(float),
        })
        # 未来数据：只包含可提前规划的字段（促销、折扣、广告预算）
        future_df = pd.DataFrame({
            'id': sku,
            'timestamp': test['date'].values,
            'is_promo': test['is_promo'].values.astype(float),
            'discount_rate': test['discount_rate'].values.astype(float),
            'ppc_fee': test['ppc_fee'].values.astype(float),
        })

        try:
            pred_df = pipeline.predict_df(
                context_df, future_df=future_df,
                prediction_length=pred_len,
                quantile_levels=[0.5],
                id_column='id', timestamp_column='timestamp', target='target',
            )
        except Exception as e:
            print(f"{sku}: 预测失败 - {e}")
            continue

        actuals = test['quantity'].values.astype(float)
        preds = np.maximum(pred_df['0.5'].values[:pred_len], 0)

        # 逐日结果
        for i in range(pred_len):
            actual = actuals[i]
            pred = preds[i]
            if actual > 0:
                accuracy = max(0, 1 - abs(pred - actual) / actual) * 100
            else:
                accuracy = 100.0 if pred == 0 else 0.0

            results.append({
                'sku': sku,
                'date': test.iloc[i]['date'].strftime('%Y-%m-%d'),
                'actual': int(actual),
                'predicted': round(pred, 1),
                'accuracy': round(accuracy, 1),
                'is_promo': int(test.iloc[i]['is_promo']),
                'discount_rate': round(test.iloc[i]['discount_rate'], 4),
                'ppc_fee': round(test.iloc[i]['ppc_fee'], 2),
                'sessions': round(test.iloc[i]['sessions'], 1),
                'ppc_clicks': round(test.iloc[i]['ppc_clicks'], 1),
                'ppc_ad_order_quantity': round(test.iloc[i]['ppc_ad_order_quantity'], 1),
                'conversion_rate': round(test.iloc[i]['conversion_rate'], 4),
            })

        elapsed = time.time() - t0
        mask = actuals > 0
        acc_arr = np.maximum(0, 1 - np.abs(preds[mask] - actuals[mask]) / actuals[mask]) * 100
        mean_acc = acc_arr.mean() if len(acc_arr) > 0 else 0
        print(f"{sku}: 准确率={mean_acc:.1f}% ({elapsed:.1f}s)")

    # 保存
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f'{OUT_DIR}/chronos2_backtest_full.csv'
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)

    # 汇总
    print(f"\n{'='*60}")
    print(f"回测完成: {df['sku'].nunique()} SKU, {len(df)} 条记录")
    print(f"整体准确率: {df['accuracy'].mean():.1f}%")
    print(f">=70%占比: {(df['accuracy'] >= 70).mean()*100:.1f}%")
    print(f"<1%占比: {(df['accuracy'] < 1).mean()*100:.1f}%")
    print(f"结果: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sku', nargs='+', help='指定SKU，不指定则跑全部')
    args = parser.parse_args()
    run_backtest(args.sku)
