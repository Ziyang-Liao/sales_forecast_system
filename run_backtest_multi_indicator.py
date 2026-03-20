#!/usr/bin/env python3
"""测试多个中间指标的两阶段Chronos-2：逐步加入ppc_clicks/ppc_ad_order_quantity/ppc_impression"""
import pandas as pd
import numpy as np
import time
import sys

DATA_DIR = '/home/ec2-user/sales_forecast_system/data'
OUT_DIR = '/home/ec2-user/sales_forecast_system/results'
ROLL_DAYS = 7
FORECAST_DAYS = 60

BASE_COVS = ['is_promo', 'discount_rate', 'ppc_fee', 'day_of_week', 'is_weekend', 'month', 'qty_yoy']
INDICATOR_COVS = ['is_promo', 'discount_rate', 'ppc_fee', 'day_of_week', 'is_weekend', 'month']

# 实验组：逐步加入更多预测指标
EXPERIMENTS = {
    'E0_baseline_2stage': ['sessions', 'conversion_rate'],
    'E1_+clicks': ['sessions', 'conversion_rate', 'ppc_clicks'],
    'E2_+clicks+adqty': ['sessions', 'conversion_rate', 'ppc_clicks', 'ppc_ad_order_quantity'],
    'E3_+clicks+adqty+imp': ['sessions', 'conversion_rate', 'ppc_clicks', 'ppc_ad_order_quantity', 'ppc_impression'],
}


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


def calc_acc(pred, actual):
    if actual > 0:
        return max(0, 1 - abs(pred - actual) / actual) * 100
    return 100.0 if pred == 0 else 0.0


def chronos_predict_indicator(pipeline, sku, current_hist, batch, target_col, covs):
    pred_len = len(batch)
    ctx = pd.DataFrame({'id': sku, 'timestamp': current_hist['date'].values,
                         'target': current_hist[target_col].values.astype(float)})
    for c in covs:
        ctx[c] = current_hist[c].values.astype(float)
    fut = pd.DataFrame({'id': sku, 'timestamp': batch['date'].values})
    for c in covs:
        fut[c] = batch[c].values.astype(float)
    try:
        pred_df = pipeline.predict_df(ctx, future_df=fut, prediction_length=pred_len,
                                       quantile_levels=[0.5], id_column='id',
                                       timestamp_column='timestamp', target='target')
        return np.maximum(pred_df['0.5'].values[:pred_len], 0)
    except:
        return np.full(pred_len, current_hist[target_col].tail(7).mean())


def main():
    from chronos import Chronos2Pipeline

    train_all = pd.read_csv(f'{DATA_DIR}/daily_train.csv', parse_dates=['date'])
    test_all = pd.read_csv(f'{DATA_DIR}/daily_test.csv', parse_dates=['date'])
    skus = pd.read_csv(f'{DATA_DIR}/sku_list.csv')['sku'].tolist()

    train_all = add_time_features(train_all)
    test_all = add_time_features(test_all)

    print("加载 Chronos-2...", flush=True)
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    print(f"多指标实验: {len(skus)} SKU\n", flush=True)

    # 先跑一遍所有SKU，收集每个实验的预测指标和销量预测
    all_indicator_acc = {ind: [] for ind in ['sessions','conversion_rate','ppc_clicks','ppc_ad_order_quantity','ppc_impression']}
    exp_results = {name: [] for name in EXPERIMENTS}
    t_total = time.time()

    for sku_idx, sku in enumerate(skus):
        t0 = time.time()
        train = train_all[train_all['sku'] == sku].sort_values('date').copy()
        test = test_all[test_all['sku'] == sku].sort_values('date').copy()
        if len(train) == 0 or len(test) == 0:
            continue
        test = test.iloc[:FORECAST_DAYS].copy()
        train['qty_yoy'] = add_yoy(train, train['date'])
        test['qty_yoy'] = add_yoy(train, test['date'])

        # 所有可能的指标都初始化pred列（历史用真实值）
        all_indicators = ['sessions', 'conversion_rate', 'ppc_clicks', 'ppc_ad_order_quantity', 'ppc_impression']
        for ind in all_indicators:
            train[f'{ind}_pred'] = train[ind].astype(float)

        # 每个实验独立的current_train
        exp_trains = {name: train.copy() for name in EXPERIMENTS}
        exp_preds = {name: [] for name in EXPERIMENTS}

        n_rolls = (len(test) + ROLL_DAYS - 1) // ROLL_DAYS
        for roll_idx, start in enumerate(range(0, len(test), ROLL_DAYS)):
            end = min(start + ROLL_DAYS, len(test))
            batch = test.iloc[start:end].copy()
            pred_len = len(batch)

            # 预测所有指标（只需预测一次，各实验共享）
            ind_preds = {}
            for ind in all_indicators:
                # 用第一个实验的current_train来预测指标（它们的历史数据相同）
                ref_train = exp_trains['E0_baseline_2stage']
                preds = chronos_predict_indicator(pipeline, sku, ref_train, batch, ind, INDICATOR_COVS)
                ind_preds[ind] = preds
                for i in range(pred_len):
                    all_indicator_acc[ind].append(calc_acc(preds[i], batch.iloc[i][ind]))
                batch[f'{ind}_pred'] = preds

            # 各实验组：用不同的指标组合做销量预测
            for exp_name, indicators in EXPERIMENTS.items():
                cur_train = exp_trains[exp_name]
                pred_cols = [f'{ind}_pred' for ind in indicators]
                sales_covs = BASE_COVS + pred_cols

                ctx = pd.DataFrame({'id': sku, 'timestamp': cur_train['date'].values,
                                     'target': cur_train['quantity'].values.astype(float)})
                fut = pd.DataFrame({'id': sku, 'timestamp': batch['date'].values})
                for c in sales_covs:
                    ctx[c] = cur_train[c].values.astype(float)
                    fut[c] = batch[c].values.astype(float)

                try:
                    pred_df = pipeline.predict_df(ctx, future_df=fut, prediction_length=pred_len,
                                                   quantile_levels=[0.5], id_column='id',
                                                   timestamp_column='timestamp', target='target')
                    preds = np.maximum(pred_df['0.5'].values[:pred_len], 0)
                except Exception as e:
                    preds = np.zeros(pred_len)

                exp_preds[exp_name].extend(preds)

                # 滚动更新
                for i in range(pred_len):
                    row = batch.iloc[i:i+1].copy()
                    row['quantity'] = preds[i]
                    for ind in indicators:
                        row[f'{ind}_pred'] = ind_preds[ind][i]
                    exp_trains[exp_name] = pd.concat([cur_train, row], ignore_index=True)
                    cur_train = exp_trains[exp_name]

        actuals = test['quantity'].values.astype(float)
        line = f"[{sku_idx+1}/{len(skus)}] {sku}:"
        for exp_name in EXPERIMENTS:
            acc = np.mean([calc_acc(exp_preds[exp_name][i], actuals[i]) for i in range(len(test))])
            line += f"  {exp_name.split('_')[0]}={acc:.1f}%"
            for i in range(len(test)):
                exp_results[exp_name].append({
                    'sku': sku, 'date': test.iloc[i]['date'].strftime('%Y-%m-%d'),
                    'actual': int(actuals[i]), 'predicted': round(float(exp_preds[exp_name][i]), 1),
                    'accuracy': round(calc_acc(exp_preds[exp_name][i], actuals[i]), 1),
                })
        print(f"{line}  ({time.time()-t0:.1f}s)", flush=True)

    # 汇总
    print(f"\n{'='*60}", flush=True)
    print(f"指标预测准确率:", flush=True)
    for ind in all_indicators:
        if all_indicator_acc[ind]:
            print(f"  {ind:30s}: {np.mean(all_indicator_acc[ind]):.1f}%", flush=True)

    print(f"\n销量预测对比:", flush=True)
    for exp_name in EXPERIMENTS:
        df = pd.DataFrame(exp_results[exp_name])
        df.to_csv(f'{OUT_DIR}/chronos2_{exp_name}.csv', index=False)
        print(f"  {exp_name:30s}: 准确率={df['accuracy'].mean():.1f}%  >=70%={((df['accuracy']>=70).mean()*100):.1f}%", flush=True)

    print(f"\n总耗时: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
