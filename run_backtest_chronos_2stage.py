#!/usr/bin/env python3
"""两阶段Chronos-2回测：先预测sessions/cr，再用预测值辅助销量预测（5个SKU快速验证）"""
import pandas as pd
import numpy as np
import time
import sys

DATA_DIR = '/home/ec2-user/sales_forecast_system/data'
OUT_DIR = '/home/ec2-user/sales_forecast_system/results'
ROLL_DAYS = 7
FORECAST_DAYS = 60
TEST_SKU_COUNT = 0  # 0=全量

# Chronos-2 基线协变量
BASE_COVS = ['is_promo', 'discount_rate', 'ppc_fee', 'day_of_week', 'is_weekend', 'month', 'qty_yoy']


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
    """用Chronos-2预测单个指标（sessions或conversion_rate）"""
    pred_len = len(batch)
    ctx = pd.DataFrame({
        'id': sku,
        'timestamp': current_hist['date'].values,
        'target': current_hist[target_col].values.astype(float),
    })
    for c in covs:
        ctx[c] = current_hist[c].values.astype(float)

    fut = pd.DataFrame({
        'id': sku,
        'timestamp': batch['date'].values,
    })
    for c in covs:
        fut[c] = batch[c].values.astype(float)

    try:
        pred_df = pipeline.predict_df(
            ctx, future_df=fut, prediction_length=pred_len,
            quantile_levels=[0.5],
            id_column='id', timestamp_column='timestamp', target='target',
        )
        return np.maximum(pred_df['0.5'].values[:pred_len], 0)
    except Exception as e:
        print(f"  {target_col}预测失败: {e}", flush=True)
        return np.full(pred_len, current_hist[target_col].tail(7).mean())


def main():
    from chronos import Chronos2Pipeline

    train_all = pd.read_csv(f'{DATA_DIR}/daily_train.csv', parse_dates=['date'])
    test_all = pd.read_csv(f'{DATA_DIR}/daily_test.csv', parse_dates=['date'])
    skus = pd.read_csv(f'{DATA_DIR}/sku_list.csv')['sku'].tolist()
    if TEST_SKU_COUNT > 0:
        skus = skus[:TEST_SKU_COUNT]

    train_all = add_time_features(train_all)
    test_all = add_time_features(test_all)

    print("加载 Chronos-2...", flush=True)
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    print(f"模型加载完成，测试 {len(skus)} 个SKU\n", flush=True)

    # 预测指标用的协变量（不含qty_yoy，因为那是销量相关的）
    INDICATOR_COVS = ['is_promo', 'discount_rate', 'ppc_fee', 'day_of_week', 'is_weekend', 'month']
    # 销量预测用的协变量 = 基线 + 预测的sessions/cr
    SALES_COVS = BASE_COVS + ['sessions_pred', 'cr_pred']

    results = []
    indicator_acc = {'sessions': [], 'conversion_rate': []}
    t_total = time.time()

    for sku_idx, sku in enumerate(skus):
        t0 = time.time()
        print(f"[{sku_idx+1}/{len(skus)}] {sku}", flush=True)

        train = train_all[train_all['sku'] == sku].sort_values('date').copy()
        test = test_all[test_all['sku'] == sku].sort_values('date').copy()
        if len(train) == 0 or len(test) == 0:
            continue
        test = test.iloc[:FORECAST_DAYS].copy()

        train['qty_yoy'] = add_yoy(train, train['date'])
        test['qty_yoy'] = add_yoy(train, test['date'])

        # 历史部分 sessions_pred/cr_pred 用真实值
        train['sessions_pred'] = train['sessions'].astype(float)
        train['cr_pred'] = train['conversion_rate'].astype(float)

        all_preds = []
        current_train = train.copy()
        n_rolls = (len(test) + ROLL_DAYS - 1) // ROLL_DAYS

        for roll_idx, start in enumerate(range(0, len(test), ROLL_DAYS)):
            end = min(start + ROLL_DAYS, len(test))
            batch = test.iloc[start:end].copy()
            pred_len = len(batch)
            print(f"  窗口 {roll_idx+1}/{n_rolls} (day {start+1}-{end})", flush=True)

            # 阶段1：Chronos-2 预测 sessions 和 conversion_rate
            sess_preds = chronos_predict_indicator(pipeline, sku, current_train, batch, 'sessions', INDICATOR_COVS)
            cr_preds = chronos_predict_indicator(pipeline, sku, current_train, batch, 'conversion_rate', INDICATOR_COVS)

            for i in range(pred_len):
                indicator_acc['sessions'].append(calc_acc(sess_preds[i], batch.iloc[i]['sessions']))
                indicator_acc['conversion_rate'].append(calc_acc(cr_preds[i], batch.iloc[i]['conversion_rate']))

            batch['sessions_pred'] = sess_preds
            batch['cr_pred'] = cr_preds

            # 阶段2：Chronos-2 用预测的 sessions/cr 辅助销量预测
            ctx = pd.DataFrame({
                'id': sku,
                'timestamp': current_train['date'].values,
                'target': current_train['quantity'].values.astype(float),
            })
            for c in SALES_COVS:
                ctx[c] = current_train[c].values.astype(float)

            fut = pd.DataFrame({
                'id': sku,
                'timestamp': batch['date'].values,
            })
            for c in SALES_COVS:
                fut[c] = batch[c].values.astype(float)

            try:
                pred_df = pipeline.predict_df(
                    ctx, future_df=fut, prediction_length=pred_len,
                    quantile_levels=[0.5],
                    id_column='id', timestamp_column='timestamp', target='target',
                )
                preds = np.maximum(pred_df['0.5'].values[:pred_len], 0)
            except Exception as e:
                print(f"  销量预测失败: {e}", flush=True)
                preds = np.zeros(pred_len)

            all_preds.extend(preds)

            # 滚动更新：用预测值回填
            for i in range(pred_len):
                row = batch.iloc[i:i+1].copy()
                row['quantity'] = preds[i]
                row['sessions_pred'] = sess_preds[i]
                row['cr_pred'] = cr_preds[i]
                current_train = pd.concat([current_train, row], ignore_index=True)

        actuals = test['quantity'].values.astype(float)
        acc_list = [calc_acc(all_preds[i], actuals[i]) for i in range(len(test))]

        for i in range(len(test)):
            results.append({
                'sku': sku, 'date': test.iloc[i]['date'].strftime('%Y-%m-%d'),
                'actual': int(actuals[i]), 'predicted': round(float(all_preds[i]), 1),
                'accuracy': round(acc_list[i], 1),
            })

        print(f"  准确率={np.mean(acc_list):.1f}%  ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(f'{OUT_DIR}/chronos2_2stage_test.csv', index=False)

    print(f"\n{'='*60}", flush=True)
    print(f"指标预测准确率:", flush=True)
    print(f"  sessions:        {np.mean(indicator_acc['sessions']):.1f}%", flush=True)
    print(f"  conversion_rate: {np.mean(indicator_acc['conversion_rate']):.1f}%", flush=True)
    print(f"\n两阶段Chronos-2 销量预测 (+sessions +cr):", flush=True)
    print(f"  准确率={df['accuracy'].mean():.1f}%  >=70%占比={((df['accuracy']>=70).mean()*100):.1f}%", flush=True)
    print(f"  (基线参考: 66.6%)", flush=True)

    # 按SKU汇总
    print(f"\n按SKU:", flush=True)
    for sku in df['sku'].unique():
        s = df[df['sku'] == sku]
        print(f"  {sku}: {s['accuracy'].mean():.1f}%", flush=True)

    print(f"\n总耗时: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
