#!/usr/bin/env python3
"""实验：残差学习 + 分位数自适应选择"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import time

DATA_DIR = '/home/ec2-user/sales_forecast_system/data'
OUT_DIR = '/home/ec2-user/sales_forecast_system/results'
ROLL_DAYS = 7
FORECAST_DAYS = 60

BASE_COVS = ['is_promo', 'discount_rate', 'ppc_fee', 'day_of_week', 'is_weekend', 'month', 'qty_yoy']
INDICATOR_COVS = ['is_promo', 'discount_rate', 'ppc_fee', 'day_of_week', 'is_weekend', 'month']
SALES_COVS = BASE_COVS + ['sessions_pred', 'cr_pred']

# 残差模型特征
RESIDUAL_FEATURES = [
    'chronos_pred', 'is_promo', 'discount_rate', 'ppc_fee',
    'day_of_week', 'is_weekend', 'month', 'qty_yoy',
    'pred_lag1_ratio', 'pred_roll7_ratio',  # 预测值与近期实际的比值
    'sessions_pred', 'cr_pred',
]

LGB_PARAMS = {
    'objective': 'regression', 'metric': 'mae', 'verbosity': -1,
    'n_estimators': 200, 'learning_rate': 0.05,
    'num_leaves': 15, 'min_child_samples': 10,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.5, 'reg_lambda': 1.0,
}


def add_time_features(df):
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['date'].dt.month
    return df


def add_yoy(train_df, target_dates):
    idx = train_df.set_index('date')['quantity']
    return [np.mean([idx.get(d - pd.DateOffset(years=1) + pd.Timedelta(days=i), 0) for i in range(-3, 4)]) for d in target_dates]


def calc_acc(pred, actual):
    if actual > 0:
        return max(0, 1 - abs(pred - actual) / actual) * 100
    return 100.0 if pred == 0 else 0.0


def chronos_predict(pipeline, sku, current_train, batch, covs, quantiles=[0.5]):
    """Chronos-2预测，支持多分位数"""
    pred_len = len(batch)
    ctx = pd.DataFrame({'id': sku, 'timestamp': current_train['date'].values,
                         'target': current_train['quantity'].values.astype(float)})
    fut = pd.DataFrame({'id': sku, 'timestamp': batch['date'].values})
    for c in covs:
        ctx[c] = current_train[c].values.astype(float)
        fut[c] = batch[c].values.astype(float)
    try:
        pred_df = pipeline.predict_df(ctx, future_df=fut, prediction_length=pred_len,
                                       quantile_levels=quantiles,
                                       id_column='id', timestamp_column='timestamp', target='target')
        result = {}
        for q in quantiles:
            result[q] = np.maximum(pred_df[str(q)].values[:pred_len], 0)
        return result
    except:
        fallback = np.full(pred_len, current_train['quantity'].tail(7).mean())
        return {q: fallback for q in quantiles}


def chronos_predict_indicator(pipeline, sku, current_train, batch, target_col):
    pred_len = len(batch)
    ctx = pd.DataFrame({'id': sku, 'timestamp': current_train['date'].values,
                         'target': current_train[target_col].values.astype(float)})
    fut = pd.DataFrame({'id': sku, 'timestamp': batch['date'].values})
    for c in INDICATOR_COVS:
        ctx[c] = current_train[c].values.astype(float)
        fut[c] = batch[c].values.astype(float)
    try:
        pred_df = pipeline.predict_df(ctx, future_df=fut, prediction_length=pred_len,
                                       quantile_levels=[0.5],
                                       id_column='id', timestamp_column='timestamp', target='target')
        return np.maximum(pred_df['0.5'].values[:pred_len], 0)
    except:
        return np.full(pred_len, current_train[target_col].tail(7).mean())


def find_best_quantile(pipeline, sku, train, sales_covs):
    """用训练期最后30天找最优分位数（无数据泄露）"""
    calib_days = 30
    if len(train) <= calib_days + 60:
        return 0.5

    calib_test = train.iloc[-calib_days:].copy()
    calib_train = train.iloc[:-calib_days].copy()

    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    q_preds = {q: [] for q in quantiles}

    current = calib_train.copy()
    for start in range(0, len(calib_test), ROLL_DAYS):
        end = min(start + ROLL_DAYS, len(calib_test))
        batch = calib_test.iloc[start:end].copy()

        # 预测sessions/cr
        sess_p = chronos_predict_indicator(pipeline, sku, current, batch, 'sessions')
        cr_p = chronos_predict_indicator(pipeline, sku, current, batch, 'conversion_rate')
        batch['sessions_pred'] = sess_p
        batch['cr_pred'] = cr_p

        result = chronos_predict(pipeline, sku, current, batch, sales_covs, quantiles)
        for q in quantiles:
            q_preds[q].extend(result[q])

        # 滚动更新用p50
        for i in range(len(batch)):
            row = batch.iloc[i:i + 1].copy()
            row['quantity'] = result[0.5][i]
            row['sessions_pred'] = sess_p[i]
            row['cr_pred'] = cr_p[i]
            current = pd.concat([current, row], ignore_index=True)

    actuals = calib_test['quantity'].values.astype(float)
    best_q, best_acc = 0.5, 0
    for q in quantiles:
        acc = np.mean([calc_acc(q_preds[q][i], actuals[i]) for i in range(len(calib_test))])
        if acc > best_acc:
            best_acc = acc
            best_q = q
    return best_q


def main():
    from chronos import Chronos2Pipeline

    train_all = pd.read_csv(f'{DATA_DIR}/daily_train.csv', parse_dates=['date'])
    test_all = pd.read_csv(f'{DATA_DIR}/daily_test.csv', parse_dates=['date'])
    skus = pd.read_csv(f'{DATA_DIR}/sku_list.csv')['sku'].tolist()

    train_all = add_time_features(train_all)
    test_all = add_time_features(test_all)

    print("加载 Chronos-2...", flush=True)
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    print(f"实验: {len(skus)} SKU\n", flush=True)

    results = {name: [] for name in ['baseline', 'residual', 'quantile_adapt', 'residual+quantile']}
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
        train['sessions_pred'] = train['sessions'].astype(float)
        train['cr_pred'] = train['conversion_rate'].astype(float)

        # === 找最优分位数 ===
        best_q = find_best_quantile(pipeline, sku, train, SALES_COVS)

        # === 滚动预测 ===
        all_preds = {'baseline': [], 'residual': [], 'quantile_adapt': [], 'residual+quantile': []}
        current_train = train.copy()

        # 残差模型需要的历史：在训练期内做leave-one-out式的Chronos预测来收集残差
        # 简化：用滚动窗口内积累的残差来训练
        residual_data = []

        for start in range(0, len(test), ROLL_DAYS):
            end = min(start + ROLL_DAYS, len(test))
            batch = test.iloc[start:end].copy()
            pred_len = len(batch)

            # 阶段1：预测sessions/cr
            sess_p = chronos_predict_indicator(pipeline, sku, current_train, batch, 'sessions')
            cr_p = chronos_predict_indicator(pipeline, sku, current_train, batch, 'conversion_rate')
            batch['sessions_pred'] = sess_p
            batch['cr_pred'] = cr_p

            # 阶段2：Chronos-2 多分位数预测
            q_list = sorted(set([0.5, best_q]))
            preds_by_q = chronos_predict(pipeline, sku, current_train, batch, SALES_COVS, q_list)

            p50 = preds_by_q[0.5]
            p_best_q = preds_by_q.get(best_q, p50)

            # baseline = p50
            all_preds['baseline'].extend(p50)
            # quantile_adapt = 最优分位数
            all_preds['quantile_adapt'].extend(p_best_q)

            # === 残差修正 ===
            # 构造残差特征
            batch_feats = batch.copy()
            batch_feats['chronos_pred'] = p50
            recent_qty = current_train['quantity'].tail(7).mean()
            roll7_qty = current_train['quantity'].tail(7).mean()
            batch_feats['pred_lag1_ratio'] = p50 / max(current_train['quantity'].iloc[-1], 0.1)
            batch_feats['pred_roll7_ratio'] = p50 / max(roll7_qty, 0.1)

            if len(residual_data) >= 14:  # 至少积累2周残差数据才训练
                res_df = pd.DataFrame(residual_data)
                res_train = res_df.dropna(subset=RESIDUAL_FEATURES)
                if len(res_train) >= 7:
                    res_model = lgb.LGBMRegressor(**LGB_PARAMS)
                    res_model.fit(res_train[RESIDUAL_FEATURES], res_train['residual'])
                    # 预测残差
                    for i in range(pred_len):
                        row_feat = batch_feats.iloc[i:i + 1][RESIDUAL_FEATURES].fillna(0)
                        residual_pred = res_model.predict(row_feat)[0]
                        corrected = max(0, p50[i] + residual_pred)
                        all_preds['residual'].append(corrected)
                        # residual+quantile: 对最优分位数也做残差修正
                        corrected_q = max(0, p_best_q[i] + residual_pred)
                        all_preds['residual+quantile'].append(corrected_q)
                else:
                    all_preds['residual'].extend(p50)
                    all_preds['residual+quantile'].extend(p_best_q)
            else:
                all_preds['residual'].extend(p50)
                all_preds['residual+quantile'].extend(p_best_q)

            # 滚动更新 + 收集残差数据
            for i in range(pred_len):
                actual_qty = batch.iloc[i]['quantity']
                residual_data.append({
                    'residual': actual_qty - p50[i],
                    'chronos_pred': p50[i],
                    'is_promo': batch.iloc[i]['is_promo'],
                    'discount_rate': batch.iloc[i]['discount_rate'],
                    'ppc_fee': batch.iloc[i]['ppc_fee'],
                    'day_of_week': batch.iloc[i]['day_of_week'],
                    'is_weekend': batch.iloc[i]['is_weekend'],
                    'month': batch.iloc[i]['month'],
                    'qty_yoy': batch.iloc[i]['qty_yoy'],
                    'sessions_pred': sess_p[i],
                    'cr_pred': cr_p[i],
                    'pred_lag1_ratio': p50[i] / max(current_train['quantity'].iloc[-1], 0.1),
                    'pred_roll7_ratio': p50[i] / max(roll7_qty, 0.1),
                })

                row = batch.iloc[i:i + 1].copy()
                row['quantity'] = p50[i]  # 用baseline回填
                row['sessions_pred'] = sess_p[i]
                row['cr_pred'] = cr_p[i]
                current_train = pd.concat([current_train, row], ignore_index=True)

        actuals = test['quantity'].values.astype(float)
        line = f"[{sku_idx + 1}/{len(skus)}] {sku} (q={best_q}):"
        for name in all_preds:
            acc = np.mean([calc_acc(all_preds[name][i], actuals[i]) for i in range(len(test))])
            line += f"  {name[:8]}={acc:.1f}%"
            for i in range(len(test)):
                results[name].append({
                    'sku': sku, 'date': test.iloc[i]['date'].strftime('%Y-%m-%d'),
                    'actual': int(actuals[i]),
                    'predicted': round(float(all_preds[name][i]), 1),
                    'accuracy': round(calc_acc(all_preds[name][i], actuals[i]), 1),
                })
        print(f"{line}  ({time.time() - t0:.1f}s)", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    for name in results:
        df = pd.DataFrame(results[name])
        acc = df['accuracy'].mean()
        ge70 = (df['accuracy'] >= 70).mean() * 100
        print(f"  {name:20s}: 准确率={acc:.1f}%  >=70%={ge70:.1f}%", flush=True)
    print(f"\n总耗时: {time.time() - t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
