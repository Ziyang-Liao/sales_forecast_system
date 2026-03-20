#!/usr/bin/env python3
"""高级集成实验：Stacking / 动态选择 / 多分位数融合"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import time

DATA_DIR = '/home/ec2-user/sales_forecast_system/data'
ROLL_DAYS = 7
FORECAST_DAYS = 60

BASE_COVS = ['is_promo', 'discount_rate', 'ppc_fee', 'day_of_week', 'is_weekend', 'month', 'qty_yoy']
INDICATOR_COVS = ['is_promo', 'discount_rate', 'ppc_fee', 'day_of_week', 'is_weekend', 'month']
SALES_COVS = BASE_COVS + ['sessions_pred', 'cr_pred']

# LightGBM lag特征
LGB_FEATURES = ['is_promo', 'discount_rate', 'ppc_fee', 'day_of_week', 'is_weekend', 'month', 'qty_yoy',
                'lag_1', 'lag_7', 'lag_14', 'lag_28', 'roll_mean_7', 'roll_mean_14', 'roll_mean_28', 'roll_std_7']

QUANTILES = [0.3, 0.4, 0.5, 0.6, 0.7]


def add_time_features(df):
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['date'].dt.month
    return df

def add_yoy(tr, dates):
    idx = tr.set_index('date')['quantity']
    return [np.mean([idx.get(d - pd.DateOffset(years=1) + pd.Timedelta(days=i), 0) for i in range(-3, 4)]) for d in dates]

def add_lag_features(df):
    df = df.copy()
    for lag in [1, 7, 14, 28]:
        df[f'lag_{lag}'] = df['quantity'].shift(lag)
    for w in [7, 14, 28]:
        df[f'roll_mean_{w}'] = df['quantity'].shift(1).rolling(w, min_periods=1).mean()
    df['roll_std_7'] = df['quantity'].shift(1).rolling(7, min_periods=1).std()
    return df

def calc_acc(p, a):
    if a > 0: return max(0, 1 - abs(p - a) / a) * 100
    return 100.0 if p == 0 else 0.0

def chronos_predict_ind(pipe, sku, ct, batch, col):
    n = len(batch)
    ctx = pd.DataFrame({'id': sku, 'timestamp': ct['date'].values, 'target': ct[col].values.astype(float)})
    fut = pd.DataFrame({'id': sku, 'timestamp': batch['date'].values})
    for c in INDICATOR_COVS:
        ctx[c] = ct[c].values.astype(float); fut[c] = batch[c].values.astype(float)
    try:
        r = pipe.predict_df(ctx, future_df=fut, prediction_length=n, quantile_levels=[0.5],
                            id_column='id', timestamp_column='timestamp', target='target')
        return np.maximum(r['0.5'].values[:n], 0)
    except:
        return np.full(n, ct[col].tail(7).mean())

def chronos_predict_multi_q(pipe, sku, ct, batch, covs, quantiles):
    n = len(batch)
    ctx = pd.DataFrame({'id': sku, 'timestamp': ct['date'].values, 'target': ct['quantity'].values.astype(float)})
    fut = pd.DataFrame({'id': sku, 'timestamp': batch['date'].values})
    for c in covs:
        ctx[c] = ct[c].values.astype(float); fut[c] = batch[c].values.astype(float)
    try:
        r = pipe.predict_df(ctx, future_df=fut, prediction_length=n, quantile_levels=quantiles,
                            id_column='id', timestamp_column='timestamp', target='target')
        return {q: np.maximum(r[str(q)].values[:n], 0) for q in quantiles}
    except:
        fb = np.full(n, ct['quantity'].tail(7).mean())
        return {q: fb for q in quantiles}

def lgb_predict_batch(ct, batch):
    """LightGBM单批次预测"""
    combined = pd.concat([ct, batch], ignore_index=True)
    combined = add_lag_features(combined)
    tr = combined.iloc[:len(ct)].dropna(subset=LGB_FEATURES)
    if len(tr) < 30:
        return np.full(len(batch), ct['quantity'].tail(7).mean())
    model = lgb.LGBMRegressor(objective='regression', metric='mae', verbosity=-1,
                               n_estimators=300, learning_rate=0.05, num_leaves=31,
                               min_child_samples=10, subsample=0.8, colsample_bytree=0.8)
    model.fit(tr[LGB_FEATURES], tr['quantity'])
    temp = combined.copy()
    preds = []
    for i in range(len(ct), len(combined)):
        row_feats = temp.iloc[i:i+1][LGB_FEATURES].fillna(0)
        pred = max(0, model.predict(row_feats)[0])
        preds.append(pred)
        temp.iloc[i, temp.columns.get_loc('quantity')] = pred
        for j in range(i+1, len(combined)):
            for lv in [1, 7, 14, 28]:
                if j - lv >= 0:
                    temp.iloc[j, temp.columns.get_loc(f'lag_{lv}')] = temp.iloc[j-lv]['quantity']
            for w in [7, 14, 28]:
                s = max(0, j-w)
                temp.iloc[j, temp.columns.get_loc(f'roll_mean_{w}')] = temp.iloc[s:j]['quantity'].mean()
            s7 = max(0, j-7)
            temp.iloc[j, temp.columns.get_loc('roll_std_7')] = temp.iloc[s7:j]['quantity'].std() if j-s7>1 else 0
    return np.array(preds)


def main():
    from chronos import Chronos2Pipeline

    train_all = pd.read_csv(f'{DATA_DIR}/daily_train.csv', parse_dates=['date'])
    test_all = pd.read_csv(f'{DATA_DIR}/daily_test.csv', parse_dates=['date'])
    skus = pd.read_csv(f'{DATA_DIR}/sku_list.csv')['sku'].tolist()
    train_all = add_time_features(train_all)
    test_all = add_time_features(test_all)

    print("加载 Chronos-2...", flush=True)
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    print(f"高级集成实验: {len(skus)} SKU\n", flush=True)

    methods = ['baseline_p50', 'trimmed_mean', 'dynamic_select', 'stacking']
    all_results = {m: [] for m in methods}
    t_total = time.time()

    for si, sku in enumerate(skus):
        t0 = time.time()
        train = train_all[train_all['sku'] == sku].sort_values('date').copy()
        test = test_all[test_all['sku'] == sku].sort_values('date').copy()
        if len(train) == 0 or len(test) == 0: continue
        test = test.iloc[:FORECAST_DAYS].copy()
        train['qty_yoy'] = add_yoy(train, train['date'])
        test['qty_yoy'] = add_yoy(train, test['date'])
        train['sessions_pred'] = train['sessions'].astype(float)
        train['cr_pred'] = train['conversion_rate'].astype(float)

        preds = {m: [] for m in methods}
        ct = train.copy()
        ct_lgb = train.copy()  # LightGBM用的历史

        # 动态选择：记录每个方法上一轮的表现
        prev_accs = {q: 50.0 for q in QUANTILES}
        prev_accs['lgb'] = 50.0

        # Stacking：积累训练数据
        stack_data = []

        for start in range(0, len(test), ROLL_DAYS):
            end = min(start + ROLL_DAYS, len(test))
            batch = test.iloc[start:end].copy()
            n = len(batch)

            # 阶段1：预测sessions/cr
            sp = chronos_predict_ind(pipe, sku, ct, batch, 'sessions')
            cp = chronos_predict_ind(pipe, sku, ct, batch, 'conversion_rate')
            batch['sessions_pred'] = sp
            batch['cr_pred'] = cp

            # 阶段2：Chronos-2 多分位数
            q_preds = chronos_predict_multi_q(pipe, sku, ct, batch, SALES_COVS, QUANTILES)

            # LightGBM预测
            lgb_preds = lgb_predict_batch(ct_lgb, batch)

            # === 方法1: baseline p50 ===
            preds['baseline_p50'].extend(q_preds[0.5])

            # === 方法2: 截尾均值（去掉最高最低，取中间3个分位数均值）===
            trimmed = np.array([q_preds[q] for q in QUANTILES])  # shape: (5, n)
            trimmed_mean = np.mean(np.sort(trimmed, axis=0)[1:-1], axis=0)  # 去掉最高最低
            preds['trimmed_mean'].extend(trimmed_mean)

            # === 方法3: 动态选择（根据上一轮表现选最优）===
            best_source = max(prev_accs, key=prev_accs.get)
            if best_source == 'lgb':
                preds['dynamic_select'].extend(lgb_preds)
            else:
                preds['dynamic_select'].extend(q_preds[best_source])

            # === 方法4: Stacking（元模型）===
            if len(stack_data) >= 14:
                sdf = pd.DataFrame(stack_data)
                feat_cols = [f'q{int(q*100)}' for q in QUANTILES] + ['lgb']
                meta = lgb.LGBMRegressor(objective='regression', metric='mae', verbosity=-1,
                                          n_estimators=100, learning_rate=0.1, num_leaves=7,
                                          min_child_samples=3, reg_lambda=1.0)
                meta.fit(sdf[feat_cols], sdf['actual'])
                for i in range(n):
                    feat = {f'q{int(q*100)}': q_preds[q][i] for q in QUANTILES}
                    feat['lgb'] = lgb_preds[i]
                    pred = max(0, meta.predict(pd.DataFrame([feat]))[0])
                    preds['stacking'].append(pred)
            else:
                preds['stacking'].extend(q_preds[0.5])

            # 更新动态选择的表现记录
            actuals_batch = batch['quantity'].values.astype(float)
            for q in QUANTILES:
                prev_accs[q] = np.mean([calc_acc(q_preds[q][i], actuals_batch[i]) for i in range(n)])
            prev_accs['lgb'] = np.mean([calc_acc(lgb_preds[i], actuals_batch[i]) for i in range(n)])

            # 积累stacking训练数据
            for i in range(n):
                row = {f'q{int(q*100)}': q_preds[q][i] for q in QUANTILES}
                row['lgb'] = lgb_preds[i]
                row['actual'] = actuals_batch[i]
                stack_data.append(row)

            # 滚动更新
            for i in range(n):
                row = batch.iloc[i:i+1].copy()
                row['quantity'] = q_preds[0.5][i]
                row['sessions_pred'] = sp[i]
                row['cr_pred'] = cp[i]
                ct = pd.concat([ct, row], ignore_index=True)
                row_lgb = batch.iloc[i:i+1].copy()
                row_lgb['quantity'] = lgb_preds[i]
                ct_lgb = pd.concat([ct_lgb, row_lgb], ignore_index=True)

        actuals = test['quantity'].values.astype(float)
        line = f"[{si+1}/{len(skus)}] {sku}:"
        for m in methods:
            acc = np.mean([calc_acc(preds[m][i], actuals[i]) for i in range(len(test))])
            all_results[m].extend([calc_acc(preds[m][i], actuals[i]) for i in range(len(test))])
            line += f"  {m[:8]}={acc:.1f}%"
        print(f"{line}  ({time.time()-t0:.1f}s)", flush=True)

    print(f"\n{'='*60}", flush=True)
    for m in methods:
        acc = np.mean(all_results[m])
        ge70 = np.mean([1 if a >= 70 else 0 for a in all_results[m]]) * 100
        print(f"  {m:20s}: 准确率={acc:.1f}%  >=70%={ge70:.1f}%", flush=True)
    print(f"\n总耗时: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
