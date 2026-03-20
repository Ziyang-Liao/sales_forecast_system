#!/usr/bin/env python3
"""多模型对比实验：XGBoost增强版、随机森林、Prophet、MLP"""
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/home/ec2-user/sales_forecast_system/data'
ROLL_DAYS = 7
FORECAST_DAYS = 60
TEST_SKUS = 5  # 0=全量

def add_time_features(df):
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['date'].dt.month
    return df

def add_yoy(train_df, target_dates):
    idx = train_df.set_index('date')['quantity']
    return [np.mean([idx.get(d - pd.DateOffset(years=1) + pd.Timedelta(days=i), 0) for i in range(-3,4)]) for d in target_dates]

def calc_acc(pred, actual):
    if actual > 0:
        return max(0, 1 - abs(pred - actual) / actual) * 100
    return 100.0 if pred == 0 else 0.0

def build_features(df, target_col='quantity'):
    """增强版特征工程"""
    df = df.copy()
    # lag特征
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    # 滚动统计
    for w in [3, 7, 14, 28]:
        df[f'roll_mean_{w}'] = df[target_col].shift(1).rolling(w, min_periods=1).mean()
        df[f'roll_std_{w}'] = df[target_col].shift(1).rolling(w, min_periods=1).std()
        df[f'roll_max_{w}'] = df[target_col].shift(1).rolling(w, min_periods=1).max()
        df[f'roll_min_{w}'] = df[target_col].shift(1).rolling(w, min_periods=1).min()
    # 趋势特征
    df['trend_7_28'] = df['roll_mean_7'] / df['roll_mean_28'].clip(lower=0.1)
    df['trend_3_7'] = df['roll_mean_3'] / df['roll_mean_7'].clip(lower=0.1)
    # 广告费变化率
    df['ppc_fee_lag1'] = df['ppc_fee'].shift(1)
    df['ppc_fee_change'] = (df['ppc_fee'] - df['ppc_fee_lag1']) / df['ppc_fee_lag1'].clip(lower=0.1)
    df['ppc_fee_roll7'] = df['ppc_fee'].shift(1).rolling(7, min_periods=1).mean()
    # 折扣变化
    df['discount_lag1'] = df['discount_rate'].shift(1)
    df['discount_change'] = df['discount_rate'] - df['discount_lag1']
    return df

ENHANCED_FEATURES = [
    'is_promo', 'discount_rate', 'ppc_fee', 'day_of_week', 'is_weekend', 'month', 'qty_yoy',
    'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_21', 'lag_28',
    'roll_mean_3', 'roll_mean_7', 'roll_mean_14', 'roll_mean_28',
    'roll_std_3', 'roll_std_7', 'roll_std_14', 'roll_std_28',
    'roll_max_7', 'roll_min_7', 'roll_max_28', 'roll_min_28',
    'trend_7_28', 'trend_3_7',
    'ppc_fee_change', 'ppc_fee_roll7',
    'discount_change',
]

def update_lags_inplace(temp, idx, total_len):
    for j in range(idx + 1, total_len):
        for lv in [1, 2, 3, 7, 14, 21, 28]:
            if j - lv >= 0:
                temp.iloc[j, temp.columns.get_loc(f'lag_{lv}')] = temp.iloc[j - lv]['quantity']
        for w in [3, 7, 14, 28]:
            s = max(0, j - w)
            vals = temp.iloc[s:j]['quantity']
            temp.iloc[j, temp.columns.get_loc(f'roll_mean_{w}')] = vals.mean()
            temp.iloc[j, temp.columns.get_loc(f'roll_std_{w}')] = vals.std() if len(vals) > 1 else 0
            if w in [7, 28]:
                temp.iloc[j, temp.columns.get_loc(f'roll_max_{w}')] = vals.max()
                temp.iloc[j, temp.columns.get_loc(f'roll_min_{w}')] = vals.min()
        rm7 = temp.iloc[j].get('roll_mean_7', 1)
        rm28 = temp.iloc[j].get('roll_mean_28', 1)
        rm3 = temp.iloc[j].get('roll_mean_3', 1)
        temp.iloc[j, temp.columns.get_loc('trend_7_28')] = rm7 / max(rm28, 0.1)
        temp.iloc[j, temp.columns.get_loc('trend_3_7')] = rm3 / max(rm7, 0.1)

def run_tree_model(model_cls, model_params, train, test, current_train):
    """通用树模型回测"""
    all_preds = []
    ct = current_train.copy()
    for start in range(0, len(test), ROLL_DAYS):
        end = min(start + ROLL_DAYS, len(test))
        batch = test.iloc[start:end].copy()
        combined = pd.concat([ct, batch], ignore_index=True)
        combined = build_features(combined)
        train_part = combined.iloc[:len(ct)].dropna(subset=ENHANCED_FEATURES)
        if len(train_part) < 30:
            all_preds.extend([0] * len(batch))
            ct = pd.concat([ct, batch], ignore_index=True)
            continue
        model = model_cls(**model_params)
        model.fit(train_part[ENHANCED_FEATURES], train_part['quantity'])
        temp = combined.copy()
        for i in range(len(ct), len(combined)):
            row_feats = temp.iloc[i:i+1][ENHANCED_FEATURES].fillna(0)
            pred = max(0, model.predict(row_feats)[0])
            all_preds.append(pred)
            temp.iloc[i, temp.columns.get_loc('quantity')] = pred
            update_lags_inplace(temp, i, len(combined))
        for i in range(len(batch)):
            row = batch.iloc[i:i+1].copy()
            row['quantity'] = all_preds[start + i]
            ct = pd.concat([ct, row], ignore_index=True)
    return all_preds

def run_prophet(train, test):
    """Prophet回测"""
    from prophet import Prophet
    all_preds = []
    ct = train[['date', 'quantity', 'is_promo', 'discount_rate', 'ppc_fee']].copy()
    ct.columns = ['ds', 'y', 'is_promo', 'discount_rate', 'ppc_fee']
    for start in range(0, len(test), ROLL_DAYS):
        end = min(start + ROLL_DAYS, len(test))
        batch = test.iloc[start:end]
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.1)
        m.add_regressor('is_promo')
        m.add_regressor('discount_rate')
        m.add_regressor('ppc_fee')
        m.fit(ct)
        future = pd.DataFrame({
            'ds': batch['date'].values,
            'is_promo': batch['is_promo'].values,
            'discount_rate': batch['discount_rate'].values,
            'ppc_fee': batch['ppc_fee'].values,
        })
        forecast = m.predict(future)
        preds = np.maximum(forecast['yhat'].values, 0)
        all_preds.extend(preds)
        for i in range(len(batch)):
            new_row = pd.DataFrame({'ds': [batch.iloc[i]['date']], 'y': [preds[i]],
                                     'is_promo': [batch.iloc[i]['is_promo']],
                                     'discount_rate': [batch.iloc[i]['discount_rate']],
                                     'ppc_fee': [batch.iloc[i]['ppc_fee']]})
            ct = pd.concat([ct, new_row], ignore_index=True)
    return all_preds

def run_mlp(train, test, current_train):
    """简单MLP回测"""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    all_preds = []
    ct = current_train.copy()
    for start in range(0, len(test), ROLL_DAYS):
        end = min(start + ROLL_DAYS, len(test))
        batch = test.iloc[start:end].copy()
        combined = pd.concat([ct, batch], ignore_index=True)
        combined = build_features(combined)
        train_part = combined.iloc[:len(ct)].dropna(subset=ENHANCED_FEATURES)
        if len(train_part) < 30:
            all_preds.extend([0] * len(batch))
            ct = pd.concat([ct, batch], ignore_index=True)
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_part[ENHANCED_FEATURES])
        model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42)
        model.fit(X_train, train_part['quantity'])
        temp = combined.copy()
        for i in range(len(ct), len(combined)):
            row_feats = temp.iloc[i:i+1][ENHANCED_FEATURES].fillna(0)
            X_test = scaler.transform(row_feats)
            pred = max(0, model.predict(X_test)[0])
            all_preds.append(pred)
            temp.iloc[i, temp.columns.get_loc('quantity')] = pred
            update_lags_inplace(temp, i, len(combined))
        for i in range(len(batch)):
            row = batch.iloc[i:i+1].copy()
            row['quantity'] = all_preds[start + i]
            ct = pd.concat([ct, row], ignore_index=True)
    return all_preds

def main():
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor

    train_all = pd.read_csv(f'{DATA_DIR}/daily_train.csv', parse_dates=['date'])
    test_all = pd.read_csv(f'{DATA_DIR}/daily_test.csv', parse_dates=['date'])
    skus = pd.read_csv(f'{DATA_DIR}/sku_list.csv')['sku'].tolist()
    if TEST_SKUS > 0:
        skus = skus[:TEST_SKUS]

    train_all = add_time_features(train_all)
    test_all = add_time_features(test_all)

    models = {
        'XGBoost增强': ('tree', xgb.XGBRegressor, {
            'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'verbosity': 0, 'random_state': 42,
        }),
        '随机森林': ('tree', RandomForestRegressor, {
            'n_estimators': 300, 'max_depth': 10, 'min_samples_leaf': 5,
            'random_state': 42, 'n_jobs': -1,
        }),
        'Prophet': ('prophet', None, None),
        'MLP神经网络': ('mlp', None, None),
    }

    print(f"多模型对比: {len(skus)} SKU, {len(models)} 模型\n", flush=True)

    all_results = {name: [] for name in models}
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
        actuals = test['quantity'].values.astype(float)

        line = f"[{sku_idx+1}/{len(skus)}] {sku}:"
        for name, (mtype, cls, params) in models.items():
            try:
                if mtype == 'tree':
                    preds = run_tree_model(cls, params, train, test, train.copy())
                elif mtype == 'prophet':
                    preds = run_prophet(train, test)
                elif mtype == 'mlp':
                    preds = run_mlp(train, test, train.copy())
                acc_list = [calc_acc(preds[i], actuals[i]) for i in range(len(test))]
                mean_acc = np.mean(acc_list)
                all_results[name].extend(acc_list)
            except Exception as e:
                mean_acc = 0
                all_results[name].extend([0] * len(test))
            short = name[:6]
            line += f"  {short}={mean_acc:.1f}%"
        print(f"{line}  ({time.time()-t0:.1f}s)", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"(基线参考: 两阶段Chronos-2 = 69.2%)\n", flush=True)
    for name in models:
        accs = all_results[name]
        if accs:
            print(f"  {name:15s}: 准确率={np.mean(accs):.1f}%  >=70%={np.mean([1 if a>=70 else 0 for a in accs])*100:.1f}%", flush=True)
    print(f"\n总耗时: {time.time()-t_total:.0f}s", flush=True)

if __name__ == '__main__':
    main()
