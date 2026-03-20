#!/usr/bin/env python3
"""测试ARIMA、SVR、岭回归/Lasso + Stacking集成"""
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/home/ec2-user/sales_forecast_system/data'
ROLL_DAYS = 7
FORECAST_DAYS = 60
TEST_SKUS = 0

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

def build_features(df):
    df = df.copy()
    for lag in [1, 7, 14, 28]:
        df[f'lag_{lag}'] = df['quantity'].shift(lag)
    for w in [7, 14, 28]:
        df[f'roll_mean_{w}'] = df['quantity'].shift(1).rolling(w, min_periods=1).mean()
    df['roll_std_7'] = df['quantity'].shift(1).rolling(7, min_periods=1).std()
    return df

FEATURES = ['is_promo', 'discount_rate', 'ppc_fee', 'day_of_week', 'is_weekend', 'month', 'qty_yoy',
            'lag_1', 'lag_7', 'lag_14', 'lag_28', 'roll_mean_7', 'roll_mean_14', 'roll_mean_28', 'roll_std_7']

def run_arima(train, test):
    """ARIMA + 滚动预测"""
    from statsmodels.tsa.arima.model import ARIMA
    preds = []
    history = train['quantity'].values.tolist()
    for start in range(0, len(test), ROLL_DAYS):
        end = min(start + ROLL_DAYS, len(test))
        for i in range(start, end):
            try:
                model = ARIMA(history, order=(7, 1, 1))
                fit = model.fit()
                pred = max(0, fit.forecast(steps=1)[0])
            except:
                pred = np.mean(history[-7:]) if history else 0
            preds.append(pred)
            history.append(pred)
    return preds

def run_sklearn_model(model, train, test):
    """通用sklearn模型回测（SVR/Ridge/Lasso）"""
    from sklearn.preprocessing import StandardScaler
    preds = []
    ct = train.copy()
    for start in range(0, len(test), ROLL_DAYS):
        end = min(start + ROLL_DAYS, len(test))
        batch = test.iloc[start:end].copy()
        combined = pd.concat([ct, batch], ignore_index=True)
        combined = build_features(combined)
        train_part = combined.iloc[:len(ct)].dropna(subset=FEATURES)
        if len(train_part) < 30:
            preds.extend([0] * len(batch))
            ct = pd.concat([ct, batch], ignore_index=True)
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_part[FEATURES])
        model.fit(X_tr, train_part['quantity'])
        temp = combined.copy()
        for i in range(len(ct), len(combined)):
            X = scaler.transform(temp.iloc[i:i+1][FEATURES].fillna(0))
            pred = max(0, model.predict(X)[0])
            preds.append(pred)
            temp.iloc[i, temp.columns.get_loc('quantity')] = pred
            # 更新lag
            for j in range(i+1, len(combined)):
                for lv in [1, 7, 14, 28]:
                    if j - lv >= 0:
                        temp.iloc[j, temp.columns.get_loc(f'lag_{lv}')] = temp.iloc[j-lv]['quantity']
                for w in [7, 14, 28]:
                    s = max(0, j-w)
                    temp.iloc[j, temp.columns.get_loc(f'roll_mean_{w}')] = temp.iloc[s:j]['quantity'].mean()
                s7 = max(0, j-7)
                temp.iloc[j, temp.columns.get_loc('roll_std_7')] = temp.iloc[s7:j]['quantity'].std() if j-s7>1 else 0
        for i in range(len(batch)):
            row = batch.iloc[i:i+1].copy()
            row['quantity'] = preds[start+i]
            ct = pd.concat([ct, row], ignore_index=True)
    return preds

def main():
    from sklearn.svm import SVR
    from sklearn.linear_model import Ridge, Lasso

    train_all = pd.read_csv(f'{DATA_DIR}/daily_train.csv', parse_dates=['date'])
    test_all = pd.read_csv(f'{DATA_DIR}/daily_test.csv', parse_dates=['date'])
    skus = pd.read_csv(f'{DATA_DIR}/sku_list.csv')['sku'].tolist()
    if TEST_SKUS > 0:
        skus = skus[:TEST_SKUS]

    train_all = add_time_features(train_all)
    test_all = add_time_features(test_all)

    models = {
        'Lasso': Lasso(alpha=1.0),
    }

    print(f"补充模型测试: {len(skus)} SKU\n", flush=True)
    all_results = {name: [] for name in models}
    t_total = time.time()

    for sku_idx, sku in enumerate(skus):
        t0 = time.time()
        train = train_all[train_all['sku']==sku].sort_values('date').copy()
        test = test_all[test_all['sku']==sku].sort_values('date').copy()
        if len(train)==0 or len(test)==0:
            continue
        test = test.iloc[:FORECAST_DAYS].copy()
        train['qty_yoy'] = add_yoy(train, train['date'])
        test['qty_yoy'] = add_yoy(train, test['date'])
        actuals = test['quantity'].values.astype(float)

        line = f"[{sku_idx+1}/{len(skus)}] {sku}:"
        for name, model in models.items():
            try:
                if name.startswith('ARIMA'):
                    preds = run_arima(train, test)
                else:
                    from sklearn.base import clone
                    preds = run_sklearn_model(clone(model), train, test)
                acc_list = [calc_acc(preds[i], actuals[i]) for i in range(len(test))]
                mean_acc = np.mean(acc_list)
                all_results[name].extend(acc_list)
            except Exception as e:
                mean_acc = 0
                all_results[name].extend([0]*len(test))
            line += f"  {name[:6]}={mean_acc:.1f}%"
        print(f"{line}  ({time.time()-t0:.1f}s)", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"(基线: 两阶段Chronos-2 = 69.2%)\n", flush=True)
    for name in models:
        accs = all_results[name]
        if accs:
            print(f"  {name:15s}: 准确率={np.mean(accs):.1f}%  >=70%={np.mean([1 if a>=70 else 0 for a in accs])*100:.1f}%", flush=True)
    print(f"\n总耗时: {time.time()-t_total:.0f}s", flush=True)

if __name__ == '__main__':
    main()
