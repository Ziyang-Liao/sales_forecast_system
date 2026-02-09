#!/usr/bin/env python3
"""
V3回测：分层策略 + 集成模型 + 残差校正
1. 按CV分层：稳定SKU用Chronos-2，高波动SKU用集成
2. 集成：Chronos-2 + LightGBM
3. 残差校正：修正系统性偏差
"""
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/home/ec2-user/sales_forecast_system/data'
OUT_DIR = '/home/ec2-user/sales_forecast_system/results'
ROLL_DAYS = 7

# CV阈值
CV_LOW = 1.0      # CV<1 稳定SKU
CV_HIGH = 1.5     # CV>1.5 高波动SKU


def add_features(df):
    """添加时间特征"""
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['date'].dt.month
    return df


def calc_acc(pred, actual):
    if actual > 0:
        return max(0, 1 - abs(pred - actual) / actual) * 100
    return 100.0 if pred == 0 else 0.0


def train_lgb_model(train_df):
    """训练LightGBM模型预测销量"""
    from lightgbm import LGBMRegressor
    
    features = ['day_of_week', 'is_weekend', 'month', 'discount_rate', 'ppc_fee',
                'sessions', 'ppc_clicks', 'ppc_ad_order_quantity', 'conversion_rate', 'is_promo']
    
    # 添加滞后特征
    df = train_df.copy()
    for lag in [1, 7, 14]:
        df[f'qty_lag{lag}'] = df['quantity'].shift(lag)
    df['qty_roll7_mean'] = df['quantity'].rolling(7).mean()
    df['qty_roll7_std'] = df['quantity'].rolling(7).std()
    
    lag_features = ['qty_lag1', 'qty_lag7', 'qty_lag14', 'qty_roll7_mean', 'qty_roll7_std']
    all_features = features + lag_features
    
    df = df.dropna()
    if len(df) < 30:
        return None, all_features
    
    X = df[all_features]
    y = df['quantity']
    
    model = LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, verbose=-1)
    model.fit(X, y)
    return model, all_features


def predict_lgb(model, features, current_train, future_row):
    """LightGBM预测"""
    if model is None:
        return None
    
    row = future_row.copy()
    # 计算滞后特征
    recent = current_train['quantity'].values
    row['qty_lag1'] = recent[-1] if len(recent) >= 1 else 0
    row['qty_lag7'] = recent[-7] if len(recent) >= 7 else 0
    row['qty_lag14'] = recent[-14] if len(recent) >= 14 else 0
    row['qty_roll7_mean'] = np.mean(recent[-7:]) if len(recent) >= 7 else np.mean(recent)
    row['qty_roll7_std'] = np.std(recent[-7:]) if len(recent) >= 7 else np.std(recent)
    
    X = pd.DataFrame([row])[features]
    return max(0, model.predict(X)[0])


def estimate_bias(preds, actuals):
    """估算预测偏差"""
    mask = actuals > 0
    if mask.sum() < 5:
        return 0
    return np.median((preds[mask] - actuals[mask]) / actuals[mask])


def main():
    from chronos import Chronos2Pipeline
    
    train_all = pd.read_csv(f'{DATA_DIR}/daily_train.csv', parse_dates=['date'])
    test_all = pd.read_csv(f'{DATA_DIR}/daily_test.csv', parse_dates=['date'])
    sku_list = pd.read_csv(f'{DATA_DIR}/sku_list.csv')
    
    train_all = add_features(train_all)
    test_all = add_features(test_all)
    
    print("加载 Chronos-2...")
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    print(f"V3回测: 分层策略 + 集成模型\n")
    
    results = []
    sku_summary = []
    
    for sku in sku_list['sku'].tolist():
        t0 = time.time()
        train = train_all[train_all['sku'] == sku].sort_values('date').copy()
        test = test_all[test_all['sku'] == sku].sort_values('date').copy()
        
        if len(train) == 0 or len(test) == 0:
            continue
        
        # 计算CV
        pos = train['quantity'].values[train['quantity'].values > 0]
        cv = pos.std() / (pos.mean() + 1e-6) if len(pos) > 10 else 0
        
        # 分层策略
        if cv < CV_LOW:
            strategy = 'chronos'
        elif cv > CV_HIGH:
            strategy = 'ensemble'
        else:
            strategy = 'chronos+bias'
        
        # 训练LightGBM（高波动SKU）
        lgb_model, lgb_features = None, []
        if strategy == 'ensemble':
            lgb_model, lgb_features = train_lgb_model(train)
        
        # 滚动预测
        all_preds = []
        all_chronos = []
        current_train = train.copy()
        
        for start in range(0, len(test), ROLL_DAYS):
            end = min(start + ROLL_DAYS, len(test))
            batch = test.iloc[start:end]
            pred_len = len(batch)
            
            # Chronos-2 预测
            ctx = pd.DataFrame({
                'id': sku,
                'timestamp': current_train['date'].values,
                'target': current_train['quantity'].values.astype(float),
                'is_promo': current_train['is_promo'].values.astype(float),
                'discount_rate': current_train['discount_rate'].values.astype(float),
                'ppc_fee': current_train['ppc_fee'].values.astype(float),
                'sessions': current_train['sessions'].values.astype(float),
                'ppc_clicks': current_train['ppc_clicks'].values.astype(float),
                'ppc_ad_order_quantity': current_train['ppc_ad_order_quantity'].values.astype(float),
                'conversion_rate': current_train['conversion_rate'].values.astype(float),
                'day_of_week': current_train['day_of_week'].values.astype(float),
                'is_weekend': current_train['is_weekend'].values.astype(float),
                'month': current_train['month'].values.astype(float),
            })
            
            fut = pd.DataFrame({
                'id': sku,
                'timestamp': batch['date'].values,
                'is_promo': batch['is_promo'].values.astype(float),
                'discount_rate': batch['discount_rate'].values.astype(float),
                'ppc_fee': batch['ppc_fee'].values.astype(float),
                'day_of_week': batch['day_of_week'].values.astype(float),
                'is_weekend': batch['is_weekend'].values.astype(float),
                'month': batch['month'].values.astype(float),
            })
            
            try:
                pred_df = pipeline.predict_df(
                    ctx, future_df=fut, prediction_length=pred_len,
                    quantile_levels=[0.5],
                    id_column='id', timestamp_column='timestamp', target='target',
                )
                chronos_preds = np.maximum(pred_df['0.5'].values[:pred_len], 0)
            except:
                chronos_preds = np.zeros(pred_len)
            
            # 根据策略决定最终预测
            batch_preds = []
            for i in range(pred_len):
                chronos_pred = chronos_preds[i]
                
                if strategy == 'ensemble' and lgb_model is not None:
                    # 集成：Chronos + LightGBM 加权
                    lgb_pred = predict_lgb(lgb_model, lgb_features, current_train, batch.iloc[i])
                    if lgb_pred is not None:
                        # 高CV用更保守的预测：取较小值
                        final_pred = min(chronos_pred, lgb_pred) * 0.9 + max(chronos_pred, lgb_pred) * 0.1
                    else:
                        final_pred = chronos_pred * 0.8  # 高CV倾向低估
                elif strategy == 'chronos+bias':
                    # 中等波动：用历史偏差校正
                    final_pred = chronos_pred
                else:
                    # 稳定SKU：直接用Chronos
                    final_pred = chronos_pred
                
                batch_preds.append(max(0, final_pred))
                all_chronos.append(chronos_pred)
            
            all_preds.extend(batch_preds)
            
            # 滚动更新
            for i in range(pred_len):
                row = batch.iloc[i:i+1].copy()
                row['quantity'] = test.iloc[start + i]['quantity']
                current_train = pd.concat([current_train, row], ignore_index=True)
        
        # 残差校正（仅对高估偏差>20%的情况）
        actuals = test['quantity'].values.astype(float)
        preds = np.array(all_preds)
        
        # 用前30天估算偏差，校正后30天
        if len(preds) >= 30 and strategy in ['ensemble', 'chronos+bias']:
            bias = estimate_bias(preds[:30], actuals[:30])
            if bias > 0.2:  # 只校正明显高估的情况
                correction = 1 / (1 + bias * 0.5)  # 温和校正
                preds[30:] = preds[30:] * correction
        
        # 记录结果
        acc_list = []
        for i in range(len(test)):
            acc = calc_acc(preds[i], actuals[i])
            acc_list.append(acc)
            results.append({
                'sku': sku,
                'date': test.iloc[i]['date'].strftime('%Y-%m-%d'),
                'actual': int(actuals[i]),
                'predicted': round(preds[i], 1),
                'accuracy': round(acc, 1),
                'strategy': strategy,
                'is_promo': int(test.iloc[i]['is_promo']),
                'discount_rate': round(test.iloc[i]['discount_rate'], 4),
                'ppc_fee': round(test.iloc[i]['ppc_fee'], 2),
            })
        
        elapsed = time.time() - t0
        mean_acc = np.mean(acc_list)
        print(f"{sku}: CV={cv:.2f} 策略={strategy:<12} 准确率={mean_acc:.1f}% ({elapsed:.1f}s)")
        
        sku_summary.append({
            'sku': sku, 'cv': round(cv, 2), 'strategy': strategy,
            'accuracy': round(mean_acc, 1)
        })
    
    # 保存
    df = pd.DataFrame(results)
    out_path = f'{OUT_DIR}/chronos2_backtest_v3.csv'
    df.to_csv(out_path, index=False)
    
    summary_df = pd.DataFrame(sku_summary)
    summary_df.to_csv(f'{OUT_DIR}/v3_summary.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"V3回测完成: {df['sku'].nunique()} SKU, {len(df)} 条记录")
    print(f"整体准确率: {df['accuracy'].mean():.1f}%")
    print(f">=70%占比: {(df['accuracy'] >= 70).mean()*100:.1f}%")
    print(f"<1%占比: {(df['accuracy'] < 1).mean()*100:.1f}%")
    
    print(f"\n策略分布:")
    for s in ['chronos', 'chronos+bias', 'ensemble']:
        sub = df[df['strategy'] == s]
        if len(sub) > 0:
            print(f"  {s:<12}: {sub['sku'].nunique():>2} SKU, 准确率={sub['accuracy'].mean():.1f}%")
    
    print(f"\n结果: {out_path}")


if __name__ == '__main__':
    main()
