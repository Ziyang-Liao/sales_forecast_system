#!/usr/bin/env python3
"""LSTM回测：用PyTorch LSTM做销量预测"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/home/ec2-user/sales_forecast_system/data'
ROLL_DAYS = 7
FORECAST_DAYS = 60
TEST_SKUS = 5  # 0=全量
SEQ_LEN = 28  # LSTM输入序列长度
EPOCHS = 100
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LR = 0.001

FEATURE_COLS = ['quantity', 'is_promo', 'discount_rate', 'ppc_fee',
                'day_of_week', 'is_weekend', 'month', 'qty_yoy']


class SalesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


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


def create_sequences(data, seq_len):
    """创建LSTM训练序列"""
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, 0])  # quantity是第0列
    return np.array(X), np.array(y)


def train_lstm(train_data, device):
    """训练LSTM模型"""
    # 标准化
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    std[std == 0] = 1
    normalized = (train_data - mean) / std

    X, y = create_sequences(normalized, SEQ_LEN)
    if len(X) < 30:
        return None, mean, std

    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SalesLSTM(X.shape[2], HIDDEN_SIZE, NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    model.train()
    best_loss = float('inf')
    patience = 10
    wait = 0
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.eval()
    return model, mean, std


def predict_step(model, seq, mean, std, device):
    """单步预测"""
    normalized = (seq - mean) / std
    x = torch.FloatTensor(normalized).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_norm = model(x).item()
    return pred_norm * std[0] + mean[0]  # 反标准化quantity


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"LSTM回测 (device={device})", flush=True)

    train_all = pd.read_csv(f'{DATA_DIR}/daily_train.csv', parse_dates=['date'])
    test_all = pd.read_csv(f'{DATA_DIR}/daily_test.csv', parse_dates=['date'])
    skus = pd.read_csv(f'{DATA_DIR}/sku_list.csv')['sku'].tolist()
    if TEST_SKUS > 0:
        skus = skus[:TEST_SKUS]

    train_all = add_time_features(train_all)
    test_all = add_time_features(test_all)

    print(f"{len(skus)} SKU, seq_len={SEQ_LEN}, hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}\n", flush=True)

    all_results = []
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

        # 滚动预测
        all_preds = []
        current_train = train.copy()

        for start in range(0, len(test), ROLL_DAYS):
            end = min(start + ROLL_DAYS, len(test))
            batch = test.iloc[start:end].copy()

            # 准备训练数据
            train_data = current_train[FEATURE_COLS].values.astype(float)
            model, mean, std = train_lstm(train_data, device)

            if model is None:
                all_preds.extend([0] * len(batch))
            else:
                # 逐天预测
                full_data = np.vstack([train_data, batch[FEATURE_COLS].values.astype(float)])
                for i in range(len(batch)):
                    idx = len(train_data) + i
                    seq = full_data[idx - SEQ_LEN:idx]
                    pred = max(0, predict_step(model, seq, mean, std, device))
                    all_preds.append(pred)
                    full_data[idx, 0] = pred  # 用预测值回填quantity

            # 滚动更新
            for i in range(len(batch)):
                row = batch.iloc[i:i + 1].copy()
                row['quantity'] = all_preds[start + i]
                current_train = pd.concat([current_train, row], ignore_index=True)

        actuals = test['quantity'].values.astype(float)
        acc_list = [calc_acc(all_preds[i], actuals[i]) for i in range(len(test))]
        mean_acc = np.mean(acc_list)
        all_results.extend(acc_list)

        print(f"[{sku_idx + 1}/{len(skus)}] {sku}: {mean_acc:.1f}%  ({time.time() - t0:.1f}s)", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"LSTM: 准确率={np.mean(all_results):.1f}%  >=70%={np.mean([1 if a >= 70 else 0 for a in all_results]) * 100:.1f}%", flush=True)
    print(f"(基线参考: 两阶段Chronos-2 = 69.2%)", flush=True)
    print(f"总耗时: {time.time() - t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
