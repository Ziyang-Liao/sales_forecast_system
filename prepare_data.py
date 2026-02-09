#!/usr/bin/env python3
"""
数据预处理：从原始xlsx生成标准化的train/test CSV
- 按SKU按日聚合
- 补全缺失日期（填0）
- 添加促销标记
- 筛选数据充足的SKU
- 输出: data/daily_train.csv, data/daily_test.csv, data/sku_list.csv
"""
import pandas as pd
import numpy as np
import os

XLSX_PATH = '/home/ec2-user/sales_forecast_system/data/uploads/美国站分层抽样_脱敏数据_20260203_181815.xlsx'
OUT_DIR = '/home/ec2-user/sales_forecast_system/data'
TRAIN_END = '2025-09-30'
TEST_START = '2025-10-01'
TEST_END = '2025-11-29'
MIN_TRAIN_DAYS = 180
MIN_TEST_DAYS = 55


def main():
    print("加载原始数据...")
    raw = pd.read_excel(XLSX_PATH)
    raw['date'] = pd.to_datetime(raw['purchase_date'], format='%Y%m%d')
    print(f"  原始: {len(raw)} 行, {raw['sku'].nunique()} SKU")

    all_train, all_test, sku_info = [], [], []

    for sku in sorted(raw['sku'].unique()):
        g = raw[raw['sku'] == sku]

        # 按日聚合
        daily = g.groupby('date').agg({
            'quantity': 'sum',
            'discount_rate': 'mean',
            'ppc_fee': 'sum',
            'sessions': 'sum',
            'ppc_clicks': 'sum',
            'ppc_ad_order_quantity': 'sum',
            'conversion_rate': 'mean',
        }).reset_index()

        # 补全缺失日期（从该SKU最早日期到TEST_END）
        full_range = pd.date_range(daily['date'].min(), TEST_END, freq='D')
        daily = daily.set_index('date').reindex(full_range).rename_axis('date').reset_index()
        daily['quantity'] = daily['quantity'].fillna(0).astype(int)
        daily['discount_rate'] = daily['discount_rate'].fillna(0)
        daily['ppc_fee'] = daily['ppc_fee'].fillna(0)
        daily['sessions'] = daily['sessions'].fillna(0)
        daily['ppc_clicks'] = daily['ppc_clicks'].fillna(0)
        daily['ppc_ad_order_quantity'] = daily['ppc_ad_order_quantity'].fillna(0)
        daily['conversion_rate'] = daily['conversion_rate'].fillna(0)

        # 促销标记（11月大促）
        daily['is_promo'] = 0
        m11 = daily['date'].dt.month == 11
        daily.loc[m11 & (daily['date'].dt.day >= 22), 'is_promo'] = 1
        daily.loc[m11 & (daily['date'].dt.day == 28), 'is_promo'] = 2  # 黑五

        daily['sku'] = sku

        # 分割
        train = daily[daily['date'] <= TRAIN_END]
        test = daily[(daily['date'] >= TEST_START) & (daily['date'] <= TEST_END)]

        # 筛选：train中有数据的天数>=MIN_TRAIN_DAYS，test天数>=MIN_TEST_DAYS
        train_nonzero = (train['quantity'] > 0).sum()
        test_nonzero = (test['quantity'] > 0).sum()
        if train_nonzero < MIN_TRAIN_DAYS or len(test) < MIN_TEST_DAYS or test_nonzero < 30:
            continue

        cv = train['quantity'].std() / (train['quantity'].mean() + 1e-6)
        sku_info.append({
            'sku': sku,
            'train_days': len(train),
            'train_nonzero_days': int(train_nonzero),
            'test_days': len(test),
            'daily_mean': round(train['quantity'].mean(), 1),
            'cv': round(cv, 2),
        })

        all_train.append(train)
        all_test.append(test)

    train_df = pd.concat(all_train, ignore_index=True)
    test_df = pd.concat(all_test, ignore_index=True)
    sku_df = pd.DataFrame(sku_info)

    # 保存
    os.makedirs(OUT_DIR, exist_ok=True)
    train_df.to_csv(f'{OUT_DIR}/daily_train.csv', index=False)
    test_df.to_csv(f'{OUT_DIR}/daily_test.csv', index=False)
    sku_df.to_csv(f'{OUT_DIR}/sku_list.csv', index=False)

    print(f"\n输出:")
    print(f"  {OUT_DIR}/daily_train.csv  ({len(train_df)} 行)")
    print(f"  {OUT_DIR}/daily_test.csv   ({len(test_df)} 行)")
    print(f"  {OUT_DIR}/sku_list.csv     ({len(sku_df)} SKU)")
    print(f"\nSKU统计:")
    print(sku_df.to_string(index=False))


if __name__ == '__main__':
    main()
