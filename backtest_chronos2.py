#!/usr/bin/env python3
"""
Chronos-2 回测脚本
使用真实协变量（促销标记、折扣率、广告费），预测销量并计算每日准确率
"""
import argparse
import pandas as pd
import numpy as np

def run_backtest(data_path: str, sku: str, test_start: str, test_end: str):
    """
    运行Chronos-2回测
    
    Args:
        data_path: 数据文件路径
        sku: SKU编码
        test_start: 测试开始日期 (YYYY-MM-DD)
        test_end: 测试结束日期 (YYYY-MM-DD)
    """
    from chronos import Chronos2Pipeline
    
    print(f"Chronos-2 回测: {sku}")
    print("="*90)
    
    # 加载模型
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    
    # 加载数据
    df = pd.read_excel(data_path)
    
    # 筛选SKU
    sku_df = df[df['sku'] == sku].copy()
    sku_df['date'] = pd.to_datetime(sku_df['purchase_date'], format='%Y%m%d')
    
    # 按日聚合
    daily = sku_df.groupby('date').agg({
        'quantity': 'sum',
        'discount_rate': 'mean',
        'ppc_fee': 'sum'
    }).reset_index()
    
    # 添加促销标记 (示例: 11月22日起为促销季，28日为黑五)
    daily['is_promo'] = 0
    daily.loc[(daily['date'].dt.month == 11) & (daily['date'].dt.day >= 22), 'is_promo'] = 1
    daily.loc[(daily['date'].dt.month == 11) & (daily['date'].dt.day == 28), 'is_promo'] = 2
    
    # 分割训练/测试
    train_df = daily[daily['date'] < test_start].copy()
    test_df = daily[(daily['date'] >= test_start) & (daily['date'] <= test_end)].copy()
    
    print(f"训练数据: {len(train_df)}天, 测试数据: {len(test_df)}天")
    
    # 准备Chronos-2格式
    # 历史数据: 包含销量+协变量
    context_df = train_df.rename(columns={'date': 'timestamp', 'quantity': 'target'})
    context_df['id'] = sku
    context_df = context_df[['id', 'timestamp', 'target', 'is_promo', 'discount_rate', 'ppc_fee']]
    
    # 未来数据: 只有协变量，不给销量
    future_df = test_df.rename(columns={'date': 'timestamp'})
    future_df['id'] = sku
    future_df = future_df[['id', 'timestamp', 'is_promo', 'discount_rate', 'ppc_fee']]
    
    # 预测
    pred_df = pipeline.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=len(test_df),
        quantile_levels=[0.5],
        id_column='id',
        timestamp_column='timestamp',
        target='target',
    )
    
    # 输出每日结果
    print(f"\n{'日期':<12} {'实际':>8} {'预测':>8} {'偏差':>10} {'准确率':>8} {'促销':>6} {'折扣率':>8} {'广告费':>10}")
    print("-"*90)
    
    for i, (_, row) in enumerate(test_df.iterrows()):
        if i >= len(pred_df):
            break
        
        actual = row['quantity']
        pred = pred_df.iloc[i]['0.5']
        bias = (pred - actual) / actual if actual > 0 else 0
        accuracy = max(0, 1 - abs(bias))
        
        promo = '黑五' if row['is_promo'] == 2 else ('促销' if row['is_promo'] == 1 else '')
        
        print(f"{row['date'].strftime('%Y-%m-%d'):<12} {actual:>8.0f} {pred:>8.0f} {bias:>+10.1%} {accuracy:>8.1%} {promo:>6} {row['discount_rate']:>8.1%} {row['ppc_fee']:>10.0f}")
    
    # 汇总
    actual_total = test_df['quantity'].sum()
    pred_total = pred_df['0.5'].sum()
    
    print("-"*90)
    print(f"汇总: 实际={actual_total:,.0f}, 预测={pred_total:,.0f}, 总偏差={(pred_total-actual_total)/actual_total:+.1%}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chronos-2 回测')
    parser.add_argument('--data', required=True, help='数据文件路径')
    parser.add_argument('--sku', required=True, help='SKU编码')
    parser.add_argument('--test-start', required=True, help='测试开始日期 (YYYY-MM-DD)')
    parser.add_argument('--test-end', required=True, help='测试结束日期 (YYYY-MM-DD)')
    
    args = parser.parse_args()
    run_backtest(args.data, args.sku, args.test_start, args.test_end)
