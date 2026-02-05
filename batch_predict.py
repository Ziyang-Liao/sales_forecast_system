#!/usr/bin/env python3
"""
高性能批量预测脚本 - 命令行版本
支持大规模SKU并行预测，数据完全本地处理
"""
import argparse
import pandas as pd
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
import warnings
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.processor import DataProcessor
from src.features.engineer import FeatureEngineer
from src.models.ensemble import EnsembleForecaster
from src.calibration.calibrator import PromotionCalibrator

warnings.filterwarnings('ignore')

def predict_sku(args):
    """预测单个SKU"""
    sku, sku_data, pred_days, future_plan = args
    try:
        processor = DataProcessor()
        fe = FeatureEngineer()
        
        # 准备数据
        df = sku_data.copy()
        df = fe.create_features(df)
        
        # 训练
        model = EnsembleForecaster(use_chronos=False)  # 批量时禁用Chronos加速
        model.fit(df, val_days=min(30, len(df)//5))
        
        # 预测
        last_date = df['date'].max()
        pred_dates = pd.date_range(last_date + timedelta(days=1), periods=pred_days)
        pred_df = pd.DataFrame({'date': pred_dates})
        pred_df = fe.create_features(pred_df, for_prediction=True)
        
        if future_plan:
            pred_df = fe.add_plan_features(pred_df, future_plan)
        
        forecast = model.predict(pred_df, pred_days)
        forecast['sku'] = sku
        
        return forecast
    except Exception as e:
        print(f"SKU {sku} 预测失败: {e}")
        return None

def batch_predict(data_path: str, output_path: str, pred_days: int = 30,
                  plan_path: str = None, max_workers: int = 4):
    """批量预测所有SKU"""
    print(f"{'='*60}")
    print(f"Amazon 销量预测系统 - 批量预测")
    print(f"{'='*60}")
    
    # 加载数据
    print(f"\n加载数据: {data_path}")
    processor = DataProcessor()
    df = processor.load_data(data_path)
    
    # 加载促销计划
    future_plan = None
    if plan_path and os.path.exists(plan_path):
        print(f"加载促销计划: {plan_path}")
        plan_df = pd.read_csv(plan_path)
        future_plan = plan_df.to_dict('records')
    
    # 按SKU分组
    if 'sku' in df.columns:
        skus = df['sku'].unique()
        print(f"SKU数量: {len(skus)}")
    else:
        skus = ['ALL']
        df['sku'] = 'ALL'
    
    # 准备任务
    tasks = []
    for sku in skus:
        sku_data = df[df['sku'] == sku].copy()
        if len(sku_data) >= 60:  # 至少60天数据
            tasks.append((sku, sku_data, pred_days, future_plan))
    
    print(f"有效SKU: {len(tasks)}")
    print(f"预测天数: {pred_days}")
    print(f"并行进程: {max_workers}")
    
    # 并行预测
    results = []
    print(f"\n开始预测...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(predict_sku, task): task[0] for task in tasks}
        
        for i, future in enumerate(as_completed(futures), 1):
            sku = futures[future]
            result = future.result()
            if result is not None:
                results.append(result)
            
            if i % 10 == 0:
                print(f"进度: {i}/{len(tasks)} ({i/len(tasks)*100:.1f}%)")
    
    # 合并结果
    if results:
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_csv(output_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"预测完成!")
        print(f"{'='*60}")
        print(f"成功预测: {len(results)}/{len(tasks)} SKU")
        print(f"总预测记录: {len(final_df)}")
        print(f"预测总销量: {final_df['prediction'].sum():,.0f}")
        print(f"结果保存: {output_path}")
    else:
        print("没有成功的预测结果")

def main():
    parser = argparse.ArgumentParser(description='Amazon 销量预测 - 批量预测')
    parser.add_argument('--data', required=True, help='历史数据路径 (CSV/Excel)')
    parser.add_argument('--output', default='forecast_result.csv', help='输出路径')
    parser.add_argument('--days', type=int, default=30, help='预测天数')
    parser.add_argument('--plan', help='促销计划文件路径 (可选)')
    parser.add_argument('--workers', type=int, default=4, help='并行进程数')
    
    args = parser.parse_args()
    batch_predict(args.data, args.output, args.days, args.plan, args.workers)

if __name__ == '__main__':
    main()
