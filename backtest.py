#!/usr/bin/env python3
"""
回测脚本 - 验证模型在历史数据上的表现
"""
import sys
import argparse
sys.path.insert(0, '/home/ec2-user/sales_forecast_system')

import pandas as pd
from src.data.processor import DataProcessor
from src.features.engineer import FeatureEngineer
from src.models.ensemble import EnsembleForecaster
from src.calibration.calibrator import PromotionCalibrator
from src.evaluation.backtester import Backtester

def run_backtest(data_path: str, mode: str = 'rolling', output: str = None):
    """运行回测"""
    print("="*60)
    print("Amazon 销量预测系统 - 回测")
    print("="*60)
    
    # 加载数据
    print(f"\n加载数据: {data_path}")
    processor = DataProcessor()
    df = processor.load_data(data_path)
    daily = processor.aggregate_daily(df)
    stats = processor.compute_stats(daily)
    
    print(f"数据: {len(daily)} 天, 日均销量: {stats['avg_daily_sales']:.0f}")
    
    # 特征工程
    fe = FeatureEngineer(stats)
    df_feat = fe.create_features(daily)
    
    # 校准器
    calibrator = PromotionCalibrator.from_data(df_feat)
    
    # 回测
    backtester = Backtester(None, fe, calibrator)
    
    if mode == 'promo':
        print("\n回测模式: 历史促销期")
        results = backtester.backtest_promotions(df_feat)
    else:
        print("\n回测模式: 滚动窗口 (每30天)")
        results = backtester.run(df_feat, pred_days=14, step_days=30)
    
    if results.empty:
        print("没有足够数据进行回测")
        return
    
    # 显示结果
    print("\n" + "="*60)
    print("回测结果")
    print("="*60)
    
    valid = results[~results.get('error', pd.Series([None]*len(results))).notna()]
    
    if len(valid) > 0:
        print(f"\n总测试期数: {len(valid)}")
        print(f"平均 MAPE: {valid['mape'].mean():.1%}")
        print(f"平均准确率 (±20%): {valid['accuracy_20'].mean():.1%}")
        print(f"平均准确率 (±30%): {valid['accuracy_30'].mean():.1%}")
        print(f"平均偏差: {valid['bias'].mean():+.1%} ({'高估' if valid['bias'].mean() > 0 else '低估'})")
        
        print("\n详细结果:")
        print("-"*60)
        for _, row in valid.iterrows():
            print(f"{row['period']:30} | 实际: {row['actual_total']:>8,.0f} | "
                  f"预测: {row['predicted_total']:>8,.0f} | "
                  f"准确率: {row['accuracy_20']:.0%} | 偏差: {row['bias']:+.1%}")
    
    # 保存
    if output:
        results.to_csv(output, index=False)
        print(f"\n结果已保存: {output}")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='回测')
    parser.add_argument('--data', required=True, help='数据文件路径')
    parser.add_argument('--mode', choices=['rolling', 'promo'], default='rolling',
                        help='rolling=滚动窗口, promo=促销期')
    parser.add_argument('--output', help='输出文件路径')
    
    args = parser.parse_args()
    run_backtest(args.data, args.mode, args.output)
