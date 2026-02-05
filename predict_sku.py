#!/usr/bin/env python3
"""
SKU级别预测脚本
输入：SKU + 未来促销计划（折扣率、广告预算）
输出：该SKU的销量预测
"""
import sys
sys.path.insert(0, '/home/ec2-user/sales_forecast_system')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.processor import DataProcessor
from src.features.engineer import FeatureEngineer
from src.models.ensemble import EnsembleForecaster

class SKUForecaster:
    """SKU级别预测器"""
    
    def __init__(self, data_path: str):
        self.processor = DataProcessor()
        self.df = self.processor.load_data(data_path)
        self.models = {}  # 每个SKU一个模型
        self.feature_engineers = {}
        
    def get_skus(self):
        """获取所有SKU列表"""
        return self.df['sku'].unique().tolist() if 'sku' in self.df.columns else []
    
    def train_sku(self, sku: str) -> dict:
        """训练单个SKU的模型"""
        sku_data = self.df[self.df['sku'] == sku].copy()
        
        if len(sku_data) < 60:
            return {'error': f'SKU {sku} 数据不足60天'}
        
        # 按日聚合
        daily = self.processor.aggregate_daily(sku_data)
        stats = self.processor.compute_stats(daily)
        
        # 特征工程
        fe = FeatureEngineer(stats)
        df_feat = fe.create_features(daily)
        
        # 训练
        model = EnsembleForecaster(use_chronos=False)
        result = model.fit(df_feat, val_days=min(30, len(df_feat)//5))
        
        # 保存
        self.models[sku] = model
        self.feature_engineers[sku] = fe
        
        return {
            'sku': sku,
            'days': len(daily),
            'avg_sales': stats['avg_daily_sales'],
            'accuracy': result['accuracy'],
            'mape': result['mape']
        }
    
    def predict_sku(self, sku: str, future_plan: list) -> pd.DataFrame:
        """
        预测单个SKU
        
        Args:
            sku: SKU编码
            future_plan: 促销计划列表
                [{'date': '2026-02-10', 'discount_rate': 0.2, 'ppc_budget': 100}, ...]
        """
        if sku not in self.models:
            self.train_sku(sku)
        
        model = self.models[sku]
        fe = self.feature_engineers[sku]
        
        # 构建预测数据
        dates = [pd.to_datetime(p['date']) for p in future_plan]
        pred_df = pd.DataFrame({'date': dates})
        pred_df = fe.create_features(pred_df, for_prediction=True)
        
        # 添加计划特征
        plan_df = pd.DataFrame(future_plan)
        plan_df['date'] = pd.to_datetime(plan_df['date'])
        pred_df = pred_df.merge(plan_df, on='date', how='left')
        
        # 填充特征
        pred_df['discount_rate'] = pred_df.get('discount_rate', 0).fillna(0)
        pred_df['ppc_fee'] = pred_df.get('ppc_budget', 50).fillna(50)
        avg_ppc = fe.stats.get('avg_ppc_fee', 50)
        pred_df['ppc_ratio'] = pred_df['ppc_fee'] / (avg_ppc + 1)
        pred_df['promo_discount'] = pred_df.get('is_major_sale', 0) * pred_df['discount_rate']
        pred_df['promo_ppc'] = pred_df.get('is_major_sale', 0) * pred_df['ppc_ratio']
        
        # 预测
        forecast = model.predict(pred_df, len(pred_df))
        forecast['sku'] = sku
        
        # 合并计划信息
        for col in ['discount_rate', 'ppc_budget']:
            if col in plan_df.columns:
                forecast = forecast.merge(plan_df[['date', col]], on='date', how='left')
        
        return forecast

def main():
    import argparse
    parser = argparse.ArgumentParser(description='SKU级别销量预测')
    parser.add_argument('--data', required=True, help='数据文件路径')
    parser.add_argument('--sku', required=True, help='要预测的SKU')
    parser.add_argument('--days', type=int, default=14, help='预测天数')
    parser.add_argument('--discount', type=float, default=0, help='计划折扣率 (0-0.5)')
    parser.add_argument('--ppc', type=float, default=50, help='计划每日广告预算')
    parser.add_argument('--promo-days', help='促销日期，逗号分隔，如 2026-02-10,2026-02-11')
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"SKU 销量预测: {args.sku}")
    print("="*60)
    
    # 初始化
    forecaster = SKUForecaster(args.data)
    
    # 检查SKU
    skus = forecaster.get_skus()
    if args.sku not in skus:
        print(f"错误: SKU '{args.sku}' 不存在")
        print(f"可用SKU: {skus[:10]}...")
        return
    
    # 训练
    print(f"\n训练模型...")
    result = forecaster.train_sku(args.sku)
    print(f"历史数据: {result['days']} 天")
    print(f"日均销量: {result['avg_sales']:.0f}")
    print(f"回测准确率: {result['accuracy']:.1%}")
    
    # 构建计划
    start = datetime.now().date() + timedelta(days=1)
    promo_dates = []
    if args.promo_days:
        promo_dates = [d.strip() for d in args.promo_days.split(',')]
    
    future_plan = []
    for i in range(args.days):
        d = start + timedelta(days=i)
        date_str = d.strftime('%Y-%m-%d')
        is_promo = date_str in promo_dates
        future_plan.append({
            'date': date_str,
            'discount_rate': args.discount if is_promo else 0,
            'ppc_budget': args.ppc,
            'is_promotion': is_promo
        })
    
    # 预测
    print(f"\n生成预测 ({args.days}天)...")
    forecast = forecaster.predict_sku(args.sku, future_plan)
    
    print(f"\n预测结果:")
    print("-"*60)
    total = 0
    for _, row in forecast.iterrows():
        d = row['date'].strftime('%Y-%m-%d')
        pred = row['prediction']
        total += pred
        discount = row.get('discount_rate', 0)
        mark = f" [促销 {discount:.0%} off]" if discount > 0 else ""
        print(f"{d}: {pred:>6.0f}{mark}")
    
    print("-"*60)
    print(f"预测总销量: {total:,.0f}")
    print(f"日均预测: {total/args.days:.0f}")

if __name__ == '__main__':
    main()
