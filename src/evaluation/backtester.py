"""回测模块 - 验证模型在历史促销期的表现"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import timedelta

class Backtester:
    """回测器 - 滚动窗口验证"""
    
    def __init__(self, model, feature_engineer, calibrator=None):
        self.model = model
        self.fe = feature_engineer
        self.calibrator = calibrator
    
    def run(self, df: pd.DataFrame, test_periods: List[Dict] = None,
            pred_days: int = 14, step_days: int = 30) -> pd.DataFrame:
        """
        运行回测
        
        Args:
            df: 带特征的历史数据
            test_periods: 指定测试期 [{'start': '2024-11-01', 'end': '2024-11-30', 'name': 'Black Friday'}]
            pred_days: 每次预测天数
            step_days: 滚动步长
        """
        results = []
        
        if test_periods:
            # 指定测试期回测
            for period in test_periods:
                result = self._backtest_period(df, period['start'], period['end'], period.get('name', ''))
                results.append(result)
        else:
            # 滚动窗口回测
            min_train = 180
            dates = df['date'].sort_values().unique()
            
            for i in range(min_train, len(dates) - pred_days, step_days):
                train_end = dates[i]
                test_start = dates[i + 1] if i + 1 < len(dates) else None
                test_end = dates[min(i + pred_days, len(dates) - 1)]
                
                if test_start:
                    result = self._backtest_period(df, str(test_start)[:10], str(test_end)[:10])
                    results.append(result)
        
        return pd.DataFrame(results)
    
    def _backtest_period(self, df: pd.DataFrame, start: str, end: str, name: str = '') -> Dict:
        """回测单个时期"""
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        
        # 分割训练/测试
        train_df = df[df['date'] < start_dt].copy()
        test_df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)].copy()
        
        if len(train_df) < 60 or len(test_df) == 0:
            return {'period': name or start, 'error': 'insufficient_data'}
        
        # 训练
        from src.models.ensemble import EnsembleForecaster
        model = EnsembleForecaster(use_chronos=False)
        model.fit(train_df, val_days=min(30, len(train_df)//5))
        
        # 预测 - 使用测试集的特征（包括促销标记）
        # 这模拟了"知道未来促销计划"的场景
        forecast = model.predict(test_df, len(test_df))
        
        # 评估
        actual = test_df['quantity'].values
        predicted = forecast['prediction'].values
        
        return {
            'period': name or f"{start} ~ {end}",
            'start': start,
            'end': end,
            'days': len(test_df),
            'actual_total': actual.sum(),
            'predicted_total': predicted.sum(),
            'mape': np.mean(np.abs(actual - predicted) / (actual + 1)),
            'accuracy_20': np.mean(np.abs(actual - predicted) <= actual * 0.2),
            'accuracy_30': np.mean(np.abs(actual - predicted) <= actual * 0.3),
            'bias': (predicted.sum() - actual.sum()) / actual.sum(),
        }
    
    def backtest_promotions(self, df: pd.DataFrame) -> pd.DataFrame:
        """专门回测历史促销期"""
        promo_periods = []
        
        # 检测历史大促
        if 'is_prime_day' in df.columns:
            prime_days = df[df['is_prime_day'] == 1]['date']
            for d in prime_days:
                promo_periods.append({
                    'start': (d - timedelta(days=1)).strftime('%Y-%m-%d'),
                    'end': (d + timedelta(days=1)).strftime('%Y-%m-%d'),
                    'name': f'Prime Day {d.year}'
                })
        
        if 'is_black_friday' in df.columns:
            bf_days = df[df['is_black_friday'] == 1]['date']
            for d in bf_days:
                promo_periods.append({
                    'start': (d - timedelta(days=2)).strftime('%Y-%m-%d'),
                    'end': (d + timedelta(days=3)).strftime('%Y-%m-%d'),
                    'name': f'Black Friday {d.year}'
                })
        
        if promo_periods:
            return self.run(df, test_periods=promo_periods)
        return pd.DataFrame()
