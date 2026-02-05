"""特征工程模块"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

# Prime Day 日期
PRIME_DAY_DATES = {
    2023: ['2023-07-11', '2023-07-12'],
    2024: ['2024-07-16', '2024-07-17'],
    2025: ['2025-07-15', '2025-07-16'],
    2026: ['2026-07-14', '2026-07-15'],
}

class FeatureEngineer:
    """特征工程主类"""
    
    def __init__(self, historical_stats: dict = None):
        self.stats = historical_stats or {}
    
    def create_features(self, df: pd.DataFrame, for_prediction: bool = False) -> pd.DataFrame:
        """创建所有特征"""
        df = df.copy()
        df = self._add_time_features(df)
        df = self._add_promotion_features(df)
        if not for_prediction:
            df = self._add_lag_features(df)
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """时间特征"""
        df['dow'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)
        df['is_q4'] = df['month'].isin([10, 11, 12]).astype(int)
        
        # 周期编码
        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        return df
    
    def _add_promotion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """促销特征"""
        # 大促日标记
        df['is_prime_day'] = 0
        df['is_black_friday'] = 0
        df['is_cyber_monday'] = 0
        
        for _, dates in PRIME_DAY_DATES.items():
            for d in dates:
                df.loc[df['date'] == pd.to_datetime(d), 'is_prime_day'] = 1
        
        # 黑五/网一 (11月第4个周四后的周五/周一)
        for year in df['date'].dt.year.unique():
            bf = self._get_black_friday(year)
            cm = bf + timedelta(days=3)
            df.loc[df['date'] == bf, 'is_black_friday'] = 1
            df.loc[df['date'] == cm, 'is_cyber_monday'] = 1
        
        df['is_major_sale'] = (df['is_prime_day'] | df['is_black_friday'] | df['is_cyber_monday']).astype(int)
        
        # 促销季节
        df['is_bf_week'] = ((df['month'] == 11) & (df['day'] >= 22)).astype(int)
        df['is_xmas_season'] = (((df['month'] == 11) & (df['day'] >= 15)) | 
                                ((df['month'] == 12) & (df['day'] <= 25))).astype(int)
        
        # 折扣特征
        if 'discount_rate' in df.columns:
            df['discount_depth'] = pd.cut(df['discount_rate'], 
                bins=[-np.inf, 0, 0.05, 0.1, 0.15, 0.2, 0.3, np.inf],
                labels=[0, 1, 2, 3, 4, 5, 6]).astype(float)
            # 折扣与促销的交互
            df['promo_discount'] = df['is_major_sale'] * df['discount_rate']
        
        # 广告与促销的交互
        if 'ppc_fee' in df.columns:
            avg_ppc = self.stats.get('avg_ppc_fee', df['ppc_fee'].mean())
            df['ppc_ratio'] = df['ppc_fee'] / (avg_ppc + 1)
            df['promo_ppc'] = df['is_major_sale'] * df['ppc_ratio']
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """滞后特征"""
        for lag in [1, 7, 14, 28]:
            df[f'lag_{lag}'] = df['quantity'].shift(lag)
        
        for w in [7, 14, 28]:
            df[f'rolling_mean_{w}'] = df['quantity'].shift(1).rolling(w, min_periods=1).mean()
            df[f'rolling_std_{w}'] = df['quantity'].shift(1).rolling(w, min_periods=1).std()
        
        df['trend_7_28'] = df['rolling_mean_7'] / (df['rolling_mean_28'] + 1)
        return df
    
    def _get_black_friday(self, year: int) -> datetime:
        """计算黑五日期"""
        nov1 = datetime(year, 11, 1)
        first_thu = nov1 + timedelta(days=(3 - nov1.weekday() + 7) % 7)
        fourth_thu = first_thu + timedelta(weeks=3)
        return fourth_thu + timedelta(days=1)
    
    def add_plan_features(self, df: pd.DataFrame, future_plan: List[dict]) -> pd.DataFrame:
        """添加未来计划特征"""
        if not future_plan:
            return df
        
        plan_df = pd.DataFrame(future_plan)
        plan_df['date'] = pd.to_datetime(plan_df['date'])
        df = df.merge(plan_df, on='date', how='left')
        
        # 填充默认值
        df['planned_discount_rate'] = df.get('discount_rate', 0).fillna(0)
        df['planned_ppc_budget'] = df.get('ppc_budget', self.stats.get('avg_ppc_fee', 50)).fillna(50)
        df['is_promotion'] = df.get('is_promotion', False).fillna(False)
        df['promotion_type'] = df.get('promotion_type', 'none').fillna('none')
        
        # 计算促销强度
        avg_ppc = self.stats.get('avg_ppc_fee', 50)
        df['ppc_ratio'] = df['planned_ppc_budget'] / (avg_ppc + 1)
        df['intensity_score'] = (df['planned_discount_rate'] * 10 + 
                                  np.clip(df['ppc_ratio'] - 1, 0, 2) +
                                  df['is_major_sale'] * 3)
        return df
