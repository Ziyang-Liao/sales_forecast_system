"""数据处理模块"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Union

class DataProcessor:
    """数据处理器 - 支持多格式输入、验证、聚合"""
    
    DATE_FORMATS = ["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]
    
    def load_data(self, source: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """加载并清洗数据"""
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        elif source.endswith('.xlsx'):
            df = pd.read_excel(source)
        else:
            df = pd.read_csv(source)
        
        # 标准化日期列
        if 'purchase_date' in df.columns:
            df['date'] = self._parse_date(df['purchase_date'])
        elif 'date' not in df.columns:
            raise ValueError("需要 purchase_date 或 date 列")
        else:
            df['date'] = pd.to_datetime(df['date'])
        
        # 确保quantity列存在
        if 'quantity' not in df.columns:
            raise ValueError("需要 quantity 列")
        
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)
        return df.sort_values('date').reset_index(drop=True)
    
    def _parse_date(self, series: pd.Series) -> pd.Series:
        """解析多种日期格式"""
        for fmt in self.DATE_FORMATS:
            try:
                return pd.to_datetime(series, format=fmt)
            except:
                continue
        return pd.to_datetime(series)
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """验证数据"""
        errors = []
        if 'date' not in df.columns:
            errors.append("缺少日期列")
        if 'quantity' not in df.columns:
            errors.append("缺少销量列")
        if df['quantity'].isna().sum() > len(df) * 0.1:
            errors.append("销量缺失超过10%")
        if len(df) < 60:
            errors.append("数据少于60天，建议至少180天")
        return len(errors) == 0, errors
    
    def aggregate_daily(self, df: pd.DataFrame, group_by: str = None) -> pd.DataFrame:
        """按日聚合"""
        agg_cols = {'quantity': 'sum'}
        numeric_cols = ['sales_amount_total_usd', 'discount_rate', 'ppc_fee', 'sessions']
        for col in numeric_cols:
            if col in df.columns:
                agg_cols[col] = 'mean' if col == 'discount_rate' else 'sum'
        
        if group_by and group_by in df.columns:
            return df.groupby(['date', group_by]).agg(agg_cols).reset_index()
        return df.groupby('date').agg(agg_cols).reset_index()
    
    def compute_stats(self, df: pd.DataFrame) -> dict:
        """计算历史统计"""
        return {
            'avg_daily_sales': df['quantity'].mean(),
            'avg_ppc_fee': df['ppc_fee'].mean() if 'ppc_fee' in df.columns else 50,
            'avg_discount_rate': df['discount_rate'].mean() if 'discount_rate' in df.columns else 0.05,
            'avg_sessions': df['sessions'].mean() if 'sessions' in df.columns else 300,
            'max_discount_rate': df['discount_rate'].max() if 'discount_rate' in df.columns else 0.3,
        }
