"""促销校准模块 - 从历史数据自动学习促销效果"""
import pandas as pd
import numpy as np
from typing import List, Dict

class PromotionCalibrator:
    """促销预测校准器 - 基于历史数据学习，不使用硬编码系数"""
    
    def __init__(self):
        self.baseline = None
        self.effects = {}  # 从数据学习的效果
        self.discount_model = None  # 折扣-销量关系
        self.ppc_model = None  # 广告-销量关系
    
    @classmethod
    def from_data(cls, df: pd.DataFrame) -> 'PromotionCalibrator':
        """从历史数据学习所有系数"""
        cal = cls()
        
        # 基准：非促销日的平均销量
        if 'is_major_sale' in df.columns:
            normal_days = df[df['is_major_sale'] == 0]
        else:
            normal_days = df
        
        cal.baseline = normal_days['quantity'].mean() if len(normal_days) > 0 else df['quantity'].mean()
        
        # 学习各类促销的提升效果
        cal._learn_promotion_effects(df)
        
        # 学习折扣-销量关系
        cal._learn_discount_effect(df)
        
        # 学习广告-销量关系
        cal._learn_ppc_effect(df)
        
        return cal
    
    def _learn_promotion_effects(self, df: pd.DataFrame):
        """从历史数据学习各类促销的提升倍数"""
        promo_cols = {
            'is_prime_day': 'prime_day',
            'is_black_friday': 'black_friday', 
            'is_cyber_monday': 'cyber_monday',
            'is_bf_week': 'bf_week',
            'is_xmas_season': 'xmas_season',
        }
        
        # 非促销日基准
        if 'is_major_sale' in df.columns:
            baseline_sales = df[df['is_major_sale'] == 0]['quantity'].mean()
        else:
            baseline_sales = df['quantity'].mean()
        
        if baseline_sales <= 0:
            baseline_sales = 1
        
        for col, name in promo_cols.items():
            if col in df.columns:
                promo_data = df[df[col] == 1]['quantity']
                if len(promo_data) >= 1:  # 至少有1天数据
                    lift = promo_data.mean() / baseline_sales
                    std = promo_data.std() / promo_data.mean() if promo_data.mean() > 0 else 0.3
                    self.effects[name] = {
                        'lift': lift,
                        'std': std,
                        'samples': len(promo_data),
                        'learned': True
                    }
    
    def _learn_discount_effect(self, df: pd.DataFrame):
        """学习折扣率与销量的关系"""
        if 'discount_rate' not in df.columns:
            self.discount_model = {'slope': 0, 'learned': False}
            return
        
        # 按折扣率分组计算平均销量
        df_valid = df[df['discount_rate'].notna() & (df['quantity'] > 0)].copy()
        if len(df_valid) < 10:
            self.discount_model = {'slope': 0, 'learned': False}
            return
        
        # 简单线性回归: 销量 = baseline * (1 + slope * discount_rate)
        x = df_valid['discount_rate'].values
        y = df_valid['quantity'].values / self.baseline
        
        # 计算斜率
        if x.std() > 0:
            slope = np.corrcoef(x, y)[0, 1] * y.std() / x.std()
            slope = max(0, min(slope, 10))  # 限制在合理范围
        else:
            slope = 0
        
        self.discount_model = {
            'slope': slope,
            'learned': True,
            'samples': len(df_valid)
        }
    
    def _learn_ppc_effect(self, df: pd.DataFrame):
        """学习广告投入与销量的关系"""
        if 'ppc_fee' not in df.columns:
            self.ppc_model = {'slope': 0, 'learned': False}
            return
        
        df_valid = df[df['ppc_fee'].notna() & (df['ppc_fee'] > 0) & (df['quantity'] > 0)].copy()
        if len(df_valid) < 10:
            self.ppc_model = {'slope': 0, 'learned': False}
            return
        
        # 计算广告弹性: 销量变化% / 广告变化%
        avg_ppc = df_valid['ppc_fee'].mean()
        x = df_valid['ppc_fee'].values / avg_ppc  # 标准化
        y = df_valid['quantity'].values / self.baseline
        
        if x.std() > 0:
            elasticity = np.corrcoef(x, y)[0, 1] * y.std() / x.std()
            elasticity = max(0, min(elasticity, 2))  # 限制在合理范围
        else:
            elasticity = 0
        
        self.ppc_model = {
            'elasticity': elasticity,
            'avg_ppc': avg_ppc,
            'learned': True,
            'samples': len(df_valid)
        }
    
    def calibrate(self, predictions: pd.DataFrame, future_plan: List[Dict]) -> pd.DataFrame:
        """校准预测结果"""
        if not future_plan:
            return predictions
        
        plan_dict = {p['date']: p for p in future_plan}
        result = predictions.copy()
        
        calibrated = []
        for _, row in result.iterrows():
            date_str = str(row['date'].date()) if hasattr(row['date'], 'date') else str(row['date'])
            plan = plan_dict.get(date_str, {})
            
            original = row['prediction']
            
            if plan.get('is_promotion', False):
                # 计算基于学习系数的预期提升
                multiplier = self._calc_multiplier(plan)
                expected = self.baseline * multiplier
                
                # 只有当模型明显低估时才校准
                if original < expected * 0.7:
                    calibrated_pred = expected
                elif original < expected * 0.9:
                    w = (original / expected - 0.7) / 0.2
                    calibrated_pred = w * original + (1 - w) * expected
                else:
                    calibrated_pred = original
            else:
                calibrated_pred = original
            
            calibrated.append(max(0, calibrated_pred))
        
        result['prediction'] = calibrated
        result['lower_bound'] = result['prediction'] * 0.8
        result['upper_bound'] = result['prediction'] * 1.2
        return result
    
    def _calc_multiplier(self, plan: Dict) -> float:
        """基于学习的系数计算预期提升倍数"""
        multiplier = 1.0
        
        # 促销类型效果
        promo_type = plan.get('promotion_type', 'none')
        if promo_type in self.effects:
            effect = self.effects[promo_type]
            if effect.get('learned'):
                multiplier *= effect['lift']
        
        # 折扣效果
        discount = plan.get('discount_rate', 0)
        if discount > 0 and self.discount_model.get('learned'):
            multiplier *= (1 + self.discount_model['slope'] * discount)
        
        # 广告效果
        ppc_ratio = plan.get('ppc_budget_ratio', 1.0)
        if ppc_ratio > 1 and self.ppc_model.get('learned'):
            multiplier *= (1 + self.ppc_model['elasticity'] * (ppc_ratio - 1))
        
        return multiplier
    
    def get_learned_effects(self) -> Dict:
        """返回学习到的效果系数"""
        return {
            'baseline': self.baseline,
            'promotion_effects': self.effects,
            'discount_model': self.discount_model,
            'ppc_model': self.ppc_model
        }
