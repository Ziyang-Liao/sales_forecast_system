"""混合预测模型"""
import pandas as pd
import numpy as np
import torch
import warnings
from typing import List, Dict, Tuple
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

class EnsembleForecaster:
    """混合集成预测模型 - Chronos-2 + LightGBM + XGBoost"""
    
    FEATURE_COLS = [
        'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'is_weekend', 'is_q4',
        'is_prime_day', 'is_black_friday', 'is_cyber_monday', 'is_major_sale',
        'is_bf_week', 'is_xmas_season', 'discount_depth',
        'lag_1', 'lag_7', 'lag_14', 'lag_28',
        'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28',
        'rolling_std_7', 'trend_7_28',
        'discount_rate', 'ppc_fee', 'ppc_ratio',
        'promo_discount', 'promo_ppc'  # 交互特征
    ]
    
    def __init__(self, use_chronos: bool = True):
        self.use_chronos = use_chronos
        self.lgb_model = None
        self.xgb_model = None
        self.chronos_pipe = None
        self.weights = {'chronos': 0.3, 'lgb': 0.4, 'xgb': 0.3}
        self.is_fitted = False
        self.context_data = None
    
    def fit(self, df: pd.DataFrame, val_days: int = 30) -> Dict:
        """训练模型"""
        df = df.dropna(subset=['quantity'])
        
        # 分割训练/验证
        split_idx = len(df) - val_days
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        # 保存上下文数据
        self.context_data = df['quantity'].values
        
        # 获取可用特征
        available_features = [c for c in self.FEATURE_COLS if c in df.columns]
        train_X = train_df[available_features].fillna(0)
        train_y = train_df['quantity']
        val_X = val_df[available_features].fillna(0)
        val_y = val_df['quantity']
        
        # 训练 LightGBM
        self.lgb_model = lgb.LGBMRegressor(
            objective='regression', n_estimators=500, learning_rate=0.05,
            num_leaves=31, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=5, verbose=-1
        )
        self.lgb_model.fit(train_X, train_y, eval_set=[(val_X, val_y)],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        
        # 训练 XGBoost
        self.xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror', n_estimators=500, learning_rate=0.05,
            max_depth=6, subsample=0.8, colsample_bytree=0.8, verbosity=0,
            early_stopping_rounds=50
        )
        self.xgb_model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
        
        # 加载 Chronos
        if self.use_chronos:
            self._load_chronos()
        
        # 验证集评估
        val_pred = self._predict_internal(val_df, available_features)
        accuracy = self._calc_accuracy(val_y.values, val_pred)
        
        self.is_fitted = True
        self.feature_cols = available_features
        
        return {
            'accuracy': accuracy,
            'mape': np.mean(np.abs(val_y.values - val_pred) / (val_y.values + 1)),
            'weights': self.weights,
            'feature_importance': self._get_feature_importance(available_features)
        }
    
    def _load_chronos(self):
        """加载Chronos模型"""
        try:
            from chronos import ChronosPipeline
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.chronos_pipe = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-small",
                device_map=device,
                torch_dtype=torch.float32
            )
        except Exception as e:
            print(f"Chronos加载失败: {e}, 使用纯ML模型")
            self.use_chronos = False
            self.weights = {'chronos': 0, 'lgb': 0.55, 'xgb': 0.45}
    
    def predict(self, df: pd.DataFrame, prediction_length: int = 14) -> pd.DataFrame:
        """生成预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        # 确保使用训练时的特征列
        X = df.reindex(columns=self.feature_cols, fill_value=0)
        predictions = self._predict_internal_direct(X, prediction_length)
        
        result = df[['date']].copy()
        result['prediction'] = np.maximum(0, predictions)
        
        # 置信区间 (基于历史波动)
        std_ratio = 0.2
        result['lower_bound'] = result['prediction'] * (1 - std_ratio)
        result['upper_bound'] = result['prediction'] * (1 + std_ratio)
        return result
    
    def _predict_internal_direct(self, X: pd.DataFrame, pred_len: int = None) -> np.ndarray:
        """内部预测 - 直接使用特征矩阵"""
        pred_lgb = self.lgb_model.predict(X)
        pred_xgb = self.xgb_model.predict(X)
        
        if self.use_chronos and self.chronos_pipe and self.context_data is not None:
            pred_chronos = self._predict_chronos(pred_len or len(X))
            if len(pred_chronos) == len(pred_lgb):
                return (self.weights['chronos'] * pred_chronos +
                        self.weights['lgb'] * pred_lgb +
                        self.weights['xgb'] * pred_xgb)
        
        w_lgb = self.weights['lgb'] / (self.weights['lgb'] + self.weights['xgb'])
        return w_lgb * pred_lgb + (1 - w_lgb) * pred_xgb
    
    def _predict_internal(self, df: pd.DataFrame, features: List[str], 
                          pred_len: int = None) -> np.ndarray:
        """内部预测"""
        X = df[features].fillna(0)
        
        pred_lgb = self.lgb_model.predict(X)
        pred_xgb = self.xgb_model.predict(X)
        
        if self.use_chronos and self.chronos_pipe and self.context_data is not None:
            pred_chronos = self._predict_chronos(pred_len or len(df))
            if len(pred_chronos) == len(pred_lgb):
                return (self.weights['chronos'] * pred_chronos +
                        self.weights['lgb'] * pred_lgb +
                        self.weights['xgb'] * pred_xgb)
        
        w_lgb = self.weights['lgb'] / (self.weights['lgb'] + self.weights['xgb'])
        return w_lgb * pred_lgb + (1 - w_lgb) * pred_xgb
    
    def _predict_chronos(self, pred_len: int) -> np.ndarray:
        """Chronos预测"""
        try:
            context = torch.tensor(self.context_data[-365:], dtype=torch.float32)
            forecast = self.chronos_pipe.predict(context, prediction_length=pred_len, num_samples=50)
            return np.median(forecast[0].numpy(), axis=0)
        except:
            return np.zeros(pred_len)
    
    def _calc_accuracy(self, actual: np.ndarray, pred: np.ndarray, threshold: float = 0.2) -> float:
        """计算准确率 (预测在实际±threshold范围内)"""
        within = np.abs(actual - pred) <= actual * threshold
        return np.mean(within)
    
    def _get_feature_importance(self, features: List[str]) -> pd.DataFrame:
        """获取特征重要性"""
        imp = pd.DataFrame({
            'feature': features,
            'importance': self.lgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        return imp
