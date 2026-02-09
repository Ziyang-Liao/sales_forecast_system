# 销量预测系统

基于 Chronos-2 模型，输入协变量预测每日销量。

## 模型效果

| 版本 | 准确率 | >=70%占比 | 说明 |
|------|--------|----------|------|
| V1 基线 | 66.7% | 57.2% | 3协变量 |
| **V2 优化** | **70.5%** | **64.3%** | 时间特征 + 滚动预测 |

## 协变量说明

### 历史数据 (context) — 完整信息
- `quantity` — 销量（目标）
- `discount_rate` — 折扣率
- `ppc_fee` — 广告费
- `sessions` — 访问量
- `ppc_clicks` — 广告点击数
- `ppc_ad_order_quantity` — 广告订单数
- `conversion_rate` — 转化率
- `is_promo` — 促销标记
- `day_of_week`, `is_weekend`, `month` — 时间特征
- `qty_yoy` — 去年同期销量

### 未来数据 (future) — 仅可规划字段
- `is_promo`, `discount_rate`, `ppc_fee` — 可提前规划
- `day_of_week`, `is_weekend`, `month`, `qty_yoy` — 可提前计算

## 数据文件

### 原始数据
- `data/uploads/美国站分层抽样_脱敏数据_20260203_181815.xlsx` — 73300行，100个SKU

### 预处理后数据
- `data/daily_train.csv` — 训练集（截止2025-09-30）
- `data/daily_test.csv` — 测试集（2025-10-01 ~ 2025-11-29），60天
- `data/sku_list.csv` — 59个SKU

## 脚本说明

| 脚本 | 用途 | 准确率 |
|------|------|--------|
| `prepare_data.py` | 数据预处理，生成train/test CSV | - |
| `run_backtest.py` | V1 基线回测 | 66.7% |
| `run_backtest_v2.py` | **V2 优化回测（推荐）** | **70.5%** |
| `run_auto_strategy.py` | 自动策略选择 | 66.8% |

## V2 优化方案

1. **时间特征工程**
   - 星期几、是否周末、月份
   - 去年同期销量

2. **滚动预测**
   - 每7天更新context，减少长期漂移
   - 用实际值滚动更新训练数据

## 用法

```bash
# 数据预处理
python3.11 prepare_data.py

# V2 回测（推荐）
python3.11 run_backtest_v2.py

# V1 基线回测
python3.11 run_backtest.py
```

## 准确率计算

```
准确率 = max(0, 1 - |预测值 - 实际值| / 实际值) × 100%
```

## 环境要求

- Python 3.11
- `chronos-forecasting >= 1.5`
- PyTorch + CUDA
- pandas, numpy, openpyxl
