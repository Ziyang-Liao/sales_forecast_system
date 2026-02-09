# 销量预测系统

基于 Amazon Chronos-2 时序模型，预测电商SKU每日销量。

## 模型效果

| 指标 | 数值 |
|------|------|
| 整体准确率 | **70.5%** |
| >=70%准确率占比 | 64.3% |
| SKU数量 | 59个 |
| 测试期 | 60天 (2025-10-01 ~ 2025-11-29) |

## 核心思路

### 协变量设计

**历史数据 (context)** — 包含完整信息，供模型学习：
- `quantity` — 销量（目标变量）
- `sessions` — 访问量
- `ppc_clicks` — 广告点击数
- `ppc_ad_order_quantity` — 广告订单数
- `conversion_rate` — 转化率
- `is_promo` — 促销标记
- `discount_rate` — 折扣率
- `ppc_fee` — 广告费
- `day_of_week`, `is_weekend`, `month` — 时间特征
- `qty_yoy` — 去年同期销量

**未来数据 (future)** — 仅可提前规划的字段：
- `is_promo`, `discount_rate`, `ppc_fee` — 可提前规划
- `day_of_week`, `is_weekend`, `month`, `qty_yoy` — 可提前计算

### 关键优化

1. **滚动预测**：每7天更新context，用实际值滚动更新，减少长期漂移
2. **时间特征**：星期几、是否周末、月份、去年同期销量
3. **无硬编码**：所有参数自动适应，可泛化到新数据

## 文件说明

### 脚本

| 文件 | 用途 |
|------|------|
| `prepare_data.py` | 数据预处理：xlsx → train/test CSV |
| `run_backtest_v2.py` | V2回测（时间特征+滚动预测） |
| `run_backtest_prod.py` | **生产版回测（推荐）** |

### 数据

| 文件 | 说明 |
|------|------|
| `data/uploads/*.xlsx` | 原始数据（73300行，100个SKU） |
| `data/daily_train.csv` | 训练集（截止2025-09-30） |
| `data/daily_test.csv` | 测试集（60天） |
| `data/sku_list.csv` | 筛选后的59个SKU |

### 结果

| 文件 | 说明 |
|------|------|
| `results/chronos2_backtest_v2.csv` | V2回测结果 |
| `results/chronos2_backtest_prod.csv` | 生产版回测结果 |

## 使用方法

```bash
# 1. 数据预处理
python3.11 prepare_data.py

# 2. 运行回测（生产版）
python3.11 run_backtest_prod.py
```

## 数据预处理逻辑

`prepare_data.py` 处理流程：

1. **按日聚合**：同一SKU同一天的多条记录合并
   - quantity, ppc_fee, sessions, ppc_clicks, ppc_ad_order_quantity → sum
   - discount_rate, conversion_rate → mean

2. **补全缺失日期**：缺失天数填0

3. **促销标记**：
   - 11月22日起：`is_promo=1`
   - 11月28日（黑五）：`is_promo=2`

4. **SKU筛选**：
   - 训练期有数据天数 >= 180天
   - 测试期 >= 55天
   - 测试期非零天数 >= 30天

## 准确率计算

```
准确率 = max(0, 1 - |预测值 - 实际值| / 实际值) × 100%
```

## 准确率分布

| 准确率区间 | 占比 |
|-----------|------|
| 90% ~ 100% | 25.5% |
| 80% ~ 90% | 21.9% |
| 70% ~ 80% | 16.7% |
| 60% ~ 70% | 11.2% |
| < 60% | 24.6% |

## 环境要求

- Python 3.11
- chronos-forecasting >= 1.5
- PyTorch + CUDA
- pandas, numpy, openpyxl

## 版本历史

| 版本 | 准确率 | 改进 |
|------|--------|------|
| V1 基线 | 66.7% | 3协变量，一次预测60天 |
| V2 | 70.5% | +时间特征，+滚动预测 |
| **生产版** | **70.5%** | 无硬编码，可泛化 |
