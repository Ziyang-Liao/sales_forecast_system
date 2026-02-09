# 电商SKU销量预测优化方案文档

## 一、项目背景

### 1.1 业务目标
预测电商平台SKU的每日销量，用于库存管理和运营决策。

### 1.2 数据概况
- **SKU数量**：59个
- **训练期**：2023-01-01 ~ 2025-09-30（约1000天/SKU）
- **测试期**：2025-10-01 ~ 2025-11-29（60天，含黑五大促）
- **数据粒度**：日级别

### 1.3 当前最优方案
- **模型**：Amazon Chronos-2（时序基础模型）
- **准确率**：70.5%
- **>=70%准确率占比**：64.3%

---

## 二、数据字段说明

### 2.1 原始数据字段（73列）
| 字段 | 说明 | 与销量相关性 |
|------|------|-------------|
| quantity | 销量（目标变量） | - |
| sessions | 访问量 | 0.439 |
| ppc_clicks | 广告点击数 | 0.469 |
| ppc_ad_order_quantity | 广告订单数 | 0.704 |
| conversion_rate | 转化率 | 0.308 |
| discount_rate | 折扣率 | 0.052 |
| ppc_fee | 广告费 | - |
| is_promo | 促销标记（0/1/2=黑五） | - |

### 2.2 协变量使用规则

**关键约束**：预测未来时，只能使用"可提前规划"的字段。

| 数据类型 | 可用字段 | 说明 |
|----------|---------|------|
| **历史数据 (context)** | 全部字段 | 模型学习用 |
| **未来数据 (future)** | is_promo, discount_rate, ppc_fee, 时间特征, 去年同期 | 可提前规划 |
| **未来不可用** | sessions, ppc_clicks, ppc_ad_order_quantity, conversion_rate | 未来不可知 |

---

## 三、当前最优实现

### 3.1 核心思路
1. **滚动预测**：每7天更新一次context，用实际值滚动更新
2. **时间特征**：day_of_week, is_weekend, month, qty_yoy（去年同期）
3. **协变量分离**：历史数据含全量特征，未来数据只含可规划字段

### 3.2 代码逻辑（简化版）
```python
from chronos import Chronos2Pipeline

pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2")

for sku in skus:
    for batch in rolling_windows(test, window=7):
        # 历史数据：全量特征
        context = {
            'target': train['quantity'],
            'sessions': train['sessions'],  # 历史可用
            'ppc_clicks': train['ppc_clicks'],
            'is_promo': train['is_promo'],
            'discount_rate': train['discount_rate'],
            'ppc_fee': train['ppc_fee'],
            'day_of_week': train['day_of_week'],
            'qty_yoy': train['qty_yoy'],
            ...
        }
        
        # 未来数据：只含可规划字段
        future = {
            'is_promo': batch['is_promo'],
            'discount_rate': batch['discount_rate'],
            'ppc_fee': batch['ppc_fee'],
            'day_of_week': batch['day_of_week'],
            'qty_yoy': batch['qty_yoy'],
            # 注意：不含 sessions, ppc_clicks 等
        }
        
        pred = pipeline.predict(context, future)
        
        # 滚动更新：将实际值加入训练集
        train = concat(train, batch_with_actuals)
```

### 3.3 准确率计算公式
```
准确率 = max(0, 1 - |预测值 - 实际值| / 实际值) × 100%
```

---

## 四、已尝试但无效的优化方案

| 方案 | 准确率 | 结论 |
|------|--------|------|
| 多分位数选择（P30/P50/P70） | 69.5% | 验证集选择不泛化 |
| 动态偏差校正 | 69.7% | 校正引入滞后 |
| Croston间歇需求模型 | 46.1% | 不适用于本数据 |
| 多SKU联合预测 | 70.5% | 无提升 |
| 滞后特征+节假日日历 | 62.8% | 引入噪音 |
| 生命周期系数校正 | 64-67% | Chronos已学习趋势，重复校正 |

---

## 五、准确率分析

### 5.1 准确率分布
| 准确率区间 | 占比 |
|-----------|------|
| 90% ~ 100% | 25.5% |
| 80% ~ 90% | 21.9% |
| 70% ~ 80% | 16.7% |
| 60% ~ 70% | 11.2% |
| < 60% | 24.6% |

### 5.2 高准确率SKU特征（>=75%，25个）
- 平均日销量：68.9
- 平均CV（变异系数）：**0.97**（低波动）
- 零销量率：0.6%

### 5.3 低准确率SKU特征（<55%，8个）
- 平均日销量：24.2
- 平均CV：**2.75**（高波动）
- 零销量率：17.5%

### 5.4 关键发现
- **CV是决定准确率的核心因素**（相关性 -0.68）
- 高CV的SKU（销量波动大）难以预测
- 8月（非促销期）和10-11月（含促销）准确率一致（~70%）

---

## 六、待探索的优化方向

### 6.1 数据层面
- 增加外部数据：品类大盘趋势、搜索热度
- 异常值处理：剔除极端销量日
- 更长的历史数据

### 6.2 模型层面
- Chronos-2 微调（官方暂不支持）
- 其他时序基础模型（TimesFM, Lag-Llama）
- 针对高CV的SKU使用专门策略

### 6.3 预测策略
- 输出预测区间而非点估计
- 高CV的SKU标记为"低置信度"，建议人工审核

---

## 七、核心问题

**如何进一步提升准确率？**

当前瓶颈：
1. 高CV的SKU（约20%）拉低整体准确率
2. 未来可用的协变量有限（只有促销、折扣、广告预算）
3. Chronos-2 已经是强基线，简单的后处理无效

期望：
- 整体准确率从 70.5% 提升到 75%+
- 或针对高CV的SKU找到更好的预测策略

---

## 八、文件结构

```
sales_forecast_system/
├── prepare_data.py          # 数据预处理
├── run_backtest_prod.py     # 生产版回测（推荐）
├── run_backtest_v2.py       # V2回测
├── data/
│   ├── uploads/*.xlsx       # 原始数据
│   ├── daily_train.csv      # 训练集
│   ├── daily_test.csv       # 测试集
│   └── sku_list.csv         # SKU列表
└── results/
    └── chronos2_backtest_prod.csv  # 回测结果
```

---

## 九、运行方式

```bash
# 数据预处理
python3.11 prepare_data.py

# 运行回测
python3.11 run_backtest_prod.py
```

---

## 十、GitHub仓库

https://github.com/Ziyang-Liao/sales_forecast_system
