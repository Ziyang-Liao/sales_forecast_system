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

**未来协变量 (future_df)** — 运营可提前规划/计算的字段：
- `is_promo` — 促销标记（运营提前排期）
- `discount_rate` — 折扣率（运营提前设定）
- `ppc_fee` — 广告总预算（运营提前规划）
- `day_of_week`, `is_weekend`, `month` — 时间特征（日历计算）
- `qty_yoy` — 去年同期销量（历史数据计算）

**历史协变量 (context)** — 用户行为产生，仅供模型学习：
- `quantity` — 销量（目标变量）
- `sessions` — 访问量（用户浏览行为）
- `ppc_clicks` — 广告点击数（用户点击行为）
- `ppc_ad_order_quantity` — 广告订单数（用户购买行为）
- `conversion_rate` — 转化率（用户购买决策）
- 以上未来协变量在历史期也包含在context中

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

## 数据说明

数据文件不包含在仓库中（含生产数据），由 `prepare_data.py` 从原始xlsx生成。以下说明各文件的格式和字段。

### 原始数据 `data/uploads/*.xlsx`

电商平台导出的SKU级别每日运营数据，每行是一个SKU在某一天的记录（同一SKU同一天可能有多条）。

`prepare_data.py` 使用的字段：

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `sku` | string | SKU编号 | SKU_001 |
| `purchase_date` | int | 日期（YYYYMMDD格式） | 20250815 |
| `quantity` | int | 销量（件） | 42 |
| `ppc_fee` | float | 广告总费用（美元） | 188.64 |
| `ppc_fee_sp` | float | SP(商品推广)广告费 | 136.86 |
| `ppc_fee_sb` | float | SB(品牌推广)广告费 | 35.41 |
| `ppc_fee_sbv` | float | SBV(品牌视频)广告费 | 0.05 |
| `ppc_fee_sd` | float | SD(展示型推广)广告费 | 16.31 |
| `ppc_sales` | float | 广告带来的销售额 | 2523.73 |
| `ppc_impression` | int | 广告展示次数 | 35974 |
| `ppc_clicks` | int | 广告点击次数 | 203 |
| `ppc_ad_order_quantity` | int | 广告带来的订单数 | 57 |
| `sessions` | int | 页面访问量 | 465 |
| `conversion_rate` | float | 转化率 | 0.1742 |
| `discount_rate` | float | 折扣率（0=无折扣） | 0.0967 |
| `ppc_fee_rate` | float | 广告费率(广告费/销售额) | 0.0478 |
| `target_costs` | float | 目标成本预算（月度） | 118.56 |
| `actual_costs` | float | 实际成本 | 189.42 |

### 处理后数据（由 `prepare_data.py` 生成）

**`data/daily_train.csv`** — 训练集（截止2025-09-30）

按SKU按日聚合后的数据，每行是一个SKU在某一天的汇总记录。

| 字段 | 聚合方式 | 说明 |
|------|----------|------|
| `date` | — | 日期 |
| `sku` | — | SKU编号 |
| `quantity` | sum | 当日销量 |
| `ppc_fee` | sum | 广告总费用 |
| `ppc_fee_sp/sb/sbv/sd` | sum | 各渠道广告费 |
| `ppc_sales` | sum | 广告销售额 |
| `ppc_impression` | sum | 广告展示次数 |
| `ppc_clicks` | sum | 广告点击次数 |
| `ppc_ad_order_quantity` | sum | 广告订单数 |
| `sessions` | sum | 访问量 |
| `target_costs` | sum | 目标成本 |
| `actual_costs` | sum | 实际成本 |
| `discount_rate` | mean | 折扣率 |
| `conversion_rate` | mean | 转化率 |
| `ppc_fee_rate` | mean | 广告费率 |
| `is_promo` | 计算 | 促销标记（0/1/2） |

示例（脱敏）：

```
date,quantity,ppc_fee,sessions,ppc_clicks,ppc_ad_order_quantity,...,discount_rate,conversion_rate,is_promo,sku
2025-08-01,42,188.64,465,203,57,...,0.0967,0.1742,0,SKU_001
2025-08-02,38,156.22,412,178,48,...,0.0850,0.1553,0,SKU_001
```

**`data/daily_test.csv`** — 测试集（2025-10-01 ~ 2025-11-29，60天），字段与训练集相同。

**`data/sku_list.csv`** — 筛选后的SKU列表

| 字段 | 说明 | 示例 |
|------|------|------|
| `sku` | SKU编号 | SKU_001 |
| `train_days` | 训练期总天数 | 458 |
| `train_nonzero_days` | 训练期有销量的天数 | 448 |
| `test_days` | 测试期天数 | 60 |
| `daily_mean` | 训练期日均销量 | 61.9 |
| `cv` | 变异系数(标准差/均值) | 0.72 |

### 数据处理流程

```
原始xlsx（73300行，100个SKU，77列）
    │
    ├─ prepare_data.py
    │   ├─ 按SKU+日期聚合（sum/mean）
    │   ├─ 补全缺失日期（填0）
    │   ├─ 添加促销标记（11月22日起=1，黑五=2）
    │   └─ SKU筛选（训练期>=180天，测试期>=55天，测试期非零>=30天）
    │
    ├─► data/daily_train.csv  （训练集，~47000行，59个SKU）
    ├─► data/daily_test.csv   （测试集，3540行，59个SKU）
    └─► data/sku_list.csv     （SKU列表，59行）
```

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
   - quantity, ppc_fee, sessions, ppc_clicks, ppc_ad_order_quantity, ppc_fee_sp/sb/sbv/sd, ppc_sales, ppc_impression, target_costs, actual_costs → sum
   - discount_rate, conversion_rate, ppc_fee_rate → mean

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

## 广告数据消融实验

在新数据（含广告分渠道）上测试了8组协变量组合，结论：

| 实验 | 准确率 | 说明 |
|------|--------|------|
| **基线(V2)** | **70.5%** | 7个未来协变量 + 4个历史协变量 |
| E1: sp+sb替代ppc_fee | 70.5% | 分渠道替代总费用，持平 |
| E2: +ppc_sales历史 | 70.5% | 加广告销售额，持平 |
| E4: +ppc_sales+impression | 70.5% | 加两个历史，持平 |
| V3: 全量广告字段 | 70.1% | 12个未来+8个历史，略降 |
| E5: 精简版 | 69.9% | 去掉discount_rate和conversion_rate，下降 |

**结论**：当前 7+4 协变量配置已是最优，广告分渠道数据对 Chronos-2 无增益。

- 广告分渠道(sp/sb/sbv/sd)与总ppc_fee线性冗余
- ppc_sales/ppc_impression与sessions/ppc_clicks高度相关
- 加入更多协变量反而引入噪声，对高广告投入SKU准确率下降明显
