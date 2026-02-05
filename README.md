# Amazon 销量预测系统

基于 Chronos-2 + LightGBM/XGBoost 的电商销量预测系统，支持协变量输入（促销标记、折扣率、广告预算等）。

## 特性

- **Chronos-2 协变量预测**: 支持输入未来促销计划、折扣率、广告预算
- **多模型集成**: Chronos-2 + LightGBM + XGBoost
- **SKU级别预测**: 针对每个SKU独立建模
- **回测验证**: 支持历史数据回测，输出每日准确率

## 环境要求

- Python >= 3.10 (Chronos-2 要求)
- CUDA (推荐，CPU也可运行)

## 安装

```bash
pip install -r requirements.txt
```

## 数据格式

### 必须字段

| 字段 | 类型 | 说明 |
|------|------|------|
| purchase_date | string | 日期，格式 YYYYMMDD |
| quantity | int | 销量 |
| sku | string | SKU编码 |

### 推荐字段（提高预测精度）

| 字段 | 类型 | 说明 |
|------|------|------|
| discount_rate | float | 折扣率 (0-1) |
| ppc_fee | float | 广告费用 |
| sessions | int | 流量 |
| sales_amount_total_usd | float | 销售额 |

## 使用方法

### 1. SKU级别预测

```bash
python predict_sku.py --sku SKU001 --days 14 --discount 0.2 --ppc 100 --promo-days "2026-02-10,2026-02-11"
```

### 2. 回测验证

```bash
python backtest.py --mode rolling
python backtest.py --mode promo
```

### 3. Chronos-2 回测（带协变量）

```bash
python backtest_chronos2.py --sku SKU001 --test-start 2025-10-01 --test-end 2025-11-29
```

### 4. Web界面

```bash
streamlit run app/main.py --server.port 8501
```

## 项目结构

```
sales_forecast_system/
├── app/                    # Streamlit Web应用
│   ├── main.py
│   └── pages/
├── src/
│   ├── data/              # 数据处理
│   │   └── processor.py
│   ├── features/          # 特征工程
│   │   └── engineer.py
│   ├── models/            # 预测模型
│   │   └── ensemble.py
│   ├── calibration/       # 促销校准
│   │   └── calibrator.py
│   └── evaluation/        # 回测评估
│       └── backtester.py
├── predict_sku.py         # SKU预测脚本
├── backtest.py            # 回测脚本
├── backtest_chronos2.py   # Chronos-2回测
└── requirements.txt
```

## 模型对比

在60天回测（含黑五促销季）中的表现：

| 模型 | 平均准确率 |
|------|-----------|
| Chronos-1 (纯时序) | ~34% |
| LGB+XGB | ~42% |
| **Chronos-2 (带协变量)** | **~62%** |

## 准确率计算

```
准确率 = 1 - |偏差|
偏差 = (预测值 - 实际值) / 实际值
```

例如：实际100，预测85 → 偏差-15% → 准确率85%

## License

MIT



## 回测结果 (2025-10-01 ~ 2025-11-29)

使用 Chronos-2 模型，输入协变量（促销标记、折扣率、广告费）预测每日销量。

共 33 个SKU，1980 条记录。

| SKU | 日期 | 实际 | 预测 | 准确率 | 促销 | 折扣率 | 广告费 |
|-----|------|------|------|--------|------|--------|--------|
| SKU_01 | 2025-10-01 | 43 | 46 | 89.30000305175781% | 0 | 8.8% | 69 |
| SKU_01 | 2025-10-02 | 55 | 42 | 69.5999984741211% | 0 | 8.2% | 39 |
| SKU_01 | 2025-10-03 | 36 | 41 | 76.4000015258789% | 0 | 9.5% | 41 |
| SKU_01 | 2025-10-04 | 29 | 41 | 25.399999618530277% | 0 | 12.0% | 54 |
| SKU_01 | 2025-10-05 | 47 | 43 | 88.0% | 0 | 12.1% | 63 |
| SKU_01 | 2025-10-06 | 31 | 45 | 23.6% | 0 | 9.6% | 72 |
| SKU_01 | 2025-10-07 | 155 | 139 | 89.0% | 0 | 6.7% | 272 |
| SKU_01 | 2025-10-08 | 119 | 149 | 71.8% | 0 | 6.8% | 278 |
| SKU_01 | 2025-10-09 | 65 | 47 | 64.6% | 0 | 10.4% | 57 |
| SKU_01 | 2025-10-10 | 32 | 43 | 40.9% | 0 | 11.3% | 45 |
