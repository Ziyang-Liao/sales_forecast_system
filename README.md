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
