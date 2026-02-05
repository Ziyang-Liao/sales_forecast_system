# 销量预测系统开发沟通记录

## 项目概述
- 目标：基于历史销售数据预测SKU级别每日销量
- 数据：S3下载的Excel文件，73,300条记录，100个SKU，1,129天数据
- 核心要求：数据不能上传外网，本地处理

## 模型演进

### 1. 初始方案：LightGBM + XGBoost
- 准确率：~42%

### 2. Chronos-1（纯时序）
- 准确率：~34%
- 问题：不支持协变量

### 3. Chronos-2（带协变量）- 当前最佳方案
- 准确率：~71.3%（SKU平均）
- 协变量：is_promo（促销标记）、discount_rate（折扣率）、ppc_fee（广告费）
- Python 3.11 环境（Chronos-2要求>=3.10）

## 回测结果分析

### 每日准确率分布（1980条记录）
| 区间 | 数量 | 占比 |
|------|------|------|
| = 0% | 69条 | 3.5% |
| 0-50% | 246条 | 12.4% |
| 50-70% | 400条 | 20.2% |
| >= 70% | 1265条 | 63.9% |

### 准确率=0的问题
- 100%都是高估（预测值是实际值的2-3倍）
- 主要集中在促销期和高波动SKU

### 低准确率SKU特征分析
| 因素 | 相关性 | 结论 |
|------|--------|------|
| 变异系数(CV) | -0.388 | **最重要**，波动越大准确率越低 |
| 历史数据天数 | +0.325 | 新品数据不足难预测 |
| 日均销量 | +0.106 | 影响较小 |

### 优化优先级（按影响×占比）
1. **高波动(CV>1.2)**：影响+5%，占比24%，优先级最高
2. 低销量(<20/天)：影响+12.9%，占比9%
3. 新品(<300天)：影响+9.3%，占比6%

## 基线对比
| 方法 | 平均准确率 |
|------|-----------|
| Chronos-2 | 71.3% |
| 30天均值 | 61.3% |
| 7天均值 | 61.8% |

Chronos-2比基线高+10%，33个SKU中胜31个。

## 校准方案探索

### 问题根源
模型对高波动SKU系统性高估，后处理校准效果有限。

### 尝试的方案
1. V1-V3：基于历史参考值校准 - 效果不明显
2. V4：历史同期×变异系数 - 效果变差
3. **待实现V5**：预测上限 = 历史同期 × 变异系数

### 用户提出的校准思路
- 对于波动大的SKU，按日期和促销以及折扣率给出变异系数
- 如果波动大的日期跟预估日期接近或吻合，系数应该一致
- 结合折扣率和促销日适当调整
- 对于波动不大的SKU，用均值的变异系数

## 代码仓库
- GitHub: https://github.com/Ziyang-Liao/sales_forecast_system (已设为私有)
- 数据已脱敏：SKU重命名(H0093→SKU_A等)，数值偏移(+13, 折扣率+5%, 广告费-28)

## 文件结构
```
sales_forecast_system/
├── app/                    # Streamlit Web应用
├── src/
│   ├── data/processor.py   # 数据处理
│   ├── features/engineer.py # 特征工程
│   ├── models/ensemble.py  # 预测模型
│   ├── calibration/calibrator.py # 促销校准
│   └── evaluation/backtester.py  # 回测评估
├── results/
│   ├── chronos2_backtest_full.csv    # 完整回测结果(33 SKU, 1980条)
│   └── chronos2_calibrated_v*.csv    # 校准版本结果
├── predict_sku.py          # SKU预测脚本
├── backtest.py             # 回测脚本
├── backtest_chronos2.py    # Chronos-2回测脚本
└── README.md               # 包含回测结果表格
```

## 准确率计算公式
```
准确率 = max(0, 1 - |偏差|)
偏差 = (预测值 - 实际值) / 实际值
```

## Chronos-2 API用法
```python
from chronos import Chronos2Pipeline
pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
pred_df = pipeline.predict_df(
    context_df,  # 历史数据：id, timestamp, target, 协变量
    future_df=future_df,  # 未来协变量（无target）
    prediction_length=60,
    quantile_levels=[0.5],
    id_column='id',
    timestamp_column='timestamp',
    target='target',
)
```

## 下一步工作
1. 完成V5校准方案测试
2. 针对高波动SKU的专项优化
3. 考虑分位数预测(P30/P50/P70)给业务更多决策空间

## 关键决策记录
- 2026-02-05: 确认不加变异系数作为协变量（对波动大的SKU有负面影响）
- 2026-02-05: 确认当前71.3%准确率方案可行，60%SKU准确率>=70%
- 2026-02-05: 校准方案持续优化中，目标是减少准确率=0的极端情况
