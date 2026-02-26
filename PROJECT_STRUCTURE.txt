sales_forecast_system/
│
├── 【核心脚本 — 生产使用】
│   ├── prepare_data.py              # 数据预处理：原始xlsx → 按日聚合 → 补缺失日期 → 促销标记 → SKU筛选 → 输出train/test CSV
│   ├── run_backtest_prod.py         # ★ 生产版回测（推荐）：Chronos-2 + 7个未来协变量 + 4个历史协变量 + 7天滚动预测
│   └── run_backtest_v2.py           # V2回测：和生产版逻辑相同，早期版本保留做对比
│
├── 【实验脚本 — 消融/调参用，不提交到GitHub】
│   ├── run_ablation_ad.py           # 广告数据消融实验：8组协变量排列组合对比（E1~E8）
│   ├── run_backtest_v3_ad.py        # V3回测：加入全量广告分渠道数据（sp/sb/sbv/sd + ppc_sales + impression）
│   ├── run_baseline_newdata.py      # 基线对照：用新数据跑V2逻辑，和V3做公平对比
│   ├── run_strict_comparison.py     # 严格对比实验：区分真正可规划 vs 事后数据
│   ├── analyze_sku_detail.py        # SKU详细分析：逐日打印预测结果+协变量，对比下降/提升SKU
│   ├── run_ablation.py              # 早期消融实验（旧数据）
│   ├── run_two_stage.py             # 两阶段预测实验
│   ├── run_yoy_exp.py               # 去年同期特征实验
│   ├── run_window_exp.py            # 滚动窗口大小实验
│   └── run_mape_optimal.py          # MAPE优化实验
│
├── 【数据目录 — 不提交到GitHub（含生产数据）】
│   └── data/
│       ├── uploads/
│       │   ├── 美国站分层抽样_含广告_脱敏数据_20260210_143426.xlsx   # ★ 最新原始数据（含广告分渠道，77列，73300行，100个SKU）
│       │   └── 美国站分层抽样_脱敏数据_20260203_181815.xlsx          # 旧原始数据（不含广告分渠道）
│       ├── daily_train.csv          # 训练集（截止2025-09-30，由prepare_data.py生成）
│       ├── daily_test.csv           # 测试集（2025-10-01 ~ 2025-11-29，60天）
│       ├── sku_list.csv             # 筛选后的59个SKU列表及统计信息
│       ├── models/                  # 模型文件存放（空）
│       └── exports/                 # 导出文件存放（空）
│
├── 【结果目录 — 不提交到GitHub（含生产数据）】
│   └── results/
│       ├── chronos2_backtest_prod.csv       # 生产版回测结果（旧数据）
│       ├── chronos2_backtest_v2.csv         # V2回测结果（旧数据）
│       ├── chronos2_backtest_v3_ad.csv      # V3回测结果（新数据，含广告分渠道）
│       ├── chronos2_baseline_newdata.csv    # 基线回测结果（新数据，V2逻辑）
│       ├── ablation_ad_results.json         # 8组消融实验汇总（准确率对比）
│       └── ad_ablation_sku_analysis.txt     # ★ SKU详细分析报告（含字段说明、协变量标注、逐日预测对比）
│
├── 【Web应用 — Streamlit前端（早期开发，暂未使用）】
│   └── app/
│       ├── main.py                  # Streamlit主入口
│       └── pages/
│           ├── 1_📊_数据上传.py      # 数据上传页面
│           ├── 2_🎯_模型训练.py      # 模型训练页面
│           ├── 3_📅_计划输入.py      # 运营计划输入页面
│           ├── 4_🔮_销量预测.py      # 销量预测页面
│           └── 5_📈_历史分析.py      # 历史分析页面
│
├── 【模块化代码 — 早期开发，暂未使用】
│   └── src/
│       ├── data/processor.py        # 数据处理模块
│       ├── features/engineer.py     # 特征工程模块
│       ├── models/ensemble.py       # 集成模型模块
│       ├── calibration/calibrator.py # 校准模块
│       └── evaluation/backtester.py # 回测评估模块
│
├── 【文档】
│   ├── README.md                    # ★ 项目说明：模型效果、协变量设计、使用方法、消融实验结论
│   ├── OPTIMIZATION_DOC.md          # 优化方案文档
│   ├── THINKING_INPUT.md            # thinking模型输入
│   ├── thinking_new.md              # thinking分析记录（不提交）
│   └── thinking_suggestions.md      # thinking建议记录（不提交）
│
├── 【配置】
│   ├── .gitignore                   # Git忽略规则（数据、结果、临时脚本、日志等）
│   └── requirements.txt             # Python依赖：chronos-forecasting, torch, pandas, numpy, openpyxl
│
└── 【临时文件 — 不提交】
    ├── ablation_ad_log.txt          # 消融实验运行日志
    ├── baseline_newdata_log.txt     # 基线回测运行日志
    ├── v3_ad_log.txt                # V3回测运行日志
    ├── backtest_60days_detail.csv   # 早期回测详情（临时）
    └── daily_backtest_result.csv    # 早期回测结果（临时）


===== 核心工作流 =====

1. python3.11 prepare_data.py          # 原始xlsx → daily_train.csv + daily_test.csv + sku_list.csv
2. python3.11 run_backtest_prod.py     # 加载Chronos-2 → 59个SKU滚动预测 → 输出结果CSV

===== 协变量配置（最终版） =====

未来协变量(future_df) — 7个，运营可提前规划：
  is_promo, discount_rate, ppc_fee, day_of_week, is_weekend, month, qty_yoy

历史协变量(context) — 4个，用户行为产生：
  sessions, ppc_clicks, ppc_ad_order_quantity, conversion_rate

===== 模型效果 =====

整体准确率: 70.5%  |  >=70%准确率占比: 64.3%  |  59个SKU  |  测试期60天
