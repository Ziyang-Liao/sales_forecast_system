# Changelog

本项目的所有重要变更记录。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

## [v3.1.0] - 2026-03-20

### Added
- 10个模型全面对比实验：Lasso/随机森林/LSTM/SVR/ARIMA/XGBoost/MLP等
- README 新增多模型准确率排行榜

### 实验结论
- 两阶段Chronos-2(69.2%)全面领先所有传统ML和从零训练的深度学习模型
- Lasso回归(52.7%)在传统ML中表现最好，但仍比Chronos-2低16.5%
- LSTM(52.6%)/随机森林(52.3%)/LightGBM(50.9%)表现接近
- ARIMA(49.2%)/XGBoost(42.0%)/MLP(32.8%)表现较差
- 预训练时序基础模型的零样本泛化能力远超从单SKU数据训练的模型

## [v3.0.0] - 2026-03-19

### Added
- `run_backtest_chronos_2stage.py` 两阶段Chronos-2回测脚本（当前最优）
- 两阶段预测方案：先用Chronos-2预测sessions/conversion_rate，再作为协变量辅助销量预测
- 多指标消融实验：测试ppc_clicks/ppc_ad_order_quantity/ppc_impression的增益

### Changed
- 成熟SKU最佳准确率从 66.6% 提升至 **69.2%**（+2.6%）
- >=70%准确率占比从 56.7% 提升至 **59.6%**（+2.9%）
- README.md 更新算法对比、模型效果表格、版本历史、准确率分布等

### 实验结论
- sessions预测准确率79.1%，conversion_rate预测准确率74.6%，可预测性强
- 仅+sessions提升+1.8%（68.4%），+sessions+cr提升+2.6%（69.2%）
- ppc_clicks(40.1%)/ppc_ad_order_quantity(36.2%)/ppc_impression(35.8%)预测准确率低，加入后无增益
- 30天预测(68.9%)与60天预测(69.2%)效果接近，缩短预测期无明显提升

## [v2.1.0] - 2026-02-26

### Added
- `prepare_data.py` 新增广告分渠道字段支持（ppc_fee_sp/sb/sbv/sd, ppc_sales, ppc_impression, target_costs, actual_costs）
- 广告数据消融实验：8组协变量排列组合对比（E1~E8）
- `PROJECT_STRUCTURE.txt` 项目结构说明文档
- `README.md` 补充完整数据文件字段说明、脱敏示例、处理流程图
- `README.md` 补充协变量分类说明（未来协变量 vs 历史协变量）
- `LICENSE` MIT开源协议
- `CHANGELOG.md` 变更日志

### Changed
- `.gitignore` 排除 results/ 目录、临时实验脚本、日志文件

### Removed
- 去掉模拟数据文件

### 实验结论
- 当前 7+4 协变量配置已是最优，广告分渠道数据对 Chronos-2 无增益
- 广告分渠道(sp/sb/sbv/sd)与总ppc_fee线性冗余
- ppc_sales/ppc_impression与sessions/ppc_clicks高度相关

## [v2.0.0] - 2026-02-09

### Added
- `run_backtest_prod.py` 生产版回测脚本（无硬编码，可泛化）
- `run_backtest_v2.py` V2回测脚本（时间特征+滚动预测）
- 时间协变量：day_of_week, is_weekend, month, qty_yoy
- 7天滚动预测机制，每轮用实际值更新context
- `OPTIMIZATION_DOC.md` 优化方案文档
- `THINKING_INPUT.md` thinking模型输入文档

### Changed
- 准确率从 V1 的 66.7% 提升至 **70.5%**
- >=70%准确率占比从 V1 提升至 64.3%

## [v1.0.0] - 2026-02-05

### Added
- 项目初始化
- `prepare_data.py` 数据预处理脚本（xlsx → train/test CSV）
- 基于 Amazon Chronos-2 的时序预测基线
- 3个协变量（is_promo, discount_rate, ppc_fee），一次预测60天
- Streamlit Web应用框架（app/）
- 模块化代码结构（src/）
- 基线准确率 **66.7%**
