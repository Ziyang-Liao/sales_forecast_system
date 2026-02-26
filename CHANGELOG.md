# Changelog

本项目的所有重要变更记录。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

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
