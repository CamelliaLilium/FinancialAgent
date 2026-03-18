# 单Agent（Finance）Base测试科研评估报告（中文）

## 1. 研究目的

本实验用于建立与多智能体 base 的可比对照基线，核心问题如下：

1. 在 FinBen 简单分类任务上，单Agent是否在准确率与效率上更优？
2. 当前误差主要来自语义判别能力，还是流程工具调用干扰？
3. 该单Agent结果是否可作为后续金融Agent迭代的可靠基准线？

## 2. 实验设置（可复现）

- Agent：`finance`
- 数据集：`Dataset/Finben/finben_test.json`
- 测试范围：全量（496）
- 超时：600 秒/样本
- 架构模式：单Agent直执行（不经过 PlanningFlow）
- 工件留存：`predictions.jsonl`、`failure_cases.jsonl`、`summary.json`、`logs/`

## 3. 关键指标

- Accuracy：0.5625
- Macro-F1：0.4407
- Correct/Total：279 / 496
- Avg latency/sample：26.52 秒

与多智能体 base（Accuracy=0.4980，Avg latency=34.308s）相比：

- 单Agent准确率更高
- 单Agent时延更低

结论：在“短文本三分类”任务上，单Agent基线更具成本效益。

## 4. 错误证据与构成

- `reasoning_or_prompt_misalignment`: 214（主误差）
- `same_args_repeated`: 26
- `label_extraction_ambiguity`: 21
- `tool_execution_error`: 21
- `repeated_tool_loop`: 15
- `timeout`: 3
- `output_format_error`: 3

解读：主要问题仍是“语义判别偏差”，而非系统级崩溃；流程层误差虽较少，但仍有优化价值。

## 5. 科研有效性评估

### 5.1 内部效度（Internal Validity）

- 优点：全链路日志、工具参数、预测输出完整留痕，可审计性较高。
- 风险：在线推理存在随机性与网络抖动。
- 建议：进行 3~5 次重复试验，报告均值、标准差、95%CI。

### 5.2 外部效度（External Validity）

- 该结果适用于“短金融语句分类”场景；
- 对复杂金融任务（多文档推理、报表计算、策略生成）的外推有限，需要单独验证。

### 5.3 构念效度（Construct Validity）

- Accuracy 与 Macro-F1 已覆盖终点表现；
- 但建议同时纳入过程指标（工具调用率、无效调用率、超时率、token成本）作为系统级质量衡量。

## 6. 作为基线的价值判断

该单Agent实验具备以下基线价值：

1. 提供了可复现、可审计的“低编排复杂度”参照；
2. 与多智能体实验形成清晰对照，便于分析“架构收益区间”；
3. 为后续复杂任务扩展提供了性能下界与效率上界的双参照。

## 7. 后续实验建议

1. 做严格消融：  
   单Agent vs 多Agent（无planning）vs 多Agent（有planning）。

2. 做任务分层评估：  
   简单分类 / 中等抽取 / 复杂多步推理，分别比较架构收益。

3. 做输出契约强化评估：  
   严格“仅输出标签”与当前输出方式对比，量化提取歧义误差影响。
