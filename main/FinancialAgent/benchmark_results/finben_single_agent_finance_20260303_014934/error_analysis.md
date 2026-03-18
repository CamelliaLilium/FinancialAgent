# FinBen 单Agent（Finance）错误分析报告（中文）

## 1. 结果总览

- Agent：`finance`
- 样本总数：496
- 正确数：279
- Accuracy：0.5625
- Macro-F1：0.4407
- 平均耗时：26.52 秒/样本
- 标签提取首尾不一致：21

与此前多智能体 base（Accuracy=0.4980，平均耗时=34.308s）相比：

- 准确率提升约 **+6.45 个百分点**
- 平均时延下降约 **7.79 秒/样本**

这说明在 FinBen 这类简单三分类任务上，单Agent流程具有更优的“性能-效率”比。

## 2. 错误分布与主要问题

### 2.1 错误标签统计

- `reasoning_or_prompt_misalignment`: 214（主导误差）
- `same_args_repeated`: 26
- `label_extraction_ambiguity`: 21
- `tool_execution_error`: 21
- `repeated_tool_loop`: 15
- `timeout`: 3
- `output_format_error`: 3

### 2.2 核心诊断

1. **语义判别偏差仍是首要瓶颈**  
   最大误差来源不是工具异常，而是模型对 hawkish/dovish/neutral 语义边界的判定偏移。

2. **neutral 偏置明显**  
   从混淆矩阵可见：  
   - `dovish -> neutral` 为 101  
   - `hawkish -> neutral` 为 80  
   模型在边界样本上偏向输出中性标签。

3. **存在轻度流程冗余**  
   尽管是单Agent，仍有较多工具调用（尤其 `python_execute` 与 `terminate`），对简单文本分类任务属于非必要开销。

## 3. 工具使用画像

- `terminate`: 488
- `python_execute`: 229
- `str_replace_editor`: 16

解释：本任务本质上是“直接分类输出”，理论上多数样本不需要工具。  
当前工具调用分布显示执行策略仍偏“工具优先”，存在进一步精简空间。

## 4. 混淆矩阵（证据）

```json
{
  "dovish": {
    "neutral": 101,
    "dovish": 22,
    "hawkish": 5,
    "none": 1
  },
  "neutral": {
    "neutral": 226,
    "hawkish": 11,
    "dovish": 8
  },
  "hawkish": {
    "dovish": 9,
    "neutral": 80,
    "hawkish": 31,
    "none": 2
  }
}
```

按类别看，`neutral` 的召回最高，而 `dovish` 和 `hawkish` 召回明显偏低，进一步支持“中性偏置”判断。

## 5. 错因分层（科研归因）

### A. 语义层（主要）

- 现象：大量边界样本被误判为 neutral。
- 原因：标签判别规则虽明确，但模型在“风险描述/历史陈述/前瞻措辞”上缺乏稳定映射。

### B. 输出层（次要）

- 现象：`label_extraction_ambiguity` 与 `output_format_error` 存在。
- 原因：模型并非总是“只输出标签”，导致提取阶段有信息损失风险。

### C. 流程层（次要）

- 现象：重复调用、循环、少量 timeout。
- 原因：任务简单但流程策略仍保留工具化执行习惯。

## 6. 本轮结论

- 单Agent在该任务上的整体表现优于多智能体工作流。  
- 但性能上限仍受“语义判别偏差”限制。  
- 后续优化重点应优先放在：  
  1) 分类判别口径稳定化  
  2) 输出格式约束（严格单标签）  
  3) 简单任务下的工具调用抑制策略
