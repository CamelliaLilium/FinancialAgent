# GHI EBITDA 纯文本任务失败根因分析

## 一、问题现象

| 项目 | 正确值 | 实际输出 |
|------|--------|----------|
| 合并净利润 | 12.68 亿元 | 120.50 亿元（虚构） |
| 所得税费用 | 3.22 亿元 | 30.25 亿元（虚构） |
| 利息支出 | 1.58 亿元 | 15.75 亿元（虚构） |
| 固定资产折旧 | 4.65 亿元 | 10.00 亿元（虚构） |
| 无形资产摊销 | 1.26 亿元 | 5.50 亿元（虚构） |
| **EBITDA** | **23.39 亿元** | **182.00 亿元** |

Finance 完全未从用户提供的财报文本中提取数值，而是编造了一组与原文无关的数字。

---

## 二、根因结论

**纯文本任务下，用户原始请求 (input_text) 从未被传入 Finance 的 step_prompt。**

因此 Finance 在 Step 0 时：
- 看不到 GHI 财报原文（含 12.68、3.22、1.58、4.65、1.26）
- 只能根据“提取合并净利润、所得税费用…”等指令凭空生成数值
- 导致 120.50、30.25 等完全虚构的结果

---

## 三、数据流追踪

### 3.1 用户输入路径

```
run_flow.py: prompt = input("Enter your prompt: ")
         → flow.execute(prompt, base64_images=base64_images)
         → PlanningFlow.execute(input_text=prompt, base64_images=...)
```

用户输入的完整 GHI 财报文本（含所有章节和数值）作为 `input_text` 传入 `PlanningFlow.execute()`。

### 3.2 Planning 阶段

- `_create_initial_plan(input_text)` 使用 `_build_plan_prompt(request)` 构建规划 prompt
- Planning 能看到完整用户请求，因此能正确生成两步计划：
  - Step 0: [finance] 提取合并净利润、所得税费用、利息支出、固定资产折旧、无形资产摊销
  - Step 1: [finance] 应用公式计算 EBITDA

### 3.3 计划存储结构

`WorkflowStateTool` 中 plan 仅存储：

```python
plan = {
    "plan_id", "title", "steps", "step_statuses", "step_notes"
}
```

**不包含用户原始请求 (input_text)**。`_format_plan()` 输出的 plan_status 只有：

- 计划标题（如 "Calculate EBITDA for GHI Group 2023"）
- 进度与步骤列表
- 步骤状态

### 3.4 Finance 执行时收到的内容

`_execute_step()` 构建的 step_prompt（非 multimodal）为：

```python
step_prompt = f"""
    CURRENT PLAN STATUS:
    {plan_status}          # 仅含计划标题、步骤、进度，无财报原文
    {prev_block}           # Step 0 时 previous_output 为空，故为空
    YOUR CURRENT TASK:
    You are now working on step {self.current_step_index}: "{step_text}"

    python_execute supports multi-step: ...
    IMPORTANT: Use numbers ONLY from PREVIOUS STEPS OUTPUT (if present) or from the image/user context. ...
"""
```

关键点：

- `plan_status`：只有计划结构，**没有** GHI 财报原文
- `prev_block`：Step 0 时为空
- `step_text`：如 "[finance] 提取合并净利润、所得税费用、利息支出、固定资产折旧、无形资产摊销"

提示中要求 "Use numbers from image/user context"，但 **user context 从未被注入**。

### 3.5 与 Multimodal 的对比

| 场景 | 数据来源 | 是否传入 executor |
|------|----------|-------------------|
| Multimodal | 图片 (base64) | ✅ 通过 `base64_image` 传入 |
| 纯文本 | 用户文本 (input_text) | ❌ 未传入 step_prompt |

多模态任务中，图片会通过 `base64_image` 传给 MultimodalAgent，模型可从图中提取数值。纯文本任务中，对应“数据源”应为 `input_text`，但 flow 从未将其加入 step_prompt。

---

## 四、为何“以前能做对”？

本次修改中，从 step_prompt 和 finance prompt 中移除了 GHI 相关示例，例如：

- `合并净利润 from 合并净利润12.68亿元`
- `利息支出1.58 vs 无形资产摊销1.26`
- 以及 finance.py 中的具体 GHI 数值示例

**之前的“正确”很可能是：**

1. 这些示例恰好与 GHI 任务一致，模型在缺乏真实上下文时，直接使用了示例中的数值（12.68、1.58 等）
2. 或模型将示例当作“模板”，误把示例当成了可用的数据源

因此，当时并非“从用户上下文正确提取”，而是“示例数值碰巧正确”。移除示例后，模型既无用户原文，也无示例可参考，只能编造数值。

---

## 五、修复方案

**在非 multimodal 步骤的 step_prompt 中显式注入用户原始请求 (input_text)。**

### 5.1 实现思路

1. 在 `execute()` 中保存 `self.input_text = input_text`
2. 在 `_execute_step()` 中，当 `step_type != "multimodal"` 时，在 step_prompt 中增加 `user_context` 块：

```python
user_context_block = ""
if self.input_text and step_type != "multimodal":
    user_context_block = f"""
USER REQUEST / 原始数据（从此处提取数值，勿编造）:
---
{self.input_text.strip()}
---
"""

step_prompt = f"""
    CURRENT PLAN STATUS:
    {plan_status}
    {user_context_block}
    {prev_block}
    YOUR CURRENT TASK:
    ...
"""
```

### 5.2 预期效果

- Finance 在 Step 0 能看到完整 GHI 财报文本
- 可从中正确提取 12.68、3.22、1.58、4.65、1.26
- 计算 EBITDA = 23.39 亿元

### 5.3 与多模态的一致性

- Multimodal：数据来自图片 → 通过 `base64_image` 传入
- 纯文本：数据来自用户文本 → 通过 `user_context_block` 传入

两者都保证 executor 能访问到真实数据源，而不是仅依赖计划结构和步骤描述。

---

## 六、总结

| 问题 | 根因 |
|------|------|
| Finance 提取到虚构数值 | step_prompt 中缺少用户原始请求 |
| 纯文本任务无数据源 | input_text 未传入 _execute_step / step_prompt |
| 以前能做对 | 依赖 prompt 中的 GHI 示例数值，移除后失效 |

**核心结论**：纯文本任务必须将 `input_text` 作为“用户上下文”注入 step_prompt，否则 Finance 无法从任何地方获取真实数值，只能编造。
