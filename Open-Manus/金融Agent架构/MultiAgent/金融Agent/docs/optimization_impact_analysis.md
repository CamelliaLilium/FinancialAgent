# 优化方案影响分析

**拟实施优化**：
1. **Planning Agent**：强调对所有关键数据的提取（不遗漏公式所需变量）
2. **执行 Agent（Multimodal、Finance）**：强调对特殊符号及单位的关注

**分析目标**：评估这两项优化能否解决 10 个失败样例中的部分问题。

---

## 一、逐案映射

| 样例 | 根因 | 优化1（Planning 全量提取） | 优化2（符号/单位） | 预期效果 |
|------|------|---------------------------|-------------------|----------|
| easy-test-0 | Planning 只要求 Interest Expense，漏 EBIT | ✅ **直接解决** | - | 高 |
| easy-test-2 | Q3/Q4 数值 + 「2%」含义理解 | ⚠️ 部分 | ⚠️ 部分 | 中 |
| easy-test-6 | Finance 变量顺序映射颠倒 | ❌ 不直接 | ❌ 不直接 | 低 |
| easy-test-14 | EMPTY_OUTPUT（代码 `\n` 转义） | ❌ 不直接 | ❌ 不直接 | 无 |
| easy-test-17 | 3.375% 未换算为 0.03375 | - | ✅ **直接解决** | 高 |
| easy-test-11 | increase/decrease 方向搞反 | ⚠️ 部分 | ⚠️ 部分 | 中 |
| easy-test-18 | 多图提取 None + EMPTY_OUTPUT | ⚠️ 部分 | ❌ 不直接 | 低 |
| easy-test-22 | revenue 与 purchase price 混同 | ⚠️ 部分 | - | 中 |
| easy-test-33 | 差值单位/口径（-3.2 vs -5000） | ⚠️ 部分 | ⚠️ 部分 | 不确定 |
| easy-test-36 | Multimodal 输出 None | ⚠️ 部分 | ❌ 不直接 | 低 |

---

## 二、详细分析

### 2.1 优化 1：Planning 强调「所有关键数据提取」

**可明显改善**：
- **easy-test-0**：若 Planning 明确要求「提取 EBIT 和 Interest Expense 两个变量」，Multimodal 会同时提取 984 和 97，问题可解决。

**可能改善**：
- **easy-test-11**：若 Planning 在提取步骤中明确列出「Cost of goods sold、Decrease in inventory、Increase in accounts payable」并注明方向（增/减），可减少 Finance 映射错误。
- **easy-test-18**：若 Planning 强调「从 image 1 提取 2010 Fair Value，从 image 2 提取 2011 Net sales」，可减少遗漏；但 image 1 中可能确实无该数据，改善有限。
- **easy-test-22**：若 Planning 明确「Step 0 提取 total purchase price（image 1），Step 1 提取 revenue（image 2），二者不同」，可降低混同；但当前 Planning 已分步，问题主要在 Multimodal 提取。
- **easy-test-36**：若 Planning 明确「必须从图中提取 2024 和 2022 归母净利润的具体数值」，可强化要求；但根因是 Multimodal 未成功解析图像，改善有限。

**难以改善**：
- **easy-test-6**：Planning 已正确，问题在 Finance 对 Step 0 输出的顺序理解。
- **easy-test-14**：根因是代码格式/转义，与 Planning 无关。
- **easy-test-33**：无图任务，Planning 已分配 [finance] 提取，问题在数值或口径理解。

---

### 2.2 优化 2：执行 Agent 强调「特殊符号与单位」

**可明显改善**：
- **easy-test-17**：若 Multimodal/Finance 明确「百分比需换算：3.375% → 0.03375」，可避免 750×3.375 的错误，直接修复。

**可能改善**：
- **easy-test-2**：若强调「goes up by 2%」需区分「2 个百分点」与「×1.02」，可减少公式理解错误。
- **easy-test-11**：若强调「Increase/Decrease 决定符号方向，不可互换」，可减少 Finance 的映射错误。
- **easy-test-33**：若强调「差值可能为比例或归一化值，注意单位」，可能改善；但 gold -3.2 的具体口径需进一步确认。

**难以改善**：
- **easy-test-0, 6, 14, 18, 22, 36**：根因不在单位/符号，优化 2 作用有限。

---

## 三、预期收益估算

| 优化 | 直接可解决 | 可能改善 | 难以改善 | 预估可修复 |
|------|------------|----------|----------|------------|
| 优化 1（Planning 全量提取） | 1 (easy-test-0) | 4 (11, 18, 22, 36) | 5 | 1–2 个 |
| 优化 2（符号/单位） | 1 (easy-test-17) | 3 (2, 11, 33) | 6 | 1–2 个 |
| **合计** | **2** | **部分重叠** | - | **约 2–3 个** |

**保守估计**：两项优化叠加，有望修复约 **2–3 个** 失败样例（如 easy-test-0、easy-test-17，以及可能的 easy-test-11 或 easy-test-2）。

---

## 四、优化实施建议

### 4.1 Planning Agent（app/prompt/planning.py 或 flow 的 _build_plan_prompt）

在现有规则中增加或强化：

```
- **Complete extraction**: For any formula (ratio, difference, sum, etc.), the extraction step MUST list EVERY variable the formula needs. Do not split into multiple steps that each extract only one variable when the formula requires two or more. Example: interest coverage = EBIT/Interest Expense → extract BOTH EBIT and Interest Expense in the same step.
- **Variable clarity**: In extraction steps, use explicit names (e.g. "extract backlog_2014 and backlog_2013" not "extract backlog values") so downstream agents can map outputs correctly.
```

### 4.2 执行 Agent（Multimodal、Finance 的 system prompt 或 step_prompt）

在现有规则中增加或强化：

```
- **Units and percentages**: When a value is given as X%, use X/100 (e.g. 3.375% → 0.03375) in calculations. Do not use the raw percentage number (3.375) as a multiplier.
- **Direction and sign**: Pay attention to "Increase" vs "Decrease", "later - earlier" vs "earlier - later". The sign of the variable matters for the formula.
- **2% and similar**: "goes up by 2%" can mean (a) add 2 percentage points, or (b) multiply by 1.02. Infer from context (share of annual sales → often percentage points).
```

### 4.3 建议不依赖本次优化的部分

- **easy-test-6**：需改进 `_format_structured_previous_output` 或 Step 输出格式，使 Finance 能明确区分 backlog_2014 与 backlog_2013。
- **easy-test-14, easy-test-18**：需在 `python_execute` 中处理字面量 `\n` 转义，避免 EMPTY_OUTPUT。
- **easy-test-18**：若 benchmark 只加载 1 张图，需检查多图加载与路由逻辑。

---

## 五、结论

| 问题 | 结论 |
|------|------|
| 两项优化能否解决部分问题？ | **可以**，预计可修复约 2–3 个失败样例。 |
| 优化 1 最有效场景 | Planning 漏掉公式所需变量（如 easy-test-0）。 |
| 优化 2 最有效场景 | 百分比、单位、符号方向理解错误（如 easy-test-17、easy-test-11）。 |
| 无法覆盖的问题 | 变量顺序映射、代码转义、多图加载、图像解析能力等，需单独处理。 |

建议优先实施这两项优化，再通过 benchmark 验证实际收益，并针对剩余失败样例做针对性修复。
