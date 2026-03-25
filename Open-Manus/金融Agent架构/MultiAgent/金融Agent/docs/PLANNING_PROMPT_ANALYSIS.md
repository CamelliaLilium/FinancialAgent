# Planning Agent Prompt 分析报告

基于 benchmark 235730 的 log，对比 Planning 实际产出的 plan 与 prompt 规则，分析**未生效规则**与**冗长/重复**部分。

---

## 一、实际 Plan 与 Prompt 规则对照

### 1.1 明确违反但未生效的规则

| 规则 (prompt 原文) | 违反样例 | 实际产出 |
|-------------------|----------|----------|
| **Interest Coverage = EBIT/Interest Expense (numerator MUST be EBIT)** | easy-test-0 | `extract total, interest_expense` → total/interest_expense |
| **Minimal extraction—do NOT copy OCR table headers** | easy-test-1 | 提取 10+ 列：interest_expense, welfare_expense, sales_tax_rate... |
| **VARIABLE consistency: Use lowercase_with_underscores** | easy-test-20, 29 | `北方特种能源集团`, `易善力`, `单片价格/元` 等中文/混合 |
| **Extraction BEFORE calculation** (隐含于 "Complete extraction") | easy-test-3 | Step 0: [finance] calculate... Step 1: [multimodal] extract... |
| **"second half" = Q3+Q4** (Disambiguate) | easy-test-2 | total_share_second_half = Q2 + Q3 + Q4_increased |
| **Match question to formula** (cash flow hedges ≠ other_income) | easy-test-32 | 提取 other_income_2011/2010，题目问 cash flow hedges |

### 1.2 规则被部分遵守

| 规则 | 遵守情况 |
|------|----------|
| [multimodal]/[finance] 标签 | 全部 plan 均有 |
| 多图分步 | easy-test-18, 22, 29 等正确分步 |
| 无图用 [finance] | easy-test-33 正确用 [finance] extract from text |
| output as 格式 | 多数遵守，easy-test-29 格式混乱 ("output as ... for 诺华") |

### 1.3 规则未被触发的场景

- **"for [entity]" 禁止**：当前 log 中已无 `for [entity]`，多为 `for 诺华` 或省略
- **NO IMAGES = NO [multimodal]**：仅 easy-test-33 无图，已遵守
- **Do NOT try to call multimodal/finance as tools**：无违反（模型未尝试调用）

---

## 二、Prompt 冗长与重复分析

### 2.1 结构概览

```
SYSTEM_PROMPT (~47 行)
├── Intro (9 行)
├── EXECUTOR ASSIGNMENT (9 条)
├── EXPLICIT DATA & CALCULATION (4 条)
├── EXTRACTION & FORMULA GUIDANCE (8 条)
├── CRITICAL RULES (6 条)
└── Available tools (3 行)

NEXT_STEP_PROMPT (~15 行)
└── DECISION RULES (10 条)
```

### 2.2 重复内容

| 主题 | 出现位置 | 重复度 |
|------|----------|--------|
| multimodal vs finance 分配 | SYSTEM 14-20 行, NEXT_STEP 1-3 | 高 |
| 无图不用 [multimodal] | SYSTEM 17, NEXT_STEP 2 | 完全重复 |
| 多图分步 | SYSTEM 21, 27, NEXT_STEP 4 | 高 |
| Multi-entity / for X | SYSTEM 22, 25, NEXT_STEP 5 | 中 |
| 显式 formula + variables | SYSTEM 26, 31, 32, NEXT_STEP 8 | 高 |
| NEVER pre-calculate | SYSTEM 40, NEXT_STEP 7 | 完全重复 |

### 2.3 冗长表述

| 位置 | 原文 | 问题 |
|------|------|------|
| SYSTEM L5 | `The initial directory is: {directory}` | Planning 阶段未使用 directory，与规划无关 |
| SYSTEM L9-10 | `In plan-synthesis phase, produce a clear plan draft and do NOT call state tools. State persistence and step status updates are controlled by PlanningFlow` | 过细的实现细节，模型不关心 |
| SYSTEM L18 | `Match executor to data source: Assign based on where the data lives—visual structure → [multimodal]; text narrative or computation → [finance].` | 与 L14-16 语义重叠 |
| SYSTEM L19-20 | `Every computation step MUST have [finance]` + `EVERY step MUST have [executor_key]` | 可合并为一条 |
| SYSTEM L44 | `[multimodal] and [finance] are STEP EXECUTORS, not tools. You only have two tools: planning and terminate. Do NOT try to call "multimodal" or "finance" as tools—they are labels you put in step text. The flow will route each step to the right executor automatically.` | 约 60 字，可压缩为 "Step labels [multimodal]/[finance] are routing hints, not tools." |
| NEXT_STEP L54-55 | `1. If image/table/chart data needed AND images are provided → assign [multimodal]... 2. If NO images provided → do NOT use [multimodal]` | 与 SYSTEM 完全重复 |
| NEXT_STEP L62-63 | `9. TASK SUCCESS = IMMEDIATE TERMINATE` + `10. NEVER REPEAT TOOL CALLS` | 执行阶段规则，与「生成 plan」关系弱 |

### 2.4 与 Flow 注入的重复

`_build_plan_prompt` 已注入：

- `**Planning rules:** Extraction: image + variables + output order... Multi-image: one [multimodal] step per image...`
- `**Images provided:** Extraction from charts/tables → [multimodal]. Computation → [finance].`
- `**Executors (use these exact keys):** [multimodal], [finance]. EVERY step MUST start with [executor_key]`

与 SYSTEM 的 EXECUTOR ASSIGNMENT、EXPLICIT DATA 高度重叠，Planning 实际收到**双份**同类规则。

---

## 三、未发挥作用的原因假设

1. **规则过多**：单次生成需兼顾 30+ 条规则，关键规则（如 EBIT、minimal extraction）被稀释
2. **位置靠后**：Interest Coverage、Minimal extraction 在 EXTRACTION & FORMULA GUIDANCE 中段，可能被后续内容覆盖
3. **表述抽象**：如 "do NOT copy OCR table headers" 未给出反例（如 easy-test-1 的错误 pattern）
4. **无强制结构**：未要求「先列出公式所需变量，再写 extraction steps」，导致 step 顺序错乱
5. **NEXT_STEP 与 plan 生成脱节**：NEXT_STEP 多针对「调用 tool」阶段，对「首次生成 plan」的约束不足

---

## 四、精简与强化建议

### 4.1 可删除/合并

- 删除 `The initial directory is: {directory}`（Planning 无关）
- 合并 L19-20 为：`Every step: [multimodal] or [finance]. Computation → [finance].`
- 将 L44 的 60 字压缩为 15 字
- NEXT_STEP 中与 SYSTEM 重复的 1-5 条可删，仅保留 6-10（merge、terminate、no repeat）

### 4.2 可前置与强化

- **Interest Coverage**：移至 EXPLICIT DATA 首条，并加反例：`WRONG: total/interest_expense. RIGHT: ebit/interest_expense`
- **Minimal extraction**：加反例：`WRONG: extract interest_expense, welfare_expense, ... (10 cols). RIGHT: extract selling_expenses_2018, selling_expenses_2019`
- **Step order**：新增一条：`Extraction steps MUST come before calculation steps that use their outputs.`

### 4.3 与 Flow 解耦

- Flow 的 Planning rules 已覆盖 executor、multi-image、extraction 格式
- 建议 SYSTEM 聚焦：**领域公式**（Interest Coverage、cash paid、second half）、**minimal extraction**、**variable 命名**（lowercase_or_chinese_consistent）、**step 顺序**
- 将「executor 分配」「多图规则」等流程性内容主要放在 Flow 注入，减少 SYSTEM 重复

---

## 五、总结

| 类型 | 数量/占比 | 建议 |
|------|-----------|------|
| 未生效规则 | 6+ 条 | 前置、加反例、压缩周围噪音 |
| 重复内容 | ~40% | 合并 SYSTEM 与 NEXT_STEP，与 Flow 解耦 |
| 冗长表述 | ~15 行 | 删除无关、压缩 60 字→15 字级 |
| 有效规则 | 多图分步、无图规则、output 格式 | 保留，可迁至 Flow |

**核心结论**：当前 prompt 存在明显冗长与重复，关键规则（EBIT、minimal extraction、step order）被稀释且缺反例，导致实际 plan 多次违反。建议做**结构性精简**并**强化少数高杠杆规则**。
