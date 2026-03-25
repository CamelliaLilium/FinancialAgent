# Planning Agent 问题深度分析

基于 `multi_agent_multi_dataset_20260316_195613` 测试日志与错误样例的详细分析。

---

## 一、问题分类概览

Planning Agent 的问题可分为三类：

| 类型 | 样本数 | 典型表现 |
|------|--------|----------|
| **计划内容错误** | 8+ | 金融概念混淆、语义理解错误、步骤设计不当 |
| **步骤标签缺失** | 3+ | 计算步骤无 `[finance]` 标签 |
| **工具误用** | 2 | 将 `[multimodal]` 当作可调用工具 |

---

## 二、计划内容错误（Plan Quality）

### 2.1 金融概念混淆

**easy-test-0**：Interest Coverage Ratio

| 项目 | 计划内容 | 标准答案 |
|------|----------|----------|
| 计划 Step 0 | `[multimodal] extract EBITDA and Interest Expense for 2018` | — |
| 计划 Step 1 | `[finance] calculate interest coverage ratio = EBITDA / Interest Expense` | — |
| 正确公式 | — | EBIT / Interest Expense = 984/97 = 10.14 |

**根因**：Planning 将 Interest Coverage Ratio 误用为 EBITDA/Interest Expense。标准定义应为 **EBIT**（Operating Income）除以 Interest Expense。计划中的错误概念直接导致 Multimodal 提取 EBITDA 并计算，得到 18.9 而非 10.14。

---

### 2.2 时间/范围语义理解错误

**easy-test-2**：second half of the year

| 项目 | 计划内容 | 标准答案 |
|------|----------|----------|
| 计划 Step 0 | `[multimodal] extract the share of annual sales for Q2, Q3, and Q4` | — |
| 计划 Step 1 | `[finance] calculate new Q4 share * 1.02, then sum Q2, Q3, and new Q4` | — |
| 正确理解 | second half = Q3 + Q4 | Q3=25, Q4=29, +2% → 25+31=56 |
| 实际执行 | Multimodal 提取 Q2=22, Q3=24, Q4=25 | 22+24+25.5=71.5（错误） |

**根因**：
1. **范围错误**：second half 应为 Q3+Q4，计划多加了 Q2。
2. **数值错误**：Multimodal 提取的 Q2/Q3/Q4 与标准答案 (Q3=25, Q4=29) 不符，可能表格列/行对应错误。
3. **公式偏差**：标准答案为 `Q3 + Q4 + 2`（Q4 增加 2 个百分点），计划写成了 `Q2+Q3+Q4*1.02`。

---

### 2.3 数据来源混淆（Text vs Image）

**easy-test-3**：unrecognized compensation cost

| 项目 | 计划内容 | 实际数据来源 |
|------|----------|--------------|
| 计划 Step 0 | `[multimodal] Extract the value of unrecognized compensation cost... as of December 31, 2007` | — |
| 文本 context | "as of december 31, 2007, there was **$37 million** of unrecognized compensation cost..." | 正确来源 |
| 图像表格 | Restricted Stock Awards 表中 "Unvested at December 31, 2007" = 1,527,831 | 错误来源 |
| 标准答案 | 37 / 1.4 = 26.429 | — |

**根因**：题目明确要求的是文本中的 "$37 million"，计划却让 [multimodal] 从图像表格提取。Multimodal 提取了 1527831，Finance 最终从文本用了 37，但计划本身未区分「文本数据」与「表格数据」的来源。

**easy-test-17**：annual interest expense for 2022 notes

| 项目 | 计划内容 | 实际数据来源 |
|------|----------|--------------|
| 计划 Step 0 | `[multimodal] Extract the annual interest expense for the 2022 notes from the provided text context` | — |
| 矛盾 | "from the provided **text** context" 却分配 [multimodal] | 逻辑矛盾 |
| 文本 | "interest on the 2022 notes of approximately **$25 million per year**" | 正确来源 |

**根因**：计划同时写了 "text context" 和 [multimodal]，混淆了数据来源与执行器。文本数据应直接由 [finance] 提取。

---

### 2.4 多步骤拆分不当

**easy-test-15**：shares repurchased in November and December 2004

| 项目 | 计划内容 | 标准答案 |
|------|----------|----------|
| 计划 Step 0 | `[multimodal] Extract shares for 11/01/04 - 11/30/04` | Nov = 5145 |
| 计划 Step 1 | `[multimodal] Extract shares for 12/01/04 - 12/31/04` | Dec = 34526 |
| 计划 Step 2 | `[finance] Sum the extracted values` | 5145+34526=39671 |
| 实际执行 | Step 0: 5145 ✓；Step 1: 34526 ✓；但 Step 1 被重复执行，且 Finance 误用 34526+34526=69052 | — |

**根因**：
1. **拆分合理**：同一表不同行，拆成两步在逻辑上可接受。
2. **执行问题**：Multimodal Step 1 重复调用，且 Finance 将两期数值混淆（可能 previous_output 结构不清晰）。
3. **计划可优化**：可合并为一步 `[multimodal] extract shares_purchased_nov and shares_purchased_dec from the table (rows 11/01-11/30 and 12/01-12/31)`，减少步骤间传递错误。

**easy-test-18**：Corporate notes 2010 + Net sales 2011（双图）

| 项目 | 计划内容 | 执行结果 |
|------|----------|----------|
| 计划 Step 0 | `[multimodal] extract Corporate notes and bonds 2010 Fair Value from image 1` | 返回 None（image 1 无该数据） |
| 计划 Step 1 | `[multimodal] extract Net sales of 2011 from image 2` | 返回 8846 |
| 计划 Step 2 | `[finance] calculate sum` | 误用 8846+8846=17692 |
| 标准答案 | 10514 | — |

**根因**：
1. **多图路由**：计划正确区分 image 1 与 image 2，但 Flow 当前只传 `base64_images[0]`，多图场景下 Step 0 和 Step 1 可能收到同一张图。
2. **步骤对应**：Finance 收到的 previous_output 未明确标注 "Step 0: xxx, Step 1: xxx"，导致误用同一数值两次。

---

### 2.5 公式/语义歧义

**easy-test-29**：诺华药物 (单片价格 - 单周期费用) × 专利到期年份

| 项目 | 计划/执行 | 标准答案 |
|------|-----------|----------|
| 计划公式 | `(单片价格 - 单周期费用) × 核心化合物专利到期时间` | — |
| Finance 计算 | (70.9 - 4466.7) × 2029 = -8919078 | — |
| 标准答案 | 8919078 | (4466.7 - 70.9) × 2029 |

**根因**："差值" 的符号有歧义。题目可能指 `|单周期费用 - 单片价格|` 或 `单周期费用 - 单片价格`（单周期费用更高）。计划未消除歧义，Finance 按字面理解为 (a - b)，得到负值。

**easy-test-14**：净利润/PE 比值再除以 PE

| 项目 | 计划/执行 | 标准答案 |
|------|-----------|----------|
| 题目 | (2024Q3净利润 / 2025预测PE) / 2026预测PE，保留两位小数 | 0.04 |
| Multimodal 提取 | 26.9, 430.9, 319.1 | — |
| 标准答案数据 | 334.5, 110.9, 83.9 | — |
| Finance 计算 | (26.9/430.9)/319.1 ≈ 0.00019 → 0.00 | — |

**根因**：Multimodal 提取的数值与标准答案不一致（可能表格结构/单位不同）。计划未明确变量名与表格列的对应关系，也未强调单位（如亿、万）。

---

### 2.6 无图场景未考虑

**easy-test-33**：税收减免与中小企业投入

| 项目 | 计划内容 | 实际输入 |
|------|----------|----------|
| 计划 Step 0 | `[multimodal] Extract 1998年4月税收减免, 1998年11月税收减免, 1999年11月中小企业投入` | — |
| 实际 | image_ref_count=0，无图像 | Step blocked |
| 结果 | Planning 仍分配 [multimodal]，Flow 因无图 block 该步 | — |

**根因**：题目可能依赖表格/图像，但输入中无图像。Planning 未考虑「无图时是否应改用 [finance] 从文本提取」或「明确标注数据缺失」。

---

### 2.7 步骤描述过于笼统

**easy-test-36**：归母净利润比例

| 项目 | 计划内容 | 执行结果 |
|------|----------|----------|
| 计划 Step 0 | `[multimodal] 从图中提取2024年预计的归母净利润和2022年的归母净利润数值` | Multimodal 返回 None, None |
| 计划 Step 1 | `使用公式 (2024年预计归母净利润 / 2022年归母净利润) * 100% 计算比例并保留两位小数` | **无 [finance] 标签** |
| step_type | Step 1 解析为 None（无 [finance]） | 回退到 Finance 执行 |

**根因**：
1. **变量名缺失**：计划未给出具体变量名（如 net_profit_2024, net_profit_2022），Multimodal 可能不知如何命名。
2. **Step 1 无标签**：计算步骤未标 [finance]，依赖系统回退。

---

## 三、步骤标签缺失

| 样本 | 步骤文本 | 问题 |
|------|----------|------|
| easy-test-1 | "Add the extracted selling expenses for 2018 and 2019" | 无 [finance]，step_type=None |
| easy-test-36 | "使用公式 (2024年预计归母净利润 / 2022年归母净利润) * 100% 计算比例..." | 无 [finance] |
| easy-test-17 | "Use the extracted value to answer the question directly" | 无 [finance] |

**根因**：Planning 未强制要求每个计算/汇总步骤都带 `[finance]`，导致步骤类型解析失败或依赖默认回退。

---

## 四、工具误用

**easy-test-3、easy-test-17**：

```
Planning selected 1 tools to use
Tools being prepared: ['multimodal']
Tool arguments: {"image": "image_1", "instruction": "Extract..."}
Result: Error: Unknown tool 'multimodal'
```

**根因**：Planning 将 `[multimodal]` 理解为可调用工具。Planning 实际只有 `planning` 和 `terminate`，`[multimodal]` 和 `[finance]` 是步骤文本中的执行器标签，由 Flow 解析并路由。

---

## 五、优化方向总结

### 5.1 Prompt 已做修改

- 金融术语：Interest Coverage Ratio 明确为 EBIT，second half 明确为 Q3+Q4
- 数据来源：文本用 [finance]，图像/表用 [multimodal]
- 步骤标签：计算步骤必须带 [finance]
- 工具澄清：[multimodal]/[finance] 为执行器标签，非可调用工具
- 步骤合并：同一来源的提取合并为一步

### 5.2 待 Flow/架构 支持

1. **多图路由**：计划中 "from image 1" / "from image 2" 时，Flow 需按步骤索引传递对应图像
2. **previous_output 结构化**：按步骤输出 `Step N: var1=val1, var2=val2`，便于 Finance 正确映射
3. **无图处理**：当 multimodal 步骤无图时，考虑回退到 [finance] 从文本提取，或明确提示 Planning 避免分配 [multimodal]

### 5.3 可选增强

1. **公式歧义消除**：对 "差值" 等歧义表述，在计划中明确符号或顺序
2. **变量名显式化**：计划中直接写出变量名，如 `extract net_profit_2024, net_profit_2022`
3. **单位提示**：涉及亿、万、% 时，在计划中注明单位换算规则
