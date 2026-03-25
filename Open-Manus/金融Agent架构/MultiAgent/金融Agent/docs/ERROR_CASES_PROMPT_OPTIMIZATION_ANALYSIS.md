# 错误样本与 Prompt 优化可行性分析

基于 215115 与 191306 的对比，逐样本分析根因及**是否可通过 prompt 优化提升正确率**。**不修改代码**，仅做分析。

---

## 一、错误样本总览

| 样本 | gold | pred | 责任阶段 | Prompt 可优化? |
|------|------|------|----------|----------------|
| easy-test-2 | 56 | 47.49 | Planning + Finance | ✓ 高 |
| easy-test-14 | 0.04 | 0.00 | Multimodal | ✓ 高 |
| easy-test-17 | 25.3125 | 25 | **Planning** | ✓ 高 |
| easy-test-18 | 10514 | 0 | Multimodal (代码格式) | △ 低 |
| easy-test-22 | 202.73 | - | Multimodal (视觉) | △ 中 |
| easy-test-29 | 8919078 | 183219 | **Planning + Multimodal** | ✓ 高 |
| easy-test-33 | -3.2 | - | 无图/数据缺失 | ✗ |
| easy-test-36 | 31.29 | - | Multimodal (视觉) | △ 中 |

---

## 二、逐样本深度分析

### 2.1 easy-test-17（25 vs 25.3125）— **Planning 主导**

#### 根因

| 运行 | Planning 的 plan | 结果 |
|------|------------------|------|
| **215115** | `[finance] Extract ... from text context (stated as $25 million per year)` | 错：直接用文本 25 |
| **191306** | `[multimodal] Extract maturity_amount and coupon_rate from '3.375% Notes due 2022' in the image table`<br>`[finance] Compute annual_interest = maturity_amount * coupon_rate` | 对：750×0.03375=25.3125 |

215115 的 Planning 将任务分给 Finance，并写明「from text context」「stated as $25 million」，导致完全依赖文本近似值，未用表格精确计算。

#### Prompt 优化方向

**Planning prompt** 增加规则：

- 当题目涉及「利息支出」「annual interest」等，且存在表格（maturity、rate 等列）时，**优先用 [multimodal] 从表格提取** maturity_amount、coupon_rate，再由 [finance] 计算。
- 文本中的「$X million」多为近似值；精确值应来自表格的 amount×rate。

**预期**：可恢复 easy-test-17 正确。

---

### 2.2 easy-test-29（183219 vs 8919078）— **Planning + Multimodal**

#### 根因

| 运行 | Planning 的 plan | Multimodal 输出 |
|------|------------------|-----------------|
| **215115** | `Extract single_price, single_cycle_cost, patent_year from the table image` | 整表 4 行 `[217,70.9,...]` |
| **191306** | `extract ... from the image table **for 诺华 (Novartis) drug**` | 单行 70.9, 4466.7, 2029 |

215115 的 plan 未限定「单药/单行」，Multimodal 整表 dump；191306 明确「for 诺华 drug」，Multimodal 只取一行。

#### Prompt 优化方向

**Planning prompt** 增加规则：

- 题目出现「该药物」「该产品」「the drug」等单实体时，extraction 步骤必须写明「for ONE row/entity」或「for [具体实体名]」，禁止整表输出。

**Multimodal prompt**（PLAN EXECUTION）补充：

- 当任务要求「该药物」「单一行」时，只提取该行，不要输出整表数组。

**预期**：可恢复 easy-test-29 正确。

---

### 2.3 easy-test-2（47.49% vs 56%）— **Planning 公式错误**

#### 根因

Planning 的 step 为：`new_q4_share = q4_share + 2`。

- 题目：「Q4 goes up by 2%」→ 应为 `q4_share * 1.02`，不是 `q4_share + 2`。
- 正确公式：`total_second_half = q3_share + (q4_share * 1.02)`。

#### Prompt 优化方向

**Planning 或 Finance prompt** 增加规则：

- 「X goes up by Y%」→ 使用 `X * (1 + Y/100)`，不要用 `X + Y`。
- 可加示例：`"2% increase" → multiply by 1.02`。

**预期**：有机会修正 easy-test-2（gold 56 的计算方式需再确认）。

---

### 2.4 easy-test-14（0.00 vs 0.04）— **Multimodal 单位**

#### 根因

- Multimodal 提取：`net_profit_2024_q3 = 26.9`（原始显示值）。
- 表中「26.9」应为「26.9 亿」→ 需 `26.9 * 1e8 = 2,690,000,000`。
- 当前 UNIT HANDLING 有「亿→*100000000」，但未在此处生效。

#### Prompt 优化方向

**Multimodal prompt**（UNIT HANDLING）补充：

- 对「净利润」「归母净利润」等常见以「亿」为单位的指标，若表中仅写数字，按「亿」处理：`26.9 → 26.9*100000000`。
- 可加示例：`净利润 26.9 → 2690000000`。

**预期**：有机会修正 easy-test-14。

---

### 2.5 easy-test-18（0 vs 10514）— **代码格式**

#### 根因

Multimodal 代码：`corporate_notes_bonds_2010 = 9300  # Source: '...'  print(...)`  
`print` 与 `#` 注释同行，被注释掉，observation 为空，Finance 得到 0。

#### Prompt 优化方向

**Multimodal prompt**（CRITICAL RULES 或代码规范）补充：

- `print()` 必须单独一行，不要与 `#` 注释写在同一行。

**预期**：可修复 easy-test-18，属低风险改动。

---

### 2.6 easy-test-22、easy-test-36 — **视觉提取失败**

#### 根因

- easy-test-22：Multimodal 输出 `revenue_2016 = None`，未从图中找到 revenue 与 total_purchase_price。
- easy-test-36：Multimodal 输出 `None None`，归母净利润未识别。

可能原因：图表布局复杂、字段名不匹配、多图对应关系不清。

#### Prompt 优化方向

**Multimodal prompt** 可尝试：

- 若首次提取为 None，提示「检查不同行/列、别名（如 revenue/income）及多图对应关系」。
- 效果不确定，可能更多依赖模型能力或图像预处理。

**预期**：有提升可能，但优先级低于 17、29、2、14、18。

---

### 2.7 easy-test-33 — **无图**

无图、数据缺失，与 prompt 无关。

---

## 三、Prompt 优化优先级与预期收益

| 优先级 | 样本 | 优化对象 | 预期收益 | 风险 |
|--------|------|----------|----------|------|
| **P0** | easy-test-17 | Planning | 高 | 低 |
| **P0** | easy-test-29 | Planning + Multimodal | 高 | 低 |
| **P1** | easy-test-2 | Planning/Finance | 中 | 低 |
| **P1** | easy-test-14 | Multimodal | 中 | 低 |
| **P2** | easy-test-18 | Multimodal（代码格式） | 高 | 极低 |
| **P3** | easy-test-22, 36 | Multimodal | 不确定 | 中 |

---

## 四、具体 Prompt 修改建议（供讨论）

### 4.1 Planning

```
# 新增规则（建议插入 CRITICAL RULES）：

- TABLE vs TEXT: When both image/table and text contain a value (e.g. interest expense), 
  prefer [multimodal] extraction from the TABLE for precise calculation (e.g. maturity×rate). 
  Text like "$25 million" may be rounded; use table when available.

- SINGLE ENTITY: When the question refers to "该药物" (the drug), "该产品", or "the X", 
  the extraction step MUST specify "for ONE row/entity" or "for [entity name]". 
  Do NOT extract the entire table as arrays.

- PERCENTAGE INCREASE: "X goes up by Y%" means X * (1 + Y/100), NOT X + Y.
```

### 4.2 Multimodal

```
# UNIT HANDLING 补充：

- For 净利润/归母净利润 in 亿: if the table shows a number without explicit unit, 
  assume 亿 and multiply by 100000000 (e.g. 26.9 → 2690000000).

# CRITICAL RULES 或代码规范补充：

- Put print() on its own line. Never write print() on the same line as a # comment.

# PLAN EXECUTION 补充（可选）：

- When the task asks for "该药物" or a single entity, extract ONE row only—do not output full table arrays.
```

### 4.3 Finance

```
# 可选补充（若公式理解错误多）：

- "X goes up by Y%" → use X * (1 + Y/100), not X + Y.
```

---

## 五、小结

| 结论 | 说明 |
|------|------|
| **可优化样本** | 17、29、2、14、18 共 5 个，其中 17、29、18 把握较大 |
| **主要改动点** | Planning：table vs text、单实体、百分比公式；Multimodal：单位、代码格式、单行提取 |
| **不建议** | 过度约束（如 211726 的 OUTPUT 限制）易导致正确率下降 |
| **建议顺序** | 先做 P0（17、29），再 P1（2、14）、P2（18），最后视情况尝试 P3（22、36） |
