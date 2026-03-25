# 正确率优化深度分析

基于 `multi_agent_multi_dataset_20260316_210940` 的 failure_cases.jsonl 与具体日志的详细分析。**先分析，不修改代码。**

---

## 一、错误概览

| 错误类型 | 样本数 | 说明 |
|----------|--------|------|
| **same_args_repeated** | 16/16 (100%) | 所有失败样本均存在重复调用 |
| **numeric_mismatch** | 10–12/16 | 数值或公式错误导致答案错误 |
| **准确率** | 37.5% (6/16) | 当前 smoke test 通过率 |

---

## 二、错误分层分析

### 2.1 第一层：重复调用 (same_args_repeated)

#### 2.1.1 成功后的重复（典型：easy-test-0）

**执行序列**：
```
Round 1: python_execute (ebitda=97+15+30+921+1363, interest=97) → 成功 '2426\n97\n'
Round 2: python_execute (ebitda=2426, interest=97) → 成功 '2426 97\n'
Round 3: python_execute (完全相同) → 成功
Round 4: python_execute (完全相同) → AntiLoop 拦截
Round 5: terminate
```

**根因**：模型在 python_execute 成功后不主动 terminate，而是继续重复相同调用，直到被 AntiLoop 拦截。

#### 2.1.2 通过变量名绕过 AntiLoop（典型：easy-test-14）

```
predicted_pe_2025 / predicted_pe_2026  → 被拦截
pe_2025_predicted / pe_2026_predicted  → 成功（不同 hash）
两种写法交替，跑满 max_steps=20
```

**根因**：AntiLoop 基于 `json.dumps(kwargs)` 的 hash，变量名微调即可绕过。

#### 2.1.3 提取失败时的无效重试（典型：easy-test-18）

```
Step 0: corporate_notes_bonds_2010_fair_value = None  → 成功（但值为 None）
Round 3–20: 相同代码 → 被拦截，模型仍不断重试
```

**根因**：模型把「提取失败（None）」当作需要重试，而不是 terminate 结束当前步骤。

#### 2.1.4 失败后不修正代码的重复（典型：easy-test-6）

```
Round 1: python_execute → [EMPTY_OUTPUT] 失败
Round 2–5: 完全相同的代码 → 被 AntiLoop 拦截
Round 6: terminate
```

**根因**：模型未根据错误信息修改代码，而是重复相同调用。

---

### 2.2 第二层：数值/公式错误 (numeric_mismatch)

#### 2.2.1 Planning 金融概念错误（easy-test-0）

| 项目 | 计划/执行 | 标准答案 |
|------|-----------|----------|
| 计划公式 | EBITDA / Interest Expense | — |
| 正确公式 | **EBIT** / Interest Expense | 984/97 = 10.14 |
| 实际输出 | 2426/97 = 25.01 | 错误 |

**根因**：Interest Coverage Ratio 标准定义是 EBIT，Planning 误用 EBITDA。

#### 2.2.2 语义/范围理解错误（easy-test-2）

| 项目 | 计划/执行 | 标准答案 |
|------|-----------|----------|
| 计划 | 提取 Q2, Q3, Q4，Q4+2% 后求和 | — |
| 正确理解 | second half = **Q3 + Q4** | 25+31=56 |
| 实际 | Q2+Q3+Q4*1.02，且提取值错误 | 49.5% |

**根因**：second half 应为 Q3+Q4，计划多加了 Q2；Multimodal 提取的 Q2/Q3/Q4 与标准答案不符。

#### 2.2.3 变量顺序/符号歧义（easy-test-6, easy-test-29）

**easy-test-6**：backlog change 2013–2014
- 标准答案：-1600（18900 - 20500）
- 模型：20500 - 18900 = 1600（方向反了）
- 另：Finance 的 python_execute 报 [EMPTY_OUTPUT]，模型未修正代码

**easy-test-29**：诺华药物 (单片价格 - 单周期费用) × 专利到期年
- 标准答案：8919078（取 |差值| 或 单周期费用 - 单片价格）
- 模型：(70.9 - 4466.7) × 2029 = -8923474（符号歧义）

#### 2.2.4 多步输出传递错误（easy-test-18）

| 步骤 | 计划 | 实际 |
|------|------|------|
| Step 0 | [multimodal] Corporate notes 2010 from image 1 | 返回 None（image 1 可能无该数据） |
| Step 1 | [multimodal] Net sales 2011 from image 2 | 返回 8846 |
| Step 2 | [finance] sum | 误用 8846+8846=17692 |
| 标准答案 | 10514 | — |

**根因**：
1. **多图路由缺陷**：Flow 对所有 multimodal 步骤只传 `base64_images[0]`，Step 0 和 Step 1 收到同一张图
2. **previous_output 结构**：Finance 收到的 `execution_result` 是拼接的原始输出，未明确标注 "Step 0: xxx, Step 1: xxx"，易混淆

#### 2.2.5 提取数值与标准答案不一致（easy-test-14, easy-test-36）

**easy-test-14**：净利润/PE 比值再除以 PE
- Multimodal 提取：26.9, 430.9, 319.1
- 标准答案数据：334.5, 110.9, 83.9
- 可能原因：表格结构/单位/列对应错误

**easy-test-36**：归母净利润比例
- Multimodal 返回：None, None（未从图中正确提取）
- 模型最终用 0.92（来源不明）

---

### 2.3 第三层：架构/流程问题

#### 2.3.1 多图场景不支持

```python
# app/flow/planning.py:584
base64_image = self.base64_images[0] if self.base64_images else None
```

所有 multimodal 步骤均使用 `base64_images[0]`，无法按 "from image 1" / "from image 2" 分配不同图像。

#### 2.3.2 python_execute [EMPTY_OUTPUT] 根因待查（easy-test-6）

代码形式正确：
```python
backlog_at_year_end_2014 = 20500  # Source: '20500'
backlog_at_year_end_2013 = 18900  # Source: '18900'
change_in_backlog = backlog_at_year_end_2014 - backlog_at_year_end_2013
print(change_in_backlog)
```

但触发 EMPTY_OUTPUT。可能方向：
- `_fix_print_after_comment` / `_wrap_last_expression` 的边界情况
- 跨 agent 的 `_global_env` 共享或重置问题
- 需本地复现并加调试日志定位

#### 2.3.3 previous_output 格式不结构化

当前 `execution_result` 为原始对话拼接，例如：
```
Observed output of cmd `python_execute` executed:\n{'observation': '18900 20500\n', 'success': True}\n...
Observed output of cmd `python_execute` executed:\n{'observation': '8846\n', 'success': True}\n...
```

Finance 需自行解析「Step 0 输出」「Step 1 输出」；多步时易混淆顺序或误用同一数值。

---

## 三、优化方向优先级

### P0：立即生效（Prompt + 机制）

| 方向 | 说明 | 预期 |
|------|------|------|
| 成功即 terminate | 已在 prompt 中强化 | 减少重复调用 |
| 禁止重复调用 | 已在 prompt 中强化 | 减少重复调用 |
| AntiLoop 成功即拦截 | `count >= 2` → `count >= 1` | 避免前 2 次重复 |
| 连续拦截强制结束 | 同一 agent 连续 N 次被拦截则自动 terminate | 避免跑满 max_steps |

### P1：短期（架构/流程）

| 方向 | 说明 | 预期 |
|------|------|------|
| 多图路由 | 按步骤索引传递 `base64_images[i]` | 修复 easy-test-18 类多图任务 |
| previous_output 结构化 | 输出 `Step N: var1=val1, var2=val2` | 减少 Finance 误用 |
| EMPTY_OUTPUT 根因 | 排查 python_execute 空输出 | 修复 easy-test-6 类问题 |

### P2：中期（Planning 质量）

| 方向 | 说明 | 预期 |
|------|------|------|
| 金融术语知识库 | Interest Coverage=EBIT、second half=Q3+Q4 等 | 减少概念错误 |
| 数据来源规则 | 文本→[finance]，图像/表→[multimodal] | 减少分配错误 |
| 歧义消除 | 对「差值」等明确符号或顺序 | 减少公式错误 |

### P3：长期（模型与能力）

| 方向 | 说明 | 预期 |
|------|------|------|
| 变量名语义去重 | 对 code 做规范化后再 hash | 减少变量名绕过 |
| 更强模型 | 8B→更大规模 | 提升理解与执行稳定性 |
| 多模态能力 | 表格/图表结构理解 | 提升提取准确率 |

---

## 四、典型样本速查

| 样本 | 主要问题 | 涉及层级 |
|------|----------|----------|
| easy-test-0 | 重复调用 + 公式错误 (EBITDA vs EBIT) | 重复 + Planning |
| easy-test-2 | 重复 + second half 范围错误 | 重复 + Planning |
| easy-test-6 | 重复 + EMPTY_OUTPUT + 公式方向 | 重复 + 工具 + Planning |
| easy-test-11 | 重复 + 数值偏差 (25265 vs 25700) | 重复 + 提取 |
| easy-test-14 | 变量名绕过 + 提取数值错误 | 重复 + 提取 |
| easy-test-18 | 多图同图 + 输出混淆 + 重复 | 架构 + 重复 |
| easy-test-22 | 重复 + 公式/数据理解错误 | 重复 + Planning |
| easy-test-29 | 重复 + 差值符号歧义 | 重复 + Planning |
| easy-test-33 | 无图仍分配 [multimodal] | Planning |
| easy-test-36 | 重复 + 提取 None | 重复 + 提取 |

---

## 五、总结

1. **重复调用**：Prompt 已强化，但需配合 AntiLoop 收紧（成功即拦截、连续拦截强制结束）才能稳定生效。
2. **数值错误**：来自 Planning 概念/公式/范围错误、提取错误、多步传递错误，需分层次改进。
3. **架构缺陷**：多图路由、previous_output 结构、EMPTY_OUTPUT 根因，需在流程层面修复。
4. **优化顺序建议**：先 P0（Prompt、AntiLoop、强制结束），再 P1（多图、结构化输出、EMPTY_OUTPUT），最后 P2/P3。
