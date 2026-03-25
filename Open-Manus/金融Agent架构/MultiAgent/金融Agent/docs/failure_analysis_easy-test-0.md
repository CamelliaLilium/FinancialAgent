# 失败案例详细分析：easy-test-0

## 1. 样本基本信息

| 字段 | 值 |
|------|-----|
| question_id | easy-test-0 |
| 问题 | What is the interest coverage ratio for the year 2018? |
| 上下文 | `<image 1>`（仅图，无文本） |
| 图片类型 | Tables |
| ground_truth | 10.144329896907216 |
| 预测结果 | 10.51 |
| 错误标签 | same_args_repeated, numeric_mismatch |

---

## 2. 关于「多模态提取任务未分配给多模态 Agent」的核实

**结论：easy-test-0 的分配是正确的。**

| 步骤 | Planning 分配 | 实际执行 | base64_image |
|------|---------------|----------|--------------|
| Step 0 | `[multimodal] extract EBITDA and Interest Expense for 2018 from image 1` | executor=**Multimodal** | ✓ 传入 |
| Step 1 | `[finance] calculate interest coverage ratio = EBITDA / Interest Expense` | executor=**Finance** | 不传入 |

- 图像提取任务已正确分配给 Multimodal Agent
- 计算任务已正确分配给 Finance Agent

---

## 3. 失败根因分析

### 3.1 概念混淆：EBITDA vs EBIT

| 来源 | 使用的指标 | 数值 |
|------|------------|------|
| **Ground truth** (python_solution) | **EBIT** | 984 |
| **Planning 要求** | EBITDA | - |
| **Multimodal 提取** | EBITDA（自算） | 1019 |

- 题目 gold 使用 **EBIT**（984），Planning 却要求提取 **EBITDA**
- Multimodal 按 EBITDA 公式计算得到 1019，与 gold 的 EBIT 984 不一致
- 正确公式：`interest_coverage_ratio = EBIT / interest_expense = 984 / 97 ≈ 10.14`

### 3.2 Multimodal 提取过程

1. **第 1 次调用**：用 Net sales 及各费用项计算 EBITDA = 1019，Interest = 97
2. **第 2 次调用**：注释提到 "Income before income taxes and other items - 984"，但代码仍用 `ebitda_2018 = 1019`
3. **第 3、4 次调用**：输出相同 `1019 97`，被 anti_loop 拦截
4. **第 5 次**：terminate(success)

### 3.3 重复调用（same_args_repeated）

- 第 3、4 次 `python_execute` 与第 2 次语义相同（仅注释不同），被 anti_loop 识别为重复
- 触发 "CRITICAL: You have been blocked 2+ times in a row. You MUST call terminate(status='success') immediately."

### 3.4 Finance 计算

- 使用 Multimodal 输出的 1019、97
- 计算：1019 / 97 = 10.505... ≈ 10.51
- 与 gold 10.14 不符，因上游 EBIT/EBITDA 混用

---

## 4. 失败链路总结

```
Planning 要求 EBITDA
    ↓
Multimodal 计算 EBITDA=1019（应为 EBIT=984）
    ↓
Multimodal 多次重复调用 → anti_loop 拦截 → terminate
    ↓
Finance 用 1019/97 → 10.51 ≠ 10.14
```

---

## 5. 改进建议

1. **Planning prompt**：对 interest coverage ratio 等常见指标，明确使用 EBIT 而非 EBITDA（或根据题目上下文判断）
2. **Multimodal prompt**：强调优先从表中直接读取 "Income before income taxes" 等现成指标，避免自行推导 EBITDA
3. **easy-test-0 本身**：分配逻辑正确，问题主要在指标理解和数值提取

---

## 6. 其他失败案例中的分配问题（供后续分析）

从 log grep 结果可见，存在**真正分配错误**的样本：

| 样本 | Step 0 执行者 | 说明 |
|------|---------------|------|
| **easy-test-3** | Finance | 上下文含图+文本，数据在文本中，用 Finance 合理，且该样本通过 |
| **easy-test-17** | Finance | 上下文含图+文本，Planning 认为数据在文本，用 Finance；gold 需 750×3.375%，Finance 用了近似 25 |
| **easy-test-36** | Multimodal→Finance | Step 1 无 `[finance]` 标签，step_type=None，fallback 到 Finance；Multimodal Step 0 输出 None None 即 terminate |

后续可重点分析：easy-test-17（是否应从图中取精确值）、easy-test-36（Step 1 未打标签 + Multimodal 未真正提取）。
