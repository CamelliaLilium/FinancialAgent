# 性能回退分析：20260315_202522

## 一、指标对比

| 指标 | 20260315_191306（改前） | 20260315_202522（改后） | 变化 |
|------|-------------------------|-------------------------|------|
| 正确率 | 50% (8/16) | **31.25% (5/16)** | **-18.75%** |
| 平均耗时 | 110.8s | 139.2s | +25.6% |
| P95 耗时 | 270.9s | 226.4s | -16.4% |
| Numeric MAE | 2382 | **1399524** | **+586x** |

---

## 二、正确样本变化

| 样本 | 191306 | 202522 |
|------|--------|--------|
| easy-test-0 | ✓ | ✗ |
| easy-test-1 | ✓ | ✓ |
| easy-test-3 | ✓ | ✓ |
| easy-test-10 | ✓ | ✓ |
| easy-test-15 | ✓ | ✓ |
| easy-test-17 | ✓ | ✓ |
| easy-test-29 | ✓ | ✗ |
| easy-test-194 | ✓ | ✗ |

**新错 3 个**：easy-test-0, easy-test-29, easy-test-194

---

## 三、回退根因分析

### 3.1 easy-test-0：Finance 公式错误

**题目**：Interest coverage ratio for 2018

**Gold**：EBIT / Interest = 984 / 97 = 10.14

**本次**：Finance 使用 (984+97)/97 = 11.14

**根因**：Finance 将 Interest Coverage 误算为 (Income Before Taxes + Interest) / Interest，正确应为 EBIT / Interest。Multimodal 提取 984、97 正确，问题在 Finance 的公式理解，**与 Multimodal prompt 修改无直接关系**，可能为随机波动。

---

### 3.2 easy-test-29：Multimodal 过度提取

**题目**：诺华药物 (单片价格 - 单周期费用) × 专利到期年份

**Gold**：70.9, 4466.7, 2029 → (4466.7-70.9)×2029 = 8919078

**本次**：Multimodal 输出整表多行：
```
[217, 70.9, 69.79, 205] [4557, 4466.7, 3908.2, 4305] [2023, 2029, 2029, 2034]
```
Finance 取 4557, 70.9, 2029 → 9102297

**根因**：「No explanation—tool calls only」可能促使 Multimodal 只输出代码、不做筛选，直接 dump 整表多行，未按 plan 只提取诺华一行。Finance 从列表中取错列（4557 而非 4466.7）。

---

### 3.3 easy-test-194：Multimodal 混行提取

**题目**：October 2018 回购股份总价值

**Gold**：90000 × 149.28 = 13435200

**本次**：Multimodal 提取 `90000 149.28 335000 159.35`（混入 November 的 335000、159.35），Finance 算出 66817450

**根因**：Multimodal 未按「仅 October」约束提取，混入多行/多列。精简 prompt 后可能弱化「按 plan 指定行列/时间筛选」的约束。

---

## 四、与 Multimodal prompt 修改的关联

| 修改点 | 可能影响 |
|--------|----------|
| "Execute the plan step ONLY" | 未强调「只提取 plan 指定项」，易过度提取 |
| "Output tool calls ONLY—no prose" | 减少解释，可能削弱「先筛选再赋值」的推理 |
| step_prompt 缩短为 "No explanation—tool calls only" | 丢失「Map labels by meaning」「Use only numbers you see」等约束 |

**结论**：prompt 精简后，对「按 plan 指定行列/实体筛选」「只提取题目要求的那一行」的约束变弱，导致 easy-test-29、easy-test-194 出现过度或混行提取。

---

## 五、建议（供后续修改参考）

1. **恢复 step_prompt 中的关键约束**：保留「Map labels by meaning」「Use only numbers you see in the image」，并明确「Extract ONLY the row/entity requested by the plan (e.g. 诺华 for Novartis)」。
2. **在 SYSTEM_PROMPT 中补充**：「Extract only the variables listed in the plan step. For tables with multiple rows, extract the row matching the plan (e.g. company name, period). Do not dump entire columns.」
3. **easy-test-0**：属 Finance 公式错误，需在 Finance prompt 中强化 Interest Coverage = EBIT/Interest 的公式定义。
4. **回滚选项**：若需快速恢复，可先回滚 Multimodal prompt 至修改前版本，再按上述方向做更小步的增量修改。
