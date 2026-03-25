# EBITDA vs EBIT 混淆根因分析与污染检查

## 1. 问题现象（easy-test-0）

| 来源 | 使用的指标 | 数值 |
|------|------------|------|
| **Ground truth** (python_solution) | **EBIT** | 984 |
| **Planning 输出** | EBITDA | 要求提取 EBITDA |
| **Multimodal 执行** | EBITDA（自算） | 1019 |
| **Finance 计算** | 1019/97 | 10.51 ≠ 10.14 |

---

## 2. 根因分析

### 2.1 财务概念区分

| 指标 | 全称 | 公式 | 常见用途 |
|------|------|------|----------|
| **EBIT** | Earnings Before Interest and Taxes | 营业利润 / Operating income / Income before income taxes | **Interest coverage ratio 标准定义**：EBIT / Interest |
| **EBITDA** | Earnings Before Interest, Taxes, Depreciation and Amortization | EBIT + D&A，或从收入逐项扣减 | 信贷契约、杠杆比率（Debt/EBITDA） |

- **Interest coverage ratio** 在 GAAP 和多数教材中定义为 **EBIT / Interest expense**
- 实务中部分信贷契约使用 **EBITDA-based interest coverage**，但 FinMMR 数据集 gold 明确使用 EBIT

### 2.2 为何模型倾向 EBITDA？

1. **Prompt 示例污染**：Planning 多处使用 `e.g. EBITDA=X+Y+Z` 作为公式示例，模型易将「公式计算」与 EBITDA 强关联
2. **预训练知识**：信贷分析、杠杆比率中 EBITDA 出现频率高，模型可能泛化到 interest coverage
3. **缺乏显式约束**：此前无「interest coverage 必须用 EBIT」的规则，模型自由选择

---

## 3. 污染源检查

### 3.1 代码库中的 EBITDA/EBIT 提及

| 文件 | 内容 | 污染类型 |
|------|------|----------|
| `app/flow/planning.py` L296 | `e.g. EBITDA=X+Y+Z` | **公式示例**：易让 Planning 在 interest coverage 场景联想到 EBITDA |
| `app/flow/planning.py` L384 | 同上 | 同上 |
| `app/flow/planning.py` L792 | `e.g. EBITDA`（最终答案呈现） | 轻度：仅影响摘要，已改为 `e.g. ratio, percentage` |
| `app/flow/planning.py` L599 | `Operating income for EBIT` | **正向**：Multimodal 步骤 prompt 已提示 EBIT 映射 |
| `app/prompt/planning.py` | 无 EBITDA | 无污染 |
| `app/prompt/multimodal.py` | `Operating income for EBIT` | 正向，已加强 interest coverage → EBIT |
| `Dataset/finmmr_easy_smoke_test.json` | `EBIT = 984`（gold） | **正确**：数据无污染 |

### 3.2 数据集检查

- **finmmr_easy_smoke_test.json**：easy-test-0 的 `python_solution` 明确使用 `EBIT = 984`
- **bizbench_test.json**：部分样本 context 含 "EBITDA"（信贷契约描述），为真实业务文本，非 prompt 污染

### 3.3 结论

- **主要污染**：Planning 的 `EBITDA=X+Y+Z` 公式示例，在 interest coverage 场景下被错误泛化
- **数据无污染**：gold 正确使用 EBIT
- **已做修复**：在 Planning、Multimodal、flow 中增加「interest coverage → EBIT」的显式规则

---

## 4. 已实施的修复

1. **Planning prompt**：增加规则「For interest coverage ratio: use EBIT, NOT EBITDA」
2. **Multimodal prompt**：增加「For interest coverage ratio: extract EBIT, NOT EBITDA. Read directly from table」
3. **Flow step_prompt**：Multimodal 步骤 prompt 中增加相同约束
4. **公式示例保留**：`EBITDA=X+Y+Z` 仍作为通用公式示例（适用于 GHI EBITDA 等任务），但通过 interest coverage 专项规则避免误用
5. **最终答案示例**：`e.g. EBITDA` 改为 `e.g. ratio, percentage`，减少不必要关联

---

## 5. 后续建议

- 若其他指标（如 second half = Q3+Q4）出现类似混淆，可建立**金融术语知识库**，在 Planning 中按问题类型注入对应规则
- 定期检查新增 prompt 中是否引入新的概念污染
