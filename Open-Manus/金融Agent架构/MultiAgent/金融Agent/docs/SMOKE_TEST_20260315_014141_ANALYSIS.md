# Smoke Test 20260315_014141 深度分析报告

## 一、测试概览

| 指标 | 20260314_195110 | 20260315_014141 | 变化 |
|------|-----------------|-----------------|------|
| 总样本 | 16 | 16 | - |
| 正确 | 8 | 8 | - |
| **正确率** | **50%** | **50%** | 持平 |
| 平均耗时 | 94.3s | 102.8s | +9% |
| P95 耗时 | 223.2s | 197.9s | -11% |
| Timeout | 1 | **0** | ✓ 消除 |
| Numeric MAE | 595313.65 | 558119.16 | -6% |
| Status | 15 ok, 1 timeout | 16 ok | 全部完成 |

### 主要变化
- **easy-test-22**：由 timeout 变为完成（169.9s），但预测 100% vs gold 202.73%，提取错误
- **easy-test-33**：由编造数值变为「无数据 + terminate failure」，Planning 误分配 [finance] 做提取

### 失败样本（8 个，与上次相同）
easy-test-2, 14, 17, 18, 22, 29, 33, 36

---

## 二、错误样例根因分析

### 1. easy-test-2：单位/语义理解错误

**题目**：Q4 goes up by 2%，求下半年销售占比（%）

**Gold**：Q3=25, Q4=29，增加 2 个百分点 → 25+29+2=56

**本次执行**：
- Planning：`new_q4_share = q4_share + 2`（正确理解为加 2 个百分点）
- Multimodal 提取：**0.24, 0.25**（误将 24%、25% 当作小数）
- Finance：`new_q4_share = 0.25 + 2 = 2.25`，`total = 0.24 + 2.25 = 2.49`
- 最终答案：**47.25%**（模型可能将 2.49 转为 47.25，或存在其他换算）

**根因**：Multimodal 将「Share of annual sales」的 24%、25% 解析为 0.24、0.25，导致后续计算全错。表格中若标注为 "24"、"25"（无 %），模型需根据上下文判断是百分比还是小数。

---

### 2. easy-test-14：单位/行列错误

**题目**：(2024Q3 净利润 / 2025 PE) / 2026 PE，保留两位小数

**Gold**：0.04（正确值约 334.5/110.9/83.9）

**本次执行**：
- Multimodal：`net_profit_2024_q3 = 26.9 * 100000000 = 2690000000`，`pe_2025=430.9`，`pe_2026=319.1`
- 计算：(2690000000/430.9)/319.1 = **19563.61**

**根因**：
1. **单位混用**：26.9 可能为亿元，模型乘以 1e8 得到 26.9 亿，但 PE 比值公式中分子应为与 PE 同量纲的数值（如亿元或元），不应直接 26.9 亿 / PE
2. **取错行列**：正确应为 334.5（或 3.345 亿）、110.9、83.9；模型取到 26.9、430.9、319.1，可能看错行或列

---

### 3. easy-test-17：公式 vs 文本摘抄

**题目**：2022 notes 年利息费用（百万美元）

**Gold**：750 × 3.375% = 25.3125 million

**本次执行**：直接采用文中 "approximately $25 million"，输出 **25**

**根因**：模型优先摘抄文本近似值，未按公式 750×0.03375 计算精确值。

---

### 4. easy-test-18：多图只传一张 + Step0 返回 None

**题目**：Corporate notes/bonds 2010 Fair Value 总和 + Net sales 2011（两图）

**Gold**：1133+9381=10514（或类似求和）

**本次执行**：
- Planning：Step0 从图1 提取 notes+bonds 2010，Step1 从图2 提取 net_sales 2011
- `max_images_per_sample=1`，两 step 均收到**同一张图**（图1）
- Step0：提取失败 → `corp_notes_bonds_fair_value_2010 = None`
- Step1：从同一图提取 → 得到 **8846**（实为 Net sales 或某单一值，非 10514）
- 最终答案：8846（错误地认为 notes+bonds 与 net_sales 均为 8846）

**根因**：
1. 只传 1 张图，无法按 step 区分图1/图2
2. Step0 对图1 的 notes+bonds 提取失败（表结构复杂或行列识别错误）
3. Step1 从同一图取到 8846，可能来自 Net sales 行，但题目要求的是两个不同指标的和

---

### 5. easy-test-22：多图只传一张 + 提取混淆

**题目**：ROI = (Indaba 9M 2016 revenue / total purchase price) × 100%

**Gold**：revenue=2286772，purchase_price=1128004 → 202.73%

**本次执行**：
- Planning：Step0 从图2 提取 revenue，Step1 从图1 提取 purchase_price
- 实际只传 1 张图，两 step 收到**同一张图**
- 两 step 均提取到 **2000000**
- 计算：2000000/2000000×100 = **100%**

**根因**：
1. 只传 1 张图，revenue 与 purchase_price 本应来自不同图，却从同一图提取
2. 可能取到同一单元格或混淆了两个指标，导致两者相同
3. 本次无 timeout，说明流程完成，但数据错误

---

### 6. easy-test-29：python_execute 空输出 + anti-loop 拦截

**题目**：(单周期费用 - 单片价格) × 专利到期年份，取整

**Gold**：(4466.7-70.9)×2029 = 8919078

**本次执行**：
- Multimodal 提取正确：70.9, 4466.7, 2029
- Finance 公式正确：`(single_cycle_cost - single_patient_price) * patent_expiry_year`
- **python_execute 返回 observation 为空**（无 print 输出）
- Finance 重复相同代码 3 次，均空输出，第 3 次被 anti-loop 拦截
- 最终答案：**82153**（来源不明，可能为模型在无输出时的推断）

**根因**：
1. **python_execute 空输出**：可能原因包括 sandbox 输出截断、print 未正确捕获、或执行环境异常
2. **82153 的来源**：4395.8×18.69≈82153，可能将 2029 误解析为 18.69 或类似值，或为模型在无观测时的错误推断
3. **anti-loop**：在 observation 为空时仍视为「成功」，导致重复调用被拦截，无法通过重试获得正确输出

---

### 7. easy-test-33：无图 + Planning 误分配 [finance] 做提取

**题目**：1998年4月/11月税收减免总和 与 1999年11月中小企业投入 的差值

**Gold**：4.6+6-7.4 = -3.2

**本次执行**：
- `has_image_refs: false`（images 为空，未用 ground_images）
- **Planning 全部分配给 [finance]**：Step0「Extract 三个数值」、Step1 求和、Step2 求差
- [finance] 无法看图，Step0 输出 `tax_reduction_april_1998=0, tax_reduction_november_1998=0, tax_investment_november_1999=0`，并注释 "values not provided"
- 后续 step 全部 `terminate(status=failure)`
- 最终答案：无法计算

**根因**：
1. **数据层**：images 为空，benchmark 未回退 ground_images，导致无图传入
2. **Planning 错误**：题目明确需从表格提取，却将提取任务分配给 [finance]，违反「有图必用 [multimodal]」的规则
3. 与 20260314 相比：上次 Planning 正确用了 [multimodal]，但因无图而编造数值；本次 Planning 直接跳过 multimodal，导致更早失败

---

### 8. easy-test-36：Multimodal 提取失败

**题目**：2024E 归母净利润 / 2022 归母净利润，保留两位小数

**Gold**：1123/3589×100 ≈ 31.29%

**本次执行**：
- Multimodal：`2024_year_net_profit = None`，`2022_year_net_profit = None`
- Finance：无法计算，输出 "Data not found"
- 最终答案：无法计算

**根因**：Multimodal 未能从图中识别 2024E、2022 年的归母净利润（可能表头/行列对应错误，或变量名与表格不一致）。

---

## 三、与 20260314 对比

| 样本 | 20260314 | 20260315 |
|------|----------|----------|
| easy-test-2 | 24,25 → 49% | 0.24,0.25 → 47.25% |
| easy-test-14 | 26.9,430.9,319.1 → 0.00 | 26.9×1e8,430.9,319.1 → 19563.61 |
| easy-test-17 | 25（同） | 25（同） |
| easy-test-18 | Step0 None，Step1 8846 | Step0 None，Step1 8846（同） |
| easy-test-22 | **Timeout** | 完成，100%（提取错误） |
| easy-test-29 | -8919078（符号错） | 82153（空输出+推断） |
| easy-test-33 | 编造 12500/18750/25000 | 无图+[finance]提取→全 failure |
| easy-test-36 | None + Data not found | None + Data not found（同） |

---

## 四、问题分类汇总

| 类别 | 样本 | 根因 |
|------|------|------|
| **数据/配置** | 33 | images 空、未用 ground_images；Planning 误分配 [finance] 做提取 |
| **多图** | 18, 22 | max_images_per_sample=1，只传 1 张图；Flow 无法按 step 传不同图 |
| **单位/行列** | 2, 14 | 百分比 vs 小数；亿元 vs 元；取错行列 |
| **公式 vs 摘抄** | 17 | 用文中近似值而非公式计算 |
| **Multimodal 提取** | 18, 36 | 表结构复杂或行列识别失败，返回 None |
| **python_execute** | 29 | 输出为空，anti-loop 拦截重试 |

---

## 五、优化方向建议（不修改代码，仅讨论）

1. **easy-test-33**：修复 images/ground_images 解析；强化 Planning prompt「有图必 multimodal 提取」
2. **easy-test-18/22**：提高 max_images_per_sample；Flow 支持按 step 索引传图
3. **easy-test-2**：Multimodal prompt 明确「Share of annual sales」为百分比，输出 24/25 而非 0.24/0.25
4. **easy-test-14**：Multimodal 明确单位（亿/元）及 PE 公式中分子量纲
5. **easy-test-17**：Finance prompt 强调「approximately」时需按公式验证
6. **easy-test-29**：排查 python_execute 空输出原因；anti-loop 对「observation 为空」是否应允许重试
7. **easy-test-36**：Multimodal 加强表头/行列对应，失败时要求根据错误修改提取逻辑
