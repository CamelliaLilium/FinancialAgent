# 错误样例深度分析报告

**分析批次**：`multi_agent_multi_dataset_20260314_122826`  
**对比基准**：`multi_agent_multi_dataset_20260311_183537`  
**分析日期**：2026-03-14

---

## 一、错因分类框架

本报告将错误根因分为三类：

| 类别 | 说明 | 典型表现 |
|------|------|----------|
| **Planning 问题** | 任务分解、步骤规划、Agent 分配错误 | 步骤遗漏、错误顺序、错误 agent 选择 |
| **模型/数据提取问题** | 多模态/表格/文本理解与数值提取错误 | 取错行列、取错单位、返回 None |
| **python_execute/工具问题** | 代码生成、执行、金融计算场景适配 | 变量名非法、元组解析、精度、符号 |

---

## 二、回退样本 (Regressed) 深度分析

### 2.1 easy-test-11 【公式符号理解错误】

| 项目 | 值 |
|------|-----|
| 题目 | How much cash did Sugarfall Corp. pay to its suppliers? |
| Gold | 25700 |
| Predicted | 26702 |
| 主错误 | same_args_repeated, numeric_mismatch |

**根因**：**模型/公式理解**，非 Planning 或工具问题。

- 正确公式（gold python_solution）：`cash_paid = COGS - decrease_in_inventory - increase_in_AP` → 27264 - 501 - 1063 = 25700
- 模型实际：`COGS + decrease_in_inventory - increase_in_AP` → 27264 + 501 - 1063 = 26702
- **原因**：库存「减少」的符号理解错误。会计上，库存减少意味着现金流出减少，应减去；模型误加。

---

### 2.2 easy-test-41 【回退】

| 项目 | 值 |
|------|-----|
| 题目 | what is the net change in the balance of employee separation... |
| Gold | -1574 |
| Predicted | 2239 - 665 或类似 |
| 主错误 | same_args_repeated, numeric_mismatch |

**根因**：**数据提取/公式**。修改前可能提取正确或计算方式不同，修改后 anti-loop 提前终止或提取路径变化导致错误。

---

### 2.3 easy-test-65, 79, 81, 113, 126, 165, 167, 193, 197 【same_args_repeated 回退】

| 共性 | 说明 |
|------|------|
| 主错误 | same_args_repeated |
| 推测根因 | **Anti-loop 提前 terminate** 导致在模型尚未给出正确最终答案前就结束 |
| 流程变化 | 修改后 terminate 增加、重复调用检测更严格，可能过早截断 |

**建议**：检查 anti-loop 与 same_args 检测的触发条件，避免在「有答案但未完成 synthesize」时提前终止。

---

### 2.4 easy-test-74, 191 【numeric_mismatch 回退】

| 项目 | 说明 |
|------|------|
| 根因 | 数值计算或提取错误，可能是模型随机性或流程变化导致 |
| 类型 | 模型/数据提取 |

---

### 2.5 easy-test-180 【唯一 Timeout 回退】

| 项目 | 值 |
|------|-----|
| 题目 | what was the percent of the change in the total other accrued... |
| Gold | -1.694 |
| 状态 | timeout (450s) |
| 主错误 | timeout |

**根因**：**流程/稳定性**。修改前能在 600s 内完成，修改后可能因：
- 多图/复杂表格导致耗时增加
- 重复调用或 anti-loop 导致无效轮次增多
- 450s 超时更严格

**建议**：单样本 trace 分析，定位耗时瓶颈。

---

## 三、持续失败样本 (Still Wrong) 错因分类

### 3.1 数据提取错误（模型/表格理解）

| 样本 | 题目 | Gold | Predicted | 根因 |
|------|------|------|-----------|------|
| **easy-test-14** | 2024年三季度净利润/2025PE/2026PE | 0.04 | 0.0 | 取错行列：26.9/430.9/319.1 vs 正确 334.5/110.9/83.9 |
| **easy-test-18** | Corporate notes 2010 Fair Value + Net sales 2011 | 10514 | 8846 | 第一次提取 0，第二次只取 8846，未正确求和两值 |
| **easy-test-22** | Rate of return on acquisition | 202.73 | None | 多图表格提取返回 None，导致后续计算失败 |
| **easy-test-36** | 2024归母净利润/2022归母净利润比例 | 31.29 | None | 提取返回 None None，模型未正确解析表格 |

**结论**：多图、多表格、复杂表头时，模型易取错行列或返回 None。

---

### 3.2 公式/符号理解错误

| 样本 | 题目 | Gold | Predicted | 根因 |
|------|------|------|-----------|------|
| **easy-test-11** | Cash paid to suppliers | 25700 | 26702 | 库存减少符号错误（见 2.1） |
| **easy-test-29** | 差值与专利到期年乘积 | 8919078 | -8919078 | 差值符号：应为 weekly_cost - single_dose，模型可能用反 |

---

### 3.3 python_execute 与金融计算场景问题

| 样本 | 题目 | Gold | Predicted | 根因 |
|------|------|------|-----------|------|
| **easy-test-33** | 1998年税收减免与1999年中小企业比较 | -3.2 | (50000,75000) | 模型输出元组，Python 代码未正确处理多值；或需先求和再比较 |
| **easy-test-19** | 德国DAX+印度SENSEX 和×法国CAC40 | 55 | 569840000 | 单位/比例理解错误：可能是归一化后计算，或题目要求不同 |
| **easy-test-20** | 市场份额总和与比较 | 3.3 | 2.0/3.4 | 比较结果格式理解：题目要「比较」结果，可能为比值或差值 |

**python_execute 相关问题**：
- 元组/多值输出：需在 prompt 中明确如何处理 `print((a,b))` 等格式
- 单位与精度：金融场景需保留小数位数、千分位等
- 变量名：避免 `2024_net_profit` 等以数字开头（已修复）

---

### 3.4 Planning 与步骤规划

| 样本 | 可能 Planning 问题 |
|------|-------------------|
| **easy-test-18** | 需两步：先提取 Corporate notes 2010 Fair Value，再提取 Net sales 2011，最后求和。Planning 可能未明确拆分为两步提取 |
| **easy-test-22** | 多图：需明确从哪张图取 acquisition cost、Goodwill、revenue。Planning 可能未指定图与字段 |
| **easy-test-33** | 需提取 1998年4月、11月、1999年11月三组数据。Planning 可能未明确「三个时间点」 |

---

## 四、错因统计汇总

| 错因类别 | Regressed | Still Wrong | 合计 |
|----------|-----------|-------------|------|
| 模型/数据提取 | 2 | 4 | 6 |
| 公式/符号理解 | 1 | 2 | 3 |
| python_execute/金融计算 | 0 | 3 | 3 |
| Planning/步骤规划 | 0 | 3 | 3 |
| Anti-loop/流程提前终止 | 11 | 0 | 11 |


---

## 五、典型错误样例筛选（用于 Smoke Test）

基于上述分析，筛选以下 **6 个** 典型样例加入 `finmmr_easy_smoke_test.json`，覆盖各类错因：

| 样本 | 错因 | 典型性 |
|------|------|--------|
| **easy-test-11** | 公式符号理解（库存减少） | Regressed，会计公式符号易错 |
| **easy-test-18** | 多值提取与求和 | 需两值相加，易只取其一 |
| **easy-test-22** | 多图 None 提取 | 多图表格提取失败 |
| **easy-test-29** | 差值符号 | 顺序影响正负 |
| **easy-test-33** | python_execute 元组/多值 | 金融多值比较 |
| **easy-test-36** | 表格 None 提取 | 归母净利润等复杂表头 |

**说明**：easy-test-14 已在 smoke test 中，easy-test-41 与 180 为回退样本，可后续按需加入。

---

## 六、Smoke Test 更新记录

已将上述 6 个典型样例加入 `Dataset/finmmr/finmmr_easy_smoke_test.json`：

- **easy-test-11**：公式符号理解
- **easy-test-18**：多值提取与求和
- **easy-test-22**：多图 None 提取
- **easy-test-29**：差值符号
- **easy-test-33**：python_execute 元组/多值
- **easy-test-36**：表格 None 提取

Smoke test 样本数：10 → 16。
