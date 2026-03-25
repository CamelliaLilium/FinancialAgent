# 金融计算优化与 python_execute 工具改进讨论

本文档基于 smoke test 错误分析，讨论如何通过优化 python_execute 及相关流程提升金融计算正确率。**暂不修改代码，仅作方案讨论。**

---

## 一、当前架构与问题定位

### 1.1 现有 python_execute 能力

| 能力 | 实现 | 状态 |
|------|------|------|
| 变量持久化 | 共享 `_global_env`，Multimodal/Finance 跨 step 复用 | ✓ 正常 |
| 输出捕获 | `sys.stdout` 重定向到 StringIO | 偶发空输出 |
| 裸表达式自动 print | `_wrap_last_expression()` | ✓ 正常 |
| 非法缩进修复 | `_normalize_indent()` | ✓ 正常 |
| 变量名 sanitize | `_sanitize_variable_names()` (2024_net_profit → net_profit_2024) | 可能破坏语义 |
| 超时控制 | timeout 参数预留，未实现 | 无 |

### 1.2 错误样例中与 python_execute 直接相关的根因

| 样本 | 根因 | 归属 |
|------|------|------|
| easy-test-2 | Multimodal 提取 0.24/0.25 而非 24/25 | **Multimodal 提取**，非 python_execute |
| easy-test-14 | 单位混用 26.9×1e8 / PE | **Multimodal 提取 + 单位理解** |
| easy-test-17 | 用文中 "约 25" 而非公式计算 | **Finance 公式选择**，非 python_execute |
| easy-test-18 | Step0 None，Step1 8846，未正确求和 | **多图 + 提取** |
| easy-test-22 | 两值均 2000000，提取混淆 | **多图 + 提取** |
| **easy-test-29** | **observation 为空 + anti-loop 拦截** | **python_execute + anti_loop** |
| easy-test-33 | 无图 + Planning 误分配 | 数据/Planning |
| easy-test-36 | Multimodal 返回 None | Multimodal 提取 |

**结论**：8 个失败样本中，**仅 easy-test-29 的根因直接与 python_execute 实现相关**（空输出 + 重试被拦截）。其余主要为 Multimodal 提取、多图、Planning、单位理解等问题。但 python_execute 作为「金融计算最后一环」，其可靠性、可观测性、与金融语义的契合度，会放大上游错误并影响整体正确率。

---

## 二、核心问题拆解

### 2.1 空输出问题（easy-test-29）

**现象**：`(4466.7-70.9)*2029` 公式正确，Multimodal 提取正确，但 `observation` 为空，`success=True`。

**可能原因**：
1. **stdout 捕获边界**：子线程、异步、或某些库内部 print 可能绕过重定向
2. **输出缓冲**：极少数情况下 flush 未触发
3. **变量名 sanitize 副作用**：若代码中有 `result` 等被误改（当前规则不应影响），需排查
4. **exec 成功但 print 未执行**：如代码被截断、换行解析错误等
5. **大数/精度**：8919078 在 Python 中为普通 int，不应有问题

**优化方向**：
- 执行后显式 `flush` stdout
- 增加「执行成功但 observation 为空」的检测，返回 `success=False` 或特殊标记，触发 anti-loop 的「失败重试」逻辑
- 可选：执行后从 `_global_env` 读取 `result` 等约定变量，作为 observation 的 fallback

### 2.2 anti-loop 与「空输出 = 成功」的矛盾

**现状**：`success=True` 时，anti-loop 将此次调用记为「成功」，重复 2 次后拦截。

**问题**：observation 为空时，对下游 Agent 而言等同于「无有效输出」，应视为**语义失败**，允许修改后重试。但当前实现将 `success=True` 一律视为成功。

**优化方向**：
- 定义「语义成功」：`success=True` 且 `observation` 非空（或包含至少一个可解析数值）
- 当 `success=True` 且 observation 为空时，返回 `success=False` 或增加 `observation_empty=True` 标记
- anti-loop：对「observation 为空」的 python_execute 调用，视为失败，允许修改代码后重试，不触发「成功重复」拦截

### 2.3 变量名 sanitize 的副作用

**现状**：`2024_year_net_profit` → `year_net_profit_2024`，避免 SyntaxError。

**风险**：
- 若代码中有 `print(2024_year_net_profit)`，sanitize 后变量名变化，可能导致 NameError 或引用错误
- 当前实现是对**整个 code 字符串**做 regex 替换，理论上 `2024_year_net_profit` 与 `year_net_profit_2024` 会同步替换，应保持一致
- 但若 LLM 生成的代码混用 `2024_net_profit` 与 `net_profit_2024`，可能产生不一致

**优化方向**：
- 记录 sanitize 映射，在 observation 或日志中提示「变量名已自动修正」
- 或：在 prompt 中强制要求使用 `net_profit_2024` 等合法命名，减少对 sanitize 的依赖

### 2.4 金融语义缺失

**现状**：python_execute 是通用 Python 执行器，无金融领域约束。

**缺失能力**：
1. **单位一致性**：亿/万/元混用（easy-test-14）时，无校验
2. **百分比语义**：24% 应存为 24 还是 0.24，依赖 LLM 理解
3. **舍入规则**：「保留两位小数」「保留整数」需 LLM 显式写 `round(x, 2)` 等
4. **公式校验**：如 `interest = principal * rate`，若 LLM 直接抄文中近似值，无机制强制用公式

**优化方向**（见第三节）：通过「金融增强层」或专用工具补充。

---

## 三、python_execute 优化方案（讨论）

### 3.1 方案 A：最小改动——可靠性增强

**目标**：解决空输出、与 anti-loop 的配合，不改变工具接口。

| 改动 | 说明 |
|------|------|
| 空输出视为失败 | `observation.strip() == ""` 且 `success=True` 时，改为 `success=False`，observation 设为 "Execution produced no output. Check that print() is used and variables are defined." |
| 显式 flush | `exec` 后 `output_buffer.flush()`，再 `getvalue()` |
| anti-loop 调整 | 对 python_execute，当 `observation` 为空时，`_is_tool_failure` 返回 True，允许修改后重试 |

**优点**：改动小，直接针对 easy-test-29。  
**缺点**：不解决单位、公式等金融语义问题。

---

### 3.2 方案 B：金融增强型 python_execute

**目标**：在现有工具上增加「金融计算契约」，提升可观测性与一致性。

#### B1. 结构化输出契约

**现状**：输出为自由文本，需 LLM 从 observation 中解析数值。

**提议**：支持「最后一行为 `return result`」时，自动将 `result` 的值附加到 observation：

```python
# 若代码最后是 return expr，除 print 外，额外附加 RETURNED: <value>
# 便于下游解析「最终答案」
```

或：约定 `__result__ = ...` 为显式结果变量，执行后自动 append 到 observation。

**优点**：减少解析歧义，便于 answer extraction。  
**缺点**：需改 prompt，约定新写法。

#### B2. 金融舍入与精度

**提议**：预置金融常用函数，减少 LLM 写错：

```python
# 在 _global_env 中注入
def round_pct(x, decimals=2): return round(x, decimals)   # 百分比
def round_currency(x, decimals=2): return round(x, decimals)  # 金额
def round_int(x): return int(round(x))  # 保留整数
```

在 description 中说明：对「保留两位小数」「保留整数」类题目，优先使用上述函数。

**优点**：统一舍入行为，减少 `round(x, 2)` 与 `int(round(x))` 的误用。  
**缺点**：依赖 LLM 选用，无法强制。

#### B3. 单位标注（可选）

**提议**：支持在赋值时标注单位，执行时做一致性检查（仅告警，不阻断）：

```python
# 例如：net_profit_2024_q3 = 26.9  # 单位: 亿元
# 若后续与 PE 相除，可提示「请确认单位一致性」
```

实现成本较高，可作为后续扩展。

---

### 3.3 方案 C：拆分——通用执行 + 金融公式库

**目标**：将「公式选择」与「公式执行」分离，降低 LLM 在公式上的出错率。

**思路**：
1. **金融公式库**：预定义常见公式（如 `interest = principal * rate`、`ROI = (revenue/cost)*100`），以结构化形式暴露（名称、参数、表达式）
2. **公式选择工具**：Planning 或 Finance 先调用 `select_formula(question_summary)`，返回匹配的公式 ID 与参数列表
3. **python_execute**：接收「公式 ID + 参数值」，从公式库取表达式并执行，保证公式正确性

**优点**：从源头减少「抄近似值」「公式写错」类问题（如 easy-test-17）。  
**缺点**：需维护公式库，覆盖有限，对开放题目泛化不足。

---

### 3.4 方案 D：双模式——快速执行 + 可验证执行

**目标**：在保持现有「自由代码」模式的同时，提供「可验证」模式。

| 模式 | 输入 | 行为 |
|------|------|------|
| 自由模式 | 任意 Python 代码 | 当前行为 |
| 可验证模式 | `{formula: "a/b", params: {a: 334.5, b: 110.9}}` | 仅执行公式，参数由上游提供，输出结构化 `{result, params_used}` |

**可验证模式**适用于：Planning 已明确公式、Multimodal 已提取参数，仅需「按公式计算」的场景。可避免 LLM 在代码中写错变量、漏写 round 等。

**优点**：对「提取正确、公式明确」的题目，可显著降低计算错误。  
**缺点**：需扩展工具接口，Flow 需支持模式选择。

---

## 四、与上游环节的协同优化

python_execute 的改进需与 Multimodal、Finance、Planning 配合，才能整体提升正确率。

### 4.1 Multimodal → python_execute

| 问题 | 优化方向 |
|------|----------|
| 百分比 24% 输出 0.24 | Multimodal prompt：明确「Share of annual sales 等占比类，输出 24 而非 0.24」 |
| 单位 26.9 亿 与 PE 混用 | Multimodal prompt：提取时标注单位；或统一为「与 PE 同量纲」再输出 |
| 取错行列 | 加强表头/行列对应，失败时要求根据错误修改 |

### 4.2 Finance → python_execute

| 问题 | 优化方向 |
|------|----------|
| 用文中近似值而非公式 | Finance prompt：遇到「approximately」「约」时，必须用 principal×rate 等公式验证 |
| 变量未重声明 | 已有「同一 block 内重声明」，可加强「从 observation 解析时，显式写 value 与 Source」 |
| 空 observation 时编造 | 已有 flow 中 `observation 为空` 的 WARNING，可加强「禁止编造」 |

### 4.3 anti-loop → python_execute

| 问题 | 优化方向 |
|------|----------|
| 空输出仍记为成功 | `_is_tool_failure`：对 python_execute，observation 为空时视为失败 |
| 成功重复 2 次即拦截 | 可考虑：当 observation 为空时，不增加「成功」计数，仅增加「调用」计数，允许更多次重试 |

---

## 五、优先级建议

| 优先级 | 方案 | 预期收益 | 实现成本 |
|--------|------|----------|----------|
| P0 | 空输出视为失败 + anti-loop 配合 | 解决 easy-test-29，避免「算对但无输出」 | 低 |
| P1 | 金融舍入预置函数 + prompt 引导 | 减少 round 误用 | 低 |
| P2 | 结构化输出契约（return/__result__） | 提升 answer 解析稳定性 | 中 |
| P3 | 可验证模式 / 公式库 | 对公式明确题目提升明显 | 高 |

---

## 六、小结

1. **直接根因**：仅 easy-test-29 与 python_execute 实现强相关（空输出 + anti-loop）。
2. **间接影响**：python_execute 的可靠性与可观测性，会放大 Multimodal/Finance 的错误。
3. **最小可行优化**：空输出视为失败 + anti-loop 对空输出按失败处理，成本低、收益明确。
4. **中期扩展**：金融舍入函数、结构化输出，提升一致性与可解析性。
5. **长期探索**：公式库、可验证模式，适合公式明确的子集题目。

建议先落地 P0，再根据后续 benchmark 决定是否推进 P1–P3。
