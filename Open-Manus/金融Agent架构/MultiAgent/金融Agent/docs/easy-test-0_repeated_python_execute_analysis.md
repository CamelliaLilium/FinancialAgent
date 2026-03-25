# easy-test-0 重复调用 python_execute 深度分析

## 1. 样本与 Ground Truth

| 字段 | 值 |
|------|-----|
| question_id | easy-test-0 |
| question | What is the interest coverage ratio for the year 2018? |
| ground_truth | 10.144329896907216 |
| python_solution | EBIT=984, interest_expense=97, ratio=EBIT/interest_expense |

**关键**：Ground Truth 使用 **EBIT**（Income before income taxes），不是 EBITDA。

---

## 2. 异常现象：Multimodal 重复调用 python_execute

### 2.1 215647 失败运行（7 次成功 + 2 次被 anti_loop 拦截）

| Step | 代码摘要 | observation | 说明 |
|------|----------|-------------|------|
| 1 | 复杂 EBITDA 计算 | 1019 97 | 误解 EBITDA |
| 2 | `ebitda_2018 = 1019  # Source: 'exact snippet'` | 1019 97 | 语义重复 |
| 3 | 另一种算法 | -386 97 | 错误 |
| 4 | `ebitda_2018 = 1019` | 1019 97 | 语义重复 |
| 5 | `ebitda_2018 = 984` | **984 97** | ✅ 正确 |
| 6 | `ebitda_2018 = 984  # Source: 'Income before...'` | 984 97 | 语义重复 |
| 7 | `ebitda_2018 = 984  # Source: 'Income before...' for 2018` | 984 97 | 语义重复 |
| 8 | `ebitda_2018 = 984`（与 step 5 完全相同） | 🛑 blocked | anti_loop 拦截 |
| 9 | 同上 | 🛑 blocked | 连续拦截，强制 terminate 提示 |
| 10 | terminate | - | 结束 |

### 2.2 214540 成功运行（3 次成功 + 2 次被拦截）

| Step | 代码摘要 | observation | 说明 |
|------|----------|-------------|------|
| 1 | 复杂计算 | -112 97 | 错误 |
| 2 | `ebitda_2018 = 984` | **984 97** | ✅ 正确 |
| 3 | `ebitda_2018 = 984  # Source: ...` | 984 97 | 语义重复 |
| 4 | 同上（完全相同） | 🛑 blocked | anti_loop 拦截 |
| 5 | 同上 | 🛑 blocked | 强制 terminate |
| 6 | terminate | - | 结束 |

---

## 3. 根本原因分析

### 3.1 模型未遵循 "ON SUCCESS, TERMINATE IMMEDIATELY"

Multimodal prompt 明确要求：
```
3. **TASK SUCCESS = IMMEDIATE TERMINATE**: As soon as python_execute returns success (observation contains numbers), call terminate(status="success") immediately.
17. **ON SUCCESS, TERMINATE IMMEDIATELY**: Once python_execute succeeds, call terminate(status="success") in the very next action.
```

**实际行为**：模型在 step 5 已得到正确 984 97 后，仍继续尝试 step 6、7、8，通过微调代码（加 Source 注释、换行）"优化"输出，而非立即 terminate。

### 3.2 anti_loop 只拦截「完全相同的参数」

```python
# app/tool/anti_loop.py
def _generate_hash(tool_name: str, kwargs: Dict[str, Any]) -> str:
    canonical_args = json.dumps(kwargs, sort_keys=True)  # 整个 code 字符串参与哈希
    payload = f"{tool_name}::{canonical_args}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()
```

- 哈希基于完整 `code` 字符串
- `ebitda_2018 = 984` 与 `ebitda_2018 = 984  # Source: 'Income before...'` 哈希不同
- 模型通过不断微调注释/格式绕过检测，导致多次「语义重复」调用

### 3.3 修复前 A 的 Step 映射错误（215647 失败的直接原因）

修复前 `_format_structured_previous_output` 提取**所有** python_execute 的 observation，按顺序编号：
- Step 0 = 第 1 次调用 → 1019 97
- Step 1 = 第 2 次调用 → 1019 97
- ...
- Step 6 = 第 7 次调用 → 984 97

Finance 的 prev_block 说明 "Map Step 0 to the first variable"，公式需要 2 个变量。Finance 取了 **Step 0: 1019 97**（第一个），而非最后的 984 97，导致 1019/97≈10.505 ≠ 10.144。

---

## 4. 代码路径梳理

```
MultimodalAgent.step()
  → ToolCallAgent.act() 解析 model 的 tool_calls
  → execute_tool("python_execute", args)
    → anti_loop_interceptor.execute_with_reflection()
      → _generate_hash(tool_name, kwargs)   # 完整 code 哈希
      → _should_block_repeat()              # 仅完全相同时拦截
      → actual_execute_func()                # 执行
      → _record_call()                       # 记录历史
  → 返回 observation 给 model
  → model 下一轮继续生成 tool_calls（未 terminate）
```

**关键点**：model 每轮独立决策，无「上一轮已成功应终止」的硬性约束，仅靠 prompt 引导。

---

## 5. 改进方向

| 方向 | 描述 | 复杂度 |
|------|------|--------|
| **语义去重** | 对 python_execute，提取 observation 中的数值，若与历史某次成功结果相同则拦截 | 中 |
| **早停启发** | 在返回给 model 的 observation 后追加 "SUCCESS: You have valid numbers. Call terminate now." | 低 |
| **强化 prompt** | 在 multimodal 中增加 "Do NOT add Source comments after first success. Terminate immediately." | 低 |
| **code 归一化** | 哈希前剥离注释、空白，使 `x=984` 与 `x=984  # Source` 视为相同 | 中 |

---

## 6. 小结

- **直接原因**：Multimodal 在首次成功后又多次微调代码（加 Source 等），anti_loop 无法识别语义重复。
- **失败链条**：修复前 A 的 Step 0 取第一个 observation（1019）→ Finance 用错数 → 答案错误。
- **已修复**：A 改为按 terminate 分割、每步取最后一次 observation，Step 0 现对应 984 97。
