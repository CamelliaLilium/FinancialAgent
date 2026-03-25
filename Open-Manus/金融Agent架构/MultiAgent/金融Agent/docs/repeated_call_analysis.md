# 重复调用问题深度分析

基于 `multi_agent_multi_dataset_20260316_210940` 测试日志的根因分析。

---

## 一、现象概览

| 指标 | 数值 |
|------|------|
| same_args_repeated | **16/16**（100%） |
| python_execute 总调用 | 91 次 |
| terminate 总调用 | 50 次 |
| 准确率 | 37.5% (6/16) |

AntiLoop 已生效（出现 "System Intercept: You have already called..."），但模型仍持续重复调用，直至被拦截或达到 max_steps。

---

## 二、典型执行序列（easy-test-0）

```
Round 1: Model → [python_execute] (ebitda=97+15+30+921+1363, interest=97)
         → 执行成功，observation: '2426\n97\n'

Round 2: Model 收到成功结果 → 仍输出 [python_execute] (ebitda=2426, interest=97)
         → 执行成功，observation: '2426 97\n'

Round 3: Model 收到成功结果 → 仍输出 [python_execute] (完全相同)
         → 执行成功

Round 4: Model 收到成功结果 → 仍输出 [python_execute] (完全相同)
         → AntiLoop 拦截："Stop repeating. Proceed to the next step or call terminate."

Round 5: Model 收到拦截消息 → 输出 [terminate]
         → 结束
```

**结论**：模型在 python_execute 成功后不会主动调用 terminate，而是继续重复 python_execute，直到被 AntiLoop 拦截后才调用 terminate。

---

## 三、根因分析

### 3.1 模型未将「成功」视为任务完成

**表现**：每次收到 `{'observation': '...', 'success': True}` 后，模型仍输出 python_execute，而不是 terminate。

**可能原因**：

1. **工作流理解偏差**：Prompt 写的是 "Call python_execute... then call terminate"。模型可能理解为「每一步都要先 python_execute 再 terminate」，而不是「整步只做一次 python_execute 然后 terminate」。
2. **输出格式干扰**：工具返回格式为 `Observed output of cmd \`python_execute\` executed:\n{'observation': '2426 97\n', 'success': True}`，模型可能更关注 observation 内容，而没把 success 当作「任务已完成」的信号。
3. **行为惯性**：模型在「提取任务」上更倾向持续调用 python_execute，缺少「成功即停止」的明确模式。

### 3.2 每轮只生成一个工具调用

日志显示每轮均为 "selected 1 tools to use"，即模型每轮只输出一个工具调用。

- 若模型能输出 `[python_execute, terminate]`，一轮即可结束。
- 实际行为是：先多次 python_execute，被拦截后才单独输出 terminate。

说明模型没有形成「成功后在同轮内输出 terminate」的习惯。

### 3.3 被拦截后通过变量名绕过 AntiLoop（easy-test-14）

```
Round 1-2: python_execute (predicted_pe_2025, predicted_pe_2026) → 成功
Round 3:   相同代码 → 被拦截
Round 4-7: 相同代码 → 继续被拦截
Round 8:   python_execute (pe_2025_predicted, pe_2026_predicted) → 成功（不同 hash）
Round 9+:  两种写法交替出现，反复被拦截或成功，直至 max_steps=20
```

**原因**：AntiLoop 用 `json.dumps(kwargs, sort_keys=True)` 做哈希，变量名不同则 code 不同，hash 不同，不会被判为重复。

模型在多次看到 "Stop repeating" 后，没有转向 terminate，而是尝试修改变量名（如 `predicted_pe_2025` → `pe_2025_predicted`），从而绕过拦截，继续重复执行。

### 3.4 提取失败时的重复重试（easy-test-18）

```
Step 0 (Multimodal): 提取 Corporate notes 2010
  Round 1-3:  python_execute (corporate_notes_bonds_2010_fair_value = None) → 成功（但值为 None）
  Round 4-20: 相同代码 → 被拦截，但模型仍不断重试
```

**原因**：模型认为「提取失败」需要重试，而不是调用 terminate 结束当前步骤。即使被 AntiLoop 拦截，仍持续输出相同或等价调用。

### 3.5 失败后不修正代码的重复（easy-test-6）

```
Finance Step: 计算 change_in_backlog
  Round 1: python_execute → [EMPTY_OUTPUT] 失败
  Round 2-5: 完全相同的代码 → 被拦截（"previously FAILED"）
  Round 6: terminate
```

**原因**：模型在首次失败后没有根据错误修改代码，而是重复相同调用。AntiLoop 正确拦截了「失败后的重复」，但模型未从错误信息中学习并修正实现。

---

## 四、流程与机制层面的因素

### 4.1 Agent 主循环

```python
while current_step < max_steps and state != FINISHED:
    step_result = await self.step()  # think → act
```

- 只有调用 terminate 才会把 `state` 设为 `FINISHED`。
- 若模型一直不输出 terminate，循环会持续到 `max_steps`。
- 没有「连续多次被拦截则强制结束」的逻辑。

### 4.2 工具结果如何反馈给模型

拦截时，模型收到的是：

```
System Intercept: You have already called `python_execute` with these same arguments 
2 times successfully. Stop repeating. Proceed to the next step or call terminate.
```

- 文案已明确要求 "call terminate"。
- 在 easy-test-0 中，模型在收到该消息后确实调用了 terminate。
- 在 easy-test-14 中，模型选择修改变量名以绕过拦截，而不是调用 terminate。

说明：部分情况下模型会遵从拦截提示，部分情况下会尝试绕过，行为不稳定。

### 4.3 AntiLoop 的拦截条件

```python
if count >= 2:  # 成功调用 2 次后才拦截
    return True, "Stop repeating. Proceed to the next step or call terminate."
```

- 前两次相同成功调用会被放行，第三次及以后才拦截。
- 因此至少会出现「2 次重复成功调用」才触发拦截。

---

## 五、根因归纳

| 层级 | 根因 | 影响 |
|------|------|------|
| **模型行为** | 成功后将「再次 python_execute」优先于「terminate」 | 每步至少多 1–2 次无效调用 |
| **模型行为** | 被拦截后通过修改变量名绕过，而非调用 terminate | 极端情况下跑满 max_steps |
| **模型行为** | 提取失败时倾向于重试而非终止 | 大量重复调用 |
| **Prompt** | 「成功即 terminate」的约束不够强、不够显眼 | 模型未形成稳定终止习惯 |
| **AntiLoop** | 基于字符串哈希，变量名微调即可绕过 | 无法阻止语义等价的重复 |
| **AntiLoop** | 需成功 2 次后才拦截 | 至少 2 次重复无法避免 |
| **流程** | 无「连续被拦截则强制结束」机制 | 模型可长期占用步数 |

---

## 六、优化方向建议

### 6.1 模型 / Prompt 层

1. **强化终止条件**：在 Multimodal/Finance 的 prompt 中明确写出「若 python_execute 已返回 success 且 observation 含有效数值，必须立即调用 terminate，不得再次调用 python_execute」。
2. **明确单次成功即完成**：强调「一次成功的 python_execute 即表示本步完成，下一步只能是 terminate」。

### 6.2 AntiLoop 层

1. **成功即拦截**：第一次成功调用后，若再次出现相同参数，直接拦截（将 `count >= 2` 改为 `count >= 1`）。
2. **语义去重**：对 python_execute 的 code 做规范化（如统一变量名、去除注释、标准化空白）后再哈希，减少通过变量名微调绕过的情况。

### 6.3 流程层

1. **连续拦截后强制结束**：若同一 agent 连续 N 次（如 2–3 次）被 AntiLoop 拦截，可自动插入 terminate 或结束当前 agent 步，避免无限循环。
2. **拦截消息增强**：在拦截文案中更强调「必须调用 terminate，不要尝试修改变量名或代码结构」。

---

## 七、总结

重复调用的核心在于：**模型未把「python_execute 成功」当作任务完成信号，而是继续输出 python_execute**；在被拦截后，部分样本会通过修改变量名绕过 AntiLoop，导致长时间无效循环。仅靠 Prompt 约束效果有限，需要配合 AntiLoop 策略调整和流程层的保护机制。
