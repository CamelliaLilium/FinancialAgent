# 耗时问题深度分析（基于 20260315_191306 执行日志）

基于对高延迟样本执行日志的逐行分析，本文档聚焦**耗时根因**与**可优化方向**，不涉及代码修改。

---

## 一、耗时分布总览

### 1.1 样本级耗时（按总耗时排序）

| 样本 | 总耗时 | 正确? | Planning | Multimodal | Finance | 其他 |
|------|--------|-------|----------|------------|---------|------|
| easy-test-3 | **290.4s** | ✓ | 140s (48%) | 139s (48%) | 10s | 1s |
| easy-test-22 | **264.3s** | ✗ | 48s (18%) | **212s (80%)** | 3s | 1s |
| easy-test-11 | 184.6s | ✗ | 120s (65%) | 57s (31%) | 7s | 1s |
| easy-test-14 | 182.5s | ✗ | 59s (32%) | 111s (61%) | 11s | 1s |
| easy-test-18 | 174.1s | ✗ | 54s (31%) | 111s (64%) | 9s | 1s |
| easy-test-17 | 92.0s | ✓ | - | - | - | - |
| easy-test-10 | **35.7s** | ✓ | 15s | 14s | 6s | 1s |

**结论**：Planning 与 Multimodal 的 LLM 调用占绝大部分耗时，Finance 通常 <15s。

---

## 二、各阶段耗时拆解（基于日志时间戳）

### 2.1 easy-test-3（290.4s，最慢）

| 阶段 | 起止时间 | 耗时 | Token (Input/Completion) | 说明 |
|------|----------|------|-------------------------|------|
| Planning create | 19:16:40 → 19:18:06 | **85.8s** | 5198 / 6260 | 长 context + 复杂计划 |
| Planning terminate | 19:18:06 → 19:19:01 | **54.4s** | 5536 / 4015 | 二次 LLM 调用 |
| Multimodal 提取 | 19:19:01 → 19:20:45 | **104.4s** | 4188 / 7479 | **大图 307KB** |
| Multimodal terminate | 19:20:45 → 19:21:20 | 34.7s | 4298 / 2601 | - |
| Finance 计算 | 19:21:20 → 19:21:23 | 2.9s | 3561 / 114 | - |
| Finance 思考(无工具) | 19:21:23 → 19:21:29 | 6.5s | 4381 / 289 | 纯推理 |
| Finance terminate | 19:21:29 → 19:21:30 | 0.8s | 5309 / 18 | - |

**根因**：
1. Planning 两次调用合计 **140s**，create 与 terminate 分开，各需一次完整 LLM 往返
2. Multimodal 单次调用 **104s**，图片 307KB，输入+生成 token 约 11.7k
3. Finance 有「思考但不调用工具」的轮次，增加约 6.5s

---

### 2.2 easy-test-22（264.3s，Multimodal 占 80%）

| 阶段 | 起止时间 | 耗时 | Token | 说明 |
|------|----------|------|-------|------|
| Planning create | 19:35:35 → 19:36:22 | 47.8s | 2195 / 3180 | - |
| **Multimodal 提取** | 19:36:22 → 19:39:12 | **169.2s** | 2110 / **12479** | 生成 1.2 万 token |
| Multimodal terminate | 19:39:12 → 19:39:55 | 42.9s | 2364 / 3300 | - |
| Finance | 19:39:55 → 19:39:58 | 2.3s | - | - |

**根因**：
1. **Multimodal 单次 Completion 达 12479 token**，流式生成耗时极长
2. 图片 47KB 不大，但模型可能对多图/复杂表格做了冗长推理
3. Planning 与 Multimodal 合计约 260s，Finance 几乎可忽略

---

### 2.3 easy-test-11（184.6s，Planning 占 65%）

| 阶段 | 起止时间 | 耗时 | Token | 说明 |
|------|----------|------|-------|------|
| Planning create | 19:29:36 → 19:31:03 | **87.4s** | 4211 / 6325 | 单次调用即 87s |
| Planning terminate | 19:31:03 → 19:31:36 | 32.8s | 4549 / 2374 | - |
| Multimodal | 19:31:36 → 19:32:28 | 51.4s | 4139 / 3838 | 图 303KB |
| Multimodal terminate | 19:32:28 → 19:32:33 | 5.5s | - | - |
| Finance | 19:32:33 → 19:32:41 | ~7s | - | - |

**根因**：Planning 两次调用合计 **120s**，为最大瓶颈。

---

### 2.4 easy-test-14（182.5s）

| 阶段 | 起止时间 | 耗时 | Token | 说明 |
|------|----------|------|-------|------|
| Planning | 19:23:06 → 19:24:06 | 59s | 3126/4238, 3566/195 | - |
| **Multimodal** | 19:24:06 → 19:25:57 | **110.9s** | 3052 / **8241** | 图 469KB，大图 |
| Finance | 19:25:57 → 19:26:09 | 11s | - | - |

**根因**：Multimodal 单次 Completion 8241 token，图片 469KB，输入与生成均较重。

---

### 2.5 easy-test-18（174.1s，多 step Multimodal）

| 阶段 | 起止时间 | 耗时 | 说明 |
|------|----------|------|------|
| Planning | 19:32:41 → 19:33:34 | 54s | - |
| Multimodal Step0 | 19:33:34 → 19:34:50 | **76.0s** | 图 87KB |
| Multimodal terminate | 19:34:50 → 19:35:02 | 11.2s | - |
| Multimodal Step1 | 19:35:02 → 19:35:13 | 11.6s | 同一张图 |
| Multimodal terminate | 19:35:13 → 19:35:25 | 11.9s | - |
| Finance | 19:35:25 → 19:35:35 | 9s | - |

**根因**：3-step 计划导致 Multimodal 被调用 **2 次**，Step0 单次 76s，两次 terminate 各约 11s。

---

### 2.6 easy-test-10（35.7s，最快正确样本）

| 阶段 | 起止时间 | 耗时 | 说明 |
|------|----------|------|------|
| Planning | 19:22:31 → 19:22:46 | **15.2s** | create+terminate 同轮 |
| Multimodal | 19:22:46 → 19:22:59 | **13.7s** | 图 377KB，但 Completion 仅 985 |
| Finance | 19:22:59 → 19:23:06 | 6s | - |

**对比**：Planning 在同一轮内完成 create+terminate，仅 15s；Multimodal Completion 仅 985 token，整体轻量。

---

## 三、耗时根因归纳

### 3.1 按阶段

| 阶段 | 主要耗时来源 | 典型样本 |
|------|--------------|----------|
| **Planning** | 1) create 与 terminate 分两次 LLM 调用<br>2) 长 context（含题目、图片描述等）<br>3) 复杂计划导致大 Completion | easy-test-3, easy-test-11 |
| **Multimodal** | 1) 大图（300KB+）导致长 Input<br>2) 大 Completion（8k–12k token）<br>3) 多 step 导致多次调用 | easy-test-22, easy-test-14, easy-test-18 |
| **Finance** | 1) 「思考但不调用工具」的轮次<br>2) 多次 terminate | 各样本均有，但占比小 |

### 3.2 按模式

| 模式 | 表现 | 说明 |
|------|------|------|
| Planning create+terminate 同轮 | 15s (easy-test-10) | 一次 LLM 调用完成 |
| Planning create 与 terminate 分轮 | 86s+54s=140s (easy-test-3) | 两次完整 LLM 往返 |
| Multimodal 小 Completion | 14s (easy-test-10, 985 token) | 快速 |
| Multimodal 大 Completion | 104–169s (8k–12k token) | 流式生成主导耗时 |
| 多 step Multimodal | 76s+12s+12s (easy-test-18) | 每 step 一次 LLM 调用 |

---

## 四、可优化方向（讨论，不修改代码）

### 4.1 Planning 阶段

| 方向 | 预期收益 | 说明 |
|------|----------|------|
| **create+terminate 合并** | 节省 1 次 LLM 调用，约 30–55s | 若 create 成功即可视为完成，无需单独 terminate 轮 |
| **缩短 Planning context** | 减少 Input token，降低首 token 延迟 | 仅保留题目与必要约束，去掉冗余说明 |
| **限制 plan 复杂度** | 降低 Completion 长度 | 如限制 step 数量、避免过长 step 描述 |

### 4.2 Multimodal 阶段

| 方向 | 预期收益 | 说明 |
|------|----------|------|
| **压缩图片或分辨率** | 减少 Input token | 307KB、469KB 图可考虑压缩或 resize |
| **限制 Multimodal 输出** | 避免 8k–12k token 的冗长推理 | prompt 要求「仅输出变量赋值代码，不要解释」 |
| **多 step 合并** | 减少 Multimodal 调用次数 | easy-test-18 类「图1 提取 + 图2 提取」可考虑一次调用多图 |
| **提前终止** | 已有有效输出时不再生成 | 检测到 python_execute 成功即可终止，避免额外 terminate 轮 |

### 4.3 Finance 阶段

| 方向 | 预期收益 | 说明 |
|------|----------|------|
| **减少「无工具」轮次** | 每轮约 5–7s | 明确要求：有结果即调用 terminate，不要纯文字总结 |
| **terminate 与计算同轮** | 节省 1 次调用 | 若支持「计算+terminate」在一次响应中完成 |

### 4.4 流程层

| 方向 | 预期收益 | 说明 |
|------|----------|------|
| **并行** | 理论可减半 | Planning 与首步 Multimodal 若独立可并行（需架构支持） |
| **缓存** | 重复题目可复用 | 相同题目+图片的 Planning 结果可缓存 |
| **模型/推理优化** | 降低单次调用延迟 | 使用更快模型、更低 max_tokens、或 speculative decoding |

---

## 五、优先级建议

| 优先级 | 方向 | 预期节省 | 实现难度 |
|--------|------|----------|----------|
| **P0** | Multimodal 限制输出长度（禁止冗长推理） | 50–100s（对 easy-test-22 类样本） | 低 |
| **P0** | Planning create+terminate 合并 | 30–55s | 中 |
| **P1** | 图片压缩/降分辨率 | 10–30s | 中 |
| **P1** | Finance 减少无工具轮次 | 5–10s | 低 |
| **P2** | 多 step Multimodal 合并 | 对 easy-test-18 类约 20s | 高 |
| **P2** | 并行 Planning+Multimodal | 理论 50% | 高 |

---

## 六、小结

1. **耗时集中在 Planning 与 Multimodal**，Finance 占比小。
2. **Planning**：create 与 terminate 分两次调用，单样本可多花 30–55s。
3. **Multimodal**：大 Completion（8k–12k token）是主要瓶颈，单次可达 100–170s。
4. **图片大小**：300KB+ 图片加重 Input 负担，但 Completion 长度影响更大。
5. **快速样本**（如 easy-test-10）证明：在 create+terminate 同轮、Multimodal 输出简短时，总耗时可控制在 35s 左右。

**首要优化目标**：限制 Multimodal 输出长度，并合并 Planning 的 create 与 terminate，可显著缩短高延迟样本耗时。
