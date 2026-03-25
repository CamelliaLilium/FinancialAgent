# Benchmark 深度分析：20260315_215115

基于 PLAN EXECUTION prompt 修改后的新一轮测试。**不修改代码**，仅做分析。

---

## 一、正确率分析

### 1.1 总体结果

| 指标 | 215115 | 191306 (基线) | 211726 (OUTPUT限制) |
|------|--------|---------------|---------------------|
| 正确数 | 8/16 | 8/16 | 4/16 |
| 正确率 | **50%** | 50% | 25% |
| numeric_mae | 558,119 | 2,381 | 1,399,526 |

**结论**：正确率恢复至基线 50%，PLAN EXECUTION 修改未造成回退。

### 1.2 样本级对比（215115 vs 191306）

| 样本 | 191306 | 215115 | 变化 |
|------|--------|-------|------|
| easy-test-0 | ✓ | ✓ | - |
| easy-test-1 | ✓ | ✓ | - |
| easy-test-2 | ✗ | ✗ | - |
| easy-test-3 | ✓ | ✓ | - |
| **easy-test-6** | ✗ (1600) | **✓ (-1600)** | **新正确** |
| easy-test-10 | ✓ | ✓ | - |
| **easy-test-11** | ✗ | **✓ (25700)** | **新正确** |
| easy-test-14 | ✗ | ✗ | - |
| easy-test-15 | ✓ | ✓ | - |
| **easy-test-17** | ✓ (25.3125) | ✗ (25) | **新错误** |
| **easy-test-18** | ✗ | ✗ | - |
| **easy-test-22** | ✗ | ✗ | - |
| **easy-test-29** | ✓ (8919078) | ✗ (183219) | **新错误** |
| easy-test-33 | ✗ | ✗ | - |
| easy-test-36 | ✗ | ✗ | - |
| easy-test-194 | ✓ | ✓ | - |

**净变化**：+2 新正确（6, 11），-2 新错误（17, 29），正确数持平。

### 1.3 错误根因分类

#### 1.3.1 easy-test-17（25 vs 25.3125）

- **根因**：Multimodal 从文本直接取「$25 million」，未从表格取 maturity=750、rate=3.375% 计算 750×0.03375=25.3125。
- **类型**：数据源选择错误（文本 vs 表格计算）。
- **PLAN EXECUTION 未覆盖**：prompt 强调「按行/实体筛选」，未强调「优先用表格计算而非文本近似」。

#### 1.3.2 easy-test-29（183219 vs 8919078）

- **根因**：Multimodal 整表 dump 四行 `[217, 70.9, 69.79, 205]` 等，Finance 用错误行计算。正确应为单药（如 70.9, 4466.7, 2029）→ (4466.7-70.9)×2029≈8.9M。
- **类型**：未按「该药物」选单行，仍整表输出。
- **PLAN EXECUTION 部分失效**：「Extract ONLY the rows/entities specified」未约束「该药物」的语义（需选单行）。

#### 1.3.3 easy-test-18（0 vs 10514）

- **根因**：Multimodal 提取了 9300、8846，但 `python_execute` 的 observation 为空。日志显示 `{'observation': '', 'success': True}`。
- **代码问题**：Step 0 代码为 `corporate_notes_bonds_2010 = 9300  # Source: '...'  print(corporate_notes_bonds_2010)`，`print` 与注释同行，被 `#` 注释掉，未执行。
- **类型**：代码格式错误（注释吞掉 print），非 prompt 问题。

#### 1.3.4 easy-test-22（无法计算 vs 202.73）

- **根因**：Multimodal 输出 `revenue_2016 = None`，未从图中提取 revenue 与 total_purchase_price。
- **类型**：视觉提取失败或未找到对应数据。

#### 1.3.5 easy-test-14（0.00 vs 0.04）

- **根因**：Multimodal 将「2024年三季度净利润」误提取为 26.9（应为 26.9 亿 = 2,690,000,000），单位未转换。
- **类型**：单位处理错误（亿→×1e8）。

#### 1.3.6 easy-test-2（47.49% vs 56%）

- **根因**：公式理解错误。「Q4 goes up by 2%」应为 Q4 份额×1.02，而非 Q4+2。正确：(0.24 + 0.25×1.02)×100 = 49.5%？ 题目要求「second half」即 Q3+Q4，Q4 增加 2% 后：(0.25×1.02)+0.24 = 0.495，即 49.5%。gold 56 可能另有计算方式。当前预测 47.49 来自 (0.25+2)+0.24 的误解。

#### 1.3.7 easy-test-33、easy-test-36

- **33**：无图，纯文本题，数据缺失。
- **36**：Multimodal 提取 None，归母净利润未正确识别。

### 1.4 PLAN EXECUTION 效果评估

| 样本 | PLAN EXECUTION 是否生效 | 说明 |
|------|-------------------------|------|
| easy-test-194 | ✓ | 仅取 October 行，13,435,200 正确 |
| easy-test-29 | ✗ | 仍整表 dump，未选单药 |
| easy-test-17 | - | 非行筛选问题，为数据源选择 |
| easy-test-6 | ✓? | 正确得到 -1600，可能符号处理改善 |

**小结**：PLAN EXECUTION 对「按行/实体筛选」类任务（如 194）有效，对「单行语义选择」（如 29 的「该药物」）约束不足；对单位、公式、数据源选择无直接帮助。

---

## 二、耗时分析

### 2.1 样本级耗时（按总耗时排序）

| 样本 | 总耗时 | 正确? | 主要阶段 |
|------|--------|-------|----------|
| easy-test-18 | **215.9s** | ✗ | Planning 63s + Multimodal×2 约 107s |
| easy-test-11 | **207.6s** | ✓ | Planning + Multimodal + Finance |
| easy-test-0 | **174.4s** | ✓ | Multimodal 119s (Completion 8878) |
| easy-test-22 | 156.9s | ✗ | Multimodal 76s (Completion 5711) |
| easy-test-14 | 154.0s | ✗ | Multimodal 64s (Completion 4785) |
| easy-test-36 | 132.2s | ✗ | - |
| easy-test-29 | 115.3s | ✗ | Multimodal 整表 4761 token |
| easy-test-10 | **35.2s** | ✓ | 全流程轻量 |
| easy-test-6 | 47.8s | ✓ | - |
| easy-test-33 | 14.4s | ✗ | 无图，快速失败 |

### 2.2 日志级耗时拆解（典型样本）

#### easy-test-0（174s，Multimodal 主导）

| 阶段 | 起止 | 耗时 | Input | Completion |
|------|------|------|-------|------------|
| Planning create | 21:51:16 → 21:51:54 | 38s | 3034 | 2870 |
| Planning terminate | 21:51:54 → 21:52:01 | 7s | 3410 | 528 |
| **Multimodal 提取** | 21:52:01 → 21:54:00 | **119s** | 3058 | **8878** |
| Multimodal terminate | 21:54:00 → 21:54:07 | 7s | 3170 | 464 |
| Finance | 21:54:07 → 21:54:10 | 3s | 2633+3436 | 95+18 |

**根因**：Multimodal 单次 Completion **8878 token**，流式生成约 119s。

#### easy-test-18（216s，多 step Multimodal）

| 阶段 | 起止 | 耗时 | Completion |
|------|------|------|------------|
| Planning create | 22:08:09 → 22:09:12 | 63s | 4786 |
| Planning terminate | 22:09:12 → 22:09:23 | 11s | 773 |
| Multimodal Step0 | 22:09:23 → 22:10:48 | **85s** | **6386** |
| Multimodal terminate | 22:10:48 → 22:11:19 | 31s | 2372 |
| Multimodal Step1 | 22:11:19 → 22:11:35 | 16s | 1100 |
| Multimodal terminate | 22:11:35 → 22:11:41 | 6s | 420 |
| Finance | 22:11:41 → 22:11:45 | 4s | 116+18 |

**根因**：3-step 计划导致 Multimodal 调用 2 次，Step0 单次 6386 token 约 85s。

#### easy-test-22（157s）

| 阶段 | 耗时 | Completion |
|------|------|------------|
| Planning (create+terminate 同轮) | ~1s | 4718 |
| Multimodal | **76s** | **5711** |
| Multimodal terminate | 13s | 948 |
| Finance | 2s | 87+18 |

**根因**：Multimodal Completion 5711 token。

#### easy-test-10（35s，最快正确样本）

- Planning create+terminate 同轮，约 15s
- Multimodal Completion 较短
- Finance 轻量

### 2.3 耗时根因归纳

| 根因 | 典型表现 | 样本 |
|------|----------|------|
| **Multimodal 大 Completion** | 4.7k–8.9k token，单次 64–119s | easy-test-0, 14, 18, 22, 29 |
| **Planning 分两次调用** | create + terminate 各一次，多 30–55s | easy-test-0, 18 |
| **多 step Multimodal** | 每 step 一次 LLM 调用 | easy-test-18 |
| **Finance 无工具轮** | 纯推理 5–7s | 部分样本 |

---

## 三、冗余 Token 与性能平衡

### 3.1 问题陈述

- **冗余 Token**：Multimodal 常生成 4k–9k token（含推理、解释），实际工具调用仅需数百 token。
- **性能平衡**：通过 prompt 限制输出（如 211726 的 OUTPUT 限制）会削弱正确率（50%→25%）。
- **结论**：单靠 prompt 难以同时达成「减少冗余」与「保持正确率」。

### 3.2 非 Prompt 优化方向（不修改代码，仅记录）

| 方向 | 预期收益 | 实现层级 |
|------|----------|----------|
| **max_tokens 配置** | 限制 Multimodal 单次输出上限（如 2048） | 配置/LLM 调用 |
| **Planning create+terminate 合并** | 节省 1 次 LLM 调用 | 流程/agent |
| **python_execute 后自动 terminate** | 检测到成功即可终止，省 terminate 轮 | 流程 |
| **图片压缩/降分辨率** | 减少 Input token | 预处理 |
| **Speculative decoding** | 加速生成 | 推理服务 |
| **代码格式校验** | 避免 print 被注释（easy-test-18） | 预处理/后处理 |

### 3.3 为何 prompt 难以兼顾

1. **禁止推理**（211726）：正确率大跌，模型需要一定推理才能正确选行/映射。
2. **仅强调 plan**（215115）：对 194 有效，对 29、17 无效，约束粒度不足。
3. **过度约束**：易与「理解题目」「单位换算」「表格计算」等需求冲突。

**建议**：优先用 **max_tokens**、**流程合并** 等非 prompt 手段降耗时，保持当前 prompt 稳定。

---

## 四、小结

1. **正确率**：50%，与基线持平。PLAN EXECUTION 对 194 有效，对 17、29 无效或部分失效。
2. **新正确**：easy-test-6（-1600）、easy-test-11（25700）。
3. **新错误**：easy-test-17（数据源选择）、easy-test-29（整表 dump）、easy-test-18（代码格式 bug）。
4. **耗时**：Multimodal 大 Completion（4.7k–8.9k token）仍是主瓶颈，单次 64–119s。
5. **冗余 Token**：仅靠 prompt 难以在降冗余与保正确率间平衡，建议从配置与流程层优化。
