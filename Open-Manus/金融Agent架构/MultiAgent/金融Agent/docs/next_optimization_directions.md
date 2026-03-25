# 接下来的优化方向详细讨论

基于最新测试报告 (20260316_214540) 与失败样本根因分析。

---

## 一、当前状态

| 指标 | 数值 |
|------|------|
| 准确率 | 43.75% (7/16) |
| 剩余失败 | 9 个 |
| 主要瓶颈 | numeric_mismatch（9 个） |
| same_args_repeated | 15/16（多为 terminate 跨步骤重复，可接受） |

---

## 二、优化方向总览

| 方向 | 预期收益 | 实现难度 | 优先级 |
|------|----------|----------|--------|
| **A. previous_output 结构化** | 修复 easy-test-18 类多步传递错误 | 中 | P1 |
| **B. 多图输入配置** | 修复 easy-test-18 双图任务 | 低 | P1 |
| **C. 无图场景处理** | 修复 easy-test-33 | 中 | P1 |
| **D. 变量名语义去重** | 减少变量名绕过 AntiLoop | 中 | P2 |
| **E. EMPTY_OUTPUT 根因排查** | 修复 easy-test-14 首次失败 | 中 | P2 |
| **F. Planning 歧义消除** | 减少 easy-test-6、29 符号错误 | 低 | P2 |
| **G. 提取失败时的终止策略** | 减少 None 时的无效重试 | 低 | P2 |
| **H. 模型/多模态能力** | 提升提取准确率 | 高 | P3 |

---

## 三、各方向详细讨论

### A. previous_output 结构化

**问题**：easy-test-18 中 Finance 收到：
```
Observed output of cmd `python_execute` executed:
{'observation': 'None\n', 'success': True}
...
Observed output of cmd `python_execute` executed:
{'observation': '8846\n', 'success': True}
```

Finance 误将两个变量都映射为 8846，得到 8846+8846=17692。

**方案**：在 Flow 层对 `execution_result` 做结构化，传给 Finance 时格式化为：
```
Step 0 (Corporate notes and bonds 2010 Fair Value): None
Step 1 (Net sales 2011): 8846
```

或更简洁：`var1=None, var2=8846`，并明确变量名与步骤对应关系。

**实现**：
- 在 `_execute_step` 的 `prev_block` 中，解析 `execution_result`，按步骤提取 observation，格式化为 `Step N: <step_text 摘要> = <value>`
- 或在每步完成时，由 Flow 维护 `step_outputs: List[Tuple[str, str]]`，拼接时用结构化格式

**权衡**：需解析 agent 返回的原始格式，可能对格式变化敏感。可先用简单正则提取 `observation` 内容。

---

### B. 多图输入配置

**根因**：benchmark 配置 `max_images_per_sample: 1`，导致双图样本（easy-test-18、22）只传入 1 张图。

**证据**：
- easy-test-18 数据集有 `["1681-1.png", "1681-2.png"]`
- 日志显示 `base64_images_count=1`
- Step 0 和 Step 1 收到同一张图（len=87420）

**方案**：
1. **配置调整**：将 `max_images_per_sample` 改为 2 或 3，使多图样本传入全部图片
2. **按需传入**：根据 Planning 计划中的 "from image N" 数量动态决定，实现稍复杂

**建议**：先改为 2，验证 easy-test-18、22 是否改善。多图路由（P1 已实现）在有多图输入时会生效。

---

### C. 无图场景处理

**问题**：easy-test-33 无图像（`images: []`），Planning 仍分配 `[multimodal]`，导致步骤被 block 或回退到错误逻辑。

**方案**：
1. **Planning 层**：在 prompt 中强调「当输入无图像时，不得分配 [multimodal]；若数据在文本中，用 [finance] 从 context 提取」
2. **Flow 层**：在执行前将「当前无图」信息注入 Planning 的 context，或 multimodal 步骤无图时自动标记为 blocked 并提示 Planning 调整计划（需多轮，实现复杂）
3. **数据层**：easy-test-33 的 `images: []` 但 `ground_images` 有图，可能是数据集设计；需确认题目意图

**建议**：优先在 Planning prompt 中加「无图时勿用 [multimodal]」规则，实现成本低。

---

### D. 变量名语义去重

**问题**：AntiLoop 基于 `json.dumps(kwargs)` 的 hash，`predicted_pe_2025` 与 `pe_2025_predicted` 被视为不同调用，可绕过拦截。

**方案**：对 python_execute 的 `code` 做规范化后再参与 hash：
- 去除注释（`# Source: ...`）
- 变量名标准化（如按字母序重排、统一命名风格）
- 或提取「语义指纹」：解析 AST，提取赋值语句的 (变量名, 值) 对，忽略变量名顺序

**权衡**：实现复杂，可能误伤合理重试（如修正变量名后的重试）。可先做轻量版：去除 `# Source:` 注释后再 hash，减少因注释微调导致的绕过。

---

### E. EMPTY_OUTPUT 根因排查

**问题**：easy-test-14 首次调用报 EMPTY_OUTPUT，第二次相同代码却成功。可能原因：
- `_fix_print_after_comment` / `_wrap_last_expression` 的边界情况
- 跨 agent 的 `_global_env` 状态
- 首次调用时 code 实际不同（如 JSON 解析差异）

**方案**：
1. 在 python_execute 中加调试日志：记录实际执行的 code、observation、是否空输出
2. 本地复现 easy-test-14 的首次调用，单步调试
3. 若为环境问题，检查 Multimodal 与 Finance 是否共享同一 PythonExecute 实例

---

### F. Planning 歧义消除

**问题**：
- easy-test-6：「change between 2013 and 2014」未明确方向，模型用 2014-2013，gold 为 2013-2014
- easy-test-29：「差值」未明确顺序，模型用 (单片价格 - 单周期费用)，gold 可能为反序或取绝对值

**方案**：在 Planning prompt 中加通用规则：
- 对「变化」「差值」等歧义表述，在步骤中明确写出公式方向，如「backlog_2013 - backlog_2014」或「|a - b|」
- 强调「若题目未明确顺序，取常见会计惯例（如 later - earlier）或注明需取绝对值」

**权衡**：不加具体金融知识，只做通用歧义消除，避免 prompt 过长。

---

### G. 提取失败时的终止策略

**问题**：Multimodal 返回 None 时，模型有时会反复重试而非 terminate。

**方案**：
1. **Prompt**：在 Multimodal 中强调「若无法从图中提取（如图中无该数据），输出 None 并立即 terminate，不要重试」
2. **Flow**：当 previous_output 含 `None` 且为提取步骤时，在传给 Finance 的 prompt 中加「Step N 提取失败，请勿用该值参与计算」

**与 A 的协同**：结构化 previous_output 后，可明确标注 `Step 0: extraction_failed`，Finance 可据此跳过或报错。

---

### H. 模型与多模态能力

**问题**：easy-test-14、36 等提取数值与 gold 不一致，或直接返回 None，属模型/多模态能力上限。

**方案**：
- 换用更大或更强的多模态模型
- 对表格类任务，可探索专门的 table extraction 工具或预处理
- 或接受当前能力边界，优先优化流程与 prompt

---

## 四、建议实施顺序

### 第一批（快速见效）
1. **B. 多图配置**：`max_images_per_sample` 改为 2
2. **C. 无图处理**：Planning prompt 加「无图勿用 [multimodal]」

### 第二批（核心流程）
3. **A. previous_output 结构化**：实现步骤输出格式化，减少 Finance 误映射
4. **F. Planning 歧义消除**：通用规则，减少符号/方向错误

### 第三批（深度优化）
5. **E. EMPTY_OUTPUT 排查**：定位并修复
6. **D. 变量名语义去重**：轻量版（去注释后 hash）
7. **G. 提取失败终止**：Prompt 与 Flow 协同

### 长期
8. **H. 模型能力**：随基础设施升级推进

---

## 五、验证策略

- 每批改动后跑 smoke test，对比准确率与失败样本
- 重点观察：easy-test-18（多图+多步）、easy-test-33（无图）、easy-test-6/29（歧义）
- 若 B 生效，easy-test-18 的 Step 0 应能从 image 1 提取 Corporate notes，而非 None
