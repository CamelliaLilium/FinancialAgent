# API 迁移（SiliconFlow → DashScope）退化原因分析

## 一、本次修改内容回顾

| 修改项 | 修改前 | 修改后 |
|--------|--------|--------|
| base_url | `https://api.siliconflow.cn/v1` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| api_key | SiliconFlow Key | *(set `SILICONFLOW_API_KEY` / `DASHSCOPE_API_KEY` in workspace `.env`)* |
| extra_body | 无 | `{"enable_thinking": False, "thinking_budget": 81920}`（所有调用） |

---

## 二、可能导致退化的根因分析

### 2.1 模型 ID 不兼容（高概率）

**问题**：当前配置使用的模型 ID 是 SiliconFlow 格式，与 DashScope 官方格式不一致。

| 配置项 | 当前值 | DashScope 官方格式 |
|--------|--------|-------------------|
| [llm] model | `Qwen/Qwen3-8B` | `qwen3-8b` |
| [llm.vision] model | `Qwen/Qwen3-VL-8B-Thinking` | **未在兼容模式模型列表中** |

**后果**：
- `Qwen/Qwen3-8B` 在 DashScope 可能被拒绝或映射到错误模型，导致 404/400 或不可预期行为
- `Qwen/Qwen3-VL-8B-Thinking`：DashScope 兼容模式文档中的模型列表**未包含** Qwen3-VL 系列，该模型可能：
  - 不存在于 DashScope
  - 需使用不同 ID（如 `qwen-vl-plus` 等商业视觉模型）
  - 导致 Vision 调用失败，进而 0-tool 空转、Planning 卡住

**证据**：ANTI_LOOP_OPTIMIZATION_EVALUATION 中 easy-test-1「Multimodal 0-tool 空转 20 轮」、easy-test-6/194「tool_usage 为空，Planning 阶段卡住」与模型/API 异常高度相关。

---

### 2.2 extra_body 参数适用范围错误（高概率）

**问题**：对所有模型（包括非 thinking 模型）统一传入 `extra_body={"enable_thinking": False, "thinking_budget": 81920}`。

| 模型类型 | 是否支持 enable_thinking | 可能行为 |
|----------|--------------------------|----------|
| qwen3-8b（文本） | **不支持** | 可能 400 报错、或静默忽略 |
| qwen3-30b-a3b-thinking 等 | 支持 | 正常关闭 thinking |
| Qwen3-VL-8B-Thinking | 视 DashScope 是否支持而定 | 未知 |

**后果**：
- 若 DashScope 对未知参数严格校验 → 请求失败、重试、超时
- 若静默忽略 → 无直接错误，但增加了请求体复杂度

**建议**：仅对明确支持 thinking 的模型传入 `extra_body`，或通过配置开关控制。

---

### 2.3 关闭 Thinking 带来的能力退化（中概率）

**依据**：`docs/QWEN3_THINKING_MODE_ANALYSIS.md` 中的结论：

| 维度 | 开启 Thinking | 关闭 Thinking |
|------|---------------|---------------|
| 回答质量 | 4.8/5.0 | 4.2/5.0（约降 12%） |
| 复杂推理 | 更强 | 较弱 |
| 多步推理、因果分析 | 更好 | 变差 |

**对 FinMMR 任务的影响**：

| 任务类型 | 关闭 Thinking 的预期影响 |
|----------|--------------------------|
| 简单表格提取 | 影响小 |
| 多表/多图关联 | 准确率可能下降 5–15% |
| 复杂公式、单位换算 | 准确率可能下降 10–20% |
| 中文金融表格 | 准确率可能下降 5–10% |

**证据**：easy-test-2「优化前正确 → 优化后数值错误（54.58 vs 56）」符合「关闭 Thinking 导致数值推理变差」的假设。

---

### 2.4 API 提供商行为差异（中概率）

| 维度 | SiliconFlow | DashScope |
|------|-------------|-----------|
| 限流策略 | 可能不同 | 新 Key 配额/TPM 可能不同 |
| 延迟特征 | 原有基准 | 可能更慢或更快 |
| 错误码/重试 | 已适配 | 需重新验证 |
| 429 处理 | 现有 tenacity 配置 | 可能需调整退避时间 |

**后果**：更多 429、超时，间接导致「Planning 卡住 5 分钟」、超时样本增加。

---

### 2.5 extra_body 传递方式（低概率）

**当前实现**：在 `params` 中直接加入 `extra_body`，通过 `**params` 传给 `client.chat.completions.create()`。

**潜在问题**：OpenAI Python SDK 对 `extra_body` 的处理可能因版本而异；部分兼容 API 可能期望参数位于请求体特定位置。若 DashScope 对 `extra_body` 结构有特殊要求，可能导致参数未正确传递。

---

## 三、退化现象与根因对应

| 退化现象 | 可能根因 | 优先级 |
|----------|----------|--------|
| easy-test-1：对→超时，0-tool 空转 20 轮 | 模型 ID 错误 / Vision 模型不可用 / 返回格式异常 | P0 |
| easy-test-2：对→数值错误 | 关闭 Thinking 导致推理能力下降 | P1 |
| easy-test-6、194：Planning 卡住，tool_usage 为空 | 模型 ID 错误 / extra_body 导致 400 / 限流超时 | P0 |
| 正确率 4→3，超时 1→3 | 上述多项叠加 | — |

---

## 四、建议的验证与修复顺序

### 4.1 先验证，再修改

1. **模型 ID 验证**：
   - 用 DashScope 官方文档确认：`qwen3-8b` 是否为正确文本模型 ID
   - 确认 Vision 模型：DashScope 是否提供 Qwen3-VL，以及正确 ID
   - 用最小请求（无 tools、无 extra_body）测试两个模型是否可正常调用

2. **extra_body 验证**：
   - 仅对 thinking 模型（若使用）测试 `enable_thinking: False`
   - 对 qwen3-8b 测试：不传 extra_body vs 传 extra_body，观察是否 400 或行为异常

3. **回归对比**：
   - 在 smoke 集上：SiliconFlow（原配置） vs DashScope（修正后配置）
   - 控制变量：timeout、数据集一致，便于归因

### 4.2 修复建议（按优先级）

| 优先级 | 建议 | 说明 |
|--------|------|------|
| P0 | 修正模型 ID | 文本模型改为 `qwen3-8b`；Vision 模型需查 DashScope 文档确认 |
| P0 | 条件化 extra_body | 仅对支持 thinking 的模型传入，或通过配置开关 |
| P1 | 评估是否保留 Thinking | 若正确率优先，可考虑对 Vision/Planning 保持 Thinking 开启 |
| P2 | 监控限流与超时 | 观察 429、超时比例，必要时调整重试与 timeout |

### 4.3 不建议的做法

- **盲目回滚**：不区分根因就全部回退，会丢失有效改动
- **盲目修改**：在未验证模型 ID、extra_body 行为前，继续改代码
- **忽视文档**：需以 DashScope 官方文档为准，确认模型列表与参数支持

---

## 五、总结

本次迁移导致退化的**最可能根因**：

1. **模型 ID 与 DashScope 不匹配**，尤其是 Vision 模型可能不可用或 ID 错误
2. **extra_body 对非 thinking 模型可能引发错误或未定义行为**
3. **关闭 Thinking 会降低复杂推理与数值任务表现**，与 easy-test-2 等退化一致

建议：**先做最小化验证**（模型 ID、extra_body、单次调用），再根据结果决定是否调整配置或代码，避免盲目修改。
