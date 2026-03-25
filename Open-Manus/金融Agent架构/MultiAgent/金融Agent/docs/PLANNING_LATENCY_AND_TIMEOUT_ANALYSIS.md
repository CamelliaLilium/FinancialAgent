# Planning 耗时、超时与网络环境分析

## 一、Planning 调用为何明显较长？

### 1.1 实际耗时案例

| 样本 | Planning 开始 | Planning 完成 | 耗时 | Input tokens | Completion tokens |
|------|---------------|---------------|------|---------------|-------------------|
| easy-test-22 | 19:33:28 | 19:34:09 | **~41s** | 2,042 | 3,650 |
| easy-test-35 | 20:10:30 | 20:14:44 | **~4分14秒** | 17,863 | 5,668 |

### 1.2 根因分析

**① 多模态任务下 Planning 使用 Vision 模型 + 传入整张图**

- Planning 阶段会调用 `planning_agent.run(plan_prompt, base64_image=base64_images[0])`
- 多模态任务时 Planning 使用 `llm_vision`（Vision 模型），并传入 base64 编码的图片
- 图片 base64 后体积很大（如 easy-test-35 的 `len=730092` 字符 ≈ 数百 KB）
- Input tokens 包含：system prompt、plan 规则、用户问题、**整张图片**
- 导致 Input 高达 17,000+ tokens

**② 模型先生成大量「思考」再输出 tool_calls**

- 当前 `tool_choice=ToolChoice.AUTO`
- 模型可自由选择：先输出纯文本 reasoning，再输出 tool_calls
- Completion 3,650～5,668 tokens 中，相当一部分是「思考过程」
- 流式输出时需等所有 token 生成完，才能得到完整 `tool_calls`
- 生成越多文本，耗时越长

**③ Vision 模型推理本身较慢**

- 图文混合输入的计算量大于纯文本
- 国内 API（如 siliconflow.cn）的 Vision 模型推理延迟通常高于纯文本

---

## 二、超时的具体情况

### 2.1 超时机制

超时**不是** OpenAI API 的 `timeout` 参数，而是 **benchmark 对整条流程的限时**：

```python
# benchmark_multi_agent_multi_dataset.py
output_text = await asyncio.wait_for(
    flow.execute(prompt, base64_images=base64_images), timeout=timeout_s  # 默认 600s
)
```

- 整个 `flow.execute()` 必须在 600s 内完成
- 超时后 `asyncio.TimeoutError`，样本记为 `status=timeout`

### 2.2 超时与 Function Calling 的关系

| 问题 | 说明 |
|------|------|
| **是否因 Function Calling 协议导致超时？** | 不直接。超时是因为**整条流程**超过 600s，而非单次 API 调用超时。 |
| **Function Calling 的间接影响** | 有。当模型不返回原生 `tool_calls`（如 Vision 模型返回纯文本）时，`_parse_content_for_tool_calls` 解析失败 → 0-tool → 空转 20 轮 → 每轮约 15–20s → 累计 300–400s，极易触发 600s 总超时。 |

### 2.3 超时样本的典型模式

- **模式 A**：Planning 慢（2–4 分钟）+ Multimodal 0-tool 空转 19 轮（每轮 ~18s）→ 轻松超过 600s
- **模式 B**：Planning 正常 + 重复调用 python_execute 20+ 次 → 单样本耗时 300–400s，接近超时
- **模式 C**：Rate limit 重试（每次 3–6s）× 多次 → 拉长总耗时

---

## 三、在 prompt 结尾加入 tool call 格式能否明显降低时间？

### 3.1 协议层面

- **OpenAI Function Calling**：模型返回的是结构化的 `tool_calls` 数组，不是自由文本
- 在 prompt 中写「请按以下格式输出」主要起**引导**作用，实际返回仍由模型和 API 协议决定

### 3.2 可采取的优化

| 手段 | 作用 | 预期效果 |
|------|------|----------|
| **在 prompt 结尾加 tool call 示例** | 让模型更清楚要输出什么，减少无效 reasoning | 可能减少 10–20% 的 completion tokens |
| **`tool_choice=required`** | 强制模型必须返回 tool_calls，减少纯文本 | 对 Planning：可显著缩短 completion；对 Multimodal：可能减少 0-tool 空转 |
| **精简 Planning prompt** | 减少 input tokens | 直接降低输入与推理时间 |
| **Planning 不传图** | 仅用文本描述任务，不传 base64_image | 大幅降低 input（17k→2k），Planning 可从 4 分钟降到 ~30s |

### 3.3 关于「直接加入需要输出的 tool call 格式」

- 若指在 prompt 中写「你必须调用 planning 工具，格式如下：`{"command":"create", "plan_id":"...", ...}`」：
  - 对**支持 function calling** 的模型：有帮助，模型更易输出正确 tool_calls
  - 对**不支持或弱支持**的模型：可能仍输出纯文本，需依赖 `_parse_content_for_tool_calls`
- 若指 `tool_choice=required`：**更直接有效**，可强制模型输出 tool_calls，减少无效文本和 0-tool 轮次

---

## 四、国内 API + VPN 与任务速率

### 4.1 当前配置

- 使用 `siliconflow.cn`（硅基流动）等国内 API
- 请求从本机直接发往国内服务器

### 4.2 VPN 对速率的影响

| 场景 | 开 VPN | 不开 VPN | 建议 |
|------|--------|----------|------|
| **国内 API（如 siliconflow、阿里云、智谱等）** | 请求路径：本机 → VPN 节点（可能在海外）→ 国内 API，多一跳，**通常更慢** | 本机 → 国内 API，路径更短，**通常更快** | **建议关闭 VPN** |
| **国际 API（OpenAI、Anthropic 等）** | 若直连被墙或不稳定，VPN 可改善连通性 | 可能无法访问或延迟高 | 视网络情况，开 VPN 可能更稳定 |

### 4.3 其他网络因素

- **带宽**：图片 base64 后体积大，上传带宽影响首包和总耗时
- **并发**：benchmark 若串行跑样本，单样本慢会拉高整体时间
- **API 限流**：429 触发重试，每次重试 3–6s，会明显拉长单样本时间

---

## 五、优化建议汇总

| 优先级 | 优化项 | 预期效果 |
|--------|--------|----------|
| P0 | Planning 多模态任务时不传图，或仅传缩略图 | Planning 从 4 分钟降到 ~30s |
| P0 | Multimodal 使用 `tool_choice=required` | 减少 0-tool 空转，降低超时率 |
| P1 | Planning 使用 `tool_choice=required` | 减少无效 reasoning，缩短 Planning 耗时 |
| P1 | 精简 Planning prompt | 降低 input tokens |
| P2 | 国内 API 时关闭 VPN | 降低网络延迟 |
| P2 | 对 429 增加退避、限流 | 减少重试带来的额外耗时 |
