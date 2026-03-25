# 计划创建阶段耗时分析

> 问题：输入 Prompt 后，产生计划需要非常久的时间。本文分析原因，不涉及代码修改。

---

## 一、计划创建流程概览

```
用户输入 → _create_initial_plan() → llm.ask_tool() → 等待 API 返回 → 解析/默认计划
```

**唯一阻塞点**：`await self.llm.ask_tool(...)` 这一轮 LLM 调用。

---

## 二、Token 体量（来自实际日志）

| 项目 | 数值 | 说明 |
|------|------|------|
| Input tokens | 833 | 系统消息 + 用户消息 + 工具 schema |
| Completion tokens | 92 | 模型生成（当前多为纯文本，非 tool call） |
| **总计** | **925** | 单次 API 调用 |

### 2.1 Input 构成（估算）

| 部分 | 估算 token | 内容 |
|------|------------|------|
| System | ~80 | "You are a planning assistant..." + agents_desc |
| User | ~450–550 | "Create a plan for: " + **完整用户请求**（GHI 财报 4 章节 + 用户问题，约 600 字符） |
| Tools | ~200–250 | workflow_state 的 name、description、parameters（7 个属性 + enum） |

用户请求为长文本（财报 + 问题），是 Input 的主要来源。

---

## 三、耗时来源分析

### 3.1 模型端推理

1. **Prefill（输入处理）**  
   - 需处理 833 个 input token  
   - 典型云模型：约 100–200 token/s  
   - 833 ÷ 150 ≈ **5–6 秒**

2. **Decode（生成）**  
   - 生成 92 个 token  
   - 典型云模型：约 50–100 token/s  
   - 92 ÷ 50 ≈ **1–2 秒**

3. **模型与配置**  
   - 当前配置：`Qwen/Qwen3-VL-8B-Instruct`，`api.siliconflow.cn`  
   - VL 模型：对纯文本任务可能略慢于纯文本模型  
   - 8B 模型：推理速度通常优于大模型，但受 API 排队和并发影响

### 3.2 网络与 API

- 网络：到 SiliconFlow 的 RTT  
- 排队：若请求在队列中等待  
- 冷启动：首次请求可能更慢  

### 3.3 工具选择与 prompt 设计

- `ToolChoice.AUTO`：模型需决定是调用工具还是返回文本  
- `workflow_state` 的 description 中写道：  
  > "avoid using this tool for initial cognitive planning"  
  即：**明确建议不要在规划阶段使用该工具**  
- 结果：模型更倾向于返回纯文本，而不是 tool call  
- 流程：几乎总是走到「默认计划」分支，而不是解析 tool call

### 3.4 总耗时估算

| 阶段 | 估算 | 说明 |
|------|------|------|
| Prefill | 5–6s | 833 input tokens |
| Decode | 1–2s | 92 tokens |
| 网络 | 0.5–2s | RTT + 排队 |
| **合计** | **6–10s** | 与「非常久」的主观感受一致 |

---

## 四、与 archive 的对比

- archive 同样使用 `ask_tool` + planning 工具  
- 若 archive 更快，可能原因：  
  1. 用户请求更短  
  2. 工具 schema 更简单  
  3. 模型或 API 不同  

当前架构中，**用户请求全文** 是 plan 阶段输入的主要部分，对 token 和耗时影响最大。

---

## 五、优化方向（仅方向，不实施）

| 方向 | 预期效果 | 实现思路 |
|------|----------|----------|
| 1. 缩短 plan 阶段输入 | 明显减少 token 与 prefill 时间 | 仅传摘要或前 N 字 + 用户问题，不传完整财报 |
| 2. 调整工具描述 | 提高 tool call 使用率 | 修改 workflow_state 描述，避免「避免使用」的表述 |
| 3. 跳过 plan 阶段 LLM | 直接消除该耗时 | 对计算类任务直接走默认计划，不调用 `ask_tool` |
| 4. 换用更快模型 | 降低 prefill/decode 时间 | 在 plan 阶段使用更小或更快的模型 |
| 5. 并行 | 缩短感知等待 | 若后续步骤可提前准备，可并行执行 |

---

## 六、结论

计划创建阶段耗时主要由以下因素决定：

1. **单次 LLM 调用**：833 input + 92 output tokens  
2. **完整用户请求**：财报 4 章节 + 问题，约 600 字符，是 Input 的主要来源  
3. **工具 schema**：workflow_state 的完整 schema 增加约 200+ token  
4. **工具描述**：当前描述引导模型避免使用该工具，导致多为纯文本回复  
5. **模型与 API**：Qwen3-VL-8B + SiliconFlow 的推理与网络延迟  

综合来看，约 6–10 秒的耗时主要来自 **prefill + decode + 网络**，其中 **prefill 占比最大**，与输入 token 数量直接相关。
