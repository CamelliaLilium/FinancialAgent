# 金融 Agent 文件结构与职责总览

> 基于 Open Manus 金融Agent架构的凝练总结

---

## 1. 顶层入口与评测

| 文件 | 作用 |
|-----|------|
| `main.py` | 单 Agent 交互入口（Manus） |
| `run_flow.py` | 多 Agent 编排入口（PlanningFlow + executors） |
| `finance_main.py` | 金融场景专用入口 |
| `benchmark_finben_single_agent.py` | 单 Agent 基线评测（FinBen） |
| `benchmark_finben_multi_agent.py` | 多 Agent 基线评测（FinBen） |
| `benchmark_multi_agent_multi_dataset.py` | 多数据集多 Agent 评测（FinMMR / BizBench / FLARE-ConvFinQA） |

---

## 2. Agent 层 (`app/agent/`)

| 文件 | 类/职责 |
|-----|--------|
| `base.py` | `BaseAgent` 抽象基类 |
| `toolcall.py` | `ToolCallAgent`：工具调用主循环、门控、重复调用保护 |
| `planning.py` | `PlanningAgent`：认知规划（任务分解、步骤生成），**不调用 workflow_state** |
| `finance.py` | `FinanceAgent`：金融执行器，工具集 `python_execute` + `str_replace_editor` + `terminate` |
| `manus.py` | `Manus`：通用执行器，同工具集 |
| `react.py` | ReAct 风格 Agent 实现 |

**职责分层**：PlanningAgent 负责“想清楚怎么做”；Finance/Manus 负责“执行具体步骤”。

---

## 3. 流程编排层 (`app/flow/`)

| 文件 | 作用 |
|-----|------|
| `base.py` | `BaseFlow` 流程基类 |
| `flow_factory.py` | `FlowFactory`：按 `FlowType` 创建 PlanningFlow 等 |
| `planning.py` | `PlanningFlow`：A/B/C 三阶段编排 |
|  | - **阶段 A**：PlanningAgent 产出计划草案（内存，不写状态） |
|  | - **阶段 B**：单点调用 `workflow_state.create` 提交计划 |
|  | - **阶段 C**：执行步骤、`mark_step` 追踪、驱动 executor |

---

## 4. 工具层 (`app/tool/`)

| 文件 | 工具 | 作用 |
|-----|------|------|
| `base.py` | `BaseTool` | 工具抽象基类、`ToolResult` / `ToolFailure` |
| `tool_collection.py` | `ToolCollection` | 工具集合管理、`execute` / `to_params` |
| `workflow_state.py` | `WorkflowStateTool` | 计划状态持久化（create / update / get / mark_step） |
| `planning.py` | `PlanningTool` | 兼容别名（deprecated），指向 `WorkflowStateTool` |
| `python_execute.py` | `PythonExecute` | 确定性计算、数据分析 |
| `str_replace_editor.py` | `StrReplaceEditor` | 文件读写，含路径/证据/存在性门控 |
| `file_operators.py` | - | 文件操作抽象（本地/沙箱） |
| `terminate.py` | `Terminate` | 任务收口 |
| `search/` | `BaiduSearch` 等 | 搜索引擎工具（当前主链路未启用） |

**主链路工具**：`workflow_state`、`python_execute`、`str_replace_editor`、`terminate`。

---

## 5. Prompt 层 (`app/prompt/`)

| 文件 | 作用 |
|-----|------|
| `planning.py` | PlanningAgent 系统提示、计划格式约束 |
| `finance.py` | FinanceAgent 系统/下一步提示，任务-工具匹配规则 |
| `manus.py` | Manus 系统/下一步提示 |
| `toolcall.py` | ToolCallAgent 基类提示，“必要时才调工具” |

---

## 6. 基础设施 (`app/`)

| 文件 | 作用 |
|-----|------|
| `config.py` | 配置加载（config.toml） |
| `llm.py` | LLM 封装（OpenAI 兼容 API） |
| `bedrock.py` | AWS Bedrock 适配 |
| `schema.py` | `Message`、`AgentState` 等数据结构 |
| `exceptions.py` | `ToolError` 等异常 |
| `logger.py` | 日志 |
| `utils/` | `files_utils`、`logger` 等工具函数 |

---

## 7. 数据与产物

| 路径 | 作用 |
|-----|------|
| `Dataset/` | 评测用数据集（Finben、finmmr、bizbench_test、flarez-confinqa_test） |
| `benchmark_results/` | 评测产物（predictions.jsonl、summary.json、error_analysis.md） |
| `workspace/` | 运行时工作目录 |
| `logs/` | 运行日志 |

---

## 8. 治理文档

| 文件 | 作用 |
|-----|------|
| `README_zh.md` | 项目入口、部署与运行说明 |
| `WORKFLOW_STATE_RULES.md` | 规划-状态分离、工具治理、StrReplaceEditor 门控 |
| `ARCHITECTURE_RESEARCH_CHANGELOG.md` | 架构改造与科研记录 |

---

## 9. 架构原则（简要）

1. **认知与状态分离**：PlanningAgent 只规划，`workflow_state` 由 PlanningFlow 单点提交。
2. **任务-工具匹配**：计算任务用 `python_execute`，纯语义任务不滥用代码/文件工具。
3. **最小必要调用**：上下文充分时避免无增益工具调用。
4. **StrReplaceEditor 门控**：需绝对路径 + 可验证文件证据 + 路径存在性检查。
