# 根本原因分析：为何当前架构相比原始 OpenManus 出现问题

## 一、问题现象

1. **计划创建极慢**：2 分钟+，且总是走默认计划
2. **Agent 不调用工具**：计算任务应使用 python_execute，但始终 0 tools
3. **其他任务被牺牲**：计算类 fast path 跳过 LLM，非计算类仍慢且无计划

## 二、根本原因

### 原因 1：WorkflowStateTool 描述与规划阶段冲突（核心）

**位置**：`app/tool/workflow_state.py` 第 9-13 行

```python
_WORKFLOW_STATE_TOOL_DESCRIPTION = """
A workflow-state tool for structured plan persistence and execution tracking.
It stores and updates plan records (create/get/update/mark_step), but it does NOT perform task decomposition reasoning.
In this architecture, plan synthesis should happen before state writes; avoid using this tool for initial cognitive planning.
"""
```

**问题**：描述中明确写有 **"avoid using this tool for initial cognitive planning"**，直接告诉模型「不要在初始规划阶段使用此工具」。

**后果**：`ask_tool` 时模型读到该描述，倾向于返回纯文本而非 tool call，导致始终走默认计划，且浪费一次 2 分钟的 LLM 调用。

**历史**：WorkflowStateTool 来自 planning→workflow_state 迁移，设计为「流程提交状态」用，假设 Phase A 由 Planner 产出 JSON、不调工具。但当前 PlanningFlow 仍用 `ask_tool` 做规划，需要模型调用该工具，与描述矛盾。

---

### 原因 2：Step Prompt 与 Executor 行为变化

| 维度 | 原始 archive | 当前 |
|------|--------------|------|
| Step prompt | `plan_status`（完整计划文本）+ `step_text` | `plan_status`（截断 ~200 字）+ `DEPENDS ON`（blackboard 变量） |
| Executor | 同一实例复用 | 每步 `model_copy` 新实例 |
| 计划上下文 | 完整进度、步骤、notes | 压缩版 |

**问题**：当前 step prompt 更短、更抽象，且每步新建 executor 并清空 memory，可能削弱模型对「当前任务 + 可用工具」的理解，导致不选 python_execute。

---

### 原因 3：计算类 fast path 的副作用

为规避慢速规划，对计算类任务直接跳过 `ask_tool`，只走默认计划。结果是：

- 非计算类任务仍要等 2 分钟，且因原因 1 仍拿不到真实计划
- 计算类任务虽快，但计划固定，无法适应其他任务类型

---

## 三、修复策略

### 1. 修复规划工具描述（必须）

在规划阶段传入**规划友好**的工具 schema，而不是直接使用 WorkflowStateTool 的默认描述。

- 新增：`WorkflowStateTool.to_planning_param()` 或等价方法，返回「用于规划」的 tool param
- 描述改为：Use this tool to create the execution plan. Call with command='create', title, and steps.
- `_create_initial_plan` 中调用 `ask_tool` 时使用该 param

### 2. 恢复 archive 式执行流程（推荐）

- 使用 `get_executor()` 返回同一实例，不再每步 `model_copy`
- Step prompt 恢复为：完整 `plan_status` + `step_text`（与 archive 一致）
- 移除 blackboard、tiered execution、stateless workers 等新增逻辑，回归简单流程

### 3. 移除计算类 fast path

- 删除 `_is_calculation_request` 及对应分支
- 所有任务统一走 `ask_tool` 规划，依赖修复后的工具描述拿到真实计划

### 4. 保留可观测性

- 保留 `_log_plan_snapshot`、步骤完成日志等，便于排查和评估

---

## 四、已实施修复（2026-03-09）

1. **WorkflowStateTool**：新增 `to_planning_param()`，返回规划友好描述
2. **PlanningFlow**：规划阶段使用 `to_planning_param()` 替代 `to_param()`
3. **PlanningFlow**：恢复 archive 结构（executor 复用、完整 plan_status、无 blackboard/tiered/stateless）
4. **移除**：计算类 fast path、blackboard、tiered execution、stateless workers
