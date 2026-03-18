# Workflow State 与工具治理规则（统一版）

本文件是当前框架的唯一治理规则文档，覆盖：

- 多智能体规划与状态追踪规则
- Agent 与工具职责边界
- 文件工具场景化门控规则
- 新增执行器的最小接入规范

---

## 1. 角色与职责

- `PlanningAgent`：认知规划（想清楚怎么做）
- `workflow_state`：状态层（把执行状态写清楚）
- `PlanningFlow`：编排层（统一提交状态与驱动执行）
- `Executor`（如 `finance`、`manus`）：执行具体任务步骤

原则：**认知与状态分离**，状态写入以 `PlanningFlow` 为单入口。

---

## 2. 阶段化执行规则（A/B/C）

### 阶段A：Plan Synthesis（规划生成）
- 主体：`PlanningAgent`
- 约束：
  - 禁止调用 `workflow_state`
  - 禁止执行业务步骤
  - 输出计划草案（结构化步骤）

### 阶段B：Plan Materialization（计划提交）
- 主体：`PlanningFlow`
- 约束：
  - 单点调用 `workflow_state.create`
  - 形成 `active_plan_id` 后才能进入执行阶段

### 阶段C：Execution Tracking（执行追踪）
- 主体：`PlanningFlow`（优先）
- 约束：
  - `mark_step` 状态更新遵循一步一更
  - 每步记录简要 notes
  - 执行器默认不改计划结构

---

## 3. 工具治理总则

### 3.1 任务-工具匹配优先
- 工具调用依据任务类型，而不是上下文长度。
- 计算任务应调用计算工具；纯语义判断任务不应滥用代码工具。

### 3.2 最小必要调用
- 上下文可直接完成的任务，避免无增益工具调用。
- 禁止重复同参数无效调用（框架已设重复调用保护）。

### 3.3 当前主链路工具
- `workflow_state`
- `python_execute`
- `str_replace_editor`
- `terminate`

---

## 4. StrReplaceEditor 场景门控

### 场景定义
- S1：纯分类/计算任务（默认禁用文件工具）
- S2：需要读取本地文件（允许 `view`）
- S3：需要写入/修改文件（允许读写命令）

### 启用条件（必须同时满足）
1. `path` 为绝对路径
2. 请求上下文中有可验证文件证据（路径或可定位文件名）
3. 路径有效性检查通过  
   - 非 `create`：目标必须存在  
   - `create`：父目录必须存在

说明：仅“用户有读文件意图”不足以启用文件工具，必须有可验证文件对象。

---

## 5. Agent 默认能力边界

- `PlanningAgent`：`terminate`（不直接写 `workflow_state`）
- `FinanceAgent`：`python_execute` + `str_replace_editor` + `terminate`
- `Manus`：`python_execute` + `str_replace_editor` + `terminate`
- `PlanningFlow`：受控调用 `workflow_state`

---

## 6. 新增执行器最小接入规范

1. 新建 `app/agent/<your_agent>.py`（继承 `ToolCallAgent`）
2. 配置 `name/description/system_prompt/next_step_prompt`
3. 明确最小工具集（只放必要工具）
4. 在 `run_flow.py` 注册并加入 `executor_keys`
5. 在计划步骤中使用 `[agent_key]` 路由

---

## 7. 评估口径（下轮复跑）

主指标：
- Accuracy / Macro-F1
- 平均耗时、P95 耗时

过程指标：
- 工具调用分布
- 重复调用率
- 工具错误率
- 步骤状态流转一致性

---

## 8. 文档维护规则

- 架构规则只更新本文件，不再新建并行规则文档。
- 所有改动过程记录在 `ARCHITECTURE_RESEARCH_CHANGELOG.md`。
- 若本文件与代码冲突，以代码为准，并在变更日志补记修正项。

# Workflow State 工具治理规范（评审版）

## 1. 文档目的

本规范用于明确多智能体架构中：

- `PlanningAgent`（规划认知层）
- `workflow_state`（状态管理层）

两者的职责边界、调用时机和执行约束。目标是降低冗余调用、提升可解释性，并为后续架构改造提供统一依据。

## 2. 核心原则

1. **认知与状态分离**  
   `PlanningAgent` 负责“想清楚怎么做”；`workflow_state` 负责“把执行状态写清楚”。

2. **规划先于状态**  
   初始规划阶段不允许调用 `workflow_state`，先形成可执行计划，再由编排层提交。

3. **单一事实源（Single Source of Truth）**  
   计划一旦进入执行期，`workflow_state` 成为唯一计划状态来源。

4. **最小必要调用**  
   仅在状态变更或审计追踪需要时调用状态工具，避免流程噪声。

## 3. 阶段化流程（A/B/C）

### 阶段 A：规划生成（Plan Synthesis）

- 主体：`PlanningAgent`
- 输入：用户任务 + 可用执行器能力描述
- 输出：内存中的“最终候选计划”（未持久化）
- 约束：
  - 禁止调用 `workflow_state`
  - 禁止执行业务步骤
  - 禁止标记步骤状态

### 阶段 B：计划提交（Plan Materialization）

- 主体：`PlanningFlow` / 编排器
- 输入：阶段 A 的最终候选计划
- 输出：`workflow_state.create` 持久化计划
- 约束：
  - 只提交一次“初始基线计划”
  - 如需修订，使用 `update` 并保留变更痕迹（notes）

### 阶段 C：执行追踪（Execution Tracking）

- 主体：`PlanningFlow`（优先）/ 执行器（受控）
- 输入：执行结果、异常、中间产物
- 输出：`mark_step/get/update` 等状态更新
- 约束：
  - 一步一更（in_progress -> completed/blocked）
  - 每步必须记录简明 `step_notes`
  - 执行器默认不直接改计划结构（除非显式白名单）

## 4. 允许调用矩阵

| 角色 | create | update | get/list | mark_step | delete |
|---|---|---|---|---|---|
| PlanningAgent（阶段A） | 禁止 | 禁止 | 禁止 | 禁止 | 禁止 |
| PlanningFlow（阶段B/C） | 允许 | 允许 | 允许 | 允许 | 受控 |
| 执行器（manus/finance） | 禁止 | 禁止 | 可选只读 | 默认禁止 | 禁止 |

说明：执行器如需标记状态，应通过编排层统一代理，避免并发写冲突和职责漂移。

## 5. 状态提交点（Commit Point）

为防止“规划结果无落地”或“执行时计划不一致”，必须设定提交点：

1. PlanningAgent 明确返回“finalized plan”
2. PlanningFlow 在单点执行 `create`
3. `active_plan_id` 锁定并进入执行循环

若未达到提交点，不得开始步骤执行。

## 6. 任务粒度控制规则（与规划质量强关联）

1. **简单任务（单步可解）**：禁止细粒度拆分  
   例：单句分类、直接数值代入计算

2. **中等任务（2~4步）**：允许必要拆分  
   要求每步有独立信息增益

3. **复杂任务（>4步）**：允许分阶段计划  
   必须标注执行器与验收条件

4. **反模式（禁止）**  
   - 同义步骤重复  
   - “先算后再算一遍验证”但无新增证据  
   - 为了流程完整性而分步

## 7. 审计与可观测要求

每个步骤至少记录以下字段（可逐步实现）：

- `step_status`
- `step_notes`（摘要，不写冗长推理）
- `executor_id`
- `elapsed_ms`
- `tool_calls_count`

建议后续扩展结构化 `step_output`（JSON）以支持可复用中间结果和误差追溯。

## 8. 评估指标（用于改造前后对照）

### 主指标
- Accuracy / Macro-F1（任务结果）
- Avg latency/sample（效率）

### 过程指标
- `workflow_state` 调用次数/样本
- `terminate` 调用次数/样本
- 重复参数调用率
- 循环率（repeated_tool_loop）
- 工具错误率

### 判定标准（建议）
- 简单任务场景下：
  - 状态工具调用显著下降
  - 时延下降
  - 准确率不下降（或提升）

## 9. 迁移策略（建议）

1. 先改规范与提示词，再改代码路径  
2. 先实现“阶段A禁调状态工具”  
3. 再实现“阶段B单点提交”  
4. 最后收紧执行器写权限与审计字段

## 10. 当前结论

将 `planning` 重命名为 `workflow_state`，并执行“阶段A禁调、阶段B提交、阶段C追踪”的分层规则，是当前架构从“可运行”走向“可治理、可验证、可扩展”的关键一步。
