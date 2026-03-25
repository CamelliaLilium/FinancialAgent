# 金融 Agent 搭建工作总体评估报告

> 对比 OpenManus 原始架构，对本次金融 Agent 搭建进行客观评价，并验证多智能体 benchmark 的正确性。

---

## 一、架构对比：金融 Agent vs OpenManus 原始

### 1.1 OpenManus 原始架构（archive/app/flow/planning.py）

| 维度 | 原始实现 |
|------|----------|
| **输入** | 仅 `input_text: str`，无多模态 |
| **计划创建** | PlanningAgent 或 flow LLM 直接调用 planning_tool |
| **执行器** | 单一 executor，按 step 顺序执行 |
| **步骤信息** | 仅 plan title + steps，无用户原文传递 |
| **Python 环境** | 每次调用独立沙箱，变量不跨步 |
| **最终答案** | `_finalize_plan` 基于 execution_result 做 LLM 摘要 |

### 1.2 当前金融 Agent 架构（app/flow/planning.py）

| 维度 | 当前实现 |
|------|----------|
| **输入** | `input_text` + `base64_images`，支持多模态 |
| **计划创建** | Planning 多模态时使用 vision 模型 + 图像，产出更精确表头 |
| **执行器** | Finance + Multimodal（可选），按 step_type 路由 |
| **步骤信息** | **user_request 注入**：非 multimodal 步骤显式传入用户原文 |
| **Python 环境** | 持久化 `_global_env`，变量跨步保留 |
| **最终答案** | `_finalize_plan(execution_result, user_request)` 做 LLM 摘要 |

### 1.3 核心差异与改进

| 改进点 | 说明 |
|--------|------|
| **纯文本数据源** | 修复 GHI EBITDA 失败：`user_request` 注入 step_prompt，Finance 可从用户原文提取数值 |
| **多模态链路** | Planning 看图 → Multimodal 提取 → Finance 计算，previous_output 传递 |
| **变量持久化** | python_execute 跨调用保留变量，支持 2 步「提取 + 计算」 |
| **Executor 标签** | 正则 `[a-zA-Z_]+` 匹配 `[finance]`，与 Planning 规则一致 |
| **Prompt 工程** | Variable-Source Anchoring、No-Guessing Policy、概念映射（multimodal） |

---

## 二、客观评价

### 2.1 优势

| 维度 | 评价 |
|------|------|
| **问题定位** | 根因清晰：MULTIMODAL_MODELS 遗漏、user_request 未注入、executor 标签不匹配等，修复成本低、收益高 |
| **架构扩展** | 在保留 PlanningFlow 主框架的前提下，增加多模态、user_request 注入、变量持久化，改动集中、可追溯 |
| **领域适配** | Finance prompt 针对金融场景（EBITDA、利息支出≠财务费用、Source 注释等）做了专门约束 |
| **可复现性** | ARCHITECTURE_RESEARCH_CHANGELOG 记录每次修改动机-改动-验证，便于复现实验 |

### 2.2 局限

| 维度 | 说明 |
|------|------|
| **任务泛化** | 当前优化主要针对「财报数值提取 + 公式计算」与「图表 QA」；对 BizBench FormulaEval（代码补全）等任务适配不足 |
| **模型依赖** | 多模态需 vision 模型且配置正确；Planning 使用 vision 增加延迟与成本 |
| **重复循环** | 部分失败样本仍出现 same_args_repeated、repeated_tool_loop，需进一步优化 stuck 检测与 early exit |
| **数据集差异** | 不同数据集格式（question/query、answer/ground_truth、task、images）需 infer_prompt/infer_gold 正确解析 |

### 2.3 与 OpenManus 的继承关系

- **保留**：PlanningFlow 主循环、PlanningTool、ToolCallAgent 基类、step 驱动执行
- **增强**：多模态输入、user_request 注入、变量持久化、Executor 路由
- **精简**：移除 sandbox、browser、MCP 等通用能力，聚焦金融分析

---

## 三、Benchmark 验证代码检查

### 3.1 运行验证结果

- **命令**：`python benchmark_multi_agent_multi_dataset.py --dataset bizbench --limit-per-dataset 1 --timeout 120`
- **结果**：Exit code 0，完整完成，输出写入 `benchmark_results/multi_agent_multi_dataset_*`

### 3.2 关键逻辑验证

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 数据集路径解析 | ✅ | `resolve_dataset_files` 支持 `Dataset/bizbench_test` 目录与单文件 |
| 样本格式 | ✅ | `infer_prompt` 支持 `query`/`question` + `context`；`infer_gold` 支持 `answer`/`ground_truth` |
| Flow 创建 | ✅ | `FlowFactory.create_flow(FlowType.PLANNING, ...)` 正确创建 PlanningFlow |
| 多模态条件 | ✅ | `multimodal_mode != "off" and "vision" in config.llm` 时添加 MultimodalAgent |
| Planning 开关 | ✅ | `--force-planning` / `--disable-planning` 覆盖 config |
| 执行调用 | ✅ | `flow.execute(prompt, base64_images=base64_images)` 正确传入 |
| 评估逻辑 | ✅ | `evaluate_prediction` 支持 numeric/text/code 三种模式 |
| 输出结构 | ✅ | predictions.jsonl、summary.json、error_analysis.md、global_summary.json |

### 3.3 数据集预设与路径

| 预设 | 路径 | 解析结果 |
|------|------|----------|
| bizbench | Dataset/bizbench_test | bizbench_test.json |
| finmmr | Dataset/finmmr | finmmr_easy_test.json, finmmr_medium_test.json, finmmr_hard_test.json |
| finmmr_easy | Dataset/finmmr/finmmr_easy_test.json | 单文件 |
| flarez | Dataset/flarez-confinqa_test | flare-convfinqa_test.json |

### 3.4 多模态数据集注意事项

- **FinMMR**：`images` 字段为绝对路径（如 `/root/autodl-tmp/datasets/FinMMR/images/1976-1.png`），需通过 `--image-root` 指定本地映射目录
- **multimodal_mode**：`off` 忽略图像；`best_effort` 有 vision 则用 direct images，否则 VL 提取；`strict` 图像缺失则失败

### 3.5 潜在问题（非阻塞）

| 问题 | 影响 | 建议 |
|------|------|------|
| loguru 兼容 | 若未安装 loguru，attach_log_capture 使用标准 logging，可能格式略有不同 | 不影响功能 |
| 元数据引用 | `workflow_final_answer_source: "planning_flow._synthesize_user_final_answer"` 实际为 `_finalize_plan` | 文档性描述，可后续统一命名 |
| BizBench FormulaEval | 任务为代码补全，Agent 误理解为执行类，导致失败 | 需针对不同 task 类型做差异化 prompt 或路由 |

---

## 四、总结

| 项目 | 结论 |
|------|------|
| **架构评价** | 在 OpenManus 基础上合理扩展，聚焦金融场景，修复了关键数据流问题（user_request 注入、多模态链路） |
| **Benchmark 运行** | 能正确执行，数据集解析、Flow 调用、评估逻辑、输出结构均正常 |
| **可运行性** | 已验证 bizbench 1 样本完整跑通；FinMMR 需配置 vision + image-root；flarez 需确认路径存在 |

**建议**：对多数据集做小规模验证（如每数据集 2–3 样本）以确认各格式兼容性；针对 BizBench FormulaEval 设计专门处理策略（如识别 task 类型并调整 Planning 规则）。
