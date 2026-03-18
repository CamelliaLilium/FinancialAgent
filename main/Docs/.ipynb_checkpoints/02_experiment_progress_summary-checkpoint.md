# 金融多模态 Agent 实验进展与方法设计总览

> 凝练总结当前实验进展，目标：基于 datasets 与 rag 构建可自主策划工具调用的金融多模态 Agent

---

## 1. 当前实验进展

### 1.1 评测框架与基线

| 框架 | 文件 | 支持数据集 | 模型 | 状态 |
|-----|------|-----------|------|------|
| Base LLM 评测 | `Base_llm_test/financial_benchmark_framework_final.py` | FinBen, BizBench, FinMMR, FLARE-ConvFinQA | Qwen3-8B / Qwen3-VL-8B 等 | 支持多模态、断点续跑 |
| 单 Agent 评测 | `benchmark_finben_single_agent.py` | FinBen | Finance / Manus | 已跑通 |
| 多 Agent 评测 | `benchmark_finben_multi_agent.py` | FinBen | PlanningFlow + Finance | 已跑通 |
| 多数据集多 Agent | `benchmark_multi_agent_multi_dataset.py` | FinMMR, BizBench, FLARE-ConvFinQA | PlanningFlow + Finance | 已跑通 |

### 1.2 代表性实验结果

| 实验 | 数据集 | 样本数 | Accuracy | 备注 |
|-----|--------|-------|----------|------|
| 单 Agent Finance | FinBen | 496 | 59.1% | Macro-F1 0.47，neutral 主导 |
| 多 Agent FinMMR Easy | finmmr_easy_test | 20 | 45% | 多模态 strict，vision 解析 17/19 |
| 多 Agent FinMMR | finmmr (easy+medium+hard) | 60 | - | 冒烟测试 |

**主要误差类型**：`reasoning_or_prompt_misalignment`、`numeric_mismatch`、`label_extraction_ambiguity`、`multimodal_vision_failure`。

### 1.3 架构改造历程（ARCHITECTURE_RESEARCH_CHANGELOG）

- **Planning → workflow_state**：认知与状态分离，单点提交
- **工具精简**：移除 web_search、ask_human、browser_use、bash 等，主链路收敛为 `workflow_state`、`python_execute`、`str_replace_editor`、`terminate`
- **StrReplaceEditor 门控**：路径/证据/存在性检查，减少误调用
- **Prompt 口径**：任务-工具匹配优先，计算任务必调 `python_execute`

---

## 2. 数据集概况 (`datasets/`)

| 数据集 | 规模 | 任务类型 | 模态 | 关键字段 |
|--------|------|----------|------|----------|
| **FinBen** | 496 test | FOMC 立场分类 (dovish/hawkish/neutral) | 文本 | question, gold |
| **BizBench** | 4,673 test | 8 类定量推理（ConvFinQA, CodeFinQA, FormulaEval 等） | 文本+表格 | question, answer, task, context |
| **FinMMR** | Easy 1.2k / Medium 1.2k / Hard 1k | 金融数值推理 | 文本+图表 | question, ground_truth, images, python_solution |
| **FLARE-ConvFinQA** | 1,490 test | 对话式金融 QA | 文本 | query (Question: ... Answer: ...) |
| **FinanceMath** | 多子集 | 数学/金融计算 | 文本 | question, answer |

**统一评测路径**：`datasets/test/*.json` 与 Agent 内 `Dataset/` 对应。

---

## 3. RAG 知识/案例库 (`rag/`)

### 3.1 数据源与规模

| 文件 | 条数 | Question 来源 |
|------|------|---------------|
| finmmr_easy/medium/hard_validation.json | 900 | `question` |
| financemath_validation.json | 200 | `question` |
| flare-convfinqa_train.json | ~8.9k | 从 `query` 提取 `Question: xxx` |
| flare-convfinqa_valid.json | ~2.2k | 同上 |
| bizbench_train.json | ~14k | `question`，含 `task` |

**总计**：约 2.6 万条，覆盖 FinMMR、FinanceMath、ConvFinQA、BizBench。

### 3.2 构建与检索

- **构建**：`build_case_library.py`，BGE-M3 embedding + Chroma 向量索引
- **检索**：`retriever.py`，`retrieve_examples(query, top_k)`，支持按 `source` / `task` 过滤
- **返回**：完整例题记录（含 answer、context、python_solution 等 metadata）

### 3.3 与 Agent 的衔接

- **当前**：RAG 尚未接入 Agent 主链路
- **规划**：以 RAG 替代 web_search，作为金融知识/案例检索主通道

---

## 4. 工具列表与 Agent 大脑

### 4.1 当前工具清单

| 工具 | 用途 | 调用者 |
|------|------|--------|
| `workflow_state` | 计划状态持久化 | PlanningFlow |
| `python_execute` | 确定性计算、数据分析 | Finance / Manus |
| `str_replace_editor` | 文件读写（含门控） | Finance / Manus |
| `terminate` | 任务收口 | Finance / Manus |

### 4.2 规划大脑与工具调用

- **PlanningAgent**：产出结构化步骤（JSON），不直接调工具
- **PlanningFlow**：统一提交 `workflow_state.create`，按步骤驱动 executor
- **Executor（Finance/Manus）**：根据步骤自主选择并调用 `python_execute` / `str_replace_editor` / `terminate`

**目标**：Agent 大脑自主策划“何时调用何工具”，减少误调用与重复调用。

---

## 5. 方法设计目标与路线

### 5.1 最终目标

设计并实现**金融场景多模态 Agent 方法**，满足：

1. **数据**：在 `datasets/` 各数据集上完成金融 QA 任务
2. **知识**：以 `rag/` 为 RAG 知识/案例库，检索相似例题辅助推理
3. **多模态**：支持文本 + 图表（FinMMR 等）
4. **自主规划**：Agent 大脑自主策划并调用工具列表中的工具

### 5.2 待完成工作

| 模块 | 内容 |
|------|------|
| **RAG 接入** | 在 FinanceAgent / PlanningFlow 中集成 `retrieve_examples`，将检索结果注入 prompt |
| **多模态增强** | 完善 vision 输入链路，降低 `multimodal_vision_failure` |
| **工具策划** | 强化 PlanningAgent 对“任务类型→工具选择”的映射，减少 `reasoning_or_prompt_misalignment` |
| **数值一致性** | 优化 `numeric_extract_strategy`，降低 `numeric_mismatch` |
| **全数据集评测** | 在 FinBen / BizBench / FinMMR / FLARE-ConvFinQA 上系统评测并对比 RAG 前后 |

### 5.3 方法设计要点

1. **RAG 注入时机**：用户 query 进入后，先检索 top-k 例题，再与 query 一并送入 PlanningAgent / FinanceAgent
2. **任务-工具映射**：计算类 → `python_execute`；文件类 → `str_replace_editor`；纯语义 → 不调工具或仅 `terminate`
3. **多模态流程**：`images` 字段 → vision 提取/编码 → 与文本拼接送入 VL 模型
4. **可审计性**：保留 `workflow_state` 单点提交、步骤 `mark_step` 追踪，便于复现与误差归因

---

## 6. 小结

- **已有**：多 Agent 架构、工具精简、FinBen/FinMMR 等多数据集评测、RAG 例题库构建与检索
- **缺口**：RAG 未接入主链路、多模态稳定性待提升、全数据集系统评测未完成
- **方向**：以 RAG 为知识/案例库、以 PlanningFlow 为编排核心，实现 Agent 大脑自主策划工具调用的金融多模态 QA 方法
