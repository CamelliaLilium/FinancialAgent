# 金融Agent框架：服务器部署与快速起步

本仓库当前文档体系（主目录）建议保持三份：

1. `README_zh.md`（本文件）：部署、运行、输入输出、产物格式  
2. `WORKFLOW_STATE_RULES.md`：架构规则与工具治理  
3. `ARCHITECTURE_RESEARCH_CHANGELOG.md`：改造与科研记录  

---

## 1. 项目定位

当前版本是面向金融任务的可控多智能体基座，目标是：

- 规划认知与状态写入解耦（`PlanningAgent` vs `workflow_state`）
- 工具空间精简，降低误调用
- 可审计、可复现、便于后续接入 RAG/多模态

---

## 2. 核心文件作用（快速索引）

### 2.1 入口与评测

- `main.py`：单Agent入口（`Manus`）
- `run_flow.py`：多Agent入口（`PlanningFlow + executors`）
- `benchmark_finben_single_agent.py`：单Agent基线评测
- `benchmark_finben_multi_agent.py`：多Agent基线评测

### 2.2 Agent 与流程

- `app/agent/finance.py`：金融执行Agent
- `app/agent/manus.py`：通用执行Agent
- `app/agent/planning.py`：规划Agent（认知层）
- `app/agent/toolcall.py`：工具调用主循环与门控
- `app/flow/planning.py`：多Agent编排与 `workflow_state` 提交点

### 2.3 工具层

- `app/tool/workflow_state.py`：计划状态持久化工具
- `app/tool/planning.py`：兼容层（deprecated shim）
- `app/tool/python_execute.py`：确定性计算工具
- `app/tool/str_replace_editor.py`：文件读写编辑工具（含证据门控）
- `app/tool/file_operators.py`：文件操作抽象（本地/沙箱）
- `app/tool/terminate.py`：任务收口工具

### 2.4 配置

- `config/config.toml`：模型、API、运行参数
- `requirements.txt`：依赖列表

---

## 3. 输入/输出命令（本地或服务器）

### 3.1 单Agent交互

```bash
python main.py
```

可选一次性输入：

```bash
python main.py --prompt "你的任务"
```

### 3.2 多Agent交互

```bash
python run_flow.py
```

终端会提示 `Enter your prompt:`，输入任务即可。

### 3.3 多Agent base实验（FinBen）

```bash
python benchmark_finben_multi_agent.py --dataset "Dataset/Finben/finben_test.json" --output-dir "benchmark_results/base_rerun_multi" --timeout 600
```

可选参数：

- `--limit`：样本数限制，0 表示全量
- `--offset`：起始样本偏移
- `--force-planning`：强制启用规划Agent
- `--disable-planning`：关闭规划Agent

### 3.4 单Agent base实验（FinBen）

```bash
python benchmark_finben_single_agent.py --agent finance --dataset "Dataset/Finben/finben_test.json" --output-dir "benchmark_results/base_rerun_single_finance" --timeout 600
```

可选 `--agent manus`。

---

## 4. 评测产物目录与格式

两类 benchmark 都会在 `--output-dir` 下按时间戳创建新目录，默认不会覆盖历史结果。

### 4.1 多Agent产物（示例目录）

`benchmark_results/finben_multi_agent_YYYYMMDD_HHMMSS/`

- `predictions.jsonl`：逐样本预测记录
- `summary.json`：汇总指标与运行参数
- `error_analysis.md`：错误分类报告
- `scientific_assessment.md`：科研评估模板
- `failure_cases.jsonl`：失败样本子集
- `logs/<sample_id>.log`：逐样本运行日志

### 4.2 单Agent产物（示例目录）

`benchmark_results/finben_single_agent_<agent>_YYYYMMDD_HHMMSS/`

文件同上，另含单Agent特有字段（如标签提取歧义统计）。

### 4.3 关键字段（简化）

- `predictions.jsonl` 常见字段：
  - `id`, `gold`, `predicted`, `is_correct`
  - `status`, `elapsed_seconds`
  - `error_tags`, `tool_usage`, `repeated_tool_args`
  - `query`, `model_output`, `log_file`
- `summary.json`：
  - `args`（命令参数）
  - `summary`（accuracy/macro_f1/latency/confusion/error tags/tool usage）

---

## 5. 服务器部署（VS Code上传后）

以下步骤适用于 Linux 服务器。

### 5.1 上传代码

推荐方式：

- VS Code Remote-SSH 直接连接服务器并打开项目目录，或
- 本地打包上传后解压到服务器目录

### 5.2 创建环境

```bash
cd /path/to/金融Agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.3 配置模型与密钥

编辑：

- `config/config.toml`

至少确认：

- `[llm] model/base_url/api_key`
- `[runflow] use_planning_agent`

### 5.4 冒烟测试

```bash
python main.py --prompt "请回答：1+1等于几？"
```

```bash
python run_flow.py
```

### 5.5 运行基线实验

```bash
python benchmark_finben_single_agent.py --agent finance --dataset "Dataset/Finben/finben_test.json" --output-dir "benchmark_results/server_single_finance" --timeout 600
python benchmark_finben_multi_agent.py --dataset "Dataset/Finben/finben_test.json" --output-dir "benchmark_results/server_multi" --timeout 600
```

---

## 6. 注意事项（强烈建议）

- 不要把真实 API Key 提交到仓库。
- 每轮实验固定参数口径（dataset/offset/limit/timeout/model）。
- 规则变更请同步更新 `WORKFLOW_STATE_RULES.md`。
- 所有方向性改动必须记录到 `ARCHITECTURE_RESEARCH_CHANGELOG.md`。

