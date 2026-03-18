# 金融 Agent Benchmark 运行指南

> **工作目录**：`main/FinancialAgent`（严禁修改 Open Manus，仅在 main 中运行）

## 前置条件

1. **安装依赖**：`pip install -r requirements.txt`（在 `main/FinancialAgent` 目录）
2. **配置模型**：编辑 `config/config.toml`，设置模型、API、密钥
3. **RAG 使用前**：`cd /root/autodl-tmp/rag && python build_case_library.py` 构建例题库（否则 `--rag-top-k > 0` 会静默跳过）

---

## 命令行参数完整说明


| 参数                           | 类型   | 默认值                 | 含义                                                            |
| ---------------------------- | ---- | ------------------- | ------------------------------------------------------------- |
| `--dataset`                  | str  | `all`               | 数据集预设：`all` | `bizbench` | `finmmr` | `flarez`                |
| `--output-dir`               | str  | `benchmark_results` | 输出根目录，每次运行在其下新建 `multi_agent_multi_dataset_<run_id>/`         |
| `--timeout`                  | int  | `600`               | 单样本超时（秒）                                                      |
| `--limit-per-dataset`        | int  | `0`                 | 每数据集文件样本数限制，`0` 表示全量                                          |
| `--offset`                   | int  | `0`                 | 每个数据集文件的起始样本索引                                                |
| `--force-planning`           | flag | -                   | 强制启用 Planning Agent（覆盖 config）                                |
| `--disable-planning`         | flag | -                   | 禁用 Planning Agent（消融用）                                        |
| `--image-root`               | str  | `""`                | 多模态图片根目录，用于重映射数据集中的图片路径                                       |
| `--multimodal-mode`          | str  | `best_effort`       | `off`：忽略图片；`best_effort`：尽量用 vision 提取；`strict`：缺图则失败         |
| `--vision-timeout`           | int  | `120`               | 单样本图片转文本提取超时（秒）                                               |
| `--max-images-per-sample`    | int  | `1`                 | 每样本最大加载图片数                                                    |
| `--numeric-extract-strategy` | str  | `any`               | 数值评分策略：`any` | `first` | `last`，研究评测建议 `any`                  |
| `--rag-top-k`                | int  | `5`                 | RAG 例题数量，`0` 表示关闭 RAG                                         |
| `--rag-persist-dir`          | str  | `""`                | Chroma 持久化目录，空则用 `{workspace}/rag/chroma_db`                  |
| `--cot-enabled`              | flag | 默认 True             | 启用 FinMMR CoT few-shot                                        |
| `--no-cot`                   | flag | -                   | 禁用 FinMMR CoT few-shot（消融用）                                   |
| `--dataset-root`             | str  | `""`                | 数据集根目录（如 `datasets/test`），设置后从 `DATASET_FILES_FROM_ROOT` 解析文件 |


---

## 数据集路径

- **默认（无 `--dataset-root`）**：使用项目内 `Dataset/bizbench_test`、`Dataset/finmmr`、`Dataset/flarez-confinqa_test`
- **指定根目录**：`--dataset-root /root/autodl-tmp/datasets/test` 或 `--dataset-root ../../datasets/test`（相对 cwd）

`--dataset-root` 下期望文件（按 preset）：


| preset     | 文件                                                                                                                             |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `all`      | `bizbench_test.json`, `finmmr_easy_test.json`, `finmmr_medium_test.json`, `finmmr_hard_test.json`, `flare-convfinqa_test.json` |
| `bizbench` | `bizbench_test.json`                                                                                                           |
| `finmmr`   | `finmmr_easy_test.json`, `finmmr_medium_test.json`, `finmmr_hard_test.json`                                                    |
| `flarez`   | `flare-convfinqa_test.json`                                                                                                    |


---

## 可运行命令示例

### 1. 冒烟测试（每数据集 2 条，快速验证）

```bash
cd /root/autodl-tmp/main/FinancialAgent
python benchmark_multi_agent_multi_dataset.py \
  --dataset all \
  --dataset-root /root/autodl-tmp/datasets/test \
  --limit-per-dataset 5 \
  --rag-top-k 0
```

### 2. 全量评测（所有数据集，CoT + RAG）

```bash
cd /root/autodl-tmp/main/FinancialAgent
python benchmark_multi_agent_multi_dataset.py \
  --dataset all \
  --dataset-root /root/autodl-tmp/datasets/test \
  --rag-top-k 5 \
  --limit-per-dataset 0
```

### 3. 单数据集全量

**FinMMR（含 CoT + RAG）**

```bash
cd /root/autodl-tmp/main/FinancialAgent
python benchmark_multi_agent_multi_dataset.py \
  --dataset finmmr \
  --dataset-root /root/autodl-tmp/datasets/test \
  --rag-top-k 5 \
  --limit-per-dataset 0
```

**BizBench**

```bash
cd /root/autodl-tmp/main/FinancialAgent
python benchmark_multi_agent_multi_dataset.py \
  --dataset bizbench \
  --dataset-root /root/autodl-tmp/datasets/test \
  --rag-top-k 5 \
  --limit-per-dataset 0
```

**ConvFinQA (flarez)**

```bash
cd /root/autodl-tmp/main/FinancialAgent
python benchmark_multi_agent_multi_dataset.py \
  --dataset flarez \
  --dataset-root /root/autodl-tmp/datasets/test \
  --rag-top-k 5 \
  --limit-per-dataset 0
```

### 4. 使用项目内 Dataset（不指定 dataset-root）

```bash
cd /root/autodl-tmp/main/FinancialAgent
python benchmark_multi_agent_multi_dataset.py \
  --dataset all \
  --limit-per-dataset 2
```

### 5. 消融实验

**Baseline（无 RAG、无 CoT）**

```bash
cd /root/autodl-tmp/main/FinancialAgent
python benchmark_multi_agent_multi_dataset.py \
  --dataset finmmr \
  --dataset-root /root/autodl-tmp/datasets/test \
  --rag-top-k 0 \
  --no-cot \
  --limit-per-dataset 0
```

**RAG 消融（CoT 保持默认）**

```bash
# 无 RAG
python benchmark_multi_agent_multi_dataset.py --dataset finmmr --dataset-root /root/autodl-tmp/datasets/test --rag-top-k 0 --limit-per-dataset 0

# RAG-5
python benchmark_multi_agent_multi_dataset.py --dataset finmmr --dataset-root /root/autodl-tmp/datasets/test --rag-top-k 5 --limit-per-dataset 0

# RAG-8
python benchmark_multi_agent_multi_dataset.py --dataset finmmr --dataset-root /root/autodl-tmp/datasets/test --rag-top-k 8 --limit-per-dataset 0
```

**CoT 消融（仅 FinMMR 有效）**

```bash
# No CoT
python benchmark_multi_agent_multi_dataset.py --dataset finmmr --dataset-root /root/autodl-tmp/datasets/test --no-cot --limit-per-dataset 0

# CoT 启用（默认）
python benchmark_multi_agent_multi_dataset.py --dataset finmmr --dataset-root /root/autodl-tmp/datasets/test --limit-per-dataset 0
```

**Planning Agent 消融**

```bash
# 禁用 Planning
python benchmark_multi_agent_multi_dataset.py --dataset bizbench --dataset-root /root/autodl-tmp/datasets/test --disable-planning --limit-per-dataset 0

# 强制启用 Planning
python benchmark_multi_agent_multi_dataset.py --dataset bizbench --dataset-root /root/autodl-tmp/datasets/test --force-planning --limit-per-dataset 0
```

### 6. 多模态与超时调优

```bash
cd /root/autodl-tmp/main/FinancialAgent
python benchmark_multi_agent_multi_dataset.py \
  --dataset finmmr \
  --dataset-root /root/autodl-tmp/datasets/test \
  --image-root /root/autodl-tmp/datasets/test/images \
  --multimodal-mode best_effort \
  --vision-timeout 180 \
  --max-images-per-sample 3 \
  --timeout 900 \
  --limit-per-dataset 0
```

### 7. 自定义输出目录与 RAG 路径

```bash
cd /root/autodl-tmp/main/FinancialAgent
python benchmark_multi_agent_multi_dataset.py \
  --dataset all \
  --dataset-root /root/autodl-tmp/datasets/test \
  --output-dir /root/autodl-tmp/benchmark_outputs \
  --rag-persist-dir /root/autodl-tmp/rag/chroma_db \
  --limit-per-dataset 0
```

---

## 输出产物


| 路径                                                 | 说明         |
| -------------------------------------------------- | ---------- |
| `{output-dir}/multi_agent_multi_dataset_{run_id}/` | 单次运行根目录    |
| `run_meta.json`                                    | 运行参数、模型、配置 |
| `global_summary.json`                              | 汇总指标       |
| `global_report.md`                                 | 可读报告       |
| `{dataset}__{file_stem}/predictions.jsonl`         | 逐样本预测      |
| `{dataset}__{file_stem}/summary.json`              | 单数据集指标     |
| `{dataset}__{file_stem}/error_analysis.md`         | 错误分析       |


---

## 表格 Ours 行填入

使用 `--dataset-root /root/autodl-tmp/datasets/test` 时，从各 `summary.json` 提取：


| 指标                  | 来源                                                     |
| ------------------- | ------------------------------------------------------ |
| BizBench Accuracy   | `test__bizbench_test/summary.json` → `accuracy`        |
| FinMMR Easy (CoT)   | `test__finmmr_easy_test/summary.json` → `accuracy`     |
| FinMMR Medium (CoT) | `test__finmmr_medium_test/summary.json` → `accuracy`   |
| FinMMR Hard (CoT)   | `test__finmmr_hard_test/summary.json` → `accuracy`     |
| ConvFinQA EmAcc     | `test__flare-convfinqa_test/summary.json` → `accuracy` |


目录名格式：`{dataset_root父目录名}__{文件stem}`，如 `test__finmmr_easy_test`。

---

## 消融实验执行顺序建议

1. **Baseline**：`--rag-top-k 0 --no-cot`
2. **RAG 消融**：`--rag-top-k 0` vs `5` vs `8`（CoT 默认）
3. **CoT 消融**：`--no-cot` vs 默认（仅 FinMMR）
4. **Planning 消融**：`--disable-planning` vs `--force-planning`（可选）

---

## 工具调用问题排查

### 现象：`no-tool empty response detected`

日志中出现 `Manus/Finance selected 0 tools to use` 且 `no-tool empty response detected`，表示模型返回了文本但未调用工具（如 `python_execute`）。

### 可能原因

1. **模型 Function Calling 支持弱**：Qwen3-VL-8B-Instruct 在 SiliconFlow 上可能未列入官方 Function Calling 支持列表，模型倾向返回文本而非 `tool_calls`。
2. **工具列表正确**：Manus/Finance 默认有 3 个工具（`python_execute`、`str_replace_editor`、`terminate`），格式符合 OpenAI 规范。
3. **API 正常**：请求成功返回，但 `response.tool_calls` 为空。

### 解决方案

1. **启用 `tool_choice=required`**：在 `config/config.toml` 的 `[runflow]` 下添加：
  ```toml
   tool_choice = "required"
  ```
   强制模型必须返回工具调用（部分模型可能仍不兼容）。
2. **更换模型**：使用 SiliconFlow 官方支持 Function Calling 的模型，如 `Qwen/Qwen2.5-7B-Instruct`、`deepseek-ai/DeepSeek-V2.5` 等。
3. **开启 DEBUG 日志**：设置 `LOG_LEVEL=DEBUG` 可查看发送的工具数量及 `tool_choice` 值。

