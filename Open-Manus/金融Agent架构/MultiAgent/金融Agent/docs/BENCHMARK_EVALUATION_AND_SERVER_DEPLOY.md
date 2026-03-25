# Benchmark 可运行性与 Agent 数据集 Base 测试评估

> 评估 `benchmark_multi_agent_multi_dataset.py` 是否可运行、是否支持服务器路径 `autodl-tmp/datasets/test`，以及是否可作为 Agent 数据集的 base 测试。

---

## 一、可运行性评估

### 1.1 运行验证结果

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 入口与依赖 | ✅ | 正常导入 app、config、flow、agents |
| 参数解析 | ✅ | parse_args 完整，无冲突 |
| 数据集路径解析 | ✅ | `resolve_dataset_files` 支持 cwd、workspace、绝对路径 |
| Flow 执行 | ✅ | 已实测 bizbench 1 样本完整跑通 |
| 输出结构 | ✅ | predictions.jsonl、summary.json、error_analysis.md、global_summary.json |

### 1.2 当前支持的路径解析

```
resolve_dataset_files 查找顺序（非绝对路径）：
1. Path.cwd() / raw          # 项目根目录下
2. config.workspace_root / raw
3. Path("/root") / raw       # 仅当 raw 以 "autodl-tmp" 开头时

支持：单文件 .json、目录（glob *.json）
```

### 1.3 服务器路径 `autodl-tmp/datasets/test` 支持

| 使用方式 | 命令示例 |
|----------|----------|
| 预设 test | `--dataset test`（需目录或 JSON 存在于 cwd/root/workspace 下） |
| 自定义路径 | `--dataset-path /root/autodl-tmp/datasets/test` |
| 自定义路径（相对） | `--dataset-path autodl-tmp/datasets/test`（cwd 为 /root 时） |

**目录结构要求**：`autodl-tmp/datasets/test` 下需有至少一个 `.json` 文件，且为 JSON 数组格式。

---

## 二、数据集格式要求

### 2.1 样本格式（兼容）

| 字段 | 用途 | 必需 |
|------|------|------|
| `query` 或 `question` | 用户问题/提示 | 至少其一 |
| `answer` 或 `ground_truth` | 标准答案 | 至少其一 |
| `context` | 上下文（如财报文本） | 可选 |
| `images` | 图片路径列表 | 可选 |
| `task` | 任务类型（如 FormulaEval） | 可选，影响评估模式 |
| `id` / `question_id` / `source_id` | 样本 ID | 可选 |

### 2.2 评估模式

| task 值 | 评估模式 | 判定逻辑 |
|---------|----------|----------|
| `formulaeval` | code | 代码子串匹配 |
| 数值型 answer | numeric | 容差匹配（any/first/last） |
| 其他 | text | 归一化精确匹配 |

### 2.3 多模态图片路径

- 样本中 `images` 可为绝对路径或相对路径
- 解析顺序：直接存在 → 相对 dataset 目录 → `dataset_dir/images/` → `--image-root`
- FinMMR 等常用：`/root/autodl-tmp/datasets/FinMMR/images/xxx.png`，需 `--image-root /root/autodl-tmp/datasets/FinMMR/images`

---

## 三、作为 Agent 数据集 Base 测试的评估

### 3.1 优势

| 维度 | 评价 |
|------|------|
| **多数据集** | 支持 bizbench、finmmr、flarez、test 等预设，可扩展 |
| **多任务类型** | numeric、code、text 三种评估模式 |
| **多模态** | 支持图像输入、vision 提取、direct base64 |
| **可配置** | timeout、limit、offset、planning 开关、multimodal 模式 |
| **输出完整** | 每样本日志、predictions、failure_cases、汇总报告 |
| **可复现** | run_meta.json 记录完整参数 |

### 3.2 局限与改进建议

| 问题 | 影响 | 建议 |
|------|------|------|
| 预设硬编码 | 新数据集需改代码 | 已加 `--dataset-path` 支持自定义路径 |
| BizBench FormulaEval | 代码补全任务与当前 Agent 设计不匹配 | 针对 task 类型做差异化 prompt 或路由 |
| 图片路径 | 不同环境路径不一致 | 统一用 `--image-root` 映射 |
| 无并行 | 顺序执行，大数据集耗时长 | 可后续加 `--workers` 并行 |

### 3.3 结论：是否可作为 Base 测试

**可以**。该 benchmark 具备：

- 统一的样本格式约定（query/question + answer/ground_truth）
- 与 PlanningFlow 的完整集成
- 多模态与多任务评估能力
- 服务器路径支持（`--dataset-path`、`test` 预设、`/root` 回退）

**建议**：

1. 在 `autodl-tmp/datasets/test` 下放置符合格式的 JSON 文件
2. 使用 `--dataset-path /root/autodl-tmp/datasets/test` 或 `--dataset test`（cwd 为 /root 时）
3. 多模态数据配置 `--image-root` 指向实际图片目录
4. 小规模验证：`--limit-per-dataset 3` 先跑通再全量

---

## 四、服务器部署示例命令

```bash
# 使用 test 预设（路径需可解析）
python benchmark_multi_agent_multi_dataset.py --dataset test --limit-per-dataset 5

# 使用自定义路径（推荐）
python benchmark_multi_agent_multi_dataset.py \
  --dataset-path /root/autodl-tmp/datasets/test \
  --limit-per-dataset 10 \
  --image-root /root/autodl-tmp/datasets/FinMMR/images

# 完整参数示例
python benchmark_multi_agent_multi_dataset.py \
  --dataset-path /root/autodl-tmp/datasets/test \
  --output-dir benchmark_results \
  --timeout 600 \
  --limit-per-dataset 0 \
  --offset 0 \
  --image-root /root/autodl-tmp/datasets \
  --multimodal-mode best_effort
```
