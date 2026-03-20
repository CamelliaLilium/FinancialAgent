# ReasoningRAG 项目设计文档

基于 ReasoningBank (arXiv:2509.25140) 的结构化记忆系统，面向金融多模态 QA 场景。

---

## 一、项目定位与实现边界

本项目将传统 RAG 从“检索相似例题”重构为“检索可迁移推理策略”。核心思想与 ReasoningBank 一致：从成功和失败经验中蒸馏高层策略，在新题上检索相关经验，再将经验中的记忆项注入推理过程。

与论文原始设置相比，当前项目有两个明确边界：

- 论文是 **streaming / online closed-loop**：任务按时间顺序到来，当前题的新经验会回写并影响后续题。
- 当前 ReasoningRag 是 **offline batch**：先对 `cases.jsonl` 全量离线建库，再用静态记忆库做测试集推理；测试阶段不做在线回写。

状态标记约定：

- **已实现**：已有代码落地。
- **设计中**：文档已确定设计，但代码尚未完成。
- **计划扩展**：当前版本不做，后续可沿论文方向演进。

系统状态总览：

| 模块 | 状态 | 说明 |
|------|------|------|
| Step 0 数据预处理 | 已实现 | `step0_preprocess.py` 已产出统一 `cases.jsonl` |
| Step 1 CoT 轨迹生成 | 已实现 | `step1_generate_trajectories.py` 已可运行 |
| Step 2 成功/失败判定 | 已实现（MVP） | 规则匹配 + free_text Judge |
| Step 3 记忆提取 | 已实现（MVP） | success / failure 独立 prompt，结构化解析输出 |
| Step 4 经验库与索引构建 | 已实现（MVP） | experience-level memory bank + FAISS + index_meta |
| 在线推理 | 设计中 | 静态库检索，不在线回写 |
| 在线闭环更新 | 计划扩展 | 对齐论文 closed-loop |
| MaTTS | 计划扩展 | 对齐论文的 memory-aware test-time scaling |

ReasoningBank 论文中最关键的三点结论如下：

- 记忆内容应是可迁移策略，而不是原始轨迹或成功案例片段。
- 失败轨迹是有效信号，论文在 WebArena-Shopping 上加入失败经验后从 46.5 提升到 49.7，而 Synapse / AWM 无法稳定受益。
- 检索数量以 top-1 experience 最稳妥，论文 Appendix C.1 显示随着检索经验数增加，性能反而下降。

---

## 二、数据集概览

三个金融多模态数据集已全部接入，合计 14,319 条记录。

| 数据集 | 条数 | answer_type 分布 |
|--------|------|-----------------|
| FinMMR | 900 | numerical: 900 |
| FinMME | 8,879 | mcq: 7,414 / numerical: 1,465 |
| FinTMM | 4,540 | free_text: 3,587 / boolean: 746 / numerical: 207 |
| **合计** | **14,319** | |

其中有图片记录 11,428 条（79.8%），无图片记录 2,891 条（FinancialTable / News 类，仅文本）。

数据来源说明：

- **FinMMR**：使用 easy / medium / hard 三个 validation split；保留图片、context、operator statistics。
- **FinMME**：使用 train split；将 base64 图片解码到磁盘，并保留 `verified_caption`、`related_sentences`、`tolerance`。
- **FinTMM**：使用 train split；通过 `FinTMMContextBuilder` 将 StockPrice / FinancialTable / News / Chart 引用解析为结构化 context 和图片路径。

---

## 三、数据流 Pipeline

```
raw/ (三个数据集原始 JSON)
    |
    | Step 0 [已实现]: step0_preprocess.py
    v
data/cases.jsonl
    |
    | Step 1 [已实现]: step1_generate_trajectories.py
    v
data/trajectories.jsonl
    |
    | Step 2 [已实现(MVP)]: step2_judge.py
    v
data/labels.jsonl
    |
    | Step 3 [已实现(MVP)]: step3_extract_memories.py
    v
data/items.jsonl
    |
    | Step 4 [已实现(MVP)]: step4_build_index.py
    v
data/reasoningbank.json + data/index.faiss
```

当前 pipeline 是离线单向构建：`cases.jsonl -> trajectories -> labels -> items -> reasoningbank`。测试阶段默认只读静态库，不在答题后将新经验回写到库中。

---

## 四、Experience 与 Memory Schema

论文原版的 memory schema 只有 `{title, description, content}` 三组件。当前项目需要区分两个层级：

- **experience-level**：一条题目及其轨迹、判定结果、关联记忆项；这是检索单元。
- **memory item-level**：从单条 experience 中抽取出的 1-3 条策略；这是注入单元。

### 4.1 Experience Schema [已实现（MVP）]

`reasoningbank.json` 中的每条 experience 计划组织为：

```json
{
  "experience_id": "exp_xxxxxxxx",
  "source_question_id": "finmme_1234",
  "query": "原题文本",
  "trajectory": "完整推理轨迹",
  "status": "success | failure",
  "memory_items": [
    {"title": "...", "description": "...", "content": "..."}
  ]
}
```

说明：

- **检索粒度对齐论文**：向量索引按 experience 建，而不是按单个 memory item 建。
- 在线阶段检索到 top-k experience 后，注入其关联的全部 memory items。
- 当前离线版本默认 `k=1`。

### 4.2 Memory Item Schema [已实现（MVP）]

memory item 采用论文原版三组件，并保留最小必要扩展字段。为保持轻量化，V1 暂不引入 `language` / `chart_type`。

```json
{
  "title": "简短标题",
  "description": "一句话适用场景描述",
  "content": "1-3 句可操作指导",
  "source": "success | failure",
  "memory_type": "strategy | warning",
  "failure_type": "读数错误 | 字段混淆 | 公式错误 | 单位混乱 | 预计算值误用 | 数据遗漏 | 范围界定错误 | 其他",
  "calc_type": ["arithmetic", "ratio", "compound", "extraction", "reasoning", "multi_step"],
  "answer_type": "numerical | mcq | boolean | free_text"
}
```

字段状态说明：

| 字段 | 状态 | 说明 |
|------|------|------|
| `title` / `description` / `content` | 已实现（MVP） | 论文原版核心三组件 |
| `source` / `memory_type` | 已实现（MVP） | success / failure 与 strategy / warning 标签 |
| `failure_type` | 已实现（MVP） | 失败经验根因标签 |
| `calc_type` / `answer_type` | 已实现（MVP） | 在 `items.jsonl` 保留自描述字段 |
| `chart_type` / `language` | 计划扩展 | V1 不启用 |

设计约束：

- 每条轨迹最多提取 **3 个**记忆项。
- success / failure 使用独立 prompt 提取。
- 禁止出现具体数值、公司名、年份等强绑定信息，避免记忆退化为示例复述。

---

## 五、离线构建各步骤

### 5.1 Step 0：数据预处理 [已实现]

`step0_preprocess.py` 将 FinMMR、FinMME、FinTMM 统一整理为 `cases.jsonl`。当前真正已落地的统一字段如下：

```json
{
  "id": "finmmr_xxx | finmme_xxx | fintmm_xxx",
  "source": "finmmr | finmme | fintmm",
  "image_paths": ["images/..."],
  "question": "...",
  "context": "...",
  "options": "... | null",
  "gold_answer": "...",
  "answer_type": "numerical | mcq | boolean | free_text",
  "calc_type": ["arithmetic", "ratio", "..."],
  "decimal_places": 2,
  "metadata": {"source_dataset": "...", "tolerance": null}
}
```

关键处理逻辑：

- **FinMMR calc_type 推导**：基于 `operator_statistics` 的算子分布自动推导；含 `/` 或 `%` 标记为 ratio，含 `**` 标记为 compound，算子数 >= 3 追加 multi_step。
- **FinMME calc_type 推导**：选择题标记为 reasoning；含百分比、增长率、ratio 等关键词时标记为 ratio，其余默认为 arithmetic。
- **FinMME 图片提取**：将 base64 图片解码到 `images/finmme/`，支持 `--skip-images` 跳过重复写入。
- **FinTMM context 构建**：将 StockPrice / FinancialTable / News / Chart 引用转为结构化文本 context，并补充 K 线图路径。
- **answer_type 分类**：FinTMM 中基于规则将答案识别为 boolean / numerical / free_text。
- **安全过滤**：丢弃 id 中含 `test` 的记录，避免测试集泄露。

运行方式：

```bash
python step0_preprocess.py
python step0_preprocess.py --skip-images
python step0_preprocess.py --raw-dir /data/raw
```

### 5.2 Step 1：CoT 轨迹生成 [已实现]

`step1_generate_trajectories.py` 使用 Gemini 为每条 case 生成推理轨迹。

| 配置项 | 值 |
|--------|-----|
| 模型 | `gemini-2.5-flash` |
| temperature | 0.7 |
| thinking_budget | 2048 |
| 接入方式 | AiHubMix Gemini 兼容接口 + `google-genai` SDK |
| 并发控制 | `asyncio.Semaphore(10)` |
| 重试策略 | 指数退避，最多 3 次 |

Prompt 结构：

1. system prefix：要求逐步完成金融推理。
2. 若有 context，注入 `=== Context ===`。
3. 若有 options，注入 `=== Options ===`。
4. 注入 `=== Question ===`。
5. 按 `answer_type` 追加输出格式约束；数值题要求严格匹配小数位。
6. 强制输出 `**Final Answer:** <answer>` 结尾。

补充说明：

- 有图题将图片与文本作为 multimodal parts 一起发送。
- `predicted` 通过正则提取 `**Final Answer:**`，并兼容 `\boxed{}` fallback。
- 支持断点续跑：已存在于 `trajectories.jsonl` 的 id 会被跳过。

### 5.3 Step 2：成功 / 失败判定 [已实现（MVP）]

当前设计与论文的关键差异在于：论文是**无 ground truth 的 test-time self-judge**，而本项目离线建库阶段拥有 `gold_answer`，因此 Step 2 以规则判定为主，仅在 free_text 场景使用 LLM Judge。这一点比论文原始设置更稳定。

判定策略：

| answer_type | 判定方法 |
|-------------|---------|
| numerical（FinMMR / FinTMM） | 精确匹配，容差 `1e-9` |
| numerical（FinMME） | `abs(predicted - gold) <= tolerance` |
| mcq | 归一化后比较字母集合 |
| boolean | 大小写归一化后精确匹配 |
| free_text | Gemini Judge，temperature=0.0 |

free_text Judge 的协议建议为：

- **输入**：question、context、trajectory、predicted、gold_answer。
- **目标**：判断预测是否在语义上等价于标准答案，并解释关键证据。
- **输出**：先给简短理由，再给 `Status: success | failure`。

文件输出仍采用 `labels.jsonl`：

```json
{"id": "...", "status": "success", "judge_reason": "..."}
```

### 5.4 Step 3：记忆提取 [已实现（MVP）]

该步骤对应论文 Appendix A.1 的 memory extraction。设计上对 success / failure 使用独立 prompt，但统一约束输出为结构化 Markdown，便于解析。

- **成功轨迹**：总结为什么成功，提取 strategy 型记忆。
- **失败轨迹**：反思失败原因，提取 warning 型记忆。
- **数量约束**：每条轨迹最多 3 个记忆项。
- **去重约束**：禁止输出语义重叠项。
- **泛化约束**：禁止提及具体公司、年份、题面字符串、具体数值。

建议输出格式：

```markdown
# Memory Item 1
## Title <title>
## Description <one-sentence summary>
## Content <1-3 sentences>
```

失败经验额外给出根因标签，当前使用 8 类：读数错误、字段混淆、公式错误、单位混乱、预计算值误用、数据遗漏、范围界定错误、其他。

实现补充：

- `items.jsonl` 中保留 `calc_type` 与 `answer_type`，作为中间产物的自描述字段。
- Step 4 组装 `reasoningbank.json` 时会忽略 memory item 上的这两个冗余字段，统一使用 experience-level metadata 作为检索过滤入口。

### 5.5 Step 4：经验库与索引构建 [已实现（MVP）]

该步骤对齐论文 Appendix A.2，但 embedding 模型采用本地 `NV-Embed-v2`，不使用论文中的 `gemini-embedding-001`。

| 配置项 | 值 |
|--------|-----|
| 检索单元 | experience |
| 注入单元 | 该 experience 关联的全部 memory items |
| Embedding model | 本地 `NV-Embed-v2/` |
| Query instruction | `Given a financial question, retrieve relevant passages that answer the query` |
| Corpus instruction | 空字符串 |
| 索引类型 | FAISS `IndexFlatIP` |
| 相似度 | L2 归一化后的 inner product，等价 cosine |
| 整合策略 | append-only，直接追加，不做 pruning（对齐论文 Appendix A.2） |

构建方式：

1. 从 `cases.jsonl` 读取原题文本，作为 experience 的 query。
2. 将 `items.jsonl` 按 `source_question_id` 分组，归并到对应 experience。
3. 以 experience 为单位计算向量并写入 `index.faiss`。
4. 输出 `index_meta.json`，保存 `faiss_row -> experience_id` 及过滤字段映射。
5. 将 experience 及其 memory items 持久化到 `reasoningbank.json`。

说明：当前文档中的 `items.jsonl` 仍是中间产物，最终检索不直接对单个 item 建索引。

---

## 六、在线推理流程 [设计中]

当前在线推理设计遵循论文的 **experience retrieval -> memory injection -> response generation** 主线，但在运行形态上保持静态库推理，不做在线闭环回写。

```
新题目 (image + question)
    |
    | 使用 NV-Embed-v2 编码当前题目
    v
experience-level semantic retrieval
    |
    | 检索 top-1 most similar experience
    | 取该 experience 关联的全部 memory items
    v
构建 Prompt
    |
    | 注入记忆项 title + content
    | 要求模型先判断每条记忆是否相关，再继续作答
    v
Qwen3-VL-8B-Instruct
    v
最终答案
```

检索设计说明：

- **默认方案 [已实现（索引侧）]**：experience-level semantic retrieval，严格对齐论文。
- **Stage 1 hard filter [已实现（索引侧）]**：仅使用 `answer_type + calc_type`，在 `index_meta.rows` 上过滤候选集。
- **Stage 1 扩展 [计划扩展]**：未来可再增加 `language` / `chart_type`，V1 暂不启用。
- **检索数量**：默认 `k=1`，依据论文 Appendix C.1；检索更多 experience 往往会引入噪声。

记忆注入方式建议参考论文 Appendix A.2，采用显式提示而非简单拼接。可使用如下模板：

```text
Below are some memory items accumulated from past solved tasks that may be helpful.
For each memory item, first briefly decide whether it is relevant to the current problem, and then continue your reasoning.
```

注入内容建议保留 `title + content`，不直接暴露原始轨迹。

当前不做的事情：

- **不在线回写**：测试集推理完成后，不将新题轨迹追加到 `reasoningbank.json`。
- **不做 streaming 更新**：后续题目不会受当前题生成的新记忆影响。

---

## 七、目录结构

当前目录中真实存在的核心内容如下：

```text
ReasoningRag/
├── NV-Embed-v2/                    # 已加入，本地 embedding 模型
├── data/                           # cases.jsonl、trajectories.jsonl 等中间产物
├── images/                         # 三个数据集的图片资源
├── raw/                            # 原始数据集 JSON
├── prepare_ab_test_data.py         # A/B 测试数据准备脚本
├── run_qwen_ab_test.py             # Qwen A/B 测试脚本
├── run.sh                          # 运行脚本
├── step0_preprocess.py             # 已实现
├── step1_generate_trajectories.py  # 已实现
└── 备份/                           # 备份目录
```

规划中的核心模块如下：

| 文件 | 状态 | 作用 |
|------|------|------|
| `step2_judge.py` | 已实现（MVP） | 成功 / 失败判定 |
| `step3_extract_memories.py` | 已实现（MVP） | 记忆提取 |
| `step4_build_index.py` | 已实现（MVP） | experience bank、FAISS 与 index_meta 构建 |
| `inference.py` | 设计中 | 静态记忆库在线推理 |

---

## 八、当前进展

当前已完成 Step 0-4 的 MVP。可稳定产出 `cases.jsonl`、`trajectories.jsonl`、`labels.jsonl`、`items.jsonl`、`reasoningbank.json`、`index.faiss` 与 `index_meta.json`。本地 `NV-Embed-v2` 已接入 experience-level 索引构建。在线 `inference.py` 仍处于设计中；论文中的在线闭环更新和 MaTTS 暂不启用。

最近一次敏捷删减决策：

- 为保持轻量与低耦合，V1 从 Stage 1 hard filter 中移除 `language` / `chart_type`，仅保留 `answer_type + calc_type`。
- `items.jsonl` 保留 `calc_type` / `answer_type` 做中间产物自描述；`reasoningbank.json` 则在 experience-level metadata 统一承载检索字段，避免在线路径重复读取与歧义。

---

## 九、扩展方向 [计划扩展]

### 9.1 在线闭环 ReasoningBank

若后续将当前系统改为 streaming 评测模式，则可以在每道新题完成后执行：trajectory -> judge -> extract -> consolidate，并将新增 experience 立即写回 memory bank，使后续题目受益于当前题经验。这一能力对应论文 Figure 2 的 closed-loop 机制。

### 9.2 MaTTS

论文提出的 MaTTS（Memory-aware Test-Time Scaling）有两种形式：

- **parallel scaling**：同一道题并行生成多条轨迹，通过 self-contrast 提炼更可靠记忆。
- **sequential scaling**：对同一道题做多轮自检 / 精修，通过 self-refinement 产出更高质量记忆。

MaTTS 的前提是“当前题产生的新记忆能够影响后续题”，因此它天然依赖在线闭环。当前 ReasoningRag 的离线静态建库方式与 MaTTS 不冲突，但不能直接复现论文中的“更多计算 -> 更好记忆 -> 更有效计算”正反馈。若后续切换到 streaming 场景，MaTTS 是最自然的下一步扩展。

