# ReasoningRAG 项目设计文档

基于 ReasoningBank (arXiv:2509.25140) 的结构化记忆系统，面向金融多模态 QA 场景。

---

## 一、项目定位与核心思想

本项目将传统 RAG（存储原始 QA 对，检索相似例题）重构为基于 **ReasoningBank** 的结构化记忆系统，核心转变为：**从"检索相似例题"到"检索可迁移推理策略"**。

ReasoningBank 论文的关键洞察：

- 从成功 **和** 失败经验中蒸馏高层可迁移推理策略，而非直接存储原始 QA 三元组。
- 记忆项以结构化 schema 存储（title / description / content），形成闭环：检索记忆 → 指导推理 → 提取新记忆 → 整合回库。
- 失败轨迹的引入对 ReasoningBank 有 +3.2 的额外提升（从 46.5 → 49.7），而对 Synapse / AWM 等基线无效甚至有害。
- 检索 top-1 记忆即可获得最优性能（k=1，论文附录 C.1 消融实验：k=1: 49.7, k=2: 46.0, k=3: 45.5），k 增大反而引入噪声。

系统由三个 LLM 角色组成（均可使用同一 backbone model）：

| 角色 | 职责 | temperature |
|------|------|-------------|
| Agent | 执行推理（CoT 轨迹生成 + 在线推理） | 0.7 |
| Extractor | 从轨迹中提取记忆项 | 1.0 |
| Judge (LLM-as-a-Judge) | 判定 free_text 类答案的正误 | 0.0 |

---

## 二、数据集概览

系统覆盖三个金融多模态数据集，合计 14,319 条记录。

| 数据集 | 条数 | answer_type 分布 |
|--------|------|-----------------|
| FinMMR | 900 | numerical: 900 |
| FinMME | 8,879 | mcq: 7,414 / numerical: 1,465 |
| FinTMM | 4,540 | free_text: 3,587 / boolean: 746 / numerical: 207 |
| **合计** | **14,319** | |

其中有图片记录 11,428 条（79.8%），无图片记录 2,891 条（FinancialTable / News 类，仅传文本）。

数据来源说明：

- **FinMMR**：使用 easy / medium / hard 三个 validation split，每条包含图表图片、结构化 operator_statistics 用于 calc_type 推导。
- **FinMME**：使用 train split，包含 base64 编码图片（预处理时解码写入磁盘）、verified_caption、related_sentences 作为 context，以及数据集自带的 tolerance 字段。
- **FinTMM**：使用 train split，source 引用指向 StockPrice / FinancialTable / News / Chart 四类辅助数据，由 `FinTMMContextBuilder` 动态解析为结构化 context 和图片路径。

---

## 三、数据流 Pipeline

```
raw/ (三个数据集原始 JSON)
    |
    | Step 0: step0_preprocess.py
    v
data/cases.jsonl (14,319 条统一格式记录)
    |
    | Step 1: step1_generate_trajectories.py (Gemini-2.5-flash, temp=0.7)
    v
data/trajectories.jsonl  {id, trajectory, predicted, answer_type, decimal_places}
    |
    | Step 2: step2_judge.py (规则匹配 + Gemini Judge,, temp=0.0)
    v
data/labels.jsonl        {id, status, judge_reason}
    |
    | Step 3: step3_extract_memories.py (Gemini Extractor, temp=1.0)
    v
data/items.jsonl         {item_id, source_id, memory_type, title, description, content, ...}
    |
    | Step 4: step4_build_index.py (text-embedding-004 + FAISS)
    v
data/reasoningbank.json + data/index.faiss
```

每一步均支持断点续跑，通过读取已有输出文件的 id 集合跳过已完成记录。

---

## 四、Memory Item Schema

每条记忆项遵循以下结构化 schema：

```json
{
  "item_id": "mem_xxxxxxxx",
  "source_question_id": "finmme_1234",
  "source": "success | failure",
  "memory_type": "strategy | warning",
  "title": "简短标题",
  "description": "一句话适用场景描述",
  "content": "1-3 句可操作指导",
  "failure_type": "读数错误 | 字段混淆 | 公式错误 | 单位混乱 | 预计算值误用 | 数据遗漏 | 范围界定错误 | 其他",
  "calc_type": ["arithmetic", "ratio", "compound", "extraction", "reasoning", "multi_step"],
  "chart_type": "table | bar | line | bar_line_dual | pie | mixed_annotation | map | other",
  "answer_type": "numerical | mcq | boolean | free_text",
  "language": "zh | en"
}
```

设计要点：

- 每条轨迹最多提取 **3 个**记忆项。
- 成功 / 失败轨迹使用**独立 prompt** 提取：成功 → strategy 型记忆，失败 → warning 型记忆。
- `failure_type` 为失败轨迹专属字段，覆盖 8 类根因分类。
- 所有策略**禁止出现**具体数值、公司名、年份，确保跨场景可迁移性。
- `calc_type` / `chart_type` 等元数据用于在线推理时的硬过滤检索（详见第六节）。

---

## 五、离线构建各步骤

### 5.1 Step 0: 数据预处理

将 FinMMR + FinMME + FinTMM 原始数据统一为 `cases.jsonl`。

每条记录的统一 schema：

```json
{
  "id": "finmmr_xxx | finmme_xxx | fintmm_xxx",
  "source": "finmmr | finmme | fintmm",
  "image_paths": ["images/finmmr/xxx.png"],
  "question": "...",
  "context": "...",
  "options": "A. ... B. ... | null",
  "gold_answer": "...",
  "answer_type": "numerical | mcq | boolean | free_text",
  "calc_type": ["arithmetic", "ratio", ...],
  "decimal_places": 2,
  "metadata": { "source_dataset": "...", "tolerance": ..., ... }
}
```

关键处理逻辑：

- **FinMMR calc_type 推导**：基于 `operator_statistics` 中的算子分布自动推导。包含 `/` 或 `%` → ratio，包含 `**` → compound，仅有 `+`/`-`/`*` → arithmetic，总算子数 >= 3 时追加 multi_step。
- **FinMME calc_type 推导**：选择题 → reasoning，含百分比/增长率关键词 → ratio，其他 → arithmetic。
- **FinMME 图片提取**：将 JSON 中 base64 编码的图片解码写入 `images/finmme/`，支持 `--skip-images` 跳过已有图片。
- **FinTMM context 构建**：`FinTMMContextBuilder` 根据 source 列表动态查询 StockPrice / FinancialTable / News 辅助数据，拼接为结构化文本 context；Chart 类 source 解析为图片路径；StockPrice 类额外尝试关联 K 线图。
- **FinTMM answer_type 分类**：基于规则——yes/no/true/false → boolean，可解析为浮点数 → numerical，其余 → free_text。
- **安全过滤**：拒绝任何 id 包含 "test" 的记录（防止测试集泄露）。
- **去重**：按 id 去重。

运行方式：

```bash
python step0_preprocess.py                      # 默认参数运行
python step0_preprocess.py --skip-images        # 跳过 FinMME 图片解码
python step0_preprocess.py --raw-dir /data/raw  # 指定原始数据目录
```

### 5.2 Step 1: CoT 轨迹生成

使用 Gemini 为每条 case 生成 Chain-of-Thought 推理轨迹。

| 配置项 | 值 |
|--------|-----|
| 模型 | `gemini-2.5-flash` |
| temperature | 0.7 |
| thinking_budget | 2048 tokens |
| 接入方式 | AiHubMix 中转站 (`https://aihubmix.com/gemini`)，`google-genai` SDK |
| 并发控制 | `asyncio.Semaphore(10)` |
| 重试策略 | 指数退避，最多 3 次（base=2s → 2s / 4s / 8s） |

Prompt 构建逻辑：

1. System prefix：`"You are a financial reasoning expert. Solve the following problem step by step."`
2. 若有 context，注入 `=== Context ===` 段。
3. 若有 options（选择题），注入 `=== Options ===` 段。
4. 注入 `=== Question ===` 段。
5. 根据 answer_type 追加格式指令（numerical 时注入精确小数位要求）。
6. 强制要求以 `**Final Answer:** <answer>` 结尾，禁止 LaTeX/boxed 格式。

图片处理：有 image_paths 时，将图片作为 multimodal content parts 与文本一起发送。

答案提取：正则匹配 `**Final Answer:**` 后的文本，fallback 提取 `\boxed{}` 格式。

断点续跑：读取已有 `trajectories.jsonl` 的 id 集合，跳过已完成记录。

运行方式：

```bash
python step1_generate_trajectories.py                # 全量运行
python step1_generate_trajectories.py --limit 5      # 仅处理前 5 条 pending
python step1_generate_trajectories.py --dry-run      # 打印第一条 prompt，不调 API
python step1_generate_trajectories.py --model gemini-2.5-pro --temperature 0.3
```

### 5.3 Step 2: 成功/失败判定

根据 answer_type 采用不同判定策略：

| answer_type | 判定方法 |
|-------------|---------|
| numerical (FinMMR / FinTMM) | 精确匹配（1e-9 容差） |
| numerical (FinMME) | `abs(predicted - gold) <= tolerance`（使用数据集自带的 tolerance 字段） |
| mcq | 提取字母集合，集合相等即为成功 |
| boolean | 大小写归一化后精确匹配 |
| free_text | Gemini Judge，temperature=0.0，确保判定确定性 |

输出 `labels.jsonl`，每条包含 `{id, status, judge_reason}`，其中 status 为 success 或 failure。

### 5.4 Step 3: 记忆提取

使用 Gemini Extractor（temperature=1.0，提高多样性）从轨迹中提取记忆项。

- **成功轨迹**：分析成功原因，提取 strategy 型记忆（可迁移推理策略）。
- **失败轨迹**：定位根因（8 类 failure_type 中选一），提取 warning 型记忆（警示信息）。

8 类失败根因：读数错误 / 字段混淆 / 公式错误 / 单位混乱 / 预计算值误用 / 数据遗漏 / 范围界定错误 / 其他。

每条轨迹最多提取 3 个记忆项，输出 `items.jsonl`。

### 5.5 Step 4: 索引构建

将记忆项编码为向量并构建检索索引。

| 配置项 | 值 |
|--------|-----|
| Embedding model | `text-embedding-004` |
| Embedding 拼接文本 | `title + description + content[:100]` |
| 建库 task_type | `retrieval_document` |
| 检索 task_type | `retrieval_query` |
| 索引类型 | FAISS `IndexFlatIP`（向量 L2 归一化后等价 cosine similarity） |
| 整合策略 | 直接追加不剪枝（论文极简方案，附录 C.2 确认剪枝无收益） |

输出 `reasoningbank.json`（记忆项全量数据）+ `index.faiss`（向量索引）。

---

## 六、在线推理流程

```
新题目 (image + question)
    |
    | 特征提取
    |   calc_type: 基于关键词规则推导
    |   language: 基于字符检测 (zh / en)
    v
两阶段检索
    |
    | Stage 1 -- 硬过滤
    |   按 calc_type 过滤候选池
    |   若过滤后不足 5 条，fallback 至全库
    |
    | Stage 2 -- 语义排序
    |   embed(question) 与候选记忆 cosine 排序
    |   分别取 strategy top-1 + warning top-1
    v
构建 Prompt
    |
    |   <参考策略> strategy 记忆的 content </参考策略>
    |   <注意事项> warning 记忆的 content </注意事项>
    v
推理模型: Qwen3-VL-8B-Instruct --> 最终答案
```

**检索数量 top-1 的依据**：论文附录 C.1 消融实验表明 k=1 性能最优（49.7），k 增大反而引入噪声导致下降（k=2: 46.0, k=3: 45.5, k=4: 44.4）。本项目检索 strategy top-1 + warning top-1，即每个类型各取一条最相关记忆。

**两阶段检索设计的考量**：

- Stage 1 硬过滤通过 calc_type 元数据缩小候选范围，避免语义相似但计算类型不匹配的记忆干扰（如 ratio 类问题检索到 extraction 类策略）。
- 5 条阈值是 fallback 保护：当特定 calc_type 的记忆条数过少时，退回全库检索以保证召回率。
- Stage 2 纯语义排序确保在同类型候选中选出与当前问题最相关的记忆。

---

## 七、目录结构

```
ReasoningRag/
├── .env                              # AIHUBMIX_API_KEY（已 gitignore）
├── .gitignore
├── run.sh                            # Pipeline 运行脚本
├── step0_preprocess.py               # Step 0: 数据预处理
├── step1_generate_trajectories.py    # Step 1: CoT 轨迹生成
├── step2_judge.py                    # Step 2: 成功/失败判定（待实现）
├── step3_extract_memories.py         # Step 3: 记忆提取（待实现）
├── step4_build_index.py              # Step 4: 索引构建（待实现）
├── inference.py                      # 在线推理（待实现）
├── raw/                              # 原始数据集 JSON
│   ├── finmmr_*_validation.json      # FinMMR 三个 split
│   ├── finmme_train.json             # FinMME train
│   ├── fintmm_train.json             # FinTMM train
│   └── fintmm_data/                  # FinTMM 辅助数据
│       ├── StockPrice.json
│       ├── FinancialTable.json
│       └── News.json
├── data/                             # Pipeline 中间产物与最终输出
│   ├── cases.jsonl                   # Step 0 输出：14,319 条统一格式
│   ├── trajectories.jsonl            # Step 1 输出：CoT 轨迹
│   ├── labels.jsonl                  # Step 2 输出：判定标签
│   ├── items.jsonl                   # Step 3 输出：记忆项
│   ├── reasoningbank.json            # Step 4 输出：记忆库
│   └── index.faiss                   # Step 4 输出：向量索引
└── images/                           # 图片资源
    ├── finmmr/                       # FinMMR 图表图片
    ├── finmme/                       # FinMME 图片（Step 0 解码生成）
    └── fintmm/                       # FinTMM Chart / K 线图
```

---

## 八、当前进展

Step 0 数据预处理已完成，输出 14,319 条记录，校验全部通过。Step 1 CoT 轨迹生成代码已完成并通过测试。Step 2-4 及在线推理模块待实现。
