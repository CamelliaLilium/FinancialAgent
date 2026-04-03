# Data Schema — ReasoningRAG

> 各数据文件的字段定义与演化说明。

---

## 1. cases.jsonl — 原始问题库

**路径**: `data/cases.jsonl`

```json
{
  "id": "finmmr_easy-validation-0",
  "question": "...",
  "context": "...",                  // 可选，财务表格/段落
  "options": "A. ... B. ...",        // 仅 MCQ 类型
  "gold_answer": "42.5",
  "answer_type": "numerical",        // numerical | mcq | boolean | free_text
  "decimal_places": 2,               // 仅 numerical
  "calc_type": ["ratio", "growth"],  // 计算类型标签
  "image_paths": ["images/xxx.png"]  // 可选，图片路径（相对 ReasoningRag/）
}
```

---

## 2. trajectories.jsonl — 单轨迹推理记录

**路径**: `data/trajectories.jsonl`

```json
{
  "id": "finmmr_easy-validation-0",
  "run_id": 0,                        // 同题第几次运行 (0=原始单次)
  "trajectory": "Step 1: ...\n...",
  "predicted": "42.5",
  "is_correct": true,
  "answer_type": "numerical",
  "decimal_places": 2,
  "model": "gemini-2.5-flash",
  "temperature": 0.0
}
```

### 扩展字段 (B1/B2 阶段填充)

```json
{
  "quality": {
    "internal_consistency": 0.8,      // 推理内部自洽性 [0,1]
    "term_overlap_rate": 0.7,         // 金融术语覆盖率 [0,1]
    "reasoning_step_count": 5,        // 推理步数 (int)
    "logical_coherence": 0.85,        // 逻辑连贯性 [0,1]
    "content_diversity": 0.6,         // 内容多样性 [0,1]
    "task_domain_relevance": 0.88,    // 任务领域相关性 [0,1]
    "instruction_alignment": 0.9      // 指令对齐程度 [0,1]
  }
}
```

> 七维质量打分来自 Fin-R1 (arXiv:2503.16252)。

---

## 3. experience.jsonl — 结构化经验库 (reasoningbank)

**路径**: `data/experience.jsonl` (B2 阶段新建)

```json
{
  "experience_id": "exp_finmmr_easy-validation-0_r0",
  "source_question_id": "finmmr_easy-validation-0",
  "query": "Calculate revenue growth rate...",   // 用于检索的查询文本
  "best_trajectory": "Step 1: ...\n...",
  "status": "success",                           // success | failure
  "contrastive": {
    "success_trajectory": "...",
    "failure_trajectory": "...",
    "success_is_correct": true,
    "failure_is_correct": false,
    "contrastive_note": "正确做法使用 [(B-A)/A]，错误做法使用 [B/A-1 后忘记减1]"
  },
  "memory_items": [
    {
      "type": "strategy",
      "text": "计算增长率时确保分母是基期值，结果减1"
    }
  ],
  "utility": 0.0                                 // D1 后续填充: 被检索后答对率
}
```

---

## 4. ab_test/ — A/B 实验数据

### targets_finmmr_100.jsonl

**路径**: `data/ab_test/targets_finmmr_100.jsonl`

cases.jsonl 的子集，用于 A/B 实验。字段同 cases.jsonl。

### manifest_finmmr_100_3shot.jsonl (IDF)

**路径**: `data/ab_test/manifest_finmmr_100_3shot.jsonl`

```json
{
  "target_id": "finmmr_easy-validation-0",
  "shots": 3,
  "retrieval": "idf",
  "examples": [
    {
      "id": "...",
      "question": "...",
      "gold_answer": "...",
      "answer_type": "..."
    }
  ]
}
```

### manifest_finmmr_100_3shot_faiss.jsonl (FAISS)

**路径**: `data/ab_test/manifest_finmmr_100_3shot_faiss.jsonl` (A1 实验产出)

字段同上，`retrieval` 字段为 `"faiss"`。

---

## 5. 实验结果文件

### results.jsonl (各实验)

```json
{
  "target_id": "finmmr_easy-validation-0",
  "gold_answer": "42.5",
  "answer_type": "numerical",
  "voted_answer": "42.5",
  "correct": true,
  "k": 5,
  "latency_sec": 12.3
}
```

### DSER results.jsonl (C1 扩展)

```json
{
  "target_id": "...",
  "chains": [
    {
      "chain_id": 0,
      "history": [
        {"response": "...", "answer": "42.5", "round": 0},
        {"response": "...", "answer": "42.5", "round": 1}
      ],
      "final_answer": "42.5",
      "error": ""
    }
  ],
  "final_answers": ["42.5", "42.5", "43.0", "42.5", "42.5"],
  "voted_answer": "42.5",
  "correct": true,
  "refine_rounds": 1
}
```

### summary.json

```json
{
  "model": "qwen3-vl-8b-instruct",
  "k": 5,
  "n": 100,
  "accuracy": 0.68,
  "correct": 68
}
```

---

## 6. FAISS 索引文件

| 文件 | 内容 |
|------|------|
| `data/index.faiss` | NV-Embed-v2 4096维 IP 索引，存储 cases.jsonl 所有问题向量 |
| `data/index_meta.json` | 索引元数据 (已删除，重建时自动生成) |

> NV-Embed-v2 模型路径: `ReasoningRag/NV-Embed-v2/`
> 维度: 4096，相似度: 内积 (cosine 经 L2 归一化后等价)
