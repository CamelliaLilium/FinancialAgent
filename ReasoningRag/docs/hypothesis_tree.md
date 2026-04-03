# Hypothesis Tree — ReasoningRAG

> 探索树：每个节点是一个可验证假设，箭头表示依赖/后继关系。

## 已确认基线

```
ZeroShot (Qwen3-VL-8B)
  └─► Baseline IDF 3-shot:  +5.67pp  ✓ CONFIRMED (p=0.0015)
        └─► 单轨迹记忆注入 (13轮):  ≈0 / 负  ✗ REFUTED
              根因: 小模型无法执行元推理 + 上下文干扰
```

---

## 分支 A — 改进检索质量

```
A1: FAISS 语义检索替换 IDF  [待验证]
  假设: NV-Embed-v2 FAISS 相似示例 > IDF 词汇匹配
  脚本: prepare_ab_test_data.py --retrieval faiss
  验收: Δ ≥ +2pp, p < 0.05

  ├─► 成立 → A2: 语义示例 + 成功轨迹 vs 纯 Q+A
  │         假设: 加入轨迹的语义示例 > 纯 Q+A 语义示例
  │
  └─► 不成立 → IDF 已是瓶颈上限，转向 B/C 分支
```

---

## 分支 B — 多轨迹生成 + 经验提取

```
B0: Self-Consistency@5 (Gemini teacher)  [待验证, 独立]
  假设: 5轨迹多数投票 > 单次贪心
  脚本: run_self_consistency.py --k 5
  验收: SC@5 accuracy > single-pass

  ├─► 成立 → B1: K=5 轨迹 + Fin-R1 七维质量打分选 best_trajectory
  │         └─► B2: 成功/失败轨迹提取结构化经验 (contrastive pairs)
  │               └─► B3: contrastive pair 注入 (同题对比)
  │                     假设: 具体对比注入 > 抽象策略注入
  │                     验收: B3 > B0-baseline, p < 0.05
  │
  └─► 不成立 → 多轨迹方向存疑，降优先级

B1 补充路径: Best-of-N + Fin-PRM (无 gold_answer 时)
  ⚠️ 通用 PRM 在金融高 N 反而 hurt，必须用 aliyun/qwen-dianjin (Fin-PRM)
  论文: arXiv:2508.15202
```

---

## 分支 C — 推理时自进化（与 B 正交）

```
C1: DSER (Qwen3-VL-8B)  [待验证, 不依赖记忆库]
  K=5 并行 Markov 链 + M=1 自精化轮次 + 多数投票
  脚本: run_dser.py --k 5 --refine-rounds 1
  验收: DSER > ZeroShot, same 100 targets
  参考: arXiv:2510.17498 (Qwen3-8B AIME超600B单次已验证)

  ├─► 成立 → C2: SE-Agent revision loop
  │         失败轨迹 → 定向 revision prompt → 新轨迹
  │         对比: C2 vs C1 (revision vs generic self-refine)
  │
  └─► 不成立 → 推理时自进化对 8B 无效，8B 路径降优先级
```

---

## 分支 D — 记忆治理（中长期）

```
D1: MEMRL 风格 — experience utility score
  成功率 = 被检索后答对次数 / 总被检索次数
  检索时: combine 语义相似度 + utility
  依赖: B3 完成并积累足够 experience 数据

D2: ReMe 风格 — 记忆去重、融合、剪枝
  新经验: 抽取 → 验证 → 去重 → 融合（非直接追加）
  依赖: D1
```

---

## 推荐执行顺序

```
第一步 (并行):
  ┌── A1: FAISS 语义检索 (~0.5天)
  ├── B0: Self-Consistency@5 (最快, 只需调 Gemini K次)
  └── C1: DSER (独立, 不依赖记忆库)

第二步: 根据 B0 决策
  B0 有增益 → B1 (多轨迹数据基础设施)
  B0 无增益 → 专注 A + C

第三步: B1 + B2 (多轨迹数据 + 经验提取)

第四步: B3 vs A2 对比

第五步: D1/D2 (长期)
```

---

## 文献支撑

| 假设/方法 | 论文 | 关键发现 |
|-----------|------|----------|
| Self-Consistency | arXiv:2203.11171 | N轨迹→多数投票提升准确率 |
| DSER | arXiv:2510.17498 | K并行链+迭代自精化，Qwen3-8B AIME超600B |
| Fin-PRM | arXiv:2508.15202 | 通用PRM金融高N反而hurt，用qwen-dianjin |
| Fin-R1 质量打分 | arXiv:2503.16252 | 7维轨迹质量标准 |
| Fino1 FinCoT | arXiv:2502.08127 | 公开金融推理数据，含backtracking+verification |
| 上下文干扰推理 | Shi et al. ICML 2023 | 不相关上下文 hurt 推理准确率 |
| 长上下文降低推理 | arXiv:2510.05381 | 上下文长度本身是噪声 |
| SE-Agent | arXiv:2508.02085 | 失败轨迹→revision prompt |
| Best-of-N + PRM | arXiv:2502.18581 | 冻结PRM打分选优 |
