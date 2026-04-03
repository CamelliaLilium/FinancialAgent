# Experiment Log — ReasoningRAG

> 本文档记录每轮实验的配置、结果与结论。按时间倒序排列，最新在前。

---

## Round 13 (已完成) — 对比注入失败根因分析

**日期**: 2026-03

**实验**: 各类记忆注入变体（自进化对比注入 v1–v13）

**配置**:
- 模型: Qwen3-VL-8B（冻结）
- 方法: 将成功/失败轨迹提取的抽象策略/警告注入 few-shot 提示
- 数据: FinMMR 100 targets

**结果**: 13轮实验，无显著提升（Δ ≈ 0 甚至负）

**根因** (文献确认):
1. 不相关上下文干扰推理 (Shi et al., ICML 2023)
2. 上下文长度本身降低准确率 (arXiv:2510.05381)
3. 8B 冻结模型无法执行"先判断记忆相关性再推理"的元推理
4. ReasoningBank 从未在小模型离线配置下验证

**结论**: 抽象策略/警告文本注入路径无效。新方向：多轨迹 + 推理时自进化。

---

## Round 0 (基线, 已完成) — ZeroShot vs IDF 3-shot

**日期**: 2026-02

**实验**: Baseline (IDF 3-shot Q+A) vs ZeroShot

**配置**:
- 模型: Qwen3-VL-8B（冻结）
- 数据: FinMMR 100 targets
- 检索: IDF 词汇匹配选 few-shot 示例

**结果**:
- ZeroShot accuracy: baseline
- IDF 3-shot: **+5.67pp** (p=0.0015, 显著)

**结论**: IDF 3-shot 有效，确认 few-shot 路径可行。

---

## 待运行实验

### A1 — FAISS 语义检索替换 IDF（当前最优先）

**状态**: 已实现，待运行

**配置**:
```bash
# 生成 FAISS 版 manifest
python prepare_ab_test_data.py --retrieval faiss --shots 3

# 运行 A/B 测试
python run_qwen_ab_test.py \
  --targets-path data/ab_test/targets_finmmr_100.jsonl \
  --manifest-path data/ab_test/manifest_finmmr_100_3shot_faiss.jsonl
```

**假设**: 语义相似示例 → 更高准确率（vs IDF）

**验收**: Δ >= +2pp, p < 0.05

---

### B0 — Self-Consistency@5（Gemini teacher）

**状态**: 已实现，待运行

**配置**:
```bash
python run_self_consistency.py \
  --k 5 \
  --model gemini-2.5-flash \
  --output-dir test/results/b0_self_consistency
```

**假设**: SC@5 多数投票 > 单次 Gemini 贪心推理

**验收**: SC@5 accuracy > single-pass

---

### C1 — DSER（Qwen3-VL-8B 推理时自进化）

**状态**: 已实现，待运行

**配置**:
```bash
python run_dser.py \
  --k 5 \
  --refine-rounds 1 \
  --output-dir test/results/c1_dser
```

**假设**: K=5 并行链 + 自精化 > 单次推理

**验收**: DSER accuracy > ZeroShot（同 100 targets）

---

## 实验结果汇总表

| 实验 | 方法 | 模型 | N | Accuracy | Δ vs ZeroShot | p | 状态 |
|------|------|------|---|----------|---------------|---|------|
| ZeroShot | 0-shot | Qwen3-VL-8B | 100 | — | 0 | — | ✓ 完成 |
| Baseline (IDF 3-shot) | IDF 词汇检索 | Qwen3-VL-8B | 100 | — | +5.67pp | 0.0015 | ✓ 完成 |
| A1 (FAISS 3-shot) | NV-Embed-v2 FAISS | Qwen3-VL-8B | 100 | — | TBD | — | 待运行 |
| B0 (SC@5) | Self-Consistency@5 | Gemini-2.5-Flash | 100 | — | TBD | — | 待运行 |
| C1 (DSER) | K=5 链 + 自精化 | Qwen3-VL-8B | 100 | — | TBD | — | 待运行 |
