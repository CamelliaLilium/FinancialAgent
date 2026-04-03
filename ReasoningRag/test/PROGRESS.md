# BizBench 文本推理实验决策记录

更新时间：2026-03-24

## 1. 项目目标与验收标准

- 主目标：在 BizBench 文本题上验证小模型的 self-evolution 是否能稳定提升推理正确率；当前关心的是 `Treatment_SF` 相对 `Baseline` 的增益，而不是单纯证明 RAG 有用。
- 对照定义：`ZeroShot` = 仅目标题；`Baseline` = 检索成功示例（Q + Context + Gold Answer）；`Treatment` = 在 Baseline 上加入轨迹；`Treatment_SF` = 在 Baseline 上加入 success/failure 经验、guardrail、memory 等 self-evolution 信息。
- 统计口径：同一 target 集合、同一 seed、同一评测脚本，采用 paired sign test。
- self-evolution 验收标准（用户确认）：`Acc(Treatment_SF) - Acc(Baseline) >= 2.0pp` 且 `sign-test p < 0.05`。
- 当前阶段判断：`Baseline > ZeroShot` 已被验证；`Treatment_SF > Baseline` 仍未被验证，且最近两轮继续为负增益。

## 2. 关键结果总表

- 已确认结论 1：RAG 检索示例本身有效。最强证据见 `ReasoningRag/test/results/pilot300_round1/summary.json`：`0.3900 -> 0.4467`，`+5.67pp`，`p=0.001514`。
- 已确认结论 2：到目前为止，没有任何一轮 `Treatment_SF` 达到验收标准；最好一次仅 `+1.67pp`，且 `p=0.726562`，不能视为有效提升。
- 已确认结论 3：`step3_extract_memories.py` 已产出可用资产（`ReasoningRag/test/pipeline/items.jsonl` 共 807 条 memory，覆盖 `ReasoningRag/test/pipeline/extract_progress.jsonl` 中 271 个 source question），但“memory 存在”不等于“memory 注入有效”。
- 已确认结论 4：目前主要阻塞不在工程闭环，而在经验表示与注入方式；也就是“怎么给经验”有问题，而不是“评不出来”。
- 已确认结论 5（H3 最新）：即使改成 ReasoningBank 风格的“短 operator + contrastive + 迭代记忆”，在 120 样本 smoke 上仍是负增益（`-0.83pp`，`p=1.0`）。这说明仅靠 prompt 侧“经验表达压缩”还不够，瓶颈已更接近模型能力/证据利用能力，而不只是 memory 文案质量。

说明：表中 `Traj/SF` 列统一记录该轮最相关的轨迹增强结果；早期轮次若只测了 `Treatment`，则该列记 `Treatment`；进入 self-evolution 阶段后记 `Treatment_SF`。

| 序 | 结果目录 | 样本 | 配置摘要 | ZeroShot | Baseline | Traj/SF | `Baseline - ZeroShot` | `Traj/SF - Baseline` | p(`Traj/SF`) | 判定 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 01 | `ReasoningRag/test/results/summary.json` | 80 | 最早 smoke，旧版对照，仅保留 baseline vs treatment 结果 | - | 0.4625 | 0.4250 | - | -3.75pp | 0.375000 | 早期负例，只能说明当时轨迹注入有害 |
| 02 | `ReasoningRag/test/results/pilot300_round1/summary.json` | 300 | 主线脚本，先验证 RAG | 0.3900 | 0.4467 | 0.4500 | +5.67pp | +0.33pp | 1.000000 | RAG 有效；轨迹-only 不显著 |
| 03 | `ReasoningRag/test/results/selfevo_round1/summary.json` | 300 | 旧主线脚本首轮 self-evo | 0.4067 | 0.4533 | 0.4300 | +4.67pp | -2.33pp | 0.167068 | 首次大样本 self-evo 直接负增益 |
| 04 | `ReasoningRag/test/results/selfevo_round2_smoke120/summary.json` | 120 | 旧主线脚本 smoke 调参 | 0.4250 | 0.4417 | 0.4167 | +1.67pp | -2.50pp | 0.375000 | 负增益持续 |
| 05 | `ReasoningRag/test/results/selfevo_round3_smoke120/summary.json` | 120 | 旧主线脚本 smoke 调参 | 0.4000 | 0.4417 | 0.4417 | +4.17pp | +0.00pp | 1.000000 | 只能止损，不能转正 |
| 06 | `ReasoningRag/test/results/selfevo_round4_smoke120/summary.json` | 120 | 旧主线脚本 smoke 调参 | 0.4250 | 0.4500 | 0.4500 | +2.50pp | +0.00pp | 1.000000 | 继续平台期 |
| 07 | `ReasoningRag/test/results/selfevo_eval_round1/summary.json` | 120 | 新快速 evaluator，基础 self-evo 变体（无 items） | 0.3583 | 0.4417 | 0.4250 | +8.33pp | -1.67pp | 0.726562 | 新脚本仍未解决负增益 |
| 08 | `ReasoningRag/test/results/selfevo_eval_round2/summary.json` | 120 | 新快速 evaluator，另一版基础 self-evo 变体（无 items） | 0.3667 | 0.4250 | 0.4250 | +5.83pp | +0.00pp | 1.000000 | 无增益 |
| 09 | `ReasoningRag/test/results/selfevo_eval_round3_items/summary.json` | 120 | 注入 step3 memory items + guardrail | 0.3583 | 0.4250 | 0.4417 | +6.67pp | +1.67pp | 0.726562 | 目前最佳，但离达标仍远 |
| 10 | `ReasoningRag/test/results/selfevo_eval_round4_items_contrastive/summary.json` | 120 | items + contrastive failure notes | 0.3583 | 0.4167 | 0.4083 | +5.83pp | -0.83pp | 1.000000 | contrastive 比 guardrail 更差 |
| 11 | `ReasoningRag/test/results/selfevo_eval_round5_typedmem/summary.json` | 120 | typed memory 匹配 + 限长注入 | 0.3583 | 0.4333 | 0.4083 | +7.50pp | -2.50pp | 0.453125 | 更强匹配仍显著伤害 |
| 12 | `ReasoningRag/test/results/selfevo_eval_round6_typedmem_3s/summary.json` | 120 | typed memory + success shot 从 5 降到 3 | 0.3667 | 0.4250 | 0.4000 | +5.83pp | -2.50pp | 0.453125 | 缩短 prompt 也未改善 |
| 13 | `ReasoningRag/test/results/h3_contrastive_ops_smoke120/summary.json` | 120 | H3：operator-level contrastive + iterative rule（`k=1`，`--no-sf-repair`） | - | 0.4167 | 0.4083 | - | -0.83pp | 1.000000 | H3 仍负增益，说明“经验压缩”未转化成稳定收益 |

## 3. 所有尝试过的方案：按时间顺序的失败分析

### 3.1 早期 smoke（`results/summary.json`）

- 结果：`Baseline=0.4625`，`Treatment=0.4250`，轨迹增强 `-3.75pp`。
- 价值：这是最早的负例，说明“把轨迹拼进去”并不会天然带来增益。
- 问题：该轮缺少完整的 `ZeroShot` 对照，且属于问题定位阶段，不能作为最终口径，但它准确预示了后续主问题：额外轨迹文本容易伤害模型。

### 3.2 先把问题拆开：先证明 RAG 有用，再看 self-evolution（`pilot300_round1`）

- 结果：`ZeroShot=0.3900`，`Baseline=0.4467`，`Treatment=0.4500`。
- 结论：检索成功示例是有效的，且这个结论已经在 300 样本上显著成立；因此后续问题被收缩为“为什么 trajectory / self-evolution 没有继续带来增益”。
- 启示：从这个节点开始，不应再怀疑整个 BizBench 文本设置是否成立，应把主要精力放到经验表示、经验筛选、经验注入方式。

### 3.3 旧主线脚本的 self-evolution 尝试（`selfevo_round1` 到 `selfevo_round4_smoke120`）

1. `selfevo_round1`（300 样本）
   - `Treatment_SF=0.4300`，相对 `Baseline=0.4533` 为 `-2.33pp`。
   - 含义：在较大样本上，naive success/failure 经验注入不是“没帮助”，而是明确偏负。

2. `selfevo_round2_smoke120`
   - `-2.50pp`，延续 round1 的负方向。
   - 含义：问题不是偶然抽样波动，而是配置层面的系统性伤害。

3. `selfevo_round3_smoke120`
   - `+0.00pp`。
   - 含义：通过缩减/改写提示，最多只能把伤害压平，仍不能形成正增益。

4. `selfevo_round4_smoke120`
   - `+0.00pp`。
   - 含义：旧主线脚本在现有轨迹表达下已经进入平台期；继续在同一路径上小修小补，收益很可能接近 0。

小结：旧主线脚本阶段已经给出很强信号：原始 success/failure 轨迹文本不是一个好的 self-evolution 载体；它可能太长、太噪、太泛，无法被 8B 级模型稳定利用。

### 3.4 新快速 evaluator 阶段：先做快迭代，再试 memory 注入（`selfevo_eval_round1` 到 `selfevo_eval_round6_typedmem_3s`）

1. `selfevo_eval_round1`
   - 基础 self-evo 变体，无 items，结果 `-1.67pp`。
   - 结论：换成更快的 evaluator 以后，负增益依然存在，说明问题不在旧脚本本身。

2. `selfevo_eval_round2`
   - 无 items 的另一版变体，结果 `+0.00pp`。
   - 结论：不加 memory 时，最好的情况也只是回到 baseline，说明单纯 guardrail / 提示调整没有找到有效增益机制。

3. `selfevo_eval_round3_items`
   - 引入 `ReasoningRag/test/pipeline/items.jsonl` 中的 strategy / warning，结果 `+1.67pp`，为当前最佳。
   - 结论：memory item 不是完全没信号，但信号太弱；它可以偶发性地帮助，但远不足以在 paired sign test 下站住。

4. `selfevo_eval_round4_items_contrastive`
   - 改为 contrastive failure notes，结果 `-0.83pp`。
   - 结论：把 failure 写成“错误 vs 正确”的对照句式，并没有帮助模型更稳地规避错误，反而增加了提示负担。

5. `selfevo_eval_round5_typedmem`
   - 加入 typed memory 选择（按 `answer_type`、`calc_type` 匹配，并做限长注入），结果 `-2.50pp`。
   - 结论：即使做更强筛选，memory 本身的表达仍然太泛；“匹配得更准”没有扭转“经验文本本身不适合作为提示载体”这个问题。

6. `selfevo_eval_round6_typedmem_3s`
   - 把 success shot 从 5 条减到 3 条，同时只保留 1 条 strategy + 1 条 warning，结果仍是 `-2.50pp`。
   - 结论：问题也不只是 prompt 太长；即使缩短，经验注入仍在改变模型的解题方式，并继续伤害准确率。

### 3.5 从最新失败中抽出的具体失败模式

| 失败模式 | 证据 | 推断 |
| --- | --- | --- |
| 经验注入诱导“过度重算”，把已有聚合值又算一遍 | `ReasoningRag/test/results/selfevo_eval_round5_typedmem/treatment_sf_results.jsonl` 中 `bizbench_test_78`：`ZeroShot=15.64` 正确，`Treatment_SF=7.82` 错；模型把总收入 `23406` 又加了一次，得到 `46812` | 经验提示把模型从“直接读取聚合值”推向“过度展开计算”，对 extraction / ratio 混合题尤其有害 |
| 额外提示污染输出，模型进入犹豫/解释模式 | `ReasoningRag/test/results/selfevo_eval_round5_typedmem/baseline_results.jsonl` 与 `.../treatment_sf_results.jsonl` 中 `bizbench_test_110` 都出现长段犹豫文本；`Treatment_SF` 甚至输出成 `Perhaps they expect...` 之类非答案片段 | 当前 prompt 组合让模型更容易“解释如何答”，而不是“给出答案”；这会放大数值题的格式失败 |
| repair / 格式约束可能把本来正确的数值改错 | `ReasoningRag/test/results/selfevo_eval_round5_typedmem/baseline_results.jsonl` 中 `bizbench_test_157=16.36` 正确；`.../treatment_sf_results.jsonl` 变成 `16.35` 错；round6 中又变成 `16.4` 错 | 当前链路没有把“格式修正”和“数值重写”分离，最后一跳会损伤精度 |
| step3 抽出的 memory 语言质量尚可，但大多是泛化启发，不是操作模板 | `ReasoningRag/test/pipeline/items.jsonl` 前几条多为 `Keyword-Driven Section Identification`、`Contextual Answer Validation` 这类一般性建议 | 对 8B 模型来说，这类通用建议很难提供超越 baseline 示例本身的新信息，只会占用上下文预算 |
| 实验配置留档不够细，削弱了失败分析效率 | `selfevo_eval_round1`、`selfevo_eval_round2` 的目录名无法完整恢复全部 prompt 细节，只能结合当时脚本状态推断是“无 items 基础变体” | 继续做 smoke 之前，需要把每轮配置显式固化到结果目录，避免重复试错 |

总体判断：当前 evidence 更支持“现有 self-evolution 经验表示会改变模型的解题行为，并经常朝错误方向改变”，而不是“经验还不够多”。继续堆更多自然语言 memory，大概率只会重复这个问题。

### 3.6 H3（ReasoningBank 式 contrastive 提炼）验证：`h3_contrastive_ops_smoke120`

- 配置：`ReasoningRag/test/results/h3_contrastive_ops_smoke120/config.json`
  - `mode=h3`（内部执行 `baseline` vs `treatment_h3`）
  - `limit=120`，`baseline_k=3`
  - `sf_success_k=3`，`sf_failure_k=1`
  - `memory_strategy=typed`，`h3_ops_k=1`
  - `h3_iterative=true`，`h3_history_limit=64`
  - `max_tokens=256`，`temperature=0`，`--no-sf-repair`
- 结果：`Baseline=0.4167`，`Treatment_H3=0.4083`，`Δ=-0.83pp`，`improved=3`，`regressed=4`，`p=1.0`。
- 解释：
  1. H3 相比旧的长文本 memory 的确更“短、硬、可执行”，但仍未带来总体收益，说明**表示压缩不是充分条件**。
  2. 与 oracle 结论一致：下一步要优先区分“证据表示问题”与“模型利用证据能力问题”。仅继续微调 operator 文案，预期收益很低。
  3. 从样本日志看，H3 的主要波动来自数值题精度、单位/比例口径、以及“可直接抽取 vs 重新计算”的策略切换，仍是同一类根因。

## 4. 根据最新情况推导的后续实验策略

- 当前进度判断：工程闭环已完成，RAG 有效性已完成，self-evolution 仍处在“表示失效”阶段；下一阶段不该再做大而全 prompt 微调，而应转为最小因子消融。

结合 H3 与 ReasoningBank 论文要点（记忆要结构化、对比式、可迭代），下一阶段策略更新如下：

1. 新主线改为“证据卡片（evidence card）”判别实验（优先级最高）
   - 设计：在检索已包含答案证据的 target 子集上，做 paired 对比：`rag_raw` vs `h3_evidence_card`。
   - 目标：判别当前失败是“提示表示问题”还是“模型能力问题”。
   - 验收门槛（oracle 建议）：覆盖量达到门槛后，若 evidence_card 相对 raw 提升显著（例如 +8pp 且 CI>0），则继续投入表示工程；若提升极小且整体准确率仍低，则转向能力侧路线。

2. H3 memory 更新机制改为“只记可验证 operator”
   - 每条 memory 统一结构：`trigger / H+ / H- / H?`（做什么、避免什么、快速检查什么）。
   - 严格触发：仅当错误类型明确且可归因（单位、口径、重算、精度）才新增；否则不写入，防止噪声增长。

3. 继续 token 预算约束
   - 注入保持 `k=1`，单条 rule 限长；题型不匹配即不注入。
   - 避免再次回到“长提示 + 多规则叠加”路径。

4. 停止条件保持不变且更严格执行
   - 任一方案若连续两轮 smoke 仍无正趋势（`improved <= regressed` 且 `Δ<=0`），立即降级优先级，不做同构微调。
   - 只有满足 `Δ>=+2pp` 且统计显著，才晋升 300 样本。

（备注）旧版“最小因子消融”路线保留为备选；当前优先级已下调到 evidence-card 判别实验之后。
