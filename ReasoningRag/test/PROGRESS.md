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

## 4. 根据最新情况推导的后续实验策略

- 当前进度判断：工程闭环已完成，RAG 有效性已完成，self-evolution 仍处在“表示失效”阶段；下一阶段不该再做大而全 prompt 微调，而应转为最小因子消融。

1. 先做严格消融，定位“到底是哪一块在伤害”
   - 固定同一批 120 target、同一 seed，只比较 4 个最小变体：`failure-only`、`strategy-only`、`failure+strategy`、`baseline+repair`。
   - 每轮都保存 `config.json` 到结果目录，至少记录：shot 数、是否启用 repair、memory 条数、memory 来源、failure 风格。
   - 目标不是马上转正，而是判断负增益来自 `warning`、`strategy`、还是 `repair`。

2. 把评测按题型拆开，不再只看总体均值
   - 直接按 manifest 中的 `calc_type` 和 `answer_type` 出分组统计：`extraction`、`arithmetic`、`ratio`、`multi_step`。
   - 特别关注：`bizbench_test_78` 这类“既能直接读聚合值、又能展开运算”的题是否系统性受害。
   - 若某个子类稳定正增益，可先把 self-evolution 限制到该子类，而不是要求全量任务统一受益。

3. 停止注入泛化自然语言 memory，改试“操作模板”
   - 不再使用类似“verify context / use keywords”这类宽泛提示；改成极短、可执行、与失败模式一一对应的 operator template。
   - 示例方向：`如果表中已有 total/aggregate 行，禁止重算全表`；`数值题优先保留原始精度，不在最后一步做口语化四舍五入`；`若答案可直接抽取，不执行额外推导`。
   - 每次最多注入 1 条模板；若 target 不满足触发条件，则完全不注入。

4. 把“答案格式修正”从“重新思考答案”中拆开
   - 在下一轮先关闭 LLM repair，改成确定性数值抽取与格式化；只有抽取失败时才进入极简 formatter。
   - 原因：现有 round5/6 说明最后一跳会把 `16.36` 改成 `16.35/16.4` 这类精度错误，repair 不能再和 reasoning 混在一起。

5. 设立晋升与停止条件，避免继续无边界调参
   - 只有当 smoke-120 同时满足 `Δ(Treatment_SF - Baseline) >= +2.0pp`、`improved_count > regressed_count`、且无明显新失败簇时，才晋升到 300 样本。
   - 如果完成上述最小消融后，仍没有任何单因子或单子类出现稳定正趋势，应暂时冻结 self-evolution 方向，转而考虑：
     - 是否需要更结构化的 experience representation；
     - 是否该把 self-evolution 放到检索侧（检索什么经验）而不是 prompt 侧（怎么写经验）；
     - 或者当前 8B 模型是否不足以稳定利用这类经验。

## 5. 假设 2 验证结果：operator template 是否优于自然语言 memory

结果目录：`ReasoningRag/test/results/h2_operator_template/`

### 实验配置
- smoke-120，seed=20260324，baseline_k=3，repair=off
- 4 组对比：B0=baseline / B1=treatment_sf（NL memory，repair off）/ B2=treatment_op（全题注入 PRECISION）/ B3=treatment_op_routed（按 calc_type 路由注入）

### Overall 结果

| 组 | mode | Acc | vs Baseline Δ | improved | regressed | p |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B0 | baseline | 0.4333 | — | — | — | — |
| B1 | treatment_sf | 0.4333 | +0.00pp | 4 | 4 | 1.0 |
| B2 | treatment_op | 0.3667 | -6.67pp | 1 | 9 | 0.0215 |
| B3 | treatment_op_routed | 0.3917 | -4.17pp | 0 | 5 | 0.0625 |

### 按 calc_type 分组

| calc_type | n | baseline | B1(NL) | B2(op) | B3(op_routed) | B2-B0 | B3-B0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| extraction | 53 | 0.7547 | 0.7170 | 0.6604 | 0.7170 | -9.43pp | -3.77pp |
| arithmetic | 36 | 0.1389 | 0.1667 | 0.1389 | 0.1111 | +0.00pp | -2.78pp |
| ratio | 18 | 0.1111 | 0.1667 | 0.0556 | 0.1111 | -5.56pp | +0.00pp |
| ratio+multi_step | 9 | 0.2222 | 0.2222 | 0.0000 | 0.1111 | -22.22pp | -11.11pp |
| arithmetic+multi_step | 3 | 0.6667 | 0.6667 | 0.6667 | 0.3333 | +0.00pp | -33.33pp |

### 结论

1. **H2 假设被否定**：operator template 不仅没有改善，反而显著伤害了模型表现。
   - `treatment_op`（全题注入 PRECISION）：overall `-6.67pp`，`p=0.0215`，是显著负增益。
   - `treatment_op_routed`（路由注入）：`-4.17pp`，伤害有所缓解但仍为负。

2. **PRECISION 模板对 extraction 题伤害最大**：extraction 从 `0.7547` 降到 `0.6604`（B2）。"保留完整精度"这条规则让模型更倾向于过度计算，而不是直接读值。路由版（B3）对 extraction 注入的是 DIRECT_READ，伤害有所缓解（`0.7170`），但仍比 baseline 差。

3. **NL memory（B1）在 repair=off 下变成了 0 增益**：这说明之前观察到的负增益有一部分来自 repair，关掉 repair 后 NL memory 至少不再明显伤害。但也不提供任何正增益。

4. **对 arithmetic/ratio 子集，没有任何方案形成正增益**：这是最关键的发现。我们原本假设 operator template 在 arithmetic/ratio 上有机会帮助，但实际连 0 增益都不稳定。

5. **核心推断**：当前 8B 模型在 BizBench 文本 QA 上的"prompt-side memory injection"路线可能已经触及天花板。无论是自然语言经验还是硬规则模板，任何额外的提示注入都倾向于干扰模型原本的解题行为，而不是帮助它。

### 对后续路线的影响

- H2 被否定后，优先做 H1（用更强模型验证是否是能力瓶颈）。
- 如果 H1 也显示更强模型同样不受益，则应考虑完全放弃 prompt-side memory injection，转向：
  - 结构化求解（让模型走固定步骤而不是自由推理）
  - 或 fine-tuning / 检索侧改造
