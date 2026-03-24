# H1 验证：能力瓶颈 vs 表示瓶颈（Gemini token-efficient）

## Goal
验证 `gemini-2.5-flash`（更强模型）在相同 operator-template 注入下是否能得到正增益，从而判断 H2 的失败是否主要由 8B 能力瓶颈导致。

## Scope
- 仅改 `test/selfevo_eval.py`：增加 Gemini backend 支持（不改评测逻辑）
- 用最省 token 的配置运行 2 组：`baseline` vs `treatment_op_routed`
- 固定 smoke 集合与 seed，输出 overall + per-calc_type 对比
- 更新 `test/PROGRESS.md` 写入 H1 结果与路线判断

## Token-Efficiency Strategy
1. 只跑 2 个模式（baseline / treatment_op_routed），不跑全模式矩阵
2. `limit=80`（先做中等样本，不直接上 120/300）
3. `baseline_k=2`（少一个示例，减少上下文 token）
4. `max_tokens=128`（限制输出长度）
5. `temperature=0.0`，减少冗余输出波动
6. `--no-sf-repair`（避免二次调用）

## Acceptance Criteria
- 能稳定跑通 Gemini backend（无 API 错误）
- 输出 summary 包含 `treatment_op_routed_vs_baseline` 与分组对比
- 给出 H1 判断：
  - 若 Gemini 出现明显正增益而 8B 为负，支持“能力瓶颈”；
  - 若 Gemini 仍不增益，倾向“prompt-side 注入路线本身受限”。

## Experiment Config
- model: `gemini-2.5-flash`
- backend: `gemini`
- api_base: `https://aihubmix.com/gemini`
- modes: `baseline`, `treatment_op_routed`
- seed: `20260324`
- limit: `80`
- baseline_k: `2`
- max_tokens: `128`
- output_dir: `test/results/h1_gemini_eff/`

## Tasks
1. 为 selfevo_eval.py 增加 Gemini backend（genai SDK）
2. 做最小 backend 联通性验证
3. 运行 H1 token-efficient 实验
4. 分析 H1 结果并更新 PROGRESS.md

## Docs Impact
- `test/PROGRESS.md` 增加 H1 结论
- `test/docs/README.md` 增加该计划索引
