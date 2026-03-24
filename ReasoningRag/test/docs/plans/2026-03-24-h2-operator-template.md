# H2 验证：operator template 是否优于自然语言 memory

## Goal
验证把自然语言经验改成极短 operator-level 模板 + 题型路由后，`Treatment` 能否首次稳定优于 `Baseline`。

## Scope
- 只改 `test/selfevo_eval.py`
- 新增 `treatment_op` 和 `treatment_op_routed` 两种模式
- 关闭 repair
- 确定性答案抽取
- 结果写入 `test/results/h2_operator_template/`
- 更新 `test/PROGRESS.md`

## Success Criteria
- 在 `arithmetic / ratio / ratio+multi_step` 子集上，operator template 组出现稳定正增益
- 同时 `extraction` 子集不被明显伤害
- 输出含 per-calc_type 分组统计

## Operator Templates (4 条)

### AGGREGATE
- 适用 calc_type: extraction, arithmetic, ratio, ratio+multi_step
- 规则: If the table already contains a total/aggregate/sum row, use it directly. Do NOT re-sum the individual line items.

### UNIT_SCALE
- 适用 calc_type: arithmetic, ratio, ratio+multi_step
- 规则: If the table header says "(in thousands)", "(in millions)", etc., keep the table's scale unless the question explicitly asks for conversion.

### DIRECT_READ
- 适用 calc_type: extraction
- 规则: If the answer can be read directly from a single cell or field, output that value exactly. Do NOT perform additional derivation or calculation.

### PRECISION
- 适用 calc_type: arithmetic, ratio, ratio+multi_step
- 规则: Preserve the precision of your intermediate calculation. Do NOT round to a "nice" number or convert to casual approximation in the final answer.

## 路由逻辑
- `treatment_op`: 全题注入（按 calc_type 选最相关的 1 条模板）
- `treatment_op_routed`: 有条件注入（仅当 calc_type 匹配时注入；extraction 类默认注入 DIRECT_READ，其余注入 PRECISION 或 AGGREGATE）

## 4 组对比

| 组 | mode | memory | 路由 | repair |
|---|---|---|---|---|
| B0 | baseline | 无 | — | off |
| B1 | treatment_sf | NL items | 现有 | off |
| B2 | treatment_op | 4 条 template | 全题 | off |
| B3 | treatment_op_routed | 4 条 template | calc_type 路由 | off |

## Tasks
1. 在 selfevo_eval.py 中新增 operator template 定义和路由逻辑
2. 新增 treatment_op / treatment_op_routed 模式
3. 新增 per-calc_type 分组统计到 summary 输出
4. 关闭 repair 默认值
5. 运行 4 组 smoke-120 实验
6. 分析结果，更新 PROGRESS.md

## Docs Impact
- `test/PROGRESS.md` 需新增 H2 验证结论
