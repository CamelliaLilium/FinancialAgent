# Planning Prompt 优化评价

## 一、优化前后对比

| 维度 | 优化前 | 优化后 |
|------|--------|--------|
| **总行数** | ~66 行 (SYSTEM 47 + NEXT 15) | ~52 行 (SYSTEM 42 + NEXT 10) |
| **结构** | 分散的 EXECUTOR、EXPLICIT、EXTRACTION、CRITICAL | 清晰的 OPERATIONAL CONSTRAINTS → EXECUTORS → GUARDRAILS → FORMAT → RULES |
| **领域公式** | 散落在 EXTRACTION & FORMULA 中 | 独立 FINANCIAL LOGIC GUARDRAILS 区块 |
| **变量命名** | "lowercase_with_underscores" 一句 | 强制 + 反例（中文→英文映射） |
| **步骤顺序** | 未明确 | 明确 "Extraction 在 Calculation 之前" |
| **依赖检查** | 无 | "公式中每个变量必须有 extraction 步骤" |

## 二、针对性改进（对应 plan_error_对照.md）

| 错误类型 | 优化前 | 优化后 |
|----------|--------|--------|
| **easy-test-0** Interest Coverage 用 total | 有 "numerator MUST be EBIT" | 在 GUARDRAILS 首条强调，更醒目 |
| **easy-test-1** 照搬表头 | 有 "Minimal extraction" | 前置为 OPERATIONAL #3，加 selling_expenses 反例 |
| **easy-test-11** Cash Paid 公式错 | 无 | 新增 GUARDRAILS: COGS - Δinv - ΔAP |
| **easy-test-32** 取错数据源 | 无 | 新增 "cash flow hedges → 提取 cash_flow_hedges，非 other_income" |
| **easy-test-2** second half | 有 "Q3+Q4" | 在 GUARDRAILS 中明确 "up by 2%" 歧义 |
| **easy-test-20** 中文变量名 | 有 "lowercase" | 强制禁中文 + 映射示例 |
| **easy-test-17** interest_rate 未提取 | 无 | 新增 Dependency Check |
| **easy-test-3** 步骤顺序错误 | 无 | 新增 Execution Order #2 |

## 三、删除/精简内容

| 删除项 | 原因 |
|--------|------|
| "State persistence and step status updates are controlled by PlanningFlow" | 实现细节，与规划无关 |
| "Match executor to data source" 等冗长描述 | 与 "Assign based on where data lives" 重复 |
| "Every computation step MUST" + "EVERY step MUST" 两条 | 合并为 EXECUTORS 简洁说明 |
| "[multimodal] and [finance] are STEP EXECUTORS... 60 字" | 压缩为 "step labels, not tools" |
| NEXT_STEP 中与 SYSTEM 重复的 1-5 条 | 精简为 5 条决策 |
| "Complete extraction" 多段 | 合并入 Minimalism + Dependency Check |

## 四、保留与架构一致性

- 仍使用 [multimodal] 和 [finance]，未引入 [ocr]/[vlm_validator]
- 步骤格式保持 "from image N extract ... output as ..." 与 "calculate ... output as ..."
- 多图、多实体、无图规则保留
- 工具仍为 planning 和 terminate

## 五、评估结论

| 维度 | 评分 | 说明 |
|------|------|------|
| **针对性** | 高 | 覆盖 plan_error_对照 中 8 类主要错误 |
| **精简度** | 高 | 行数减少 ~20%，结构更清晰 |
| **可执行性** | 中 | 需 benchmark 验证实际效果 |
| **架构兼容** | 高 | 未引入不存在的 executor，格式兼容 |

**建议**：运行 benchmark 验证 easy-test-0, 1, 2, 11, 17, 20, 32 等典型错误样例的改善情况。
