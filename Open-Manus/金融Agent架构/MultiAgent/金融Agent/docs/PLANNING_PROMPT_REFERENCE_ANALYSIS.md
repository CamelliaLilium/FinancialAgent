# 参考 Prompt 适用性分析

## 参考结构与当前架构对照

| 参考项 | 参考内容 | 当前架构 | 是否采纳 |
|--------|----------|----------|----------|
| Executor | [ocr], [vlm_validator], [finance] | [multimodal], [finance] | ❌ 无 [ocr]/[vlm_validator]，保留 [multimodal]/[finance] |
| 步骤顺序 | Define→OCR→VLM→Python | 无 Define 步骤 | ✅ 采纳「Extraction 在 Calculation 之前」 |
| Variable Naming | lowercase_snake_case, 无中文 | 已有 lowercase，但未禁中文 | ✅ 采纳，强化禁中文 |
| Minimalism | 只提取公式所需变量 | 已有类似规则 | ✅ 采纳，前置并强化 |
| Financial Guardrails | Interest Coverage, Cash Paid, H2 等 | 已有部分 | ✅ 采纳，整理为独立区块 |
| Plan Template | Step 0 Define formula | 无独立 Define | ❌ 不采纳，公式写在 [finance] 中 |
| NEXT ACTION | Formula First, Dependency Check | 无 | ✅ 采纳，精简并入 |

## 选取的优化点

1. **Variable Naming**：强制 lowercase_snake_case，禁止中文/特殊字符（解决 easy-test-20, 29）
2. **Execution Order**：明确 extraction 步骤必须在 calculation 之前
3. **Financial Logic Guardrails**：独立区块，含 Interest Coverage、Cash Paid、Growth、Second Half
4. **Minimalism**：前置为 OPERATIONAL CONSTRAINTS，强调「只提取公式所需」
5. **Dependency Check**：公式中每个变量必须有对应 extraction 步骤
6. **精简**：删除与 Flow 重复内容、冗长表述
