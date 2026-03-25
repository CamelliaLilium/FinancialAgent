# finance_extraction_skill 重构设计：VLM 优先 + OCR 校验

## 一、当前设计的问题回顾

| 问题 | 说明 |
|------|------|
| OCR 无法做定向提取 | OCR 是「全图 dump」，只能输出整张图的全部文本，无法根据 plan step 只提取 interest_expense_2018、ebit_2018 等特定变量 |
| DeepSeek OCR 不输出 regions | prompt 允许 "omit the Regions section"，导致 regions 经常为空，无法做基于 bbox 的 Point-Check |
| 变量-区域匹配失败 | 依赖 regions 做语义匹配；regions 为空时退化为 markdown 行搜索，对「行标签+年份列」表格结构支持差 |
| 结果 | Phase 1/2 大量失败，从未进入持久化，Multimodal 无有效 fallback |

---

## 二、你提出的新流程（VLM 优先 + OCR 校验）

```
1. VLM 先做计划要求的定向提取（理解 step，从图中找特定变量）
2. OCR 做全局数据提取（全图文本）
3. 校验：VLM 提取的值是否在 OCR 的全局数据中出现
4a. 匹配 → 变量存储
4b. 不匹配 → 用 OCR 中相关片段 + 图像，让 VLM 重新提取，然后直接存储（避免拥塞）
```

---

## 三、设计合理性分析

### 3.1 OCR 与 VLM 的能力对比

| 能力 | OCR | VLM |
|------|-----|-----|
| 定向提取 | ❌ 只能全图 dump | ✅ 可理解 "extract interest_expense_2018" 并定位 |
| 语义理解 | ❌ 纯文本 | ✅ 理解表格结构、行列对应、单位 |
| 输出形式 | 固定格式（markdown/regions） | 灵活（可按 prompt 要求格式化） |
| 依赖 bbox | 可选，常为空 | 不依赖 |

**结论**：用 VLM 做「定向提取」、OCR 做「全局参考」是合理的分工。

### 3.2 为何 VLM 优先更合适？

- **当前流程**：OCR 全图 → 从 OCR 结果中匹配变量 → 失败率高（regions 空、匹配逻辑弱）
- **新流程**：VLM 定向提取 → 用 OCR 全图做存在性校验 → 不依赖 regions，校验逻辑简单（数值是否出现在 OCR 文本中）

VLM 能理解 "interest_expense_2018" 对应表格中的 "Interest Expense" 行 + "2018" 列，无需我们写复杂的变量-区域映射。

### 3.3 校验逻辑（无需 regions）

```
VLM 提取: interest_expense_2018 = 123.45
OCR markdown: "... Interest Expense ... 123.45 ..." 或 "123.45" 出现在某行

匹配规则：VLM 的数值（或规范化后的形式）是否出现在 OCR 的 markdown 中
- 精确： "123.45" in ocr_text
- 宽松： 移除千分位、统一小数位后比较
- 单位： VLM 若已换算（如 万→×10000），OCR 可能仍是 "12.345万"，需在比较时考虑
```

不需要 bbox，只需文本包含性检查。

### 3.4 不匹配时的 fallback（避免拥塞）

当 VLM 的值在 OCR 中找不到时：

1. **可能原因**：VLM 幻觉、单位不一致、OCR 漏字、表格结构复杂
2. **处理**：从 OCR markdown 中截取与变量相关的行（如含 "interest"、"expense"、"2018" 的行），连同图像一起给 VLM
3. **Prompt**：`Given this OCR excerpt from the table: [lines]. What is the value for interest_expense_2018? Reply with ONLY the number.`
4. **结果**：直接存储 VLM 的第二次提取结果，不再做递归校验，避免死循环

这样既利用 OCR 提供上下文，又保证最终一定能写入变量。

---

## 四、与当前流程的对比

| 维度 | 当前（OCR 优先） | 新设计（VLM 优先） |
|------|------------------|---------------------|
| 定向提取 | 依赖 regions + 语义匹配，常失败 | VLM 直接定向提取 |
| 校验依赖 | regions/bbox，常为空 | OCR markdown 文本，稳定可得 |
| 失败时 | 返回 FAILURE，无后续 | OCR 片段 + 二次 VLM → 直接存储 |
| 拥塞风险 | 无 fallback，Multimodal 反复失败 | 有明确 fallback，保证可持久化 |

---

## 五、实现要点

### 5.1 流程伪代码

```python
async def execute(variables, base64_image, ...):
    # Phase 1: VLM 定向提取
    vlm_extracted = await _vlm_extract_variables(base64_image, variables)
    # vlm_extracted: {var_name: (raw_str, converted_float) or None}

    # Phase 2: OCR 全局提取
    ocr_result = await self._ocr.execute(base64_image=base64_image)
    markdown = ocr_result.get("markdown", "")

    # Phase 3: 校验 + 补全
    extracted = {}
    for var_name, (raw, num) in vlm_extracted.items():
        if num is None:
            # VLM 未提取到，走 OCR 引导的二次提取
            num = await _vlm_extract_with_ocr_context(base64_image, markdown, var_name)
        elif not _value_in_ocr(num, raw, markdown):
            # 校验不通过，用 OCR 片段引导重算
            num = await _vlm_extract_with_ocr_context(base64_image, markdown, var_name)
        if num is not None:
            extracted[var_name] = num

    # Phase 4: 持久化（与现有一致）
    if extracted:
        shared_py._run_code(...)
    else:
        return FAILURE
```

### 5.2 关键函数

- **`_vlm_extract_variables(image, variables)`**：一次调用 VLM，prompt 要求按变量名逐行返回数值（格式如 `var_name=value`），支持单位换算说明。
- **`_value_in_ocr(num, raw, markdown)`**：检查 `num` 或 `raw` 的规范形式是否出现在 `markdown` 中（可做千分位、小数位、单位的归一化）。
- **`_vlm_extract_with_ocr_context(image, markdown, var_name)`**：从 markdown 中筛出与 `var_name` 相关的行（含变量名片段），拼成 prompt，与 image 一起传给 VLM，要求只返回该变量的数值。

### 5.3 匹配规则

```python
def _value_in_ocr(num: float, raw: str, markdown: str) -> bool:
    # 数值形式
    num_str = re.sub(r"\.?0+$", "", f"{num:.4f}")  # 去尾零
    num_compact = str(num).replace(".", "")
    # 原始形式（如 "12.345万"）
    raw_clean = re.sub(r"[^\d.]", "", raw)
    return (
        num_str in markdown or
        str(int(num)) in markdown or
        raw_clean in markdown or
        str(num) in markdown
    )
```

---

## 六、总结

| 问题 | 结论 |
|------|------|
| 设计是否合理？ | **合理**，VLM 优先 + OCR 校验能解决当前 regions 缺失和匹配失败问题 |
| 能否优化现有流程？ | **能**，定向提取、校验、fallback 都有明确改进 |
| 实现复杂度 | 中等，需新增 VLM 提取 prompt、校验函数、OCR 上下文筛选逻辑 |

建议按此设计重构 `finance_extraction_skill`，优先实现「VLM 定向提取 + OCR 存在性校验」，再补充「不匹配时 OCR 引导二次提取」的 fallback。
