# OCR 多图任务与 Multimodal 提取能力分析

## 一、Benchmark 差异是否属于正常 LLM 波动？

### 对比数据（以实际 log 为准）

| 版本 | 正确数 | 准确率 | 说明 |
|------|--------|--------|------|
| 014335 | 10/21 | 47.62% | 修改前基线 |
| 021625 | 5/21 | 23.81% | 修改后（prompt 已回退） |
| **022959** | **9/21** | **42.86%** | 回退 prompt 后实际运行 |

### 结论

- **5 个样本的差异不属于正常 LLM 波动**。在 21 样本规模下，典型随机波动通常为 ±1~2 个样本。
- 5 个样本的差异（约 24% 准确率下降）更可能是由 prompt 修改引起的系统性变化，而非随机噪声。
- **022959 实际为 9/21 正确**，回退后介于 014335(10) 与 021625(5) 之间，与 014335 差 1 个样本，属于正常波动范围。详见 `benchmark_results/multi_agent_multi_dataset_20260319_022959/DETAILED_ANALYSIS.md`。

---

## 二、公式优化瓶颈与 Multimodal 增强方向

### 当前瓶颈

1. **Prompt 优化已接近上限**：规划、公式、变量映射等规则已较完善，但公式类任务仍易出错。
2. **OCR 仅提供文本**：OCR 将表格转为 Markdown，Planning 基于文本规划，**Multimodal 执行时仍依赖视觉模型看图**。若 OCR 质量不佳或表格结构复杂，Planning 可能误判，Multimodal 也可能误读。
3. **感知与认知解耦的局限**：OCR 解耦后 Planning 不看图，但 Multimodal 执行时仍看图。若 OCR 与视觉模型对同一张图的解读不一致，会形成「规划基于 A 理解、执行基于 B 理解」的错位。

### 建议的 Multimodal 增强方向

| 方向 | 说明 | 预期收益 |
|------|------|----------|
| **表格结构化提取** | 使用专门的表格识别模型（如 Table Transformer）输出行列结构，而非纯 Markdown OCR | 更稳定的表格解析，减少行列错位 |
| **结构化输出约束** | 要求 Multimodal 在提取时输出 `{variable: value}` 的 JSON，而非自由 python_execute | 更易与 plan 对应，减少变量名/顺序错误 |
| **多图显式绑定** | 在 step prompt 中明确「当前图片对应 OCR 的 [Image N] 内容」，便于核对 | 降低 Planning 与 OCR 的图-内容错配 |
| **OCR + 视觉双路校验** | 对关键单元格，同时用 OCR 与视觉模型提取，不一致时触发重试或人工确认 | 提高关键数值可靠性 |
| **表格专用 OCR** | 针对金融表格优化 OCR prompt（如强调表头、单位、年份列） | 减少漏读、误读 |

---

## 三、OCR 多图任务在 Planning 中的处理流程

### 3.1 数据流概览

```
Dataset (images: [path1, path2])
    ↓
Benchmark: resolve_image_path → resolved = [p1, p2]
    ↓
load_images_as_base64(resolved) → base64_images = [img1, img2]
    ↓
Flow.execute(base64_images=base64_images)
    ↓
OCR: ocr.execute(base64_images=base64_images)
    ↓
ocr_results = [text_from_img0, text_from_img1]  # 按 index 严格对应
    ↓
_build_plan_prompt: "--- [Image 1] ---\n{text0}\n--- [Image 2] ---\n{text1}"
    ↓
Planning Agent (仅收到 OCR 文本，不传图)
```

### 3.2 图片与 OCR 的对应关系

**严格对应关系：**

| 层级 | 索引 | 对应关系 |
|------|------|----------|
| Dataset | `images[0]`, `images[1]` | 原始图片路径顺序 |
| base64_images | `[0]`, `[1]` | 与 dataset 顺序一致 |
| OcrExtract batch | `for i, img in enumerate(base64_images)` | `results[i]["index"] = i` |
| ocr_results | `ocr_results[0]`, `ocr_results[1]` | `sorted(results, key=index)` 保证顺序 |
| Planning prompt | `--- [Image 1] ---`, `--- [Image 2] ---` | `i+1` 对应 ocr_results[i] |

**结论：每一张图片的 OCR 内容与对应图片是严格一一对应的。**

- `ocr_results[i]` 来自 `base64_images[i]`
- Planning 看到的 `--- [Image N] ---` 对应 `ocr_results[N-1]`
- 执行时 `_get_image_index_for_step` 解析 "from image 1" → `base64_images[0]`

### 3.3 潜在风险点

1. **Dataset 中 images 顺序**：若 `images` 数组顺序与题目中 `<image 1>`, `<image 2>` 的语义不一致，会出错。FinMMR 数据集通常按题目顺序排列。

2. **Planning 的变量-图片映射**：Planning 仅凭 OCR 文本判断「变量 X 在 Image 1 还是 Image 2」。若 OCR 文本不完整或歧义，可能误分配。

3. **Multimodal 执行时**：每步只收到一张图（`base64_image`），但 step 文本可能写 "from image 1" 或 "from image 2"。`_get_image_index_for_step` 解析 step 文本决定用哪张图，逻辑正确。

### 3.4 关键代码位置

| 功能 | 文件 | 行号/逻辑 |
|------|------|-----------|
| OCR 批量提取 | `app/tool/ocr.py` | `for i, img in enumerate(base64_images)` → `results.append({..., "index": i})` |
| Flow 组装 ocr_results | `app/flow/planning.py` | `ocr_results = [item.get("text","") for item in sorted(results, key=lambda x: x.get("index",0))]` |
| Planning prompt 注入 | `app/flow/planning.py` | `for i, text in enumerate(self.ocr_results): lines.append(f"--- [Image {i+1}] ---")` |
| 执行时图片路由 | `app/flow/planning.py` | `_get_image_index_for_step` 解析 "image N" → `base64_images[N-1]` |

---

## 四、总结与建议

1. **图片-OCR 对应**：当前实现已保证每张图片的 OCR 内容与对应图片严格对应，无需额外修改。
2. **公式优化瓶颈**：单纯依赖 prompt 和 OCR 的优化空间有限，建议在 **Multimodal 图片数据提取能力** 上做增强（表格结构化、结构化输出、双路校验等）。
3. **多图任务**：Pipeline 本身支持多图，且 OCR 与执行阶段的图片索引一致。问题更可能出在 OCR 质量、Planning 的变量分配或 Multimodal 的提取准确度上。
