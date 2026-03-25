# OCR 内嵌与非多模态任务影响分析

## 一、OCR 预处理触发条件

```python
# app/flow/planning.py execute()
if base64_images and len(base64_images) > 0 and getattr(config, "ocr_config", None):
    # 执行 OCR 预处理
```

**结论**：仅当同时满足以下条件时才会执行 OCR 预处理：
1. `base64_images` 非空（存在图片输入）
2. `config.ocr_config` 已配置（config.toml 中有 [ocr] 段）

## 二、非多模态任务是否受影响

| 任务类型 | base64_images | OCR 预处理 | Planning 输入 | 影响 |
|----------|---------------|------------|---------------|------|
| 纯文本（无图） | None 或 [] | 不执行 | 仅 request 文本 | **无影响** |
| 单图多模态 | [img1] | 执行 | request + OCR 文本 | 正常 |
| 多图多模态 | [img1, img2] | 执行 | request + OCR 文本 | 正常 |

**非多模态任务**（如纯金融计算、文本问答、无图 BizBench 等）：
- `base64_images` 为 None 或空列表
- OCR 预处理**不会执行**
- Planning 按原逻辑工作，不注入 OCR 内容
- `ocr_results` 保持 None，`_create_plan_via_planning_agent` 不传图时仅当 ocr_results 存在；对无图任务 ocr_results 为 None，Planning 不传图（本就没有图）

**结论**：OCR 内嵌形式**不会影响**非多模态任务的正常处理。

## 三、Agent 调用 OCR 的适用场景

- **MultimodalAgent** 已注册 `OcrExtract` 工具
- 在 multimodal 步骤中，Agent 可调用 `ocr_extract(use_context_image=True)` 使用当前步骤图片
- 也可显式传入 `base64_image` 或 `base64_images`（如从文件路径解析后传入）
- 后续 Skill 搭建时，可将 OCR 作为可选工具注入到需要图像文本提取的 Agent 中
