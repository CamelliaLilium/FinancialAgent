# Planning 感知-认知解耦优化方案分析

## 一、目标

在多模态任务中，将**感知**（从图像提取文本）与**认知**（规划）解耦：
- **感知**：OCR 先提取图像中的文本
- **认知**：Planning Agent 基于 OCR 文本生成计划，不再直接看图

---

## 二、当前流程 vs 优化后流程

### 2.1 当前流程

```
execute(input_text, base64_images)
    ↓
_create_initial_plan(request)
    ↓
_build_plan_prompt(request)  → plan_prompt
    ↓
planning_agent.run(plan_prompt, base64_image=img1, base64_images=[img1,img2])
    ↓
Planning 使用 vision 模型「看图」理解内容，生成步骤
```

**问题**：Planning 依赖 vision 模型读图，易受模型波动影响；多图时易漏看或混淆。

### 2.2 优化后流程

```
execute(input_text, base64_images)
    ↓
[新增] 若 base64_images 存在且 OCR 配置可用：
    ocr_results = await OcrExtract().execute(base64_images=base64_images)
    self.ocr_results = [r["text"] for r in ocr_results["results"]]
    ↓
_create_initial_plan(request)
    ↓
_build_plan_prompt(request)  → plan_prompt（内含 OCR 文本）
    ↓
planning_agent.run(plan_prompt)  ← 不再传 base64_image/images
    ↓
Planning 基于纯文本（含 OCR 结果）生成步骤，无需 vision
```

---

## 三、需改动部分清单

### 3.1 PlanningFlow 类（`app/flow/planning.py`）

| 位置 | 改动类型 | 说明 |
|------|----------|------|
| **类属性** | 新增 | `ocr_results: Optional[List[str]] = None`，存储每张图的 OCR 文本 |
| **execute()** | 修改 | 在 `_create_initial_plan` 前增加 OCR 预处理逻辑 |
| **_build_plan_prompt()** | 修改 | 当 `ocr_results` 非空时，在 prompt 中插入「Image N 内容」块 |
| **_create_plan_via_planning_agent()** | 修改 | 当 `ocr_results` 可用时，不传 `base64_image`/`base64_images` |
| **_create_plan_via_flow_llm()** | 修改 | 当 `ocr_results` 可用时，在 user_message 中附加 OCR 内容 |

### 3.2 依赖与导入

| 文件 | 改动 |
|------|------|
| `app/flow/planning.py` | 新增 `from app.tool import OcrExtract` |
| `app/config.py` | 已有 `ocr_config`，无需改 |

### 3.3 不改动部分

- **PlanningAgent**：接口不变，仍为 `run(request, base64_image=..., base64_images=...)`；当不传图时，仅用 request 文本
- **步骤执行**：本阶段不改 `_execute_step`，Multimodal 仍按原逻辑执行
- **OcrExtract 工具**：已支持批量，无需改

---

## 四、详细设计

### 4.1 OCR 预处理逻辑（execute 内）

```python
# 伪代码
self.ocr_results = None
if self.base64_images and len(self.base64_images) > 0:
    if getattr(config, "ocr_config", None):
        try:
            ocr = OcrExtract()
            r = await ocr.execute(base64_images=self.base64_images)
            if r.get("success") and "results" in r:
                self.ocr_results = [item.get("text", "") for item in r["results"]]
        except Exception as e:
            logger.warning(f"OCR preprocessing failed: {e}")
            # 失败时 ocr_results 保持 None，回退到原逻辑（传图给 Planning）
```

**回退策略**：OCR 配置缺失、调用失败、超时时，`ocr_results = None`，Planning 继续接收图像（保持兼容）。

### 4.2 _build_plan_prompt 中插入 OCR 内容

插入位置：在 `**Task:**` 与 `**Planning rules:**` 之间。

```python
if self.ocr_results and len(self.ocr_results) > 0:
    lines.extend(["", "**Content extracted from images (OCR):**"])
    for i, text in enumerate(self.ocr_results):
        lines.append(f"\n--- Image {i+1} ---")
        lines.append(text.strip() if text else "(No text extracted)")
    lines.append("")
```

### 4.3 _create_plan_via_planning_agent 条件分支

```python
if self.ocr_results is not None:
    # 感知已解耦：Planning 仅需文本
    await planning_agent.run(plan_prompt, base64_image=None, base64_images=None)
else:
    # 回退：传图给 Planning
    base64_image = self.base64_images[0] if self.base64_images else None
    base64_images = self.base64_images if len(self.base64_images or []) > 1 else None
    await planning_agent.run(plan_prompt, base64_image=base64_image, base64_images=base64_images)
```

### 4.4 _create_plan_via_flow_llm 条件分支

当前 `user_message` 仅为 `"Create a reasonable plan... {request}"`。当 `ocr_results` 可用时，附加 OCR 内容：

```python
user_content = f"Create a reasonable plan... {request}"
if self.ocr_results:
    user_content += "\n\n**Content from images (OCR):**\n"
    for i, t in enumerate(self.ocr_results):
        user_content += f"\n--- Image {i+1} ---\n{t or '(empty)'}\n"
user_message = Message.user_message(user_content)
```

---

## 五、边界与回退

| 场景 | 行为 |
|------|------|
| 无图 | 不调用 OCR，`ocr_results=None`，Planning 按原逻辑 |
| OCR 配置缺失 | 不调用 OCR，回退到传图 |
| OCR 调用失败 | `ocr_results=None`，回退到传图 |
| OCR 部分失败 | 当前设计为全量成功才设 `ocr_results`；若需部分成功，可扩展 |
| 单图 | 同样走 OCR 批量接口（`base64_images=[img]`），`ocr_results` 长度为 1 |

---

## 六、后续阶段预留

- `self.ocr_results` 将在后续「数据提取 Skill」中复用，供 Multimodal 步骤使用
- 本阶段仅完成 Planning 解耦，不修改 `_execute_step`
