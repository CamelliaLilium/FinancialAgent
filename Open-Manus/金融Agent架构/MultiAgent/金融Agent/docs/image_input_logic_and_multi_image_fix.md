# 图片输入逻辑与多图样本修复方案

## 一、当前图片输入逻辑（全链路）

### 1. 数据层（Dataset）

```json
{
  "context": "<image 1>\n<image 2>",
  "images": [
    "/path/to/image1.png",
    "/path/to/image2.png"
  ]
}
```

- `images`：图片路径数组，顺序对应 `<image 1>`, `<image 2>`, ...
- `context`：可能包含 `<image 1>`、`<image 2>` 占位符，表示问题涉及多图

---

### 2. Benchmark 加载（`build_augmented_prompt`）

```python
# benchmark_multi_agent_multi_dataset.py:289-292
image_refs_raw = sample.get("images", []) or []
image_refs = [str(x) for x in image_refs_raw if x]
if max_images_per_sample > 0:
    image_refs = image_refs[:max_images_per_sample]  # 截断！
```

- **关键**：`max_images_per_sample` 默认 **1**，会截断为只加载第一张图
- 多图样本（如 easy-test-18、22）的 image 2 被丢弃

---

### 3. 解析与 Base64 编码

```python
# 313-330 行
for ref in image_refs:
    p = resolve_image_path(ref, ...)
    resolved.append(p)
base64_images = load_images_as_base64(resolved)  # List[str]
```

- 返回 `base64_images: List[str]`，顺序与 `images` 数组一致
- `base64_images[0]` = image 1, `base64_images[1]` = image 2

---

### 4. Flow 入口（`PlanningFlow.execute`）

```python
# app/flow/planning.py:149-162
async def execute(self, input_text: str, base64_images: Optional[List[str]] = None):
    self.base64_images = base64_images  # 存储到 flow 实例
```

- `base64_images` 为 `None` 或 `List[str]`
- 单图时 `len(base64_images)==1`，多图时 `len(base64_images)>=2`

---

### 5. Planning 阶段（创建计划）

```python
# planning.py:351-354
base64_image = self.base64_images[0] if self.base64_images else None
await planning_agent.run(plan_prompt, base64_image=base64_image)
```

- **Planning 只收到第一张图**，用于理解任务
- 若 `max_images=1`，Planning 看不到 image 2，无法规划「from image 2」的步骤

---

### 6. 执行阶段（`_execute_step`）

```python
# planning.py:623-635
if self.base64_images and step_type == "multimodal":
    img_idx = self._get_image_index_for_step(step_info)  # 从步骤文本解析
    base64_image = (
        self.base64_images[img_idx]
        if img_idx < len(self.base64_images)
        else self.base64_images[0]
    )
step_result = await executor.run(step_prompt, base64_image=base64_image)
```

**`_get_image_index_for_step`**：

```python
# planning.py:523-532
def _get_image_index_for_step(self, step_info: dict) -> int:
    text = (step_info.get("text") or "").strip()
    m = re.search(r"(?:from\s+)?image\s*(\d+)", text, re.IGNORECASE)
    if m:
        return max(0, int(m.group(1)) - 1)  # "image 2" → 1
    return 0  # 未指定则默认 image 1
```

- 步骤文本含 `"from image 1"` / `"image 2"` 时，选对应图片
- 未指定时默认 `img_idx=0`（第一张）

---

### 7. Agent 接收（`BaseAgent.run`）

```python
# app/agent/base.py:116-141
async def run(self, request, base64_image: Optional[str] = None):
    self.update_memory("user", request, base64_image=base64_image)
```

- **每次只传一张图**（`base64_image: str`）
- Multimodal 每步只看到一张图

---

## 二、多图样本失败原因

| 环节 | 问题 | 影响 |
|------|------|------|
| **max_images_per_sample=1** | 只加载第一张图 | image 2 从未进入 flow |
| **Planning 只收 image 1** | 无法知道有 image 2 | 不会写 "from image 2" |
| **步骤默认 image 1** | 未指定时用 img_idx=0 | 多步 multimodal 都看同一张图 |

**easy-test-22 典型链**：
1. 样本有 2 张图，context 含 `<image 1>\n<image 2>`
2. `max_images=1` → 只加载 image 1
3. Planning 只看到 image 1（purchase price, goodwill），看不到 image 2（revenue）
4. 计划只写 1 个 multimodal 步骤，且 revenue 在 image 2 → 永远拿不到

---

## 三、修复方案讨论

### 方案 A：提高 `max_images_per_sample` 默认值

```python
# benchmark_multi_agent_multi_dataset.py
default=2  # 或按 sample["images"] 长度动态设置
```

**优点**：改动小，多图样本能拿到两张图  
**缺点**：单图样本也会加载 2 张（若 images 数组有 2 个元素则无影响；若只有 1 个则 `[:2]` 仍为 1 张）

---

### 方案 B：按样本实际图像数动态设置

```python
image_refs = [str(x) for x in image_refs_raw if x]
# 不截断，或 max_images = min(max_images_per_sample, len(image_refs))
if max_images_per_sample > 0:
    image_refs = image_refs[:max(max_images_per_sample, len(image_refs))]
```

更合理：`max_images_per_sample` 为上限，实际加载数 = min(上限, 样本图像数)。

---

### 方案 C：Planning 阶段传入多图信息

即使 Planning 不直接看 image 2，也可通过 **prompt 提示** 告知有多图：

```python
# 在 plan_prompt 中注入
if self.base64_images and len(self.base64_images) > 1:
    plan_prompt += "\n\nNOTE: This task has {} images. Use 'from image 1' / 'from image 2' in step text to specify which image each [multimodal] step should use.".format(len(self.base64_images))
```

**作用**：引导 Planning 写出 "from image 1"、"from image 2" 的步骤。

---

### 方案 D：LLM 多图 API 支持（可选）

若底层 LLM 支持一次传入多图（如 `images: [img1, img2]`），可考虑：
- 对「需要联合看图」的任务，一次性传多图
- 对「按图分步提取」的任务，保持当前「每步单图」路由

需确认 qwen3-8b 等多模态 API 是否支持多图输入。

---

## 四、已实现优化（2024-03）

1. **Benchmark 单图/多图区分**：  
   - 单图（`len(images)==1`）：只加载 1 张  
   - 多图（`len(images)>1`）：加载 `min(max_images_per_sample, len(images))` 张

2. **Schema + LLM 多图支持**：  
   - `Message.base64_images: Optional[List[str]]`  
   - `format_messages` 支持多图，逐张追加 `image_url`

3. **Planning 多图传入**：  
   - 多图时 `base64_images` 传入全部图，Planning 可阅读并分配任务  
   - 单图时仍用 `base64_image` 单张

4. **Planning prompt 注入**：  
   - 多图时注入：`"Multi-image (N images): ... write 'from image 1' / 'from image 2' in each [multimodal] step"`

5. **执行阶段**：  
   - `_get_image_index_for_step` 按步骤文本解析，每步 Multimodal 仍只收单张图
