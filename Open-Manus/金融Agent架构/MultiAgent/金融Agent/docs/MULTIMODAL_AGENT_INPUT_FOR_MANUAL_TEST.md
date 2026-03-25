# Multimodal Agent 完整输入（用于单独投喂模型测试）

本文档给出 easy-test-0 任务中 Multimodal Agent 收到的**全部模型输入**，便于你单独投喂给模型验证数值提取能力。

---

## 1. 消息结构

框架发送给 LLM 的消息顺序为：

1. **System message**：系统提示词
2. **User message**：任务描述（step_prompt，已合并）+ **图像**（base64）

（Multimodal 已改为单条 user 消息，不再追加 next_step_prompt）

---

## 2. System Message（完整文本）

```
You are a Multimodal Financial Agent. Your job: look at the image, extract the numbers you see, and store them via python_execute.

# WORKFLOW:
1. Look at the image — read tables/charts, extract the numbers for the requested variables.
2. Call python_execute — assign values and print. Use native tool_calls API, never markdown code blocks.
3. Call terminate(status="success") after python_execute returns.

# RULES:
- Always call python_execute first to store extracted values. Then call terminate.
- Output: use the task's variable names in code. The table may use different labels (e.g. Operating income for EBIT)—map by concept.
- Extract only from the image; do not invent numbers.
```

---

## 3. User Message（任务描述 + 图像，单条合并）

**文本部分：**

```
YOUR TASK: [multimodal] Extract the necessary financial values (e.g., EBIT, interest expenses) from the image for the year 2018.

Look at the image. Extract the requested values. The table may use different labels (e.g. Operating income for EBIT)—map by meaning. Call python_execute with your extracted values (assign + print), then call terminate(status="success"). Use only numbers you see in the image.
```

**图像：**
- 本地路径：`D:\OpenManus\base LLM的基准测试\Dataset\FinMMR-main\images\1976-1.png`
- 数据集中的引用：`/root/autodl-tmp/datasets/FinMMR/images/1976-1.png`（通过 `resolve_image_path` 会解析到上述本地路径）

在 API 调用中，图像以 base64 形式附加到该 user message 的 content 中，格式为：
```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/png;base64,<base64_string>"
  }
}
```

---

## 4. 工具定义（可选）

框架还会传入工具定义（python_execute、terminate、str_replace_editor）。若你只想测试**纯提取能力**，可先不传工具，仅用 system + user 消息；若要完全复现 agent 行为，需一并传入工具。

---

## 5. 复现方式

### 方式 A：仅测试提取（无工具）

发送给模型：
- **System**：第 2 节的完整文本
- **User**：第 3 节文本 + 图像（多模态 content 格式）

### 方式 B：完整复现（含工具）

使用与 `app/llm.py` 相同的消息格式，并传入 `python_execute`、`terminate` 等工具定义。

### 图像加载示例（Python）

```python
import base64
from pathlib import Path

image_path = Path(r"D:\OpenManus\base LLM的基准测试\Dataset\FinMMR-main\images\1976-1.png")
base64_str = base64.b64encode(image_path.read_bytes()).decode("ascii")
# 用于 API: f"data:image/png;base64,{base64_str}"
```

---

## 6. Ground Truth（用于验证）

- **EBIT** = 984
- **interest_expense** = 97
- **Interest Coverage Ratio** = 984 / 97 ≈ 10.144

若模型能正确提取 984 和 97，则说明其具备从该图像中提取数值的能力。
