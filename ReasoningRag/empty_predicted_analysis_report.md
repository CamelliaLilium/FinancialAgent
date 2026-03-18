# Empty Predicted 分析报告

## 一、概况

| 指标 | 数值 |
|------|------|
| 总 empty predicted 条数 | **59** |
| 可恢复（扩展正则） | **57** |
| 不可恢复 | **2** |

## 二、分布

### 按数据源
| 来源 | 条数 |
|------|------|
| finmmr | 38 |
| finmme | 21 |

### 按 answer_type
| 类型 | 条数 |
|------|------|
| numerical | 42 |
| mcq | 17 |

## 三、根因分析

### 3.1 当前 `extract_predicted` 逻辑（step1_generate_trajectories.py）

```python
def extract_predicted(text: str) -> str:
    m = re.search(r"\*\*Final Answer:\*\*\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().rstrip(".").split("\n")[0]
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    return m.group(1).strip().rstrip(".") if m else ""
```

仅匹配两种格式：
1. `**Final Answer:** <answer>`
2. `\boxed{<answer>}`

### 3.2 模型实际输出格式（导致 empty 的变体）

| 格式变体 | 示例 | 条数（约） |
|----------|------|------------|
| `Final Answer:`（无 `**`） | `Final Answer: 11.32` | 多数 |
| `The final answer is X` | `The final answer is 22.98` | 多数 |
| `Final Answer: Data Missing` | 模型声明数据缺失 | 1 |
| 模型幻觉/乱码 | 重复数字串 | 1 |
| `$\\boxed{}$` 空内容 | MCQ 未给出选项 | 若干 |

## 四、不可恢复的 2 条

| id | 情况 |
|----|------|
| finmmr_medium-validation-18 | `Final Answer: Data Missing`，模型明确表示无法计算 |
| finmmr_hard-validation-253 | 轨迹末尾为重复数字串 `8252264340...`，疑似幻觉/截断 |

## 五、修复方案（不修改代码时的可选操作）

### 方案 A：离线后处理脚本（推荐）

编写独立脚本 `recover_empty_predicted.py`，对 `trajectories.jsonl` 做一次性修复：

1. 读取 `trajectories.jsonl`
2. 对每条 `predicted == ""` 的记录，用扩展正则从 `trajectory` 中尝试提取
3. 若提取到有效值，则更新 `predicted` 字段
4. 输出到新文件 `trajectories_recovered.jsonl` 或原地覆盖（需备份）

**扩展提取规则建议**（按优先级）：
- `**Final Answer:**`（保持现有）
- `\boxed{...}`（保持现有）
- `Final Answer:`（无 `**`，大小写不敏感）
- `The final answer is X`（取最后一次出现，过滤 "the percentage calculated" 等元描述）
- `$\\boxed{...}$`（LaTeX 形式）

**过滤规则**：对 `Data Missing`、`无法计算`、`N/A` 等视为无效，保持 empty。

### 方案 B：修改 step1 的 `extract_predicted`（需改代码）

在 `step1_generate_trajectories.py` 中扩展 `extract_predicted`，加入上述 fallback 模式。仅影响**新生成**的轨迹，对已有 59 条需配合方案 A 做离线修复。

### 方案 C：对 2 条不可恢复记录的处理

- **finmmr_medium-validation-18**：可手动设为 `predicted: "Data Missing"`，或保持空，由 step2 Judge 按规则处理（如视为 failure）
- **finmmr_hard-validation-253**：建议保持 empty，或标记为需重新生成（若支持重跑单条）

## 六、验证

运行分析脚本：
```bash
python analyze_empty_predicted.py
```

可查看：
- 57 条可恢复样本
- 2 条不可恢复的轨迹结尾
- 与 `cases.jsonl` 中 gold 的对比（recovered 与 gold 不同多为模型答错，非提取错误）

## 七、注意事项

1. **恢复值 ≠ 正确答案**：扩展提取只是把模型写出的答案填回 `predicted`，与 gold 不一致时说明模型本身答错。
2. **MCQ 特殊格式**：部分 MCQ 输出 `**`、`$\\boxed{}$` 等，提取结果可能需额外清洗（如去 `**`、过滤空 boxed）。
3. **向后兼容**：若采用方案 A，建议先备份 `trajectories.jsonl`，再生成 `trajectories_recovered.jsonl` 供 step2 使用。
