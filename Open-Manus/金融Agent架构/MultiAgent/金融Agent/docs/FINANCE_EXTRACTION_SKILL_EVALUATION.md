# 金融数值提取 Skill 架构评估

> 针对「OCR 感知 + VLM Point-Check + 变量持久化」的金融数值提取 Skill 构想，从问题覆盖、科研适配性、可完善点三方面进行客观评估。

---

## 一、Skill 架构回顾

```
输入：step 文本 + base64_image(s) + 变量列表 (var_a, var_b, ...)
  ↓
1. DeepSeek OCR 提取结构化数据（含 bbox）
  ↓
2. 8B VLM 传入 Box 坐标进行 Point-Check（交叉验证）
  ↓
3. 清洗 + 单位换算
  ↓
4. python_execute 写入变量到全局环境
  ↓
输出：SUCCESS + 变量就绪 | FAILURE + 具体原因
```

---

## 二、能否解决当前错误？

### 2.1 覆盖度分析

| 错误类别 | 样本 | Skill 能否缓解 | 说明 |
|----------|------|----------------|------|
| **3.1 提取错误/数据源混淆** | easy-test-6, 10, 27 | ✅ **可缓解** | OCR 结构化 + Point-Check 可减少「错行/错列」；easy-test-6 若 OCR 明确提取 backlog 行、VLM 校验 bbox 区域，可避免与 net_sales 混淆 |
| **3.2 多图字段缺失/错图** | easy-test-18, 22, 36 | ✅ **可缓解** | 变量→图片映射由 step 明确；缺失时返回 FAILURE 而非 None，避免 3.3 |
| **3.3 违反 FORBIDDEN（输出 None）** | easy-test-22, 36, 194 | ✅ **可消除** | Skill 闭环：提取失败 → 返回 FAILURE，不写入 None；Multimodal 不再有机会输出 None |
| **3.4 公式/概念理解错误** | easy-test-5, 14, 17, 20, 31, 33 | ❌ **不覆盖** | 属于 Planning/Finance 层，Skill 专注提取，合理 |

**结论**：Skill 能针对性解决 3.1、3.2、3.3，与「数值提取」问题高度匹配；3.4 由公式层承担，符合职责划分。

### 2.2 与现有 OCR 的衔接

当前 `app/tool/ocr.py` 已支持：

- **regions**：`{text, bbox: [x,y,w,h]}`，归一化坐标 0~1
- **markdown**：表格/段落文本

**现状**：Flow 仅将 `text` 传给 Planning，**regions 未下游使用**。Skill 可直接复用 OCR 的 regions，无需重复实现 OCR。

**注意**：DeepSeek OCR 的 prompt 允许「无法精确估计坐标时省略 Regions」。若 regions 经常为空，Point-Check 无法执行。建议：

- 强化 OCR prompt，要求表格单元格必须输出 bbox；或
- 引入表格专用模型（如 Table Transformer）输出 cell-level bbox，作为 Skill 的备选输入。

---

## 三、是否适合金融 Agent 科研工作？

### 3.1 科研价值

| 维度 | 评价 |
|------|------|
| **问题定义** | 金融 Agent 的「表格数值提取」是明确、可复现的问题，有 FinMMR 等 benchmark |
| **方法新颖性** | OCR + VLM Point-Check 形成「感知-校验」双阶段，与纯 VLM 端到端提取形成对比，可写成方法贡献 |
| **可复现性** | 需明确：OCR 模型、bbox 格式、VLM 校验 prompt、单位换算规则 |
| **可评估性** | 可增加指标：提取准确率（变量级）、Point-Check 通过率、与 gold 的数值误差 |

### 3.2 论文叙事建议

1. **动机**：纯 VLM 提取易出现错行、错图、None 输出，影响下游金融计算。
2. **方法**：提出「OCR 结构化提取 + VLM 区域校验」的金融数值提取 Skill，实现感知与校验解耦。
3. **实验**：对比 (a) 纯 VLM；(b) OCR-only；(c) OCR + VLM Point-Check；报告提取准确率与端到端任务准确率。
4. **消融**：Point-Check 的贡献（有/无 Point-Check 的对比）。

### 3.3 与 OpenManus 的契合度

- **Skill 作为工具**：可注册为 `finance_extraction_skill`，供 Multimodal 或 Flow 调用，符合 OpenManus 的 tool 抽象。
- **闭环输出**：SUCCESS/FAILURE 与具体原因，便于 Flow 做分支（重试、fallback、报错）。
- **变量持久化**：通过 `python_execute` 写入共享环境，与现有 Finance Agent 的计算流程一致。

---

## 四、可完善的部分

### 4.1 变量 ↔ 区域映射

**问题**：step 要求「extract net_sales_2019, gross_profit_2018」，OCR 输出的是表格单元格（可能无变量名）。如何映射？

**建议**：

- **方案 A**：OCR 输出时要求「表头/行标签 + 数值 + bbox」，由 Planning 或 Skill 内逻辑做语义匹配（如 "Net Sales 2019" → net_sales_2019）。
- **方案 B**：Skill 接收 `(variable_name, semantic_hint)`，例如 `("net_sales_2019", "Net sales for 2019")`，用 hint 在 OCR 的 text 中检索，再取对应 bbox 做 Point-Check。
- **方案 C**：若 OCR 输出结构化表格（行列索引），Planning 在 step 中显式指定 `"from row 2, col 3"`，Skill 按行列取 cell。

### 4.2 多图任务的变量-图绑定

**问题**：easy-test-18 中 corporate_notes_bonds 在 image 1，net_sales 在 image 2。Skill 需知道每个变量对应哪张图。

**建议**：Flow 调用 Skill 时传入 `[(var, image_index), ...]`，例如 `[("corporate_notes_bonds_2010_fair_value", 0), ("net_sales_2011", 1)]`，由 step 解析 "from image 1" / "from image 2" 得到。Skill 内部按 image_index 选择图片和 OCR 结果。

### 4.3 Point-Check 的粒度与成本

**问题**：若 regions 很多，每个都做 Point-Check 会增加 VLM 调用次数和延迟。

**建议**：

- 仅对「step 要求的变量」对应的 region 做 Point-Check，而非全表。
- Point-Check 的 prompt 尽量简短：「此区域是否显示数值 X？（是/否）」，降低 token 消耗。
- 可选：对高置信度（如 OCR 与表格结构强一致）的 cell 跳过 Point-Check，只对歧义或关键数值校验。

### 4.4 失败时的降级策略

**问题**：Skill 返回 FAILURE 时，Flow 如何应对？

**建议**：

- **策略 1**：直接 terminate(status='failure')，向用户报告「某变量提取失败」。
- **策略 2**：Fallback 到当前 Multimodal 的纯 VLM 提取，作为兜底。
- **策略 3**：重试一次（如 OCR 重跑、或换不同 bbox 解析方式）。

在论文中可对比「无降级」「Fallback 到 VLM」两种策略对端到端准确率的影响。

### 4.5 单位换算的确定性

**问题**：「执行清洗和单位换算」依赖规则（万→×10000，亿→×1e8）。若表格中单位标注不清，易出错。

**建议**：

- 单位换算规则显式化、可配置（如 YAML/JSON），便于复现和消融。
- 若无法确定单位，Skill 返回 FAILURE 并注明「单位歧义」，而非猜测。

### 4.6 与现有 Flow 的集成方式

**选项**：

| 方式 | 说明 | 复杂度 |
|------|------|--------|
| **A. Skill 作为 Multimodal 的工具** | Multimodal 在 extraction step 中调用 Skill，成功则用其输出，失败则自行提取 | 中 |
| **B. Skill 替代 Multimodal 的 extraction** | Flow 在 [multimodal] step 时直接调 Skill，不再调 Multimodal | 低 |
| **C. 并行双路** | OCR+Skill 与 Multimodal 并行，取置信度高的结果 | 高 |

建议先实现 **B**，逻辑清晰，便于做对照实验（有/无 Skill）。

---

## 五、总结

| 问题 | 结论 |
|------|------|
| **能否解决数值提取问题？** | 能。对 3.1、3.2、3.3 类错误有直接缓解作用，且与 3.4 职责分离合理。 |
| **是否适合金融 Agent 科研？** | 适合。问题清晰、方法有对比空间、可复现可评估，可形成独立方法贡献。 |
| **关键完善点** | 变量-区域映射、多图变量-图绑定、Point-Check 粒度与成本、失败降级策略、单位换算确定性、Flow 集成方式。 |

**建议实施顺序**：先实现最小闭环（OCR→变量映射→python_execute），验证提取准确率提升；再加入 Point-Check 做消融；最后完善多图、单位、降级等细节。
