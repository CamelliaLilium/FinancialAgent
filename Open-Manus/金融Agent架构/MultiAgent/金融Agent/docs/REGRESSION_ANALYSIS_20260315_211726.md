# 回退分析：20260315_211726（Multimodal OUTPUT 限制）

## 一、总体对比

| 指标 | 基线 (191306) | 本次 (211726) | 变化 |
|------|---------------|---------------|------|
| **正确数** | 8/16 | 4/16 | **-4** |
| **正确率** | 50% | 25% | **-25pp** |
| 平均耗时 | 110.8s | 106.1s | -4.7s |
| numeric_mae | 2381.6 | 1399525.9 | 暴增 |
| same_args_repeated | 15 | 16 | +1 |
| numeric_mismatch | 8 | 12 | +4 |

**结论**：加入 Multimodal OUTPUT 限制后，正确率从 50% 降至 25%，出现明显回退。

---

## 二、样本级对比（正确→错误）

| 样本 | 基线 (191306) | 本次 (211726) | 根因简述 |
|------|---------------|---------------|----------|
| **easy-test-0** | ✓ 10.14 | ✗ 9.84 | Finance 变量名 typo：`ebti_2018` 而非 `interest_expense_2018` |
| **easy-test-17** | ✓ 25.3125 | ✗ 25 | Multimodal 从文本取 25，未从表格取 750×0.03375 |
| **easy-test-194** | ✓ 13,435,200 | ✗ 54,357,000 | Multimodal 混行：Oct+Nov 合计，未仅取 October |
| **easy-test-29** | ✓ 8,919,078 | ✗ 183,219 | Multimodal 整表 dump，未按「单片价格-单周期费用」选单行 |

---

## 三、根因分析

### 3.1 本次修改

在 `app/prompt/multimodal.py` 中新增：

```python
# OUTPUT (STRICT):
- Output tool calls ONLY. NO explanations, NO reasoning, NO step-by-step prose.
- Be efficient: extract values and call tools directly.
```

### 3.2 与 202522 回退的相似性

| 现象 | 202522（精简 prompt） | 211726（OUTPUT 限制） |
|------|----------------------|------------------------|
| easy-test-0 | 错 | 错 |
| easy-test-17 | 错 | 错 |
| easy-test-194 | 错 | 错 |
| easy-test-29 | 错 | 错 |

两次回退都涉及：**按 plan 指定行/实体筛选** 的约束被弱化。

### 3.3 具体机制

1. **easy-test-0**  
   - 基线：Multimodal 正确提取 984、97；Finance 正确计算 10.14。  
   - 本次：Multimodal 仍正确；Finance 代码中 `interest_expense_2018` 误写为 `ebti_2018`，导致 NameError 或错误结果。  
   - 推测：「NO reasoning」使 Finance 在快速写代码时更容易出现变量名 typo。

2. **easy-test-17**  
   - 基线：Multimodal 从表格取 maturity=750、rate=0.03375，Finance 算出 25.3125。  
   - 本次：Multimodal 直接从文本取「$25 million」，未按表格计算。  
   - 推测：「Be efficient」促使模型优先用文本，跳过表格映射与计算。

3. **easy-test-194**  
   - 基线：Multimodal 仅取 October 行（90,000 × 149.28 = 13,435,200）。  
   - 本次：Multimodal 取 Oct+Nov 两行合计（54,357,000）。  
   - 推测：为追求「efficient」，未按 plan 中「during October 2018」做行级筛选。

4. **easy-test-29**  
   - 基线：Multimodal 提取单行 (70.9, 4466.7, 2029)，Finance 正确计算。  
   - 本次：Multimodal 整表 dump 四行，Finance 用错误行计算。  
   - 推测：「NO reasoning」削弱了「按题目选单药」的约束，导致整表输出。

---

## 四、结论与建议

### 4.1 结论

- **OUTPUT 限制**（禁止解释、推理、步骤说明）在 Multimodal 上产生了与 202522 类似的副作用：
  - 弱化「按 plan 指定行/实体筛选」
  - 鼓励「快速输出」而非「按题意精确提取」
- 正确率从 50% 降至 25%，回退幅度大，**不建议保留当前形式的 OUTPUT 限制**。

### 4.2 建议

| 选项 | 说明 |
|------|------|
| **A. 回滚 OUTPUT 限制** | 移除 `# OUTPUT (STRICT)` 段，恢复 50% 基线 |
| **B. 弱化限制** | 改为「优先输出工具调用，必要时可简短说明」，避免完全禁止推理 |
| **C. 仅限 Finance** | OUTPUT 限制只加在 Finance prompt，Multimodal 保持原样 |
| **D. 用 max_tokens 替代** | 不改 prompt，通过配置降低 Multimodal 的 max_tokens 控制输出长度 |

**推荐**：先执行 **A（回滚）**，再考虑 **D** 作为不依赖 prompt 的耗时优化手段。
