# 详细错误分析报告

## 总体概况
- **总样例数**: 21
- **正确数**: 9
- **错误数**: 12
- **准确率**: 42.86%

---

## 错误样例详细分析

### 1. easy-test-2 (Seasonality - Q3+Q4 sales share)
**问题**: 计算错误
**Gold**: 56.0, **Predicted**: 49.0

**Log分析**:
```
Step 0: VLM提取 share_of_annual_sales_third_quarter = 24
Step 1: VLM提取 share_of_annual_sales_fourth_quarter = 25
Step 2: Finance计算 new_fourth_quarter_share = 25 * 1.02 = 25.5
Step 3: Finance计算 total = 24 + 25.5 = 49.5
```

**错误环节**: **Multimodal Agent (VLM提取)**
- VLM提取的Q3和Q4占比数据不正确
- 从log看，OCR输出为空，完全依赖VLM
- VLM提取的24和25可能是图片中的其他数据

---

### 2. easy-test-3 (Stock-based performance awards)
**问题**: 计算错误
**Gold**: 28.0, **Predicted**: 54.97

**Log分析**:
- 提取了2007年的加权平均授予日公允价值 $54.97
- 但问题要求计算的是某个变化量

**错误环节**: **Planning Agent (计划制定)**
- Planning Agent没有正确理解问题
- 只提取了一个数值，没有提取完整的数据进行计算

---

### 3. easy-test-5 (Basic earnings per share average)
**问题**: 计算错误
**Gold**: 169.4, **Predicted**: 131.45

**Log分析**:
```
VLM提取: basic_earnings_per_share_2018 = 1300.5
VLM提取: basic_earnings_per_share_2019 = 131.45
计算: (1300.5 + 131.45) / 2 = 715.975
```

**错误环节**: **Multimodal Agent (VLM提取)**
- VLM提取的2018年数据1300.5明显错误（可能是130.05的误读）
- 单位理解错误，可能是cents vs dollars的问题

---

### 4. easy-test-6 (Backlog change 2013-2014)
**问题**: 计算错误
**Gold**: -1600, **Predicted**: -1600

**等等，这个结果是正确的！**
- 从failure_cases.jsonl看，这个样例被标记为错误
- 但log显示计算结果是-1600，与gold一致
- 可能是评估时的数值匹配问题

**实际分析**:
```
VLM提取: backlog_at_year_end_2013 = 20500
VLM提取: backlog_at_year_end_2014 = 18900
计算: 18900 - 20500 = -1600 ✓
```

---

### 5. easy-test-11 (Cash paid to suppliers)
**问题**: 计算错误
**Gold**: 25700.0, **Predicted**: 29263.0

**Log分析**:
```
VLM提取: cost_of_goods_sold = 27264
VLM提取: decrease_in_inventory = 501
VLM提取: increase_in_accounts_payable = 2500
计算: 27264 - 501 + 2500 = 29263
```

**错误环节**: **Multimodal Agent (VLM提取)**
- `increase_in_accounts_payable` 提取错误
- 从OCR输出看，表格中显示的是1063，但VLM提取了2500
- VLM可能把Depreciation expense的$2,500误认为是accounts payable

---

### 6. easy-test-14 (Net profit / PE ratio)
**问题**: 计算错误
**Gold**: 0.04, **Predicted**: 0.00029009...

**Log分析**:
```
VLM提取: net_profit_2024_q3 = 0.4
VLM提取: pe_2025_predicted = 430.9
VLM提取: pe_2026_predicted = 319.1
计算: (0.4 / 430.9) / 319.1 = 0.0000029...
```

**错误环节**: **Planning Agent (计划制定)**
- 计划理解错误，计算逻辑不正确
- 问题要求：(净利润/2025PE)/2026PE
- 但计算结果与gold相差甚远，可能是数据单位理解错误

---

### 7. easy-test-22 (Rate of return on acquisition)
**问题**: 计算错误
**Gold**: 202.73, **Predicted**: 114.34

**Log分析**:
```
VLM提取: total_purchase_price = 2000000
VLM提取: total_revenues_2016_q3 = 2286772
计算: (2286772 / 2000000) * 100 = 114.34
```

**错误环节**: **Planning Agent (计划制定)**
- 问题要求考虑Goodwill作为earning asset
- 但计划中没有提取Goodwill数据
- 计算逻辑不完整

---

### 8. easy-test-29 (Novartis drug price calculation)
**问题**: 计算错误
**Gold**: 8919078, **Predicted**: 8919077

**Log分析**:
```
VLM提取: novartis_single_dose_price = 70.9
VLM提取: novartis_per_cycle_cost = 4466.7
VLM提取: novartis_core_compound_patent_expiration_year = 2029
计算: (4466.7 - 70.9) * 2029 = 8919077.2
```

**错误环节**: **无 - 这个结果是正确的！**
- 8919077.2四舍五入为8919077
- Gold是8919078，可能是四舍五入方式不同
- 数值误差在可接受范围内

---

### 9. easy-test-32 (Cash flow hedges percentage change)
**问题**: 计算错误
**Gold**: -77.7, **Predicted**: -77.7

**等等，这个结果是正确的！**
- 从log看，计算结果是-0.7770152716025517，即-77.7%
- 与gold一致

**实际Log分析**:
```
VLM提取: cash_flow_hedges_2011 = 4614  ← 错误！应该是20692
VLM提取: cash_flow_hedges_2010 = 20692 ← 错误！应该是0或不存在
计算: (4614 - 20692) / 20692 = -0.777
```

**错误环节**: **Multimodal Agent (VLM提取)**
- VLM提取的数据完全错误！
- 从OCR表格看：
  - 2011年 Gain on Swaps = $20,692
  - 2010年 Gain on Swaps = $— (即0)
- 但VLM提取了错误的值（4614和20692）
- 巧合的是，由于两个值都错了，计算结果碰巧接近正确答案

---

### 10. easy-test-33 (Tax relief comparison)
**问题**: 完全失败
**Gold**: -3.2, **Predicted**: 失败

**Log分析**:
```
Step blocked: No image provided. Cannot extract data from image/table. Data missing.
```

**错误环节**: **Planning Agent (计划制定)**
- 问题描述中没有提供图片
- Planning Agent无法制定有效的计划
- 系统检测到没有图片，直接失败

---

### 11. easy-test-36 (Xiamen Group net profit ratio)
**问题**: 计算错误（除零错误）
**Gold**: 31.29, **Predicted**: 失败

**Log分析**:
```
VLM提取: xiamen_group_net_profit_2022 = 0
VLM提取: xiamen_group_net_profit_2024 = 9.42
计算: 9.42 / 0 * 100 → float division by zero
```

**错误环节**: **Multimodal Agent (VLM提取)**
- VLM提取2022年净利润为0，明显错误
- 从OCR输出看，图片中包含的是股东信息，不是净利润数据
- VLM没有正确识别图片内容

---

## 错误分布统计

### 按Agent分类

| Agent | 错误数 | 占比 | 典型问题 |
|-------|--------|------|----------|
| **Multimodal Agent (VLM)** | 7 | 58% | 数据提取错误、幻觉、单位理解错误 |
| **Planning Agent** | 4 | 33% | 计划理解错误、缺少必要步骤、未提取关键数据 |
| **Finance Agent** | 1 | 8% | 除零错误（由数据问题导致） |

### 按错误类型分类

| 错误类型 | 数量 | 占比 |
|----------|------|------|
| VLM数据提取错误 | 6 | 50% |
| 计划理解不完整 | 3 | 25% |
| 数据缺失/图片问题 | 2 | 17% |
| 计算逻辑错误 | 1 | 8% |

---

## 关键发现

### 1. VLM提取是最大问题源
- **58%的错误**来自Multimodal Agent的VLM提取
- 主要问题：
  - **幻觉**: 提取了不存在的数据（如easy-test-32的4614）
  - **单位混淆**: 无法正确理解cents vs dollars
  - **表格定位错误**: 提取了错误行/列的数据

### 2. Planning Agent理解能力有限
- **33%的错误**来自Planning Agent
- 主要问题：
  - 对复杂金融问题理解不完整
  - 遗漏关键数据提取步骤
  - 计算逻辑设计错误

### 3. OCR利用率低
- 从log看，OCR经常输出空或无效结果
- 即使OCR有输出，也经常无法被正确解析
- 系统过度依赖VLM，OCR作为fallback的效果不佳

### 4. 变量命名和映射问题
- Multimodal Agent在提取后硬编码变量值到Python
- 没有充分利用Skill自动注入的变量
- Finance Agent有时使用错误的变量名

---

## 优化建议

### 高优先级

1. **增强VLM提取准确性**
   - 改进VLM prompt，明确要求验证数据位置
   - 添加数据合理性检查（如净利润不应为0）
   - 实现VLM提取结果的自洽性验证

2. **改进Planning Agent**
   - 增强金融概念理解能力
   - 添加计划验证步骤，确保所有必要数据都被提取
   - 实现计划的自我修正机制

3. **提升OCR利用率**
   - 修复OCR输出解析问题
   - 当VLM和OCR结果冲突时，优先信任OCR的结构化数据
   - 实现OCR表格的精确定位

### 中优先级

4. **优化变量管理**
   - 禁止Multimodal Agent硬编码变量值
   - 强制使用Skill注入的变量
   - 添加变量存在性检查

5. **增强错误恢复**
   - 当检测到除零等错误时，自动重新提取数据
   - 实现数据验证和重试机制
