# 最新测试报告分析

**测试批次**: `multi_agent_multi_dataset_20260316_214540`  
**对比基准**: `multi_agent_multi_dataset_20260316_210940`（P0/P1 优化前）

---

## 一、总体指标对比

| 指标 | 优化前 (210940) | 最新 (214540) | 变化 |
|------|-----------------|---------------|------|
| **准确率** | 37.5% (6/16) | **43.75% (7/16)** | +1 正确 |
| **same_args_repeated** | 16/16 (100%) | 15/16 (93.75%) | -1 |
| **numeric_mismatch** | 10 | 9 | -1 |
| **python_execute 调用** | 91 | **44** | **-52%** |
| **terminate 调用** | 50 | 54 | +4 |
| **平均延迟 (s)** | 18.59 | **14.40** | **-22.5%** |
| **P95 延迟 (s)** | 42.41 | 25.67 | -39.5% |
| **平均 LLM 调用** | 10.38 | **7.19** | **-30.7%** |

---

## 二、P0 优化效果验证

### 2.1 成功即拦截生效

- **easy-test-0**：出现 `"1 time(s) successfully"` 后即被拦截（此前需 2 次成功才拦截）
- **easy-test-14**：首次成功后被拦截，触发 `CRITICAL: You have been blocked 2+ times...` 后模型调用 terminate

### 2.2 连续拦截强制 terminate 生效

- **easy-test-14** 日志：连续 2 次被拦截后收到 CRITICAL 提示，随后调用 terminate，避免跑满 max_steps

### 2.3 工具调用大幅减少

- **python_execute**：91 → 44（约减半），重复调用明显减少
- **easy-test-6**：由 5 次 python_execute 降为 2 次，不再出现 EMPTY_OUTPUT 死循环

---

## 三、通过样本 (7/16)

| 样本 | 说明 |
|------|------|
| easy-test-0 | Interest Coverage Ratio：正确使用 EBIT/Interest = 10.14 |
| easy-test-1 | 销售费用合计 2018+2019 = 24539 |
| easy-test-3 | 未确认补偿成本 26.429 |
| easy-test-4, 5, 7, 8... | 其他通过样本 |

**easy-test-0 改进**：此前因 EBITDA 误用失败，本次正确使用 984（EBIT），可能受益于 Planning 或随机波动。

---

## 四、失败样本分析 (9/16)

### 4.1 仍含 same_args_repeated（15 个）

| 样本 | 主要问题 | 备注 |
|------|----------|------|
| easy-test-2 | 重复 + 数值错误 | second half 应为 Q3+Q4，提取/公式错误 |
| easy-test-6 | 重复 + 符号错误 | 输出 1600，gold -1600（方向反） |
| easy-test-11 | 重复 + 数值偏差 | 25265 vs 25700 |
| easy-test-14 | 重复 + 提取/计算错误 | EMPTY_OUTPUT 后修正，但最终 0.00 vs 0.04 |
| easy-test-18 | 重复 + 多步传递错误 | Corporate None + Net 8846，误算 17692/26538 |
| easy-test-22 | 重复 + 公式错误 | 56.4% vs 202.7% |
| easy-test-29 | 重复 + 符号歧义 | -43958 vs 8919078 |
| easy-test-33 | 重复 + 无图分配 [multimodal] | 5000 vs -3.2 |

### 4.2 仅 numeric_mismatch（1 个）

| 样本 | 说明 |
|------|------|
| easy-test-36 | 无重复调用，Multimodal 返回 None，Finance 正确报 "Data not found" |

---

## 五、典型改进案例

### easy-test-6

- **优化前**：EMPTY_OUTPUT → 重复相同代码 4 次 → 被拦截 → terminate
- **最新**：Multimodal 提取 18900 20500 → terminate；Finance 计算 1600 → terminate
- **结论**：流程正常结束，无死循环；数值仍错（符号反），属 Planning/公式问题

### easy-test-14

- **优化前**：变量名绕过 AntiLoop，跑满 max_steps=20
- **最新**：首次 EMPTY_OUTPUT → 修正后成功 → 被拦截 1 次即停 → CRITICAL 提示 → terminate
- **结论**：P0 连续拦截机制生效，避免长时间无效循环

### easy-test-36

- **优化前**：same_args_repeated + numeric_mismatch
- **最新**：仅 numeric_mismatch，无重复调用
- **结论**：Prompt 与 AntiLoop 对「成功后立即 terminate」的约束生效

---

## 六、待改进方向

### 6.1 重复调用（15/16 仍存在）

- **terminate 重复**：`{"status": "success"}` 被多次调用，属跨步骤正常行为，可保持现状
- **python_execute 重复**：已明显减少，但部分样本仍有 2–3 次重复，可继续强化 prompt

### 6.2 数值/公式错误（9 个）

- **Planning 概念**：second half、差值方向、公式定义等
- **多图/多步**：easy-test-18 仍存在 Corporate None + Net 8846 的传递与聚合问题
- **无图场景**：easy-test-33 无图仍分配 [multimodal]

### 6.3 多图路由（P1）

- easy-test-18 为双图任务，需确认 Planning 是否生成 "from image 1" / "from image 2"
- 若已生成，需验证 Flow 是否正确按索引传递图片

---

## 七、结论

1. **P0 效果明显**：python_execute 调用减半，延迟与 LLM 调用显著下降，连续拦截机制按预期工作。
2. **准确率提升**：37.5% → 43.75%，+1 正确样本。
3. **剩余问题**：以 numeric_mismatch 为主，需从 Planning 质量、多图路由、无图处理等方面继续优化。
