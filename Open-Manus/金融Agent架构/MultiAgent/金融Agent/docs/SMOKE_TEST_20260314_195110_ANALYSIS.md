# Smoke Test 20260314_195110 深度分析报告

## 一、测试概览

| 指标 | 数值 |
|------|------|
| 总样本 | 16 |
| 正确 | 8 |
| **正确率** | **50%** |
| 平均耗时 | 94.3s |
| P95 耗时 | 223.2s |
| Timeout | 1 (easy-test-22) |
| Numeric MAE | 595313.65 |

### 正向变化
- **easy-test-11**：由错变对 ✓（COGS - decrease - increase 公式正确应用）

### 失败样本（8 个）
easy-test-2, 14, 17, 18, 22, 29, 33, 36

---

## 二、根因分类与典型样例

### 1. 数据/配置层：图片未传入（easy-test-33）

**现象**：`has_image_refs: false`，Multimodal 无图可看，直接编造数值。

**根因**：数据集 `finmmr_easy_smoke_test.json` 中 easy-test-33 的字段为：
```json
"images": [],
"ground_images": ["/root/autodl-tmp/datasets/FinMMR/images/4231-1.png"]
```
Benchmark 仅使用 `sample.get("images", [])`，未回退到 `ground_images`。

**日志证据**：
```
base64_images_count=0
Step 0 passing base64_image to executor: no (len=0)
```
Multimodal 输出编造值：`tax_reduction_apr_1998 = 12500, tax_reduction_nov_1998 = 18750`  
正确值应为：4.6, 6, 7.4（来自表格）。

**优化方向**：
- Benchmark：当 `images` 为空且 `ground_images` 非空时，使用 `ground_images` 作为图片源
- 或：修复 smoke test 数据集，确保 `images` 字段与 `ground_images` 一致

---

### 2. 多图任务：只传一张图 + 固定传第一张（easy-test-18, easy-test-22）

**easy-test-18**：
- 题目需要 2 张图：图1 → Corporate notes/bonds 2010 Fair Value；图2 → Net sales 2011
- 当前配置 `max_images_per_sample: 1`，只加载 1 张图
- Flow 中 `base64_image = self.base64_images[0]`，所有 multimodal 步骤都收到同一张图
- Step 0 从「图1」提取 → 返回 `None None`（可能表结构复杂或取错行列）
- Step 2 需要「图2」提取 Net sales，但收到的仍是图1 → 无法正确提取

**easy-test-22**：
- 2 张图，复杂 ROI 计算（revenue/purchase price）
- 同样只传 1 张图，且 300s 内未完成 → timeout

**优化方向**：
- 多图题目：`max_images_per_sample` 至少等于题目所需图片数，或按需动态设置
- Flow：按 step 语义传递对应图片（如 step 指定 "from image 2" 则传 `base64_images[1]`），而非始终传 `base64_images[0]`

---

### 3. 公式/语义理解错误（easy-test-2, easy-test-17）

**easy-test-2**：「Q4 goes up by 2%」
- Planning 正确：`new Q4_share = Q4_share + 2%`
- Multimodal 提取：Q3=24, Q4=25（gold 为 25, 29，提取有偏差）
- Finance 将 `+ 2%` 理解为 `+ 2`（加 2 个百分点），得到 24+27=51；gold 为 56（25+29+2）
- 问题：① 提取数值错误；② "2%" 的语义（加 2 个百分点 vs 乘 1.02）需在 prompt 中明确

**easy-test-17**：「annual interest expense for 2022 notes」
- 文本：`$750 million of 3.375% notes` → 年利息 = 750 × 0.03375 = 25.3125 million
- 模型直接取文中 "approximately $25 million" → 25，未按公式计算

**优化方向**：
- Planning/Finance prompt：对「X%」「X million」「approximately」等表述，强调需按公式推导而非直接摘抄
- 对「goes up by 2%」等歧义表述，在 prompt 中给出常见金融语义（如「增加 2 个百分点」）

---

### 4. 符号/顺序错误（easy-test-29）

**现象**：差值 70.9 - 4466.7 = -4395.8，乘以 2029 → -8919078；gold 为 8919078（取绝对值或顺序相反）。

**根因**：题目「单片价格与单周期费用之间的差值」未明确顺序，模型计算了 A-B，而 gold 可能为 |A-B| 或 B-A。

**优化方向**：
- 题目若存在顺序歧义，在 prompt 中提示「若为比较类，考虑绝对值或明确顺序」
- 或：在评估时对「仅符号相反」的结果做特殊处理（需与业务一致）

---

### 5. 数据提取失败（easy-test-14, easy-test-36）

**easy-test-14**：
- 需：2024Q3 净利润、2025 预测 PE、2026 预测 PE
- 提取到：26.9, 430.9, 319.1
- 正确应为：334.5, 110.9, 83.9（取错行列或表格区域）

**easy-test-36**：
- Multimodal 返回 `None None`，后续 Finance 报 "Data not found"
- 归母净利润 2022/2024E 未从图中正确识别

**优化方向**：
- Multimodal prompt：强调表头、行列对应关系，失败时要求根据错误信息修改提取逻辑而非重复相同调用
- 对表格类题目，可增加「先描述表结构再取值」的约束

---

### 6. Anti-loop 与 same_args_repeated

**现象**：14 个样本出现 `same_args_repeated`，多为 `terminate` 或相同 `python_execute` 被重复调用。

**影响**：
- 提前终止：easy-test-2 中 Finance 重复执行相同代码 3 次后被 anti-loop 拦截，可能尚未输出最终答案
- 流程冗余：多次 `terminate` 增加 token 消耗

**优化方向**：
- 区分「重复无意义调用」与「重试」：对 python_execute，若上次 observation 为空或异常，允许修改后重试
- 对 terminate，可考虑首次成功即不再计入 same_args

---

## 三、问题优先级建议

| 优先级 | 问题 | 影响样本 | 建议 |
|--------|------|----------|------|
| P0 | 图片源：images 空时用 ground_images | easy-test-33 | Benchmark 或数据集修复 |
| P0 | 多图：max_images_per_sample 与按 step 传图 | easy-test-18, 22 | Benchmark + Flow |
| P1 | 公式语义（%、approximately） | easy-test-2, 17 | Planning/Finance prompt |
| P1 | 符号/顺序歧义 | easy-test-29 | Prompt 或评估策略 |
| P2 | 表格行列提取准确性 | easy-test-14, 36 | Multimodal prompt |
| P2 | Anti-loop 与重试策略 | 多个 | anti_loop / 流程逻辑 |

---

## 四、与 20260311 对比（easy_test 200 样本）

| 指标 | 20260311 | 20260314 (easy_test) | 变化 |
|------|----------|----------------------|------|
| 正确率 | 58% | 67% | +9% |
| 平均耗时 | 177.5s | 91.8s | -48% |
| P95 耗时 | 600s | 216.5s | -64% |
| Timeout | 24 | 1 | 大幅减少 |

Smoke test 正确率 50% 低于 easy_test 67%，主因：
1. Smoke 样本为刻意挑选的易错/典型样例，本身更难
2. 多图、无图、公式歧义等集中在 smoke 中，暴露了配置与 prompt 的短板

---

## 五、后续讨论要点

1. **easy-test-33 无图**：优先修复 `images`/`ground_images` 解析，确保 smoke 覆盖有图样本
2. **easy-test-18/22 多图**：是否在 benchmark 中提高 `max_images_per_sample`，以及 Flow 是否支持「按 step 选图」
3. **easy-test-2 公式**：「Q4+2%」的标准化语义是否写入 prompt
4. **easy-test-29 符号**：是否在评估或 prompt 中处理「仅符号相反」的情况
5. **Anti-loop**：是否放宽对「修改后重试」的限制，避免误杀有效重试
