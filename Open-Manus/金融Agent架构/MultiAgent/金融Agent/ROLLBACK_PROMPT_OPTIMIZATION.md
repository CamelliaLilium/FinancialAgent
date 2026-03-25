# Prompt 优化回退说明

## 回退命令（pre_opt4，最新一轮）

```powershell
cd "d:\OpenManus\金融Agent"
copy /Y "app\prompt\planning.py.pre_opt4" "app\prompt\planning.py"
copy /Y "app\prompt\finance.py.pre_opt4" "app\prompt\finance.py"
copy /Y "app\prompt\multimodal.py.pre_opt4" "app\prompt\multimodal.py"
```

## pre_opt4 优化摘要（极简、无冗余、无数据污染）

| 文件 | 变更 |
|------|------|
| `app/prompt/planning.py` | %, 百分点 行扩展：second half=Q3+Q4；date-specific rows 提取该行全部列 |
| `app/prompt/multimodal.py` | NEVER output None 强化：先提取，找不到则 print('Missing: [var]')，禁止 None |
| `app/prompt/finance.py` | No-Guessing 扩展：禁止用 text 替代；image-sourced 仅用 [multimodal] 输出 |

---

## 回退命令（pre_opt3，上一轮）

```powershell
copy /Y "app\prompt\planning.py.pre_opt3" "app\prompt\planning.py"
copy /Y "app\prompt\finance.py.pre_opt3" "app\prompt\finance.py"
copy /Y "app\prompt\multimodal.py.pre_opt3" "app\prompt\multimodal.py"
```

## pre_opt3 优化摘要

| 文件 | 变更 |
|------|------|
| `app/prompt/planning.py` | 数据来自图/上下文必须安排提取；多实体表写明 for [entity]；多图指定 variable→image、year；%, 百分点 歧义澄清 |
| `app/prompt/multimodal.py` | 时间/行严格匹配；区分 单周期 vs 年度 |
| `app/prompt/finance.py` | 计算前检查 None/Missing；python_execute 内重声明变量；禁止用其他变量替代缺失变量 |

---

## 回退命令（pre_opt2，上一轮）

```powershell
copy /Y "app\prompt\planning.py.pre_opt2" "app\prompt\planning.py"
copy /Y "app\prompt\finance.py.pre_opt2" "app\prompt\finance.py"
copy /Y "app\prompt\multimodal.py.pre_opt2" "app\prompt\multimodal.py"
copy /Y "app\flow\planning.py.pre_opt2" "app\flow\planning.py"
```

## pre_opt2 优化摘要

| 文件 | 变更 |
|------|------|
| `app/prompt/planning.py` | Compare 优先比值；Accounting 优先文本提取、勿自建公式 |
| `app/prompt/multimodal.py` | Multi-image 按 step 指定 image 提取；Units 在输出中标注 |
| `app/prompt/finance.py` | None/Missing 时报告并终止；Compare 优先比值；Units 统一后再算 |
| `app/flow/planning.py` | 有 OCR 时：Use OCR content 识别表结构、实体、变量-图片映射 |

---

## 回退命令（pre_opt，更早一轮）

```powershell
copy /Y "app\prompt\planning.py.pre_opt" "app\prompt\planning.py"
copy /Y "app\prompt\finance.py.pre_opt" "app\prompt\finance.py"
copy /Y "app\prompt\multimodal.py.pre_opt" "app\prompt\multimodal.py"
copy /Y "app\flow\planning.py.pre_opt" "app\flow\planning.py"
```
