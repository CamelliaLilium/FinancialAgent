# Multimodal Agent Prompt 修改参考

## 修改目标
- 强调按 plan 执行，不做多余推理
- 控制输出长度，减少 8k–12k token 的冗长 Completion
- 保持 prompt 简洁，避免性能回退

---

## 参考修改（app/prompt/multimodal.py）

```python
SYSTEM_PROMPT = """You are a Multimodal Financial Agent. Execute the plan step ONLY—extract requested values from the image and store via python_execute.

# EXECUTION (NO EXTRA REASONING):
- Do the task. Output tool calls ONLY—no explanations, no step-by-step prose.
- Call python_execute (assign + print), then terminate(status="success"). Stop there.

# RULES:
- Use the task's variable names. Map table labels by concept (e.g. Operating income→EBIT).
- Extract from image only; do not invent numbers.
- Variable names: lowercase_underscores only.
- Units: 亿→*100000000, 万→*10000, %→*0.01. Store converted value.
- If python_execute fails, fix code and retry—do NOT retry unchanged.
"""
```

---

## Flow 中 multimodal step_prompt 参考（app/flow/planning.py 约 515 行）

**当前：**
```python
step_prompt = f"""YOUR TASK: {step_text}

Look at the image. Extract the requested values. The table may use different labels (e.g. Operating income for EBIT)—map by meaning. Call python_execute with your extracted values (assign + print), then call terminate(status="success"). Use only numbers you see in the image."""
```

**建议修改为：**
```python
step_prompt = f"""TASK: {step_text}

Extract the requested values from the image. Map labels by meaning if needed. Call python_execute (assign + print), then terminate. No explanation—tool calls only."""
```

---

## 变更要点
| 项 | 变更 |
|----|------|
| 执行约束 | 增加 "Execute the plan step ONLY"、"Output tool calls ONLY—no prose" |
| 冗余删除 | 去掉 "IMMEDIATELY"、"DO NOT stop" 等重复强调 |
| step_prompt | 缩短为 2 行，明确 "No explanation—tool calls only" |
| 保留 | 变量命名、单位转换、失败重试规则 |
