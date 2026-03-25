# -*- coding: utf-8 -*-
"""分析「正常执行」样本的正确率，排除流程异常"""
import json
from pathlib import Path
from collections import Counter

path = Path(__file__).parent / "benchmark_results/multi_agent_multi_dataset_20260311_183537/finmmr__finmmr_easy_test/predictions.jsonl"
records = []
with path.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

# 流程异常 tag（架构/执行问题，不反映智能体能力）
FLOW_ANOMALY_TAGS = {
    "timeout",
    "runtime_error",
    "empty_output",
    "rate_limit",
    "repeated_tool_loop",
    "tool_execution_error",
    "multimodal_input_missing",
    "multimodal_vision_failure",
    "same_args_repeated",
}

# 正常执行 = status=ok 且 无上述流程异常 且 zero_tool_rounds 不过多
def is_clean_execution(r):
    if r["status"] != "ok":
        return False
    tags = set(r.get("error_tags", []))
    if tags & FLOW_ANOMALY_TAGS:
        return False
    if r.get("zero_tool_rounds", 0) >= 5:
        return False
    return True


clean = [r for r in records if is_clean_execution(r)]
clean_correct = [r for r in clean if r["is_correct"]]

print("=== 正常执行样本统计（排除流程异常）===")
print(f"总样本: {len(records)}")
print(f"正常执行（无流程异常、zero_tool<5）: {len(clean)}")
print(f"正常执行且答案正确: {len(clean_correct)}")
if clean:
    acc = len(clean_correct) / len(clean) * 100
    print(f"正常执行正确率: {acc:.1f}% ({len(clean_correct)}/{len(clean)})")
print()

# 被排除样本的分布
excl_by_reason = Counter()
for r in records:
    if r["status"] != "ok":
        excl_by_reason["status!ok:" + r["status"]] += 1
    elif not is_clean_execution(r):
        tags = set(r.get("error_tags", [])) & FLOW_ANOMALY_TAGS
        z = r.get("zero_tool_rounds", 0)
        if z >= 5:
            excl_by_reason["zero_tool_rounds>=5"] += 1
        for t in tags:
            excl_by_reason[t] += 1

print("=== 被排除的流程异常分布 ===")
for k, v in excl_by_reason.most_common():
    print(f"  {k}: {v}")

# 放宽 same_args_repeated：仅排除严重流程异常
FLOW_ANOMALY_STRICT = {
    "timeout", "runtime_error", "empty_output", "rate_limit",
    "repeated_tool_loop", "tool_execution_error",
    "multimodal_input_missing", "multimodal_vision_failure",
}

def is_clean_execution_relaxed(r):
    if r["status"] != "ok":
        return False
    tags = set(r.get("error_tags", []))
    if tags & FLOW_ANOMALY_STRICT:
        return False
    if r.get("zero_tool_rounds", 0) >= 5:
        return False
    return True

clean_relaxed = [r for r in records if is_clean_execution_relaxed(r)]
clean_relaxed_correct = [r for r in clean_relaxed if r["is_correct"]]
print()
print("=== Relaxed (exclude severe only, keep same_args_repeated) ===")
print(f"Clean: {len(clean_relaxed)}, Correct: {len(clean_relaxed_correct)}")
if clean_relaxed:
    print(f"Accuracy: {len(clean_relaxed_correct)/len(clean_relaxed)*100:.1f}%")

# Tag overlap: same_args_repeated vs correct
same_only = [r for r in records if r["status"]=="ok" and "same_args_repeated" in r.get("error_tags",[]) and "repeated_tool_loop" not in r.get("error_tags",[]) and r.get("zero_tool_rounds",0)<5]
same_correct = [r for r in same_only if r["is_correct"]]
print()
print("=== same_args_repeated only (no repeated_tool_loop, zero_tool<5) ===")
print(f"Count: {len(same_only)}, Correct: {len(same_correct)}")
if same_only:
    print(f"Accuracy: {len(same_correct)/len(same_only)*100:.1f}%")
