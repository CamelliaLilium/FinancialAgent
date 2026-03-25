# Single-Agent Base Benchmark Scientific Assessment

## 1) Experimental Setup

- Agent: `manus`
- Dataset: `Dataset/Finben/finben_test.json`
- Sample range: offset=0, limit=1
- Timeout per sample: 120s
- Architecture mode: single-agent direct execution (no PlanningFlow)

## 2) Core Metrics

- Accuracy: 0.0000
- Macro-F1: 0.0000
- Correct/Total: 0/1
- Avg latency/sample: 7.919s

## 3) Failure Taxonomy Evidence

- reasoning_or_prompt_misalignment: 1

## 4) Reproducibility Checklist

- Keep model/config fixed across baseline comparisons.
- Persist full artifacts: predictions, failure_cases, logs, summary.
- Run multiple trials and report confidence intervals.
