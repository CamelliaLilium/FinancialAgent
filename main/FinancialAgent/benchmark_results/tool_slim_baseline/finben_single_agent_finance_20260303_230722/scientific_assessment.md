# Single-Agent Base Benchmark Scientific Assessment

## 1) Experimental Setup

- Agent: `finance`
- Dataset: `Dataset/Finben/finben_test.json`
- Sample range: offset=0, limit=8
- Timeout per sample: 60s
- Architecture mode: single-agent direct execution (no PlanningFlow)

## 2) Core Metrics

- Accuracy: 0.6250
- Macro-F1: 0.3758
- Correct/Total: 5/8
- Avg latency/sample: 5.881s

## 3) Failure Taxonomy Evidence

- reasoning_or_prompt_misalignment: 3

## 4) Reproducibility Checklist

- Keep model/config fixed across baseline comparisons.
- Persist full artifacts: predictions, failure_cases, logs, summary.
- Run multiple trials and report confidence intervals.
