# Single-Agent Base Benchmark Scientific Assessment

## 1) Experimental Setup

- Agent: `finance`
- Dataset: `Dataset/Finben/finben_test.json`
- Sample range: offset=0, limit=0
- Timeout per sample: 600s
- Architecture mode: single-agent direct execution (no PlanningFlow)

## 2) Core Metrics

- Accuracy: 0.5907
- Macro-F1: 0.4709
- Correct/Total: 293/496
- Avg latency/sample: 21.047s

## 3) Failure Taxonomy Evidence

- reasoning_or_prompt_misalignment: 203
- label_extraction_ambiguity: 34
- tool_execution_error: 1

## 4) Reproducibility Checklist

- Keep model/config fixed across baseline comparisons.
- Persist full artifacts: predictions, failure_cases, logs, summary.
- Run multiple trials and report confidence intervals.
