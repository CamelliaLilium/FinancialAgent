# Single-Agent Base Benchmark Scientific Assessment

## 1) Experimental Setup

- Agent: `finance`
- Dataset: `Dataset/Finben/finben_test.json`
- Sample range: offset=0, limit=40
- Timeout per sample: 120s
- Architecture mode: single-agent direct execution (no PlanningFlow)

## 2) Core Metrics

- Accuracy: 0.4000
- Macro-F1: 0.2614
- Correct/Total: 16/40
- Avg latency/sample: 15.607s

## 3) Failure Taxonomy Evidence

- reasoning_or_prompt_misalignment: 23
- same_args_repeated: 3
- label_extraction_ambiguity: 2
- timeout: 1
- output_format_error: 1
- repeated_tool_loop: 1
- tool_execution_error: 1

## 4) Reproducibility Checklist

- Keep model/config fixed across baseline comparisons.
- Persist full artifacts: predictions, failure_cases, logs, summary.
- Run multiple trials and report confidence intervals.
