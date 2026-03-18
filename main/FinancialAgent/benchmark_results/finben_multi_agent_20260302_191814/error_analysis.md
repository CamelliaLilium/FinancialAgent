# FinBen Multi-Agent Benchmark Report

- Total samples: 1
- Correct: 0
- Accuracy: 0.0000
- Avg latency/sample (s): 29.385

## Error Breakdown

- same_args_repeated: 1
- reasoning_or_prompt_misalignment: 1

## Tool Usage

- terminate: 4
- planning: 1

## Confusion Matrix

```json
{
  "dovish": {
    "neutral": 1
  }
}
```

## Research-Oriented Failure Interpretation

- `same_args_repeated` / `repeated_tool_loop`: tool policy or step-boundary control is weak.
- `likely_unnecessary_web_search`: prompt/tool routing mismatch for in-context solvable tasks.
- `output_format_error`: output contract under-specified (classification label not strictly enforced).
- `reasoning_or_prompt_misalignment`: model reasoning drift or prompt instruction ambiguity.
- `tool_execution_error`: tool reliability gap or missing task-fit tools.
