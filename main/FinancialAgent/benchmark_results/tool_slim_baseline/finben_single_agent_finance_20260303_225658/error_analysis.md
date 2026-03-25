# FinBen Single-Agent Benchmark Report

- Agent: `finance`
- Total samples: 40
- Correct: 16
- Accuracy: 0.4000
- Macro-F1: 0.2614
- Avg latency/sample (s): 15.607
- Label extraction first/last mismatch count: 2

## Error Breakdown

- reasoning_or_prompt_misalignment: 23
- same_args_repeated: 3
- label_extraction_ambiguity: 2
- timeout: 1
- output_format_error: 1
- repeated_tool_loop: 1
- tool_execution_error: 1

## Tool Usage

- terminate: 39
- python_execute: 16

## Confusion Matrix

```json
{
  "dovish": {
    "neutral": 12,
    "dovish": 2,
    "hawkish": 1
  },
  "neutral": {
    "neutral": 14,
    "none": 1,
    "hawkish": 1
  },
  "hawkish": {
    "neutral": 9
  }
}
```
