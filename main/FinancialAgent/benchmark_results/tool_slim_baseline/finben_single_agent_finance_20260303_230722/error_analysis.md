# FinBen Single-Agent Benchmark Report

- Agent: `finance`
- Total samples: 8
- Correct: 5
- Accuracy: 0.6250
- Macro-F1: 0.3758
- Avg latency/sample (s): 5.881
- Label extraction first/last mismatch count: 0

## Error Breakdown

- reasoning_or_prompt_misalignment: 3

## Tool Usage

- terminate: 8
- python_execute: 1

## Confusion Matrix

```json
{
  "dovish": {
    "neutral": 3,
    "dovish": 1
  },
  "neutral": {
    "neutral": 4
  }
}
```
