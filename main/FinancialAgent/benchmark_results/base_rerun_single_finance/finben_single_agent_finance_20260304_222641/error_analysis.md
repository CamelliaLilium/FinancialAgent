# FinBen Single-Agent Benchmark Report

- Agent: `finance`
- Total samples: 496
- Correct: 293
- Accuracy: 0.5907
- Macro-F1: 0.4709
- Avg latency/sample (s): 21.047
- Label extraction first/last mismatch count: 34

## Error Breakdown

- reasoning_or_prompt_misalignment: 203
- label_extraction_ambiguity: 34
- tool_execution_error: 1

## Tool Usage

- terminate: 495

## Confusion Matrix

```json
{
  "dovish": {
    "neutral": 90,
    "hawkish": 13,
    "dovish": 26
  },
  "neutral": {
    "neutral": 234,
    "hawkish": 7,
    "dovish": 4
  },
  "hawkish": {
    "neutral": 86,
    "dovish": 3,
    "hawkish": 33
  }
}
```
