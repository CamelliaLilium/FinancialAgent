# Multi-Agent Dataset Report

- Dataset file: `/root/autodl-tmp/datasets/test/flare-convfinqa_test.json`
- LLM model: `Qwen/Qwen3-VL-8B-Instruct`
- Planning enabled: `True`
- Multimodal mode: `best_effort`
- Total samples: 2
- Correct: 2
- Accuracy: 1.0000
- Avg latency/sample (s): 38.126
- P50 latency/sample (s): 38.126
- P95 latency/sample (s): 44.044
- Avg LLM calls/sample: 6.0
- Avg zero-tool rounds/sample: 2.0
- Avg blocked marks/sample: 0.0
- Multimodal samples: 0
- Multimodal resolved samples: 0
- Multimodal resolution rate: None
- Vision extraction ok: 0
- Vision extraction failed: 0
- Numeric MAE: 0.0
- Numeric eval samples: 2
- Numeric parse ambiguity count: 0

## Status Breakdown

- ok: 2

## Error Tag Breakdown

- correct: 2

## Tool Usage


## Research Notes

- `timeout` / `runtime_error`: stability and serving-layer risks.
- `repeated_tool_loop` / `same_args_repeated`: workflow-control inefficiency.
- `prediction_parse_failure`: output contract mismatch (answer extraction risk).
- `multimodal_input_missing`: image path resolution gap or data mounting issue.
- `multimodal_vision_failure`: vision extraction call failed or timed out.
- `numeric_mismatch` / `code_mismatch` / `text_mismatch`: reasoning or grounding error.
- `numeric_mae` helps quantify magnitude, not only correctness rate.
