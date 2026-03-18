# Multi-Agent Dataset Report

- Dataset file: `/root/autodl-tmp/datasets/test/finmmr_medium_test.json`
- LLM model: `Qwen/Qwen3-VL-8B-Instruct`
- Planning enabled: `True`
- Multimodal mode: `best_effort`
- Total samples: 5
- Correct: 2
- Accuracy: 0.4000
- Avg latency/sample (s): 92.687
- P50 latency/sample (s): 59.235
- P95 latency/sample (s): 183.985
- Avg LLM calls/sample: 8.0
- Avg zero-tool rounds/sample: 3.0
- Avg blocked marks/sample: 0.0
- Multimodal samples: 4
- Multimodal resolved samples: 4
- Multimodal resolution rate: 1.0
- Vision extraction ok: 4
- Vision extraction failed: 0
- Numeric MAE: 798.776457
- Numeric eval samples: 5
- Numeric parse ambiguity count: 4

## Status Breakdown

- ok: 5

## Error Tag Breakdown

- numeric_mismatch: 3
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
