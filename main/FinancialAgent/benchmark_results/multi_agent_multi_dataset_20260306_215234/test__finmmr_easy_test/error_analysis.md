# Multi-Agent Dataset Report

- Dataset file: `/root/autodl-tmp/datasets/test/finmmr_easy_test.json`
- LLM model: `Qwen/Qwen3-VL-8B-Instruct`
- Planning enabled: `True`
- Multimodal mode: `best_effort`
- Total samples: 5
- Correct: 3
- Accuracy: 0.6000
- Avg latency/sample (s): 42.83
- P50 latency/sample (s): 34.476
- P95 latency/sample (s): 85.331
- Avg LLM calls/sample: 7.2
- Avg zero-tool rounds/sample: 2.6
- Avg blocked marks/sample: 0.0
- Multimodal samples: 5
- Multimodal resolved samples: 5
- Multimodal resolution rate: 1.0
- Vision extraction ok: 5
- Vision extraction failed: 0
- Numeric MAE: 1.915066
- Numeric eval samples: 5
- Numeric parse ambiguity count: 1

## Status Breakdown

- ok: 5

## Error Tag Breakdown

- correct: 3
- numeric_mismatch: 2

## Tool Usage


## Research Notes

- `timeout` / `runtime_error`: stability and serving-layer risks.
- `repeated_tool_loop` / `same_args_repeated`: workflow-control inefficiency.
- `prediction_parse_failure`: output contract mismatch (answer extraction risk).
- `multimodal_input_missing`: image path resolution gap or data mounting issue.
- `multimodal_vision_failure`: vision extraction call failed or timed out.
- `numeric_mismatch` / `code_mismatch` / `text_mismatch`: reasoning or grounding error.
- `numeric_mae` helps quantify magnitude, not only correctness rate.
