# Multi-Agent Dataset Report

- Dataset file: `/root/autodl-tmp/datasets/test/finmmr_hard_test.json`
- LLM model: `Qwen/Qwen3-VL-8B-Instruct`
- Planning enabled: `True`
- Multimodal mode: `best_effort`
- Total samples: 5
- Correct: 1
- Accuracy: 0.2000
- Avg latency/sample (s): 106.879
- P50 latency/sample (s): 52.55
- P95 latency/sample (s): 258.896
- Avg LLM calls/sample: 8.8
- Avg zero-tool rounds/sample: 3.4
- Avg blocked marks/sample: 0.6
- Multimodal samples: 5
- Multimodal resolved samples: 5
- Multimodal resolution rate: 1.0
- Vision extraction ok: 5
- Vision extraction failed: 0
- Numeric MAE: 829.0528
- Numeric eval samples: 5
- Numeric parse ambiguity count: 2

## Status Breakdown

- ok: 5

## Error Tag Breakdown

- numeric_mismatch: 4
- correct: 1

## Tool Usage


## Research Notes

- `timeout` / `runtime_error`: stability and serving-layer risks.
- `repeated_tool_loop` / `same_args_repeated`: workflow-control inefficiency.
- `prediction_parse_failure`: output contract mismatch (answer extraction risk).
- `multimodal_input_missing`: image path resolution gap or data mounting issue.
- `multimodal_vision_failure`: vision extraction call failed or timed out.
- `numeric_mismatch` / `code_mismatch` / `text_mismatch`: reasoning or grounding error.
- `numeric_mae` helps quantify magnitude, not only correctness rate.
