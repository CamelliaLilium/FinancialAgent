# Multi-Agent Dataset Report

- Dataset file: `/root/autodl-tmp/Open Manus /金融Agent架构/金融Agent/Dataset/finmmr/finmmr_medium_test.json`
- LLM model: `Qwen/Qwen3-VL-8B-Instruct`
- Planning enabled: `True`
- Multimodal mode: `strict`
- Total samples: 20
- Correct: 4
- Accuracy: 0.2000
- Avg latency/sample (s): 279.973
- P50 latency/sample (s): 238.133
- P95 latency/sample (s): 600.033
- Multimodal samples: 19
- Multimodal resolved samples: 19
- Multimodal resolution rate: 1.0
- Vision extraction ok: 19
- Vision extraction failed: 0
- Numeric MAE: 705.583884
- Numeric eval samples: 17

## Status Breakdown

- ok: 18
- timeout: 2

## Error Tag Breakdown

- numeric_mismatch: 13
- correct: 4
- timeout: 2
- empty_output: 2
- prediction_parse_failure: 1

## Tool Usage


## Research Notes

- `timeout` / `runtime_error`: stability and serving-layer risks.
- `repeated_tool_loop` / `same_args_repeated`: workflow-control inefficiency.
- `prediction_parse_failure`: output contract mismatch (answer extraction risk).
- `multimodal_input_missing`: image path resolution gap or data mounting issue.
- `multimodal_vision_failure`: vision extraction call failed or timed out.
- `numeric_mismatch` / `code_mismatch` / `text_mismatch`: reasoning or grounding error.
- `numeric_mae` helps quantify magnitude, not only correctness rate.
