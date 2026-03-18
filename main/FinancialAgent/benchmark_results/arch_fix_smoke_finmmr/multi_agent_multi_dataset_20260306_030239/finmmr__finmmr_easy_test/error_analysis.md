# Multi-Agent Dataset Report

- Dataset file: `/root/autodl-tmp/Open Manus /金融Agent架构/金融Agent/Dataset/finmmr/finmmr_easy_test.json`
- LLM model: `Qwen/Qwen3-VL-8B-Instruct`
- Planning enabled: `True`
- Multimodal mode: `strict`
- Total samples: 20
- Correct: 9
- Accuracy: 0.4500
- Avg latency/sample (s): 78.997
- P50 latency/sample (s): 43.947
- P95 latency/sample (s): 290.478
- Avg LLM calls/sample: 7.1
- Avg zero-tool rounds/sample: 2.65
- Avg blocked marks/sample: 0.05
- Multimodal samples: 19
- Multimodal resolved samples: 19
- Multimodal resolution rate: 1.0
- Vision extraction ok: 17
- Vision extraction failed: 2
- Numeric MAE: 715.833549
- Numeric eval samples: 17
- Numeric parse ambiguity count: 6

## Status Breakdown

- ok: 18
- runtime_error: 2

## Error Tag Breakdown

- correct: 9
- numeric_mismatch: 8
- runtime_error: 2
- empty_output: 2
- multimodal_vision_failure: 2
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
