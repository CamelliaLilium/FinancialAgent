# Multi-Agent Dataset Report

- Dataset file: `/root/autodl-tmp/Open Manus /金融Agent架构/金融Agent/Dataset/finmmr/finmmr_hard_test.json`
- LLM model: `Qwen/Qwen3-VL-8B-Instruct`
- Planning enabled: `True`
- Multimodal mode: `strict`
- Total samples: 20
- Correct: 3
- Accuracy: 0.1500
- Avg latency/sample (s): 64.624
- P50 latency/sample (s): 33.868
- P95 latency/sample (s): 190.41
- Avg LLM calls/sample: 6.7
- Avg zero-tool rounds/sample: 2.55
- Avg blocked marks/sample: 0.3
- Multimodal samples: 20
- Multimodal resolved samples: 20
- Multimodal resolution rate: 1.0
- Vision extraction ok: 16
- Vision extraction failed: 4
- Numeric MAE: 309970988.337334
- Numeric eval samples: 14
- Numeric parse ambiguity count: 9

## Status Breakdown

- ok: 16
- runtime_error: 4

## Error Tag Breakdown

- numeric_mismatch: 11
- runtime_error: 4
- empty_output: 4
- multimodal_vision_failure: 4
- correct: 3
- prediction_parse_failure: 2

## Tool Usage


## Research Notes

- `timeout` / `runtime_error`: stability and serving-layer risks.
- `repeated_tool_loop` / `same_args_repeated`: workflow-control inefficiency.
- `prediction_parse_failure`: output contract mismatch (answer extraction risk).
- `multimodal_input_missing`: image path resolution gap or data mounting issue.
- `multimodal_vision_failure`: vision extraction call failed or timed out.
- `numeric_mismatch` / `code_mismatch` / `text_mismatch`: reasoning or grounding error.
- `numeric_mae` helps quantify magnitude, not only correctness rate.
