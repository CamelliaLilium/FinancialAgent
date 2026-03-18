# Multi-Agent Dataset Report

- Dataset file: `/root/autodl-tmp/Open Manus /金融Agent架构/金融Agent/Dataset/flarez-confinqa_test/flare-convfinqa_test.json`
- LLM model: `Qwen/Qwen3-VL-8B-Instruct`
- Planning enabled: `True`
- Multimodal mode: `strict`
- Total samples: 20
- Correct: 4
- Accuracy: 0.2000
- Avg latency/sample (s): 303.906
- P50 latency/sample (s): 257.895
- P95 latency/sample (s): 600.021
- Multimodal samples: 0
- Multimodal resolved samples: 0
- Multimodal resolution rate: None
- Vision extraction ok: 0
- Vision extraction failed: 0
- Numeric MAE: 1061.387635
- Numeric eval samples: 17

## Status Breakdown

- ok: 17
- timeout: 3

## Error Tag Breakdown

- numeric_mismatch: 13
- correct: 4
- timeout: 3
- empty_output: 3

## Tool Usage


## Research Notes

- `timeout` / `runtime_error`: stability and serving-layer risks.
- `repeated_tool_loop` / `same_args_repeated`: workflow-control inefficiency.
- `prediction_parse_failure`: output contract mismatch (answer extraction risk).
- `multimodal_input_missing`: image path resolution gap or data mounting issue.
- `multimodal_vision_failure`: vision extraction call failed or timed out.
- `numeric_mismatch` / `code_mismatch` / `text_mismatch`: reasoning or grounding error.
- `numeric_mae` helps quantify magnitude, not only correctness rate.
