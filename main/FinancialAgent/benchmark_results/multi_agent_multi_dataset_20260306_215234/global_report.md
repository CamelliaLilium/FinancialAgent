# Multi-Dataset Multi-Agent Benchmark Report

- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Planning enabled: `True`
- Dataset selection: `all`
- Overall accuracy: 0.4800
- Overall correct/total: 12/25

## Dataset Comparison

- `/root/autodl-tmp/datasets/test/bizbench_test.json`: acc=0.4000, correct=2/5, avg=258.218s, p95=332.021s, numeric_mae=None, mm_resolve_rate=None, mm_vision_failed=0
- `/root/autodl-tmp/datasets/test/finmmr_easy_test.json`: acc=0.6000, correct=3/5, avg=42.83s, p95=85.331s, numeric_mae=1.915066, mm_resolve_rate=1.0, mm_vision_failed=0
- `/root/autodl-tmp/datasets/test/finmmr_hard_test.json`: acc=0.2000, correct=1/5, avg=106.879s, p95=258.896s, numeric_mae=829.0528, mm_resolve_rate=1.0, mm_vision_failed=0
- `/root/autodl-tmp/datasets/test/finmmr_medium_test.json`: acc=0.4000, correct=2/5, avg=92.687s, p95=183.985s, numeric_mae=798.776457, mm_resolve_rate=1.0, mm_vision_failed=0
- `/root/autodl-tmp/datasets/test/flare-convfinqa_test.json`: acc=0.8000, correct=4/5, avg=93.12s, p95=271.247s, numeric_mae=400.715194, mm_resolve_rate=None, mm_vision_failed=0

## Scientific Guidance

- 先按数据集分别分析错因分布，避免被总体均值掩盖。
- 对同一数据集至少重复 3 次运行，报告均值与标准差。
- 对失败样本进行分层抽样（数值、代码、文本）做人工复核。
- 对高延迟样本追踪 tool 调用链，识别可裁剪步骤。
