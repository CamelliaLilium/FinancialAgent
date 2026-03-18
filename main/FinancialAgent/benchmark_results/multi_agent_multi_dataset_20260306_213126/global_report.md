# Multi-Dataset Multi-Agent Benchmark Report

- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Planning enabled: `True`
- Dataset selection: `all`
- Overall accuracy: 0.6000
- Overall correct/total: 6/10

## Dataset Comparison

- `/root/autodl-tmp/datasets/test/bizbench_test.json`: acc=0.5000, correct=1/2, avg=192.605s, p95=197.345s, numeric_mae=None, mm_resolve_rate=None, mm_vision_failed=0
- `/root/autodl-tmp/datasets/test/finmmr_easy_test.json`: acc=1.0000, correct=2/2, avg=76.996s, p95=129.886s, numeric_mae=0.002165, mm_resolve_rate=1.0, mm_vision_failed=0
- `/root/autodl-tmp/datasets/test/finmmr_hard_test.json`: acc=0.0000, correct=0/2, avg=46.707s, p95=50.587s, numeric_mae=129.947, mm_resolve_rate=1.0, mm_vision_failed=0
- `/root/autodl-tmp/datasets/test/finmmr_medium_test.json`: acc=0.5000, correct=1/2, avg=42.67s, p95=45.937s, numeric_mae=0.170711, mm_resolve_rate=1.0, mm_vision_failed=0
- `/root/autodl-tmp/datasets/test/flare-convfinqa_test.json`: acc=1.0000, correct=2/2, avg=38.126s, p95=44.044s, numeric_mae=0.0, mm_resolve_rate=None, mm_vision_failed=0

## Scientific Guidance

- 先按数据集分别分析错因分布，避免被总体均值掩盖。
- 对同一数据集至少重复 3 次运行，报告均值与标准差。
- 对失败样本进行分层抽样（数值、代码、文本）做人工复核。
- 对高延迟样本追踪 tool 调用链，识别可裁剪步骤。
