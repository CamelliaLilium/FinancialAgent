#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp

# ── RAG Pipeline ──
python -m rag.processor.solution_generator --discover --top-k 3
python -m rag.processor.inference --batch-test --top-k 3
python -m rag.processor.generate_pilot_report \
    --solutions enhanced_solutions_20260313_102057.json \
    --inference inference_results_20260313.json --top-k 3
python -m rag.processor.test_example_tokens
python -m rag.processor.test_two_phase

# ── ReasoningRag Pipeline ──
cd /root/autodl-tmp/ReasoningRag

# Step 0: preprocess (all args have sensible defaults, override as needed)
python step0_preprocess.py
# python step0_preprocess.py --skip-images --raw-dir /other/raw

# Step 1: generate trajectories (test 3 first, then full run)
python step1_generate_trajectories.py --limit 3
# 打印第一条 pending case 的完整 prompt
python step1_generate_trajectories.py --dry-run
# step1 run full
python step1_generate_trajectories.py --model gemini-2.5-flash --temperature 0.7
# Step 1 Test
python run_qwen_ab_test.py --backend openai

python ReasoningRag/step2_judge.py --data-dir ReasoningRag/data --resume
python ReasoningRag/step3_extract_memories.py --data-dir ReasoningRag/data --resume
python ReasoningRag/step4_build_index.py --data-dir ReasoningRag/data --embed-model-path ReasoningRag/NV-Embed-v2

python ReasoningRag/step2_judge.py --data-dir ReasoningRag/data --limit 50
python ReasoningRag/step3_extract_memories.py --data-dir ReasoningRag/data --resume
python ReasoningRag/step4_build_index.py --data-dir ReasoningRag/data --embed-model-path ReasoningRag/NV-Embed-v2

