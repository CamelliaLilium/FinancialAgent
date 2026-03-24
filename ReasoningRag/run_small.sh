#!/usr/bin/env bash
set -euo pipefail

python step1_generate_trajectories.py --limit 10 --teacher qwen
python step2_judge.py --data-dir data --limit 10 --resume
python step3_extract_memories.py --data-dir data --limit 10 --resume
python step4_build_index.py --data-dir data --embed-backend hash

python retrieve_top1.py \
  --data-dir data \
  --query "What is the net profit margin?" \
  --answer-type numerical \
  --calc-type ratio
