#!/usr/bin/env bash
set -euo pipefail

python step0_preprocess.py
python step1_generate_trajectories.py --teacher gemini --temperature 0.7
python step2_judge.py --data-dir data --resume
python step3_extract_memories.py --data-dir data --resume
python step4_build_index.py --data-dir data --embed-model-path NV-Embed-v2
