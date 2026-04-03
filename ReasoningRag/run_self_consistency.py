#!/usr/bin/env python3
"""
B0 — Self-Consistency@K with Gemini teacher model.

For each target question, generates K independent trajectories at temperature>0,
then selects the answer by majority vote. Compares against single greedy inference
to validate whether multi-trajectory scaling helps.

Usage:
    python run_self_consistency.py --dry-run
    python run_self_consistency.py --k 5 --backend gemini
    python run_self_consistency.py --k 5 --output-dir test/results/sc5_gemini
"""

import argparse
import asyncio
import json
import mimetypes
import os
import re
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

SYSTEM_PROMPT = (
    "You are a financial reasoning expert. "
    "Solve the problem step by step. "
    "You MUST end with exactly: **Final Answer:** <your answer>"
)

ANSWER_INSTRUCTIONS = {
    "numerical": "Output the final answer as a single number.",
    "mcq": "Output only the letter(s) of the correct option(s).",
    "boolean": "Output only Yes or No.",
    "free_text": "Output a concise answer.",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="B0: Self-Consistency@K with Gemini teacher model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--targets-path", type=Path,
                   default=ROOT / "data" / "ab_test" / "targets_finmmr_100.jsonl")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "test" / "results" / "b0_self_consistency")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--k", type=int, default=5, help="trajectories per question")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--thinking-budget", type=int, default=1024)
    p.add_argument("--api-base-url", default="https://aihubmix.com/gemini")
    p.add_argument("--semaphore-limit", type=int, default=5)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict], mode: str = "w"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def extract_final_answer(text: str) -> str:
    m = re.search(r"\*\*Final Answer:\*\*\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().splitlines()[0].rstrip(".")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def parse_float(value: str) -> float | None:
    cleaned = (value or "").strip().replace(",", "").replace("$", "").replace("%", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    return float(m.group(0)) if m else None


def is_correct(target: dict, predicted: str) -> bool:
    gold = str(target.get("gold_answer", ""))
    atype = target.get("answer_type")
    if atype == "numerical":
        gn, pn = parse_float(gold), parse_float(predicted)
        return gn is not None and pn is not None and abs(gn - pn) <= 1e-9
    if atype == "mcq":
        return set(re.findall(r"[A-Z]", gold.upper())) == set(re.findall(r"[A-Z]", predicted.upper()))
    if atype == "boolean":
        def norm(v): return "yes" if v.strip().lower() in {"yes", "true"} else "no"
        return norm(gold) == norm(predicted)
    return gold.strip().lower() == predicted.strip().lower()


def majority_vote(answers: list[str]) -> str:
    """Return most common answer; ties broken by first occurrence."""
    if not answers:
        return ""
    counts = Counter(answers)
    return counts.most_common(1)[0][0]


def build_prompt(target: dict) -> str:
    parts = []
    if target.get("context"):
        parts.append(f"=== Context ===\n{target['context']}")
    if target.get("options"):
        parts.append(f"=== Options ===\n{target['options']}")
    parts.append(f"=== Question ===\n{target['question']}")
    atype = target.get("answer_type", "")
    instr = ANSWER_INSTRUCTIONS.get(atype, "")
    dp = target.get("decimal_places")
    if atype == "numerical" and dp is not None:
        instr = f"Output the final answer as a number with exactly {dp} decimal places."
    if instr:
        parts.append(instr)
    parts.append(
        "\nThink step by step. End with exactly:\n**Final Answer:** <your answer>"
    )
    return "\n\n".join(parts)


def load_image_part(rel_path: str) -> types.Part | None:
    full = ROOT / rel_path
    if not full.is_file():
        return None
    mime, _ = mimetypes.guess_type(str(full))
    return types.Part.from_bytes(data=full.read_bytes(), mime_type=mime or "image/png")


async def call_gemini_once(
    client: genai.Client,
    target: dict,
    config: types.GenerateContentConfig,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    backoff_base: float = 2.0,
) -> str:
    parts: list[types.Part] = []
    for img_rel in target.get("image_paths", []):
        part = load_image_part(img_rel)
        if part:
            parts.append(part)
    parts.append(types.Part.from_text(text=build_prompt(target)))
    contents = [types.Content(role="user", parts=parts)]

    for attempt in range(1, max_retries + 1):
        try:
            async with semaphore:
                resp = await client.aio.models.generate_content(
                    model=config._model, contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=config.temperature,
                        thinking_config=config.thinking_config,
                        response_mime_type="text/plain",
                    ),
                )
            return resp.text or ""
        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep(backoff_base ** attempt)
            else:
                return f"ERROR: {e}"
    return ""


async def process_target(
    client: genai.Client,
    target: dict,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    config_template,
) -> dict:
    target_id = target["id"]
    tasks = [
        call_gemini_once(client, target, config_template, semaphore, args.max_retries)
        for _ in range(args.k)
    ]
    trajectories = await asyncio.gather(*tasks)
    answers = [extract_final_answer(t) for t in trajectories]
    voted = majority_vote([a for a in answers if a and not a.startswith("ERROR")])
    return {
        "target_id": target_id,
        "gold_answer": target.get("gold_answer", ""),
        "answer_type": target.get("answer_type", ""),
        "trajectories": list(trajectories),
        "answers": answers,
        "voted_answer": voted,
        "correct": is_correct(target, voted),
        "k": args.k,
    }


async def run(args: argparse.Namespace):
    targets = read_jsonl(args.targets_path)
    if args.limit:
        targets = targets[:args.limit]

    results_path = args.output_dir / "results.jsonl"
    done_ids: set[str] = set()
    if args.resume and results_path.exists():
        done_ids = {r["target_id"] for r in read_jsonl(results_path)}
    pending = [t for t in targets if t["id"] not in done_ids]

    print(f"Targets: {len(targets)}, pending: {len(pending)}, K={args.k}, model={args.model}")

    if args.dry_run:
        print(f"\n--- Dry-run prompt for {targets[0]['id']} ---")
        print(build_prompt(targets[0]))
        return

    api_key = os.getenv("AIHUBMIX_API_KEY", "")
    if not api_key or api_key.startswith("sk-REPLACE"):
        raise RuntimeError("Set AIHUBMIX_API_KEY in .env")

    client = genai.Client(api_key=api_key, http_options={"base_url": args.api_base_url})

    # Store model name on config (workaround: use a simple namespace)
    class _Cfg:
        _model = args.model
        temperature = args.temperature
        thinking_config = types.ThinkingConfig(thinking_budget=args.thinking_budget)

    semaphore = asyncio.Semaphore(args.semaphore_limit)
    config_template = _Cfg()

    all_results: list[dict] = []
    open_mode = "a" if args.resume else "w"
    with open(results_path, open_mode, encoding="utf-8") as out:
        for i, target in enumerate(pending, 1):
            t0 = time.time()
            result = await process_target(client, target, args, semaphore, config_template)
            result["latency_sec"] = round(time.time() - t0, 2)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            all_results.append(result)
            status = "✓" if result["correct"] else "✗"
            print(f"  [{i}/{len(pending)}] {target['id']} → {result['voted_answer']!r} {status} "
                  f"| answers={result['answers']} | {result['latency_sec']}s")

    # Summarize
    all_rows = read_jsonl(results_path)
    accuracy = sum(1 for r in all_rows if r["correct"]) / max(len(all_rows), 1)
    summary = {
        "model": args.model,
        "k": args.k,
        "temperature": args.temperature,
        "n": len(all_rows),
        "accuracy": round(accuracy, 4),
        "correct": sum(1 for r in all_rows if r["correct"]),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(f"\nSC@{args.k} accuracy: {accuracy:.4f} ({summary['correct']}/{summary['n']})")
    print(f"Results: {results_path}")


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
