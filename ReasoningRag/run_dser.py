#!/usr/bin/env python3
"""
C1 — DSER: Deep Self-Evolving Reasoning with Qwen3-VL-8B (frozen).

Runs K parallel Markov chains per question. Within each chain, the model
generates an initial answer and then does up to M self-refinement iterations.
Final answer is chosen by majority vote across all K chains.

Reference: arXiv:2510.17498 (adapted for frozen inference-time use).

Usage:
    python run_dser.py --dry-run
    python run_dser.py --k 5 --refine-rounds 1
    python run_dser.py --k 5 --refine-rounds 2 --output-dir test/results/c1_dser
"""

import argparse
import base64
import io
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError as e:
    raise RuntimeError("pip install openai") from e

try:
    from PIL import Image
except ImportError:
    Image = None

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

SYSTEM_PROMPT = (
    "You are a financial reasoning expert. "
    "Solve the given problem step by step with careful numerical reasoning. "
    "End your response with exactly: **Final Answer:** <your answer>"
)

REFINE_PROMPT = (
    "Please carefully review your previous solution. "
    "Check each calculation step for errors, especially numerical precision. "
    "If you find a mistake, correct it. If your solution is correct, confirm it. "
    "End with exactly: **Final Answer:** <your answer>"
)

ANSWER_INSTRUCTIONS = {
    "numerical": "Output the final answer as a single number.",
    "mcq": "Output only the letter(s) of the correct option(s).",
    "boolean": "Output only Yes or No.",
    "free_text": "Output a concise answer.",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="C1: DSER — frozen Qwen3-VL-8B K parallel chains + self-refinement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--targets-path", type=Path,
                   default=ROOT / "data" / "ab_test" / "targets_finmmr_100.jsonl")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "test" / "results" / "c1_dser")
    p.add_argument("--model", default="qwen3-vl-8b-instruct")
    p.add_argument("--api-base", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p.add_argument("--k", type=int, default=5, help="parallel chains per question")
    p.add_argument("--refine-rounds", type=int, default=1,
                   help="self-refinement rounds per chain (0=no refinement)")
    p.add_argument("--temperature", type=float, default=0.6,
                   help="sampling temperature for initial generation")
    p.add_argument("--refine-temperature", type=float, default=0.3,
                   help="sampling temperature for refinement rounds")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-workers", type=int, default=5,
                   help="thread pool size for concurrent chains")
    p.add_argument("--image-max-side", type=int, default=768)
    p.add_argument("--image-jpeg-quality", type=int, default=85)
    p.add_argument("--request-retries", type=int, default=3)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


# ── I/O helpers ─────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ── Answer extraction / scoring ─────────────────────────────────────────────

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
    valid = [a for a in answers if a and not a.startswith("ERROR")]
    if not valid:
        return answers[0] if answers else ""
    return Counter(valid).most_common(1)[0][0]


# ── Image encoding ───────────────────────────────────────────────────────────

def encode_image(path: Path, max_side: int, quality: int) -> str:
    if Image is None or max_side <= 0:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
            path.suffix.lower().lstrip("."), "image/png"
        )
        return f"data:{mime};base64,{data}"
    with Image.open(path) as img:
        img = img.convert("RGBA")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side))
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img, mask=img.split()[-1])
        buf = io.BytesIO()
        bg.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_image_blocks(target: dict, args: argparse.Namespace) -> list[dict]:
    blocks = []
    for rel in target.get("image_paths", []):
        p = ROOT / rel
        if p.exists():
            blocks.append({"type": "image_url", "image_url": {
                "url": encode_image(p, args.image_max_side, args.image_jpeg_quality),
                "detail": "auto",
            }})
    return blocks


# ── Prompt builders ──────────────────────────────────────────────────────────

def build_initial_user_content(target: dict, args: argparse.Namespace) -> list[dict]:
    atype = target.get("answer_type", "")
    instr = ANSWER_INSTRUCTIONS.get(atype, "")
    dp = target.get("decimal_places")
    if atype == "numerical" and dp is not None:
        instr = f"Output the final answer as a number with exactly {dp} decimal places."

    text_parts = []
    if target.get("context"):
        text_parts.append(f"=== Context ===\n{target['context']}")
    if target.get("options"):
        text_parts.append(f"=== Options ===\n{target['options']}")
    text_parts.append(f"=== Question ===\n{target['question']}")
    if instr:
        text_parts.append(instr)
    text_parts.append("Think step by step. End with: **Final Answer:** <your answer>")

    content: list[dict] = [{"type": "text", "text": "\n\n".join(text_parts)}]
    content.extend(build_image_blocks(target, args))
    return content


# ── Single chain execution ───────────────────────────────────────────────────

def run_chain(
    client: OpenAI,
    target: dict,
    args: argparse.Namespace,
    chain_id: int,
) -> dict:
    """Run one Markov chain: initial generation + M refinement rounds."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": build_initial_user_content(target, args)})

    history: list[dict] = []  # [{response, answer}]
    error = ""

    # Initial generation
    for attempt in range(1, args.request_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=120,
            )
            text = resp.choices[0].message.content or ""
            answer = extract_final_answer(text)
            history.append({"response": text, "answer": answer, "round": 0})
            messages.append({"role": "assistant", "content": text})
            break
        except Exception as e:
            error = str(e)
            if attempt < args.request_retries:
                time.sleep(2 ** attempt)
            else:
                history.append({"response": f"ERROR: {e}", "answer": "", "round": 0})
                return {"chain_id": chain_id, "history": history, "final_answer": "", "error": error}

    # Self-refinement rounds
    for round_num in range(1, args.refine_rounds + 1):
        messages.append({"role": "user", "content": REFINE_PROMPT})
        for attempt in range(1, args.request_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.refine_temperature,
                    max_tokens=args.max_tokens,
                    timeout=120,
                )
                text = resp.choices[0].message.content or ""
                answer = extract_final_answer(text)
                history.append({"response": text, "answer": answer, "round": round_num})
                messages.append({"role": "assistant", "content": text})
                break
            except Exception as e:
                error = str(e)
                if attempt < args.request_retries:
                    time.sleep(2 ** attempt)
                else:
                    history.append({"response": f"ERROR: {e}", "answer": "", "round": round_num})

    final_answer = history[-1]["answer"] if history else ""
    return {"chain_id": chain_id, "history": history, "final_answer": final_answer, "error": error}


def run_target(client: OpenAI, target: dict, args: argparse.Namespace) -> dict:
    """Run K parallel chains and return aggregated result."""
    t0 = time.time()
    chains: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(run_chain, client, target, args, i): i
            for i in range(args.k)
        }
        for fut in as_completed(futures):
            chains.append(fut.result())

    chains.sort(key=lambda c: c["chain_id"])
    final_answers = [c["final_answer"] for c in chains]
    voted = majority_vote(final_answers)

    return {
        "target_id": target["id"],
        "gold_answer": target.get("gold_answer", ""),
        "answer_type": target.get("answer_type", ""),
        "chains": chains,
        "final_answers": final_answers,
        "voted_answer": voted,
        "correct": is_correct(target, voted),
        "k": args.k,
        "refine_rounds": args.refine_rounds,
        "latency_sec": round(time.time() - t0, 2),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    targets = read_jsonl(args.targets_path)
    if args.limit:
        targets = targets[:args.limit]

    results_path = args.output_dir / "results.jsonl"
    done_ids: set[str] = set()
    if args.resume and results_path.exists():
        done_ids = {r["target_id"] for r in read_jsonl(results_path)}
    pending = [t for t in targets if t["id"] not in done_ids]

    print(f"DSER C1 | model={args.model} K={args.k} M={args.refine_rounds}")
    print(f"Targets: {len(targets)}, pending: {len(pending)}")

    if args.dry_run:
        content = build_initial_user_content(targets[0], args)
        print(f"\n--- Dry-run for {targets[0]['id']} ---")
        for block in content:
            if block.get("type") == "text":
                print(block["text"])
        print(f"\nK={args.k} chains × (1 initial + {args.refine_rounds} refine) = "
              f"{args.k * (1 + args.refine_rounds)} total API calls per question")
        return

    api_key = os.getenv("QWEN_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set QWEN_API_KEY in .env")

    client = OpenAI(api_key=api_key, base_url=args.api_base)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if args.resume else "w"

    with open(results_path, open_mode, encoding="utf-8") as out:
        for i, target in enumerate(pending, 1):
            result = run_target(client, target, args)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            status = "✓" if result["correct"] else "✗"
            print(f"  [{i}/{len(pending)}] {target['id']} → {result['voted_answer']!r} {status} "
                  f"| answers={result['final_answers']} | {result['latency_sec']}s")

    all_rows = read_jsonl(results_path)
    accuracy = sum(1 for r in all_rows if r["correct"]) / max(len(all_rows), 1)
    summary = {
        "model": args.model,
        "k": args.k,
        "refine_rounds": args.refine_rounds,
        "temperature": args.temperature,
        "n": len(all_rows),
        "accuracy": round(accuracy, 4),
        "correct": sum(1 for r in all_rows if r["correct"]),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(f"\nDSER K={args.k} M={args.refine_rounds} accuracy: {accuracy:.4f} "
          f"({summary['correct']}/{summary['n']})")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
