#!/usr/bin/env python3
"""
Step 1 - Generate CoT trajectories via Gemini or Qwen.

Usage:
    python step1_generate_trajectories.py                 # full run with defaults
    python step1_generate_trajectories.py --teacher qwen # use qwen3-vl-8b-instruct
    python step1_generate_trajectories.py --limit 5      # test with first N pending cases
    python step1_generate_trajectories.py --dry-run      # print prompt for first case, no API call
    python step1_generate_trajectories.py -h             # show all options
"""

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

GEMINI_MODEL = "gemini-2.5-flash"
QWEN_MODEL = "qwen3-vl-8b-instruct"
QWEN_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ---------------------------------------------------------------------------
# Argument parser — every tuneable parameter lives here
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate CoT trajectories via Gemini or Qwen",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── I/O paths ──
    p.add_argument("--data-dir", type=Path, default=ROOT / "data",
                   help="directory containing cases.jsonl")
    p.add_argument("--cases-file", default="cases.jsonl",
                   help="input filename inside --data-dir")
    p.add_argument("--trajectories-file", default="trajectories.jsonl",
                   help="output filename inside --data-dir (append mode)")

    # ── Teacher & generation ──
    p.add_argument("--teacher", choices=["gemini", "qwen"], default="gemini",
                   help="teacher model provider")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="sampling temperature")
    p.add_argument("--thinking-budget", type=int, default=2048,
                   help="thinking token budget for Gemini thinking mode only")

    # ── API ──
    p.add_argument("--api-base-url", default="https://aihubmix.com/gemini",
                   help="Gemini-compatible API base URL (used only when --teacher gemini)")

    # ── Concurrency & retry ──
    p.add_argument("--semaphore-limit", type=int, default=10,
                   help="max concurrent API requests")
    p.add_argument("--max-retries", type=int, default=3,
                   help="max retry attempts per case on transient errors")
    p.add_argument("--backoff-base", type=float, default=2.0,
                   help="exponential backoff base in seconds")

    # ── Run control ──
    p.add_argument("--limit", type=int, default=0,
                   help="process only first N pending cases (0 = all)")
    p.add_argument("--dry-run", action="store_true",
                   help="print prompt for first case, no API call")
    p.add_argument("--progress-interval", type=int, default=50,
                   help="print progress every N completed cases")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Prompt constants (not worth CLI-ifying, but easy to find here)
# ---------------------------------------------------------------------------

SYSTEM_PREFIX = "You are a financial reasoning expert. Solve the following problem step by step.\n\n"

ANSWER_INSTRUCTIONS = {
    "numerical": "Output the final answer as a single number.",
    "mcq": "Output only the letter(s) of the correct option(s).",
    "boolean": "Output only Yes or No.",
    "free_text": "Output a concise answer.",
}

SUFFIX = (
    "\nFinal Answer must follow the precision requirement stated in the question exactly. "
    "If no precision is specified, match the number of decimal places in standard financial reporting.\n\n"
    "Think step by step. You MUST end your response with exactly this line:\n"
    "**Final Answer:** <your answer>\n"
    "Do not use LaTeX, boxed notation, or any other format for the final answer line."
)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(case: dict) -> str:
    parts: list[str] = [SYSTEM_PREFIX]
    if case.get("context"):
        parts.append(f"=== Context ===\n{case['context']}\n")
    if case.get("options"):
        parts.append(f"=== Options ===\n{case['options']}\n")
    parts.append(f"=== Question ===\n{case['question']}\n")
    at = case.get("answer_type", "")
    instr = ANSWER_INSTRUCTIONS.get(at, "")
    dp = case.get("decimal_places")
    if at == "numerical" and dp is not None:
        instr = f"Output the final answer as a number with exactly {dp} decimal places."
    if instr:
        parts.append(instr)
    parts.append(SUFFIX)
    return "\n".join(parts)


def extract_predicted(text: str) -> str:
    m = re.search(r"\*\*Final Answer:\*\*\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().rstrip(".").split("\n")[0]
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    return m.group(1).strip().rstrip(".") if m else ""


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image_part(rel_path: str) -> types.Part | None:
    full = ROOT / rel_path
    if not full.is_file():
        return None
    mime, _ = mimetypes.guess_type(str(full))
    with open(full, "rb") as f:
        data = f.read()
    return types.Part.from_bytes(data=data, mime_type=mime or "image/png")


def encode_image_data_url(rel_path: str) -> str | None:
    full = ROOT / rel_path
    if not full.is_file():
        return None
    mime, _ = mimetypes.guess_type(str(full))
    with open(full, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime or 'image/png'};base64,{data}"


def build_qwen_user_content(case: dict) -> list[dict]:
    content = [{"type": "text", "text": build_prompt(case)}]
    for img_rel in case.get("image_paths", []):
        data_url = encode_image_data_url(img_rel)
        if data_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": data_url, "detail": "auto"},
            })
    return content


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_completed_ids(traj_path: Path) -> set[str]:
    done: set[str] = set()
    if traj_path.exists():
        with open(traj_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


# ---------------------------------------------------------------------------
# Core async worker
# ---------------------------------------------------------------------------

async def process_one(
    client,
    case: dict,
    config,
    file_lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
    outfile,
    progress: dict,
    args: argparse.Namespace,
):
    case_id = case["id"]

    contents = None
    if args.teacher == "gemini":
        content_parts: list[types.Part] = []
        for img_rel in case.get("image_paths", []):
            part = load_image_part(img_rel)
            if part:
                content_parts.append(part)
        content_parts.append(types.Part.from_text(text=build_prompt(case)))
        contents = [types.Content(role="user", parts=content_parts)]

    trajectory = ""
    predicted = ""
    error_msg = ""

    for attempt in range(1, args.max_retries + 1):
        try:
            async with semaphore:
                if args.teacher == "gemini":
                    response = await client.aio.models.generate_content(
                        model=GEMINI_MODEL, contents=contents, config=config,
                    )
                    trajectory = response.text or ""
                else:
                    trajectory = await asyncio.to_thread(call_qwen_generate, client, case, args)
            predicted = extract_predicted(trajectory)
            break
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            if attempt < args.max_retries:
                await asyncio.sleep(args.backoff_base ** attempt)
            else:
                trajectory = f"ERROR: {error_msg}"
                predicted = ""

    record = {
        "id": case_id,
        "trajectory": trajectory,
        "predicted": predicted,
        "answer_type": case.get("answer_type", ""),
        "decimal_places": case.get("decimal_places"),
    }
    async with file_lock:
        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
        outfile.flush()

    progress["done"] += 1
    done, total = progress["done"], progress["total"]
    status = "OK" if not error_msg else f"ERR({error_msg[:60]})"
    if done % args.progress_interval == 0 or done == total or error_msg:
        elapsed = time.time() - progress["start"]
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  [{done}/{total}] {case_id} → {status} "
              f"| predicted={predicted[:40]!r} "
              f"| {rate:.1f} cases/s, ETA {eta / 60:.0f}m")


def call_qwen_generate(client, case: dict, args: argparse.Namespace) -> str:
    if OpenAI is None:
        raise RuntimeError("Missing dependency `openai`. Install it with `pip install openai`.")

    completion = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": build_qwen_user_content(case)}],
        temperature=args.temperature,
        max_tokens=2048,
        timeout=600,
    )
    message = completion.choices[0].message.content
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts = []
        for item in message:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts).strip()
    return str(message)


def build_client_and_config(args: argparse.Namespace):
    if args.teacher == "gemini":
        api_key = os.getenv("AIHUBMIX_API_KEY", "")
        if not api_key or api_key.startswith("sk-REPLACE"):
            print("ERROR: Set your real API key in .env (AIHUBMIX_API_KEY)")
            sys.exit(1)

        client = genai.Client(
            api_key=api_key,
            http_options={"base_url": args.api_base_url},
        )
        config = types.GenerateContentConfig(
            temperature=args.temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=args.thinking_budget),
            response_mime_type="text/plain",
        )
        return client, config

    if OpenAI is None:
        print("ERROR: Missing dependency `openai`. Install it with `pip install openai`")
        sys.exit(1)

    api_key = os.getenv("QWEN_API_KEY", "")
    if not api_key:
        print("ERROR: Set your real API key in .env (QWEN_API_KEY)")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=QWEN_API_BASE_URL)
    return client, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace):
    cases_path = args.data_dir / args.cases_file
    traj_path = args.data_dir / args.trajectories_file

    with open(cases_path, encoding="utf-8") as f:
        all_cases = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(all_cases)} cases from {cases_path}")

    done_ids = load_completed_ids(traj_path)
    pending = [c for c in all_cases if c["id"] not in done_ids]
    print(f"Already completed: {len(done_ids)}, pending: {len(pending)}")

    if args.limit:
        pending = pending[:args.limit]
        print(f"Limited to first {args.limit} pending cases")

    if args.dry_run:
        if pending:
            c = pending[0]
            print(f"\n{'=' * 60}\nDry-run prompt for: {c['id']}\n{'=' * 60}")
            print(f"Image parts: {len(c.get('image_paths', []))}")
            print(build_prompt(c))
        return

    client, config = build_client_and_config(args)

    if not pending:
        print("Nothing to do — all cases completed.")
        return

    semaphore = asyncio.Semaphore(args.semaphore_limit)
    file_lock = asyncio.Lock()
    progress = {"done": 0, "total": len(pending), "start": time.time()}
    print(f"\nStarting generation: {len(pending)} cases, teacher={args.teacher}, semaphore={args.semaphore_limit}")

    outfile = open(traj_path, "a", encoding="utf-8")
    try:
        tasks = [process_one(client, case, config, file_lock, semaphore, outfile, progress, args)
                 for case in pending]
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        outfile.close()

    elapsed = time.time() - progress["start"]
    print(f"\nDone. {progress['done']}/{progress['total']} cases in {elapsed / 60:.1f} min")

    error_count = empty_pred = 0
    with open(traj_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["trajectory"].startswith("ERROR:"):
                error_count += 1
            if not rec["predicted"]:
                empty_pred += 1
    print(f"Errors: {error_count}, Empty predicted: {empty_pred}")


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
