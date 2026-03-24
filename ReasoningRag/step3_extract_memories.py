#!/usr/bin/env python3
"""
Step 3 - Extract memory items from judged trajectories.

Usage:
    python step3_extract_memories.py
    python step3_extract_memories.py --limit 100
    python step3_extract_memories.py --resume
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

FAILURE_TYPES = [
    "读数错误",
    "字段混淆",
    "公式错误",
    "单位混乱",
    "预计算值误用",
    "数据遗漏",
    "范围界定错误",
    "其他",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract memory items from success/failure trajectories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--cases-file", default="cases.jsonl")
    p.add_argument("--trajectories-file", default="trajectories.jsonl")
    p.add_argument("--labels-file", default="labels.jsonl")
    p.add_argument("--items-file", default="items.jsonl")
    p.add_argument("--progress-file", default="extract_progress.jsonl")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--api-base-url", default="https://aihubmix.com/gemini")
    p.add_argument("--api-key-env", default="AIHUBMIX_API_KEY")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--backoff-base", type=float, default=2.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--progress-interval", type=int, default=100)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_done_source_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for row in read_jsonl(path):
        sid = row.get("source_question_id")
        if sid and row.get("status") in {"done", "empty", "error"}:
            done.add(sid)
    return done


def write_progress_record(handle, source_question_id: str, status: str, item_count: int, note: str = "") -> None:
    record = {
        "source_question_id": source_question_id,
        "status": status,
        "item_count": item_count,
        "note": note[:300],
    }
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def trim_text(text: str, limit: int) -> str:
    t = (text or "").strip()
    return t if len(t) <= limit else t[:limit].rstrip() + "..."


def build_common_context(case: dict, traj: dict) -> str:
    parts = [
        f"Question: {case.get('question', '')}",
    ]
    if case.get("context"):
        parts.append(f"Context: {trim_text(str(case['context']), 3000)}")
    if case.get("options"):
        parts.append(f"Options: {case['options']}")

    parts.extend(
        [
            f"Gold Answer: {case.get('gold_answer', '')}",
            f"Predicted Answer: {traj.get('predicted', '')}",
            "Trajectory:",
            trim_text(str(traj.get("trajectory", "")), 8000),
        ]
    )
    return "\n".join(parts)


def build_success_prompt(case: dict, traj: dict) -> str:
    return (
        "You are an expert in financial reasoning memory extraction.\n"
        "Given a SUCCESSFUL trajectory, extract transferable strategy memories.\n\n"
        "Rules:\n"
        "1) Extract at most 3 memory items.\n"
        "2) Do not output overlapping or repetitive items.\n"
        "3) Do not mention specific company names, years, exact numbers, or dataset IDs.\n"
        "4) Keep each content actionable and concise (1-3 sentences).\n\n"
        "Output must use this exact markdown schema for each item:\n"
        "# Memory Item i\n"
        "## Title <title>\n"
        "## Description <one sentence>\n"
        "## Content <1-3 sentences>\n\n"
        f"{build_common_context(case, traj)}"
    )


def build_failure_prompt(case: dict, traj: dict) -> str:
    taxonomy = " | ".join(FAILURE_TYPES)
    return (
        "You are an expert in financial reasoning memory extraction.\n"
        "Given a FAILED trajectory, extract preventive warning memories.\n\n"
        "Rules:\n"
        "1) Extract at most 3 memory items.\n"
        "2) Do not output overlapping or repetitive items.\n"
        "3) Do not mention specific company names, years, exact numbers, or dataset IDs.\n"
        "4) Keep each content actionable and concise (1-3 sentences).\n"
        f"5) For each item, choose exactly one FailureType from: {taxonomy}.\n\n"
        "Output must use this exact markdown schema for each item:\n"
        "# Memory Item i\n"
        "## Title <title>\n"
        "## Description <one sentence>\n"
        "## Content <1-3 sentences>\n"
        "## FailureType <one label from taxonomy>\n\n"
        f"{build_common_context(case, traj)}"
    )


def extract_line_value(block: str, key: str) -> str:
    m = re.search(rf"(?im)^##\s*{re.escape(key)}\s*(.*)$", block)
    return m.group(1).strip() if m else ""


def extract_content_value(block: str) -> str:
    m = re.search(r"(?ims)^##\s*Content\s*(.+?)(?=^##\s*\w+|\Z)", block)
    if not m:
        return ""
    text = m.group(1).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def normalize_failure_type(raw: str, is_failure: bool) -> str | None:
    if not is_failure:
        return None
    value = (raw or "").strip()
    if value in FAILURE_TYPES:
        return value
    for t in FAILURE_TYPES:
        if t in value:
            return t
    return "其他"


def parse_memory_markdown(text: str, is_failure: bool) -> list[dict]:
    if not text.strip():
        return []

    blocks = re.split(r"(?im)^#\s*Memory Item\b.*$", text)
    parsed: list[dict] = []
    for block in blocks[1:]:
        title = extract_line_value(block, "Title")
        description = extract_line_value(block, "Description")
        content = extract_content_value(block)
        failure_raw = extract_line_value(block, "FailureType")
        failure_type = normalize_failure_type(failure_raw, is_failure)

        if not title or not description or not content:
            continue

        parsed.append(
            {
                "title": title[:120],
                "description": description[:240],
                "content": content[:1200],
                "failure_type": failure_type,
            }
        )

    return parsed[:3]


def call_extractor(
    client: genai.Client,
    model: str,
    prompt: str,
    temperature: float,
    max_retries: int,
    backoff_base: float,
) -> str:
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="text/plain",
    )

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            return resp.text or ""
        except Exception as e:  # pylint: disable=broad-except
            last_error = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                time.sleep(backoff_base ** attempt)

    return f"ERROR: {last_error}"


def make_item_id(source_question_id: str, idx: int) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", source_question_id)
    return f"mem_{clean}_{idx:02d}"


def main() -> None:
    args = parse_args()
    cases_path = args.data_dir / args.cases_file
    traj_path = args.data_dir / args.trajectories_file
    labels_path = args.data_dir / args.labels_file
    items_path = args.data_dir / args.items_file
    progress_path = args.data_dir / args.progress_file

    cases = {row["id"]: row for row in read_jsonl(cases_path)}
    trajectories = {row["id"]: row for row in read_jsonl(traj_path)}
    labels = read_jsonl(labels_path)

    done_ids = load_done_source_ids(progress_path) if args.resume else set()
    pending = [row for row in labels if row.get("id") in cases and row.get("id") in trajectories and row.get("id") not in done_ids]
    if args.limit:
        pending = pending[:args.limit]

    print(f"Loaded cases={len(cases)}, trajectories={len(trajectories)}, labels={len(labels)}, pending={len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {args.api_key_env}")

    client = genai.Client(
        api_key=api_key,
        http_options={"base_url": args.api_base_url},
    )

    open_mode = "a" if args.resume else "w"
    total_items = 0
    empty_cases = 0
    error_cases = 0
    started = time.time()

    with open(items_path, open_mode, encoding="utf-8") as out, open(progress_path, open_mode, encoding="utf-8") as progress_out:
        for i, label in enumerate(pending, start=1):
            cid = label["id"]
            case = cases[cid]
            traj = trajectories[cid]
            status = (label.get("status") or "failure").strip().lower()
            is_failure = status != "success"

            prompt = build_failure_prompt(case, traj) if is_failure else build_success_prompt(case, traj)
            raw = call_extractor(
                client=client,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                max_retries=args.max_retries,
                backoff_base=args.backoff_base,
            )

            had_error = raw.startswith("ERROR:")
            if had_error:
                parsed = []
            else:
                parsed = parse_memory_markdown(raw, is_failure=is_failure)

            if not parsed:
                empty_cases += 1
                if had_error:
                    error_cases += 1

            calc_type = case.get("calc_type")
            if not isinstance(calc_type, list):
                calc_type = []

            answer_type = case.get("answer_type", "")

            for idx, item in enumerate(parsed, start=1):
                record = {
                    "item_id": make_item_id(cid, idx),
                    "source_question_id": cid,
                    "source": "failure" if is_failure else "success",
                    "memory_type": "warning" if is_failure else "strategy",
                    "title": item["title"],
                    "description": item["description"],
                    "content": item["content"],
                    "failure_type": item["failure_type"],
                    "calc_type": calc_type,
                    "answer_type": answer_type,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_items += 1

            out.flush()

            progress_status = "done" if parsed else ("error" if had_error else "empty")
            progress_note = raw[:300] if had_error else ""
            write_progress_record(progress_out, cid, progress_status, len(parsed), progress_note)

            if i % args.progress_interval == 0 or i == len(pending):
                elapsed = time.time() - started
                rate = i / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{i}/{len(pending)}] items={total_items}, empty_cases={empty_cases}, error_cases={error_cases}, rate={rate:.1f}/s"
                )

    print("Done.")
    print(f"Output: {items_path}")
    print(f"Progress: {progress_path}")
    print(f"cases={len(pending)}, items={total_items}, empty_cases={empty_cases}, error_cases={error_cases}")


if __name__ == "__main__":
    main()
