#!/usr/bin/env python3
"""
Step 2 - Judge trajectories as success/failure.

Usage:
    python step2_judge.py
    python step2_judge.py --limit 100
    python step2_judge.py --resume
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import types


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Judge trajectories with rules + LLM for free_text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--cases-file", default="cases.jsonl")
    p.add_argument("--trajectories-file", default="trajectories.jsonl")
    p.add_argument("--labels-file", default="labels.jsonl")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--api-base-url", default="https://aihubmix.com/gemini")
    p.add_argument("--api-key-env", default="AIHUBMIX_API_KEY")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--backoff-base", type=float, default=2.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--progress-interval", type=int, default=200)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_completed_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for row in read_jsonl(path):
        rid = row.get("id")
        if rid:
            done.add(rid)
    return done


def parse_float(value: str) -> float | None:
    text = (value or "").strip().replace(",", "")
    text = text.replace("$", "").replace("%", "")
    text = text.replace("USD", "").replace("TWD", "")
    text = text.replace("usd", "").replace("twd", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def normalize_mcq(value: str) -> set[str]:
    return set(re.findall(r"[A-Z]", (value or "").upper()))


def normalize_boolean(value: str) -> str:
    lowered = (value or "").strip().lower()
    if lowered in {"yes", "true"}:
        return "yes"
    if lowered in {"no", "false"}:
        return "no"
    return lowered


def judge_rule(case: dict, predicted: str) -> tuple[str, str]:
    answer_type = case.get("answer_type")
    gold = str(case.get("gold_answer", ""))

    if not predicted.strip():
        return "failure", "empty predicted"

    if answer_type == "numerical":
        gold_num = parse_float(gold)
        pred_num = parse_float(predicted)
        if gold_num is None or pred_num is None:
            return "failure", "numerical parse failed"

        tol = None
        metadata = case.get("metadata", {})
        if isinstance(metadata, dict):
            tol = metadata.get("tolerance")

        if tol is not None:
            try:
                tolerance = float(tol)
            except (TypeError, ValueError):
                tolerance = 1e-9
        else:
            tolerance = 1e-9

        ok = abs(pred_num - gold_num) <= tolerance
        reason = f"|pred-gold|={abs(pred_num - gold_num):.6g}, tol={tolerance:.6g}"
        return ("success" if ok else "failure"), reason

    if answer_type == "mcq":
        g = normalize_mcq(gold)
        p = normalize_mcq(predicted)
        ok = g == p
        return ("success" if ok else "failure"), f"gold={sorted(g)}, pred={sorted(p)}"

    if answer_type == "boolean":
        g = normalize_boolean(gold)
        p = normalize_boolean(predicted)
        ok = g == p
        return ("success" if ok else "failure"), f"gold={g}, pred={p}"

    return "unknown", "rule not applicable"


def build_free_text_prompt(case: dict, traj: dict) -> str:
    parts = [
        "You are an expert evaluator for financial QA answers.",
        "Judge whether the predicted answer is semantically equivalent to the gold answer.",
        "Return exactly two lines:",
        "Reason: <short reason>",
        "Status: success OR failure",
        "",
        f"Question: {case.get('question', '')}",
    ]
    if case.get("context"):
        parts.append(f"Context: {case['context']}")
    if case.get("options"):
        parts.append(f"Options: {case['options']}")
    parts.extend([
        f"Gold Answer: {case.get('gold_answer', '')}",
        f"Predicted Answer: {traj.get('predicted', '')}",
        "Trajectory:",
        str(traj.get("trajectory", ""))[:6000],
    ])
    return "\n".join(parts)


def parse_judge_output(text: str) -> tuple[str, str]:
    status_match = re.search(r"status\s*:\s*(success|failure)", text, re.IGNORECASE)
    if status_match:
        status = status_match.group(1).lower()
    else:
        status = "failure"

    reason_match = re.search(r"reason\s*:\s*(.+)", text, re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()
    else:
        reason = text.strip().replace("\n", " ")[:200]
    return status, reason[:300]


def judge_free_text(
    client: genai.Client,
    model: str,
    case: dict,
    traj: dict,
    max_retries: int,
    backoff_base: float,
) -> tuple[str, str]:
    prompt = build_free_text_prompt(case, traj)
    config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="text/plain",
    )

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            text = response.text or ""
            return parse_judge_output(text)
        except Exception as e:  # pylint: disable=broad-except
            last_error = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                time.sleep(backoff_base ** attempt)

    return "failure", f"judge_error: {last_error[:220]}"


def main() -> None:
    args = parse_args()
    cases_path = args.data_dir / args.cases_file
    traj_path = args.data_dir / args.trajectories_file
    labels_path = args.data_dir / args.labels_file

    cases = {row["id"]: row for row in read_jsonl(cases_path)}
    trajectories = read_jsonl(traj_path)

    done_ids = load_completed_ids(labels_path) if args.resume else set()
    pending = [row for row in trajectories if row.get("id") in cases and row.get("id") not in done_ids]
    if args.limit:
        pending = pending[:args.limit]

    print(f"Loaded cases={len(cases)}, trajectories={len(trajectories)}, pending={len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    need_free_text = any(cases[row["id"]].get("answer_type") == "free_text" for row in pending)
    client = None
    if need_free_text:
        api_key = os.getenv(args.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {args.api_key_env}")
        client = genai.Client(
            api_key=api_key,
            http_options={"base_url": args.api_base_url},
        )

    open_mode = "a" if args.resume else "w"
    success_count = 0
    failure_count = 0
    rule_count = 0
    llm_count = 0

    started = time.time()
    with open(labels_path, open_mode, encoding="utf-8") as out:
        for idx, traj in enumerate(pending, start=1):
            cid = traj["id"]
            case = cases[cid]
            predicted = str(traj.get("predicted", ""))

            status, reason = judge_rule(case, predicted)
            if status != "unknown":
                judge_source = "rule"
                rule_count += 1
            else:
                if client is None:
                    raise RuntimeError("LLM client is not initialized for free_text judging")
                status, reason = judge_free_text(
                    client=client,
                    model=args.model,
                    case=case,
                    traj=traj,
                    max_retries=args.max_retries,
                    backoff_base=args.backoff_base,
                )
                judge_source = "llm"
                llm_count += 1

            if status == "success":
                success_count += 1
            else:
                failure_count += 1

            record = {
                "id": cid,
                "status": status,
                "judge_source": judge_source,
                "judge_reason": reason,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            if idx % args.progress_interval == 0 or idx == len(pending):
                elapsed = time.time() - started
                rate = idx / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{idx}/{len(pending)}] success={success_count}, failure={failure_count}, "
                    f"rule={rule_count}, llm={llm_count}, rate={rate:.1f}/s"
                )

    print("Done.")
    print(f"Output: {labels_path}")
    print(f"success={success_count}, failure={failure_count}, rule={rule_count}, llm={llm_count}")


if __name__ == "__main__":
    main()
