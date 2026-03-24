#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError as exc:
    raise RuntimeError("Missing dependency `openai`. Install with `pip install openai`.") from exc


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Self-evolution A/B evaluator for BizBench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--targets-path", type=Path, required=True)
    p.add_argument("--example-pool-path", type=Path, required=True)
    p.add_argument("--manifest-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mode", choices=["all", "baseline", "treatment_sf", "zeroshot"], default="all")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260324)
    p.add_argument("--baseline-k", type=int, default=3)
    p.add_argument("--sf-success-k", type=int, default=5)
    p.add_argument("--sf-failure-k", type=int, default=1)
    p.add_argument("--sf-style", choices=["guardrail", "contrastive"], default="guardrail")
    p.add_argument("--sf-repair", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--model", default="qwen3-vl-8b-instruct")
    p.add_argument("--api-base", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--request-retries", type=int, default=2)
    return p.parse_args()


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


def parse_float(value: str) -> float | None:
    cleaned = (value or "").strip().replace(",", "")
    cleaned = cleaned.replace("$", "").replace("%", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
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


def is_correct(record: dict, predicted: str) -> bool:
    gold = str(record.get("gold_answer", ""))
    answer_type = record.get("answer_type")
    if answer_type == "numerical":
        g = parse_float(gold)
        p = parse_float(predicted)
        return g is not None and p is not None and abs(g - p) <= 1e-9
    if answer_type == "mcq":
        return normalize_mcq(gold) == normalize_mcq(predicted)
    if answer_type == "boolean":
        return normalize_boolean(gold) == normalize_boolean(predicted)
    return gold.strip().lower() == (predicted or "").strip().lower()


def extract_final_answer(text: str) -> str:
    m = re.search(r"\*\*Final Answer:\*\*\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().splitlines()[0].rstrip(".")
    m = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().splitlines()[0].rstrip(".")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def answer_instruction(answer_type: str) -> str:
    if answer_type == "numerical":
        return "Output a single number only."
    if answer_type == "mcq":
        return "Output option letter(s) only."
    if answer_type == "boolean":
        return "Output only Yes or No."
    return "Output a concise direct answer."


def build_baseline_user(target: dict, examples: list[dict]) -> str:
    chunks = [
        "You will see worked examples and one target problem.",
        "Use examples as method references, but solve target from target context only.",
    ]
    for i, ex in enumerate(examples, start=1):
        lines = [f"=== Example {i} ===", f"Question: {ex['question']}"]
        if ex.get("context"):
            lines.append(f"Context: {ex['context']}")
        if ex.get("options"):
            lines.append(f"Options: {ex['options']}")
        lines.append(f"Gold Answer: {ex['gold_answer']}")
        chunks.append("\n".join(lines))

    target_lines = ["=== Target Problem ===", f"Question: {target['question']}"]
    if target.get("context"):
        target_lines.append(f"Context: {target['context']}")
    if target.get("options"):
        target_lines.append(f"Options: {target['options']}")
    target_lines.append(answer_instruction(target.get("answer_type", "")))
    target_lines.append("End with exactly one line: **Final Answer:** <answer>")
    chunks.append("\n".join(target_lines))
    return "\n\n".join(chunks)


def build_failure_guardrails(target: dict, failures: list[dict], style: str) -> str:
    if style == "contrastive":
        rows = ["=== Failure-derived Notes ==="]
        for i, ex in enumerate(failures, start=1):
            rows.append(
                f"- Case {i}: wrong={ex.get('predicted','')}; correct={ex.get('gold_answer','')}; "
                "Do not repeat this mismatch pattern."
            )
        rows.append("- Recompute from target context only.")
        rows.append("- Always output strict final answer format.")
        return "\n".join(rows)

    rows = ["=== Failure-derived Guardrails ==="]
    rows.append("- Never copy numbers from examples; recompute from target context.")
    if target.get("answer_type") == "numerical":
        rows.append("- Verify sign, unit scale, and decimal precision before final answer.")
    if any("empty predicted" in (ex.get("judge_reason") or "").lower() for ex in failures):
        rows.append("- Never leave the final answer blank.")
    rows.append("- End with exactly one line: **Final Answer:** <answer>.")
    return "\n".join(rows)


def build_sf_user(target: dict, successes: list[dict], failures: list[dict], style: str) -> str:
    prompt = build_baseline_user(target, successes)
    return f"{prompt}\n\n{build_failure_guardrails(target, failures, style)}"


def call_chat(client: OpenAI, model: str, system_prompt: str, user_prompt: str, temperature: float, top_p: float, max_tokens: int, retries: int) -> str:
    last = None
    for _ in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=600,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # pylint: disable=broad-except
            last = e
            time.sleep(1.5)
    raise RuntimeError(f"chat failed: {last}")


def maybe_repair_answer(client: OpenAI, args: argparse.Namespace, target: dict, raw_text: str, predicted: str) -> tuple[str, str]:
    at = target.get("answer_type")
    if at == "numerical" and parse_float(predicted) is not None:
        return raw_text, predicted
    if at == "mcq" and normalize_mcq(predicted):
        return raw_text, predicted
    if at == "boolean" and normalize_boolean(predicted) in {"yes", "no"}:
        return raw_text, predicted

    repair_user = (
        "Rewrite the previous answer into strict final format only.\n"
        f"Required output type: {at}.\n"
        f"Previous output:\n{raw_text}\n\n"
        "Output exactly one line: **Final Answer:** <answer>"
    )
    repaired = call_chat(
        client,
        args.model,
        "You are a strict answer formatter.",
        repair_user,
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        retries=args.request_retries,
    )
    repaired_pred = extract_final_answer(repaired)
    return repaired, repaired_pred


def run_mode(mode: str, args: argparse.Namespace, client: OpenAI, targets: dict[str, dict], examples: dict[str, dict], manifests: list[dict]) -> Path:
    out_path = args.output_dir / f"{mode}_results.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    ordered = manifests[:]
    rng.shuffle(ordered)
    if args.limit:
        ordered = ordered[:args.limit]

    with open(out_path, "w", encoding="utf-8") as out:
        for idx, m in enumerate(ordered, start=1):
            target = targets[m["target_id"]]
            sids = m.get("example_ids", [])
            fids = m.get("failure_example_ids", [])
            succ = [examples[x] for x in sids if x in examples][: args.sf_success_k]
            fail = [examples[x] for x in fids if x in examples][: args.sf_failure_k]

            if mode == "zeroshot":
                user = build_baseline_user(target, [])
            elif mode == "baseline":
                user = build_baseline_user(target, succ[: args.baseline_k])
            else:
                user = build_sf_user(target, succ, fail, args.sf_style)

            started = time.time()
            error = ""
            try:
                raw = call_chat(
                    client,
                    args.model,
                    "You are a financial reasoning expert.",
                    user,
                    args.temperature,
                    args.top_p,
                    args.max_tokens,
                    args.request_retries,
                )
                pred = extract_final_answer(raw)
                if mode == "treatment_sf" and args.sf_repair:
                    raw2, pred2 = maybe_repair_answer(client, args, target, raw, pred)
                    raw = raw2
                    pred = pred2
                ok = is_correct(target, pred)
            except Exception as e:  # pylint: disable=broad-except
                raw = ""
                pred = ""
                ok = False
                error = str(e)

            rec = {
                "mode": mode,
                "target_id": target["id"],
                "predicted": pred,
                "gold_answer": target.get("gold_answer", ""),
                "answer_type": target.get("answer_type", ""),
                "correct": ok,
                "latency_sec": round(time.time() - started, 3),
                "error": error,
                "response_text": raw,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{mode} {idx}/{len(ordered)}] {target['id']} -> {pred!r} | correct={ok}")
    return out_path


def p_value_sign(improved: int, regressed: int) -> float:
    n = improved + regressed
    if n == 0:
        return 1.0
    k = min(improved, regressed)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def pair_summary(a_rows: dict[str, dict], b_rows: dict[str, dict], a_name: str, b_name: str) -> dict:
    ids = sorted(set(a_rows) & set(b_rows))
    improved = 0
    regressed = 0
    for tid in ids:
        a_ok = a_rows[tid]["correct"]
        b_ok = b_rows[tid]["correct"]
        if (not a_ok) and b_ok:
            improved += 1
        elif a_ok and (not b_ok):
            regressed += 1
    a_acc = sum(1 for tid in ids if a_rows[tid]["correct"]) / len(ids)
    b_acc = sum(1 for tid in ids if b_rows[tid]["correct"]) / len(ids)
    return {
        "paired_count": len(ids),
        f"{a_name}_accuracy": round(a_acc, 4),
        f"{b_name}_accuracy": round(b_acc, 4),
        "delta": round(b_acc - a_acc, 4),
        "improved_count": improved,
        "regressed_count": regressed,
        "sign_test_p_value": round(p_value_sign(improved, regressed), 6),
    }


def main() -> None:
    args = parse_args()
    api_key = os.getenv("QWEN_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing QWEN_API_KEY in .env")

    targets = {r["id"]: r for r in read_jsonl(args.targets_path)}
    examples = {r["id"]: r for r in read_jsonl(args.example_pool_path)}
    manifests = read_jsonl(args.manifest_path)
    client = OpenAI(api_key=api_key, base_url=args.api_base)

    if args.mode == "all":
        modes = ["zeroshot", "baseline", "treatment_sf"]
    else:
        modes = [args.mode]

    paths = {}
    for mode in modes:
        paths[mode] = run_mode(mode, args, client, targets, examples, manifests)

    rows = {m: {r["target_id"]: r for r in read_jsonl(p)} for m, p in paths.items()}
    summary = {"modes": modes}
    if {"zeroshot", "baseline"}.issubset(rows):
        summary["baseline_vs_zeroshot"] = pair_summary(rows["zeroshot"], rows["baseline"], "zeroshot", "baseline")
    if {"baseline", "treatment_sf"}.issubset(rows):
        summary["treatment_sf_vs_baseline"] = pair_summary(rows["baseline"], rows["treatment_sf"], "baseline", "treatment_sf")
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
