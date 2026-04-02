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
    p.add_argument("--items-path", type=Path, default=None,
                   help="optional extracted memory items jsonl")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--mode",
        choices=["all", "baseline", "treatment_sf", "zeroshot", "h3", "treatment_h3"],
        default="all",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260324)
    p.add_argument("--baseline-k", type=int, default=3)
    p.add_argument("--sf-success-k", type=int, default=5)
    p.add_argument("--sf-failure-k", type=int, default=1)
    p.add_argument("--sf-style", choices=["guardrail", "contrastive"], default="guardrail")
    p.add_argument("--sf-repair", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--memory-strategy", choices=["simple", "typed"], default="typed")
    p.add_argument("--memory-max-strategy", type=int, default=2)
    p.add_argument("--memory-max-warning", type=int, default=1)
    p.add_argument("--h3-ops-k", type=int, default=1,
                   help="max strategy/warning operator templates in H3 mode")
    p.add_argument("--h3-max-rule-chars", type=int, default=120,
                   help="max chars for a single synthesized operator rule")
    p.add_argument("--h3-iterative", action=argparse.BooleanOptionalAction, default=False,
                   help="enable lightweight iterative operator updates during treatment_h3 run")
    p.add_argument("--h3-history-limit", type=int, default=64,
                   help="max number of iterative rules retained in memory")
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


def build_items_map(path: Path | None) -> dict[str, list[dict]]:
    if path is None or not path.exists():
        return {}
    out: dict[str, list[dict]] = {}
    for row in read_jsonl(path):
        sid = row.get("source_question_id")
        if not sid:
            continue
        out.setdefault(sid, []).append(row)
    return out


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
        rows.append("- If context says '(in thousands/millions)', keep table scale unless the question asks conversion.")
    if any("empty predicted" in (ex.get("judge_reason") or "").lower() for ex in failures):
        rows.append("- Never leave the final answer blank.")
    rows.append("- End with exactly one line: **Final Answer:** <answer>.")
    return "\n".join(rows)


def memory_match_score(item: dict, target: dict) -> int:
    score = 0
    target_answer_type = target.get("answer_type")
    if target_answer_type and item.get("answer_type") == target_answer_type:
        score += 3

    target_calc = set(target.get("calc_type") or [])
    item_calc = set(item.get("calc_type") or [])
    score += len(target_calc & item_calc)

    content_len = len((item.get("content") or "").strip())
    if 40 <= content_len <= 260:
        score += 1
    return score


def pick_memory_items(
    source_examples: list[dict],
    expected_type: str,
    target: dict,
    items_map: dict[str, list[dict]],
    limit: int,
    strategy: str,
) -> list[str]:
    if limit <= 0:
        return []

    candidates: list[tuple[int, int, str]] = []
    for ex in source_examples:
        for item in items_map.get(ex.get("id", ""), []):
            if item.get("memory_type") != expected_type:
                continue
            text = (item.get("content") or "").strip()
            if not text:
                continue
            if strategy == "typed":
                score = memory_match_score(item, target)
            else:
                score = 0
            candidates.append((score, -len(text), text))

    candidates.sort(reverse=True)
    out: list[str] = []
    seen = set()
    for _, _, text in candidates:
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def build_memory_section(
    target: dict,
    successes: list[dict],
    failures: list[dict],
    items_map: dict[str, list[dict]],
    strategy: str,
    max_strategy: int,
    max_warning: int,
) -> str:
    lines = ["=== Retrieved Experience Memory ==="]
    success_notes = pick_memory_items(successes, "strategy", target, items_map, max_strategy, strategy)
    failure_notes = pick_memory_items(failures, "warning", target, items_map, max_warning, strategy)

    for note in success_notes:
        lines.append(f"- Strategy: {note}")
    for note in failure_notes:
        lines.append(f"- Warning: {note}")
    if len(lines) == 1:
        lines.append("- Strategy: compute from target context only and verify output format.")
    return "\n".join(lines)


def squash_line(text: str, max_chars: int) -> str:
    one = re.sub(r"\s+", " ", (text or "").strip())
    if not one:
        return ""
    if len(one) <= max_chars:
        return one
    return one[: max_chars - 3].rstrip() + "..."


def template_from_item(item: dict, max_chars: int) -> str:
    answer_type = item.get("answer_type") or "unknown"
    calc_type = "+".join(item.get("calc_type") or []) or "general"
    title = squash_line(item.get("title") or "", max_chars)
    content = squash_line(item.get("content") or "", max_chars)
    core = title or content
    if not core:
        return ""
    return f"[{answer_type}|{calc_type}] {core}"


def summarize_iterative_rule(target: dict, predicted: str, gold: str) -> str | None:
    answer_type = target.get("answer_type")
    if answer_type == "numerical":
        g = parse_float(gold)
        p = parse_float(predicted)
        if p is None:
            return "For numerical targets: output a single numeric value only, no explanation text."
        if g is None:
            return None
        if abs(g - p) <= 1e-9:
            return None
        if abs(g - p) <= 0.1:
            return "For numerical targets: preserve source precision and avoid extra rounding in final step."
        return "For numerical targets: avoid re-computation if the required value is directly stated in context."

    if answer_type == "boolean":
        return "For boolean targets: answer strictly with Yes or No only."

    if answer_type == "mcq":
        return "For MCQ targets: output option letter(s) only, no additional words."

    if not (predicted or "").strip():
        return "Never leave final answer empty; always output strict final answer line."
    return None


def build_h3_operator_section(
    target: dict,
    successes: list[dict],
    failures: list[dict],
    items_map: dict[str, list[dict]],
    args: argparse.Namespace,
    iter_rules: list[str],
) -> str:
    strategies = pick_memory_items(
        successes,
        "strategy",
        target,
        items_map,
        max(1, args.h3_ops_k),
        args.memory_strategy,
    )
    warnings = pick_memory_items(
        failures,
        "warning",
        target,
        items_map,
        max(1, args.h3_ops_k),
        args.memory_strategy,
    )

    lines = ["=== H3 Contrastive Operator Memory ==="]
    for idx, text in enumerate(strategies[: args.h3_ops_k], start=1):
        item = {
            "title": text,
            "content": text,
            "answer_type": target.get("answer_type"),
            "calc_type": target.get("calc_type") or [],
        }
        lines.append(f"- OP{idx} Prefer: {template_from_item(item, args.h3_max_rule_chars)}")
    for idx, text in enumerate(warnings[: args.h3_ops_k], start=1):
        item = {
            "title": text,
            "content": text,
            "answer_type": target.get("answer_type"),
            "calc_type": target.get("calc_type") or [],
        }
        lines.append(f"- OP{idx} Avoid: {template_from_item(item, args.h3_max_rule_chars)}")

    if iter_rules:
        for i, rule in enumerate(iter_rules[-args.h3_ops_k:], start=1):
            lines.append(f"- OP-Iter{i}: {squash_line(rule, args.h3_max_rule_chars)}")

    lines.append("- Execute exactly one best-matching operator; do not narrate reasoning.")
    lines.append("- If target contains direct value, prefer extraction over recomputation.")
    return "\n".join(lines)


def build_sf_user(target: dict, successes: list[dict], failures: list[dict], style: str, items_map: dict[str, list[dict]], args: argparse.Namespace) -> str:
    prompt = build_baseline_user(target, successes)
    memory = build_memory_section(
        target,
        successes,
        failures,
        items_map,
        strategy=args.memory_strategy,
        max_strategy=args.memory_max_strategy,
        max_warning=args.memory_max_warning,
    )
    return f"{prompt}\n\n{memory}\n\n{build_failure_guardrails(target, failures, style)}"


def build_h3_user(
    target: dict,
    successes: list[dict],
    failures: list[dict],
    items_map: dict[str, list[dict]],
    args: argparse.Namespace,
    iter_rules: list[str],
) -> str:
    prompt = build_baseline_user(target, successes)
    op_mem = build_h3_operator_section(target, successes, failures, items_map, args, iter_rules)
    extra = (
        "=== Output Contract ===\n"
        "- Return exactly one line: **Final Answer:** <answer>\n"
        "- No chain-of-thought, no extra explanation."
    )
    return f"{prompt}\n\n{op_mem}\n\n{extra}"


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


def run_mode(
    mode: str,
    args: argparse.Namespace,
    client: OpenAI,
    targets: dict[str, dict],
    examples: dict[str, dict],
    manifests: list[dict],
    items_map: dict[str, list[dict]],
) -> Path:
    out_path = args.output_dir / f"{mode}_results.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    ordered = manifests[:]
    rng.shuffle(ordered)
    if args.limit:
        ordered = ordered[:args.limit]

    iter_rules: list[str] = []

    with open(out_path, "w", encoding="utf-8") as out:
        for idx, m in enumerate(ordered, start=1):
            target = dict(targets[m["target_id"]])
            if m.get("calc_type"):
                target["calc_type"] = m.get("calc_type")
            sids = m.get("example_ids", [])
            fids = m.get("failure_example_ids", [])
            succ = [examples[x] for x in sids if x in examples][: args.sf_success_k]
            fail = [examples[x] for x in fids if x in examples][: args.sf_failure_k]

            if mode == "zeroshot":
                user = build_baseline_user(target, [])
            elif mode == "baseline":
                user = build_baseline_user(target, succ[: args.baseline_k])
            elif mode in {"h3", "treatment_h3"}:
                user = build_h3_user(target, succ, fail, items_map, args, iter_rules)
            else:
                user = build_sf_user(target, succ, fail, args.sf_style, items_map, args)

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
                if mode in {"treatment_sf", "h3", "treatment_h3"} and args.sf_repair:
                    raw2, pred2 = maybe_repair_answer(client, args, target, raw, pred)
                    raw = raw2
                    pred = pred2
                ok = is_correct(target, pred)

                if mode in {"h3", "treatment_h3"} and args.h3_iterative and not ok:
                    rule = summarize_iterative_rule(target, pred, target.get("gold_answer", ""))
                    if rule and rule not in iter_rules:
                        iter_rules.append(rule)
                        if len(iter_rules) > args.h3_history_limit:
                            iter_rules = iter_rules[-args.h3_history_limit :]
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
                "prompt_chars": len(user),
                "calc_type": target.get("calc_type", []),
            }
            if mode in {"h3", "treatment_h3"}:
                rec["h3_iter_rules_count"] = len(iter_rules)
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
    items_map = build_items_map(args.items_path)
    client = OpenAI(api_key=api_key, base_url=args.api_base)

    if args.mode == "all":
        modes = ["zeroshot", "baseline", "treatment_sf"]
    elif args.mode == "h3":
        modes = ["baseline", "treatment_h3"]
    else:
        modes = [args.mode]

    config_payload = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
    }
    write_json(args.output_dir / "config.json", config_payload)

    paths = {}
    for mode in modes:
        paths[mode] = run_mode(mode, args, client, targets, examples, manifests, items_map)

    rows = {m: {r["target_id"]: r for r in read_jsonl(p)} for m, p in paths.items()}
    summary = {"modes": modes}
    if {"zeroshot", "baseline"}.issubset(rows):
        summary["baseline_vs_zeroshot"] = pair_summary(rows["zeroshot"], rows["baseline"], "zeroshot", "baseline")
    if {"baseline", "treatment_sf"}.issubset(rows):
        summary["treatment_sf_vs_baseline"] = pair_summary(rows["baseline"], rows["treatment_sf"], "baseline", "treatment_sf")
    if {"baseline", "treatment_h3"}.issubset(rows):
        summary["treatment_h3_vs_baseline"] = pair_summary(
            rows["baseline"],
            rows["treatment_h3"],
            "baseline",
            "treatment_h3",
        )
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
