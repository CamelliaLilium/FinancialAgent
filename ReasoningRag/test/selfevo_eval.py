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

try:
    from google import genai
    from google.genai import types
except ImportError as exc:
    raise RuntimeError("Missing dependency `google-genai`. Install with `pip install google-genai`.") from exc


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
    p.add_argument("--mode", choices=[
        "all", "baseline", "treatment_sf", "zeroshot",
        "treatment_op", "treatment_op_routed", "h2", "h1",
    ], default="all")
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
    p.add_argument("--model", default="qwen3-vl-8b-instruct")
    p.add_argument("--backend", choices=["openai", "gemini"], default="openai")
    p.add_argument("--api-base", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p.add_argument("--api-key-env", default="",
                   help="if empty, use QWEN_API_KEY for openai and AIHUBMIX_API_KEY for gemini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--gemini-thinking-budget", type=int, default=0)
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


# ---------------------------------------------------------------------------
# Operator Templates (H2)
# ---------------------------------------------------------------------------

OPERATOR_TEMPLATES = {
    "AGGREGATE": (
        "RULE: If the table already contains a total / aggregate / sum row, "
        "use that value directly. Do NOT re-sum the individual line items."
    ),
    "UNIT_SCALE": (
        "RULE: If the table header says '(in thousands)', '(in millions)', etc., "
        "keep the table's unit scale in your answer unless the question explicitly asks for conversion."
    ),
    "DIRECT_READ": (
        "RULE: If the answer can be read directly from a single cell or field in the context, "
        "output that exact value. Do NOT perform additional derivation or calculation."
    ),
    "PRECISION": (
        "RULE: Preserve the full precision of your intermediate calculation in the final answer. "
        "Do NOT round to a 'nice' number or convert to a casual approximation."
    ),
}

# calc_type → ordered list of template keys to try (first applicable wins)
CALC_TYPE_TEMPLATE_MAP: dict[tuple[str, ...], list[str]] = {
    ("extraction",): ["DIRECT_READ", "AGGREGATE"],
    ("arithmetic",): ["AGGREGATE", "UNIT_SCALE", "PRECISION"],
    ("ratio",): ["AGGREGATE", "UNIT_SCALE", "PRECISION"],
    ("ratio", "multi_step"): ["AGGREGATE", "UNIT_SCALE", "PRECISION"],
    ("arithmetic", "multi_step"): ["AGGREGATE", "UNIT_SCALE", "PRECISION"],
    ("reasoning",): ["PRECISION"],
}

# For unrouted mode, every target gets this single best-effort template
DEFAULT_TEMPLATE_KEY = "PRECISION"


def select_operator_template(calc_type: tuple[str, ...], routed: bool) -> str | None:
    """Return a single operator-template string, or None if routing says skip."""
    if not routed:
        # unrouted: always inject the default
        return OPERATOR_TEMPLATES[DEFAULT_TEMPLATE_KEY]

    keys = CALC_TYPE_TEMPLATE_MAP.get(calc_type)
    if keys:
        return OPERATOR_TEMPLATES[keys[0]]
    # unknown calc_type → inject nothing when routed
    return None


def build_op_user(target: dict, examples: list[dict], calc_type: tuple[str, ...], routed: bool) -> str:
    """Build prompt for treatment_op / treatment_op_routed modes."""
    prompt = build_baseline_user(target, examples)
    rule = select_operator_template(calc_type, routed)
    if rule is None:
        return prompt
    section = (
        "=== Operator Rule (follow strictly) ===\n"
        f"{rule}\n"
        "Apply this rule to the target problem above. "
        "End with exactly one line: **Final Answer:** <answer>"
    )
    return f"{prompt}\n\n{section}"


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


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key_env:
        key = os.getenv(args.api_key_env, "")
        if key:
            return key
    default_env = "QWEN_API_KEY" if args.backend == "openai" else "AIHUBMIX_API_KEY"
    return os.getenv(default_env, "")


def build_client(args: argparse.Namespace):
    api_key = resolve_api_key(args)
    if not api_key:
        if args.api_key_env:
            raise RuntimeError(f"Missing {args.api_key_env} in .env")
        default_env = "QWEN_API_KEY" if args.backend == "openai" else "AIHUBMIX_API_KEY"
        raise RuntimeError(f"Missing {default_env} in .env")

    if args.backend == "openai":
        return OpenAI(api_key=api_key, base_url=args.api_base)

    return genai.Client(api_key=api_key, http_options={"base_url": args.api_base})


def call_chat(
    client,
    backend: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    gemini_thinking_budget: int,
    retries: int,
) -> str:
    last = None
    for _ in range(retries):
        try:
            if backend == "openai":
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

            prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"
            cfg = types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=gemini_thinking_budget),
                response_mime_type="text/plain",
            )
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=cfg,
            )
            return resp.text or ""
        except Exception as e:  # pylint: disable=broad-except
            last = e
            time.sleep(1.5)
    raise RuntimeError(f"chat failed: {last}")


def maybe_repair_answer(client, args: argparse.Namespace, target: dict, raw_text: str, predicted: str) -> tuple[str, str]:
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
        args.backend,
        args.model,
        "You are a strict answer formatter.",
        repair_user,
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        gemini_thinking_budget=args.gemini_thinking_budget,
        retries=args.request_retries,
    )
    repaired_pred = extract_final_answer(repaired)
    return repaired, repaired_pred


def run_mode(mode: str, args: argparse.Namespace, client, targets: dict[str, dict], examples: dict[str, dict], manifests: list[dict], items_map: dict[str, list[dict]]) -> Path:
    out_path = args.output_dir / f"{mode}_results.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    ordered = manifests[:]
    rng.shuffle(ordered)
    if args.limit:
        ordered = ordered[:args.limit]

    with open(out_path, "w", encoding="utf-8") as out:
        for idx, m in enumerate(ordered, start=1):
            target = dict(targets[m["target_id"]])
            if m.get("calc_type"):
                target["calc_type"] = m.get("calc_type")
            sids = m.get("example_ids", [])
            fids = m.get("failure_example_ids", [])
            succ = [examples[x] for x in sids if x in examples][: args.sf_success_k]
            fail = [examples[x] for x in fids if x in examples][: args.sf_failure_k]

            calc_type = tuple(target.get("calc_type") or [])

            if mode == "zeroshot":
                user = build_baseline_user(target, [])
            elif mode == "baseline":
                user = build_baseline_user(target, succ[: args.baseline_k])
            elif mode in ("treatment_op", "treatment_op_routed"):
                routed = mode == "treatment_op_routed"
                user = build_op_user(target, succ[: args.baseline_k], calc_type, routed)
            else:
                user = build_sf_user(target, succ, fail, args.sf_style, items_map, args)

            started = time.time()
            error = ""
            try:
                raw = call_chat(
                    client,
                    args.backend,
                    args.model,
                    "You are a financial reasoning expert. Return exactly one line: **Final Answer:** <answer>. Do not include any explanation.",
                    user,
                    args.temperature,
                    args.top_p,
                    args.max_tokens,
                    args.gemini_thinking_budget,
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
                "calc_type": list(calc_type),
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


def group_accuracy(rows_by_id: dict[str, dict], manifest_meta: dict[str, tuple]) -> dict[str, dict]:
    from collections import defaultdict
    groups: dict[str, list[bool]] = defaultdict(list)
    for tid, rec in rows_by_id.items():
        ct = manifest_meta.get(tid, ())
        key = "+".join(ct) if ct else "unknown"
        groups[key].append(bool(rec["correct"]))
    out = {}
    for key, vals in sorted(groups.items()):
        acc = sum(vals) / len(vals) if vals else 0.0
        out[key] = {"count": len(vals), "correct": sum(vals), "accuracy": round(acc, 4)}
    return out


def pair_summary_grouped(
    a_rows: dict[str, dict], b_rows: dict[str, dict],
    a_name: str, b_name: str,
    manifest_meta: dict[str, tuple],
) -> dict[str, dict]:
    from collections import defaultdict
    groups: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for tid in sorted(set(a_rows) & set(b_rows)):
        ct = manifest_meta.get(tid, ())
        key = "+".join(ct) if ct else "unknown"
        groups[key].append((bool(a_rows[tid]["correct"]), bool(b_rows[tid]["correct"])))
    out = {}
    for key, pairs in sorted(groups.items()):
        n = len(pairs)
        a_acc = sum(1 for a, _ in pairs if a) / n if n else 0.0
        b_acc = sum(1 for _, b in pairs if b) / n if n else 0.0
        improved = sum(1 for a, b in pairs if not a and b)
        regressed = sum(1 for a, b in pairs if a and not b)
        out[key] = {
            "count": n,
            f"{a_name}_acc": round(a_acc, 4),
            f"{b_name}_acc": round(b_acc, 4),
            "delta": round(b_acc - a_acc, 4),
            "improved": improved,
            "regressed": regressed,
        }
    return out


def dump_config(args: argparse.Namespace, modes: list[str]) -> dict:
    return {
        "modes": modes,
        "seed": args.seed,
        "limit": args.limit,
        "baseline_k": args.baseline_k,
        "sf_success_k": args.sf_success_k,
        "sf_failure_k": args.sf_failure_k,
        "sf_style": args.sf_style,
        "sf_repair": args.sf_repair,
        "memory_strategy": args.memory_strategy,
        "memory_max_strategy": args.memory_max_strategy,
        "memory_max_warning": args.memory_max_warning,
        "backend": args.backend,
        "model": args.model,
        "api_base": args.api_base,
        "api_key_env": args.api_key_env,
        "temperature": args.temperature,
        "gemini_thinking_budget": args.gemini_thinking_budget,
        "items_path": str(args.items_path) if args.items_path else None,
    }


def main() -> None:
    args = parse_args()

    targets = {r["id"]: r for r in read_jsonl(args.targets_path)}
    examples = {r["id"]: r for r in read_jsonl(args.example_pool_path)}
    manifests = read_jsonl(args.manifest_path)
    items_map = build_items_map(args.items_path)
    client = build_client(args)

    manifest_meta: dict[str, tuple] = {}
    for m in manifests:
        manifest_meta[m["target_id"]] = tuple(m.get("calc_type", []))

    if args.mode == "all":
        modes = ["zeroshot", "baseline", "treatment_sf"]
    elif args.mode == "h1":
        modes = ["baseline", "treatment_op_routed"]
    elif args.mode == "h2":
        modes = ["baseline", "treatment_sf", "treatment_op", "treatment_op_routed"]
    else:
        modes = [args.mode]

    write_json(args.output_dir / "config.json", dump_config(args, modes))

    paths = {}
    for mode in modes:
        paths[mode] = run_mode(mode, args, client, targets, examples, manifests, items_map)

    rows = {m: {r["target_id"]: r for r in read_jsonl(p)} for m, p in paths.items()}
    summary: dict = {"modes": modes}

    if {"zeroshot", "baseline"}.issubset(rows):
        summary["baseline_vs_zeroshot"] = pair_summary(rows["zeroshot"], rows["baseline"], "zeroshot", "baseline")

    baseline_rows = rows.get("baseline")
    for treat_mode in ["treatment_sf", "treatment_op", "treatment_op_routed"]:
        if baseline_rows and treat_mode in rows:
            key = f"{treat_mode}_vs_baseline"
            summary[key] = pair_summary(baseline_rows, rows[treat_mode], "baseline", treat_mode)
            summary[f"{key}_by_calc_type"] = pair_summary_grouped(
                baseline_rows, rows[treat_mode], "baseline", treat_mode, manifest_meta,
            )

    for m_name, m_rows in rows.items():
        summary[f"{m_name}_by_calc_type"] = group_accuracy(m_rows, manifest_meta)

    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
