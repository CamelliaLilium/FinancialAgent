#!/usr/bin/env python3
"""
Prepare A/B test data for FinMMR or BizBench.

Usage:
    python prepare_ab_test_data.py --dataset-name bizbench --test-file ../datasets/test/bizbench_test.json
"""

import argparse
import json
import math
import random
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "the", "to", "was", "what", "when",
    "which", "with", "would", "year", "years",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare normalized A/B test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-name", choices=["finmmr", "bizbench"], default="finmmr",
                   help="target dataset and example source")
    p.add_argument("--test-root", type=Path, default=ROOT.parent / "datasets" / "test",
                   help="directory containing held-out test datasets")
    p.add_argument("--test-file", type=Path, default=None,
                   help="override test file path")
    p.add_argument("--finmmr-image-root", type=Path,
                   default=ROOT.parent / "datasets" / "FinMMR" / "images",
                   help="directory containing FinMMR test images")
    p.add_argument("--cases-path", type=Path, default=ROOT / "data" / "cases.jsonl",
                   help="existing case pool")
    p.add_argument("--trajectories-path", type=Path, default=ROOT / "data" / "trajectories.jsonl",
                   help="trajectory pool")
    p.add_argument("--labels-path", type=Path, default=ROOT / "data" / "labels.jsonl",
                   help="labels for success/failure status")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data" / "ab_test",
                   help="output directory")
    p.add_argument("--images-subdir", default="test_targets/finmmr",
                   help="subdir under ReasoningRag/images for copied target images")
    p.add_argument("--target-count", type=int, default=100,
                   help="number of held-out target questions")
    p.add_argument("--shots", type=int, default=3,
                   help="success examples per target")
    p.add_argument("--failure-shots", type=int, default=0,
                   help="failure examples per target")
    p.add_argument("--seed", type=int, default=20260318,
                   help="random seed")
    p.add_argument("--only-success-trajectory", action="store_true",
                   help="keep only success trajectories in example pool")
    p.add_argument("--leakage-check", choices=["none", "question", "question_answer"], default="question",
                   help="exclude leaked examples")
    p.add_argument("--copy-images", action=argparse.BooleanOptionalAction, default=True,
                   help="copy target images into ReasoningRag/images")
    p.add_argument("--dry-run", action="store_true",
                   help="show planned outputs without writing files")
    return p.parse_args()


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def tokenize(text: str) -> set[str]:
    tokens = TOKEN_RE.findall((text or "").lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


def parse_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).strip().replace(",", "")
    cleaned = cleaned.replace("$", "").replace("%", "")
    cleaned = cleaned.replace("USD", "").replace("TWD", "")
    cleaned = cleaned.replace("usd", "").replace("twd", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def classify_answer_type(answer, options: str | None) -> str:
    if options:
        s = str(answer).strip().upper()
        letters = set(re.findall(r"[A-Z]", s))
        if letters:
            return "mcq"
    lowered = str(answer).strip().lower()
    if lowered in {"yes", "no", "true", "false"}:
        return "boolean"
    if parse_float(answer) is not None:
        return "numerical"
    return "free_text"


def get_decimal_places(answer) -> int | None:
    if parse_float(answer) is None:
        return None
    s = str(answer).strip().replace(",", "")
    if "e" in s.lower():
        s = format(float(s), "f")
    if "." not in s:
        return 0
    frac = s.split(".", 1)[1].rstrip("0")
    return len(frac)


def derive_calc_type_finmmr(stats: dict | None) -> list[str]:
    if not stats:
        return ["extraction"]
    ops = stats.get("operator_statistics", {}).get("operators", {})
    if not ops or all(v == 0 for v in ops.values()):
        return ["extraction"]
    tags = []
    has_div = ops.get("/", 0) > 0 or ops.get("%", 0) > 0
    has_pow = ops.get("**", 0) > 0
    has_add_sub = ops.get("+", 0) > 0 or ops.get("-", 0) > 0
    has_mul = ops.get("*", 0) > 0
    total_ops = sum(ops.values())
    if has_pow:
        tags.append("compound")
    if has_div:
        tags.append("ratio")
    if (has_add_sub or has_mul) and not has_div and not has_pow:
        tags.append("arithmetic")
    if not tags:
        tags.append("arithmetic")
    if total_ops >= 3:
        tags.append("multi_step")
    return tags


def derive_calc_type_bizbench(program: str | None, task: str, question: str, answer_type: str) -> list[str]:
    if answer_type != "numerical":
        return ["reasoning"]
    text = f"{question} {task}".lower()
    ratio_keywords = ("percent", "percentage", "ratio", "rate", "margin", "yield", "growth")
    has_ratio_words = any(kw in text for kw in ratio_keywords)
    if not program:
        if task == "SEC-NUM":
            return ["extraction"]
        return ["ratio"] if has_ratio_words else ["arithmetic"]
    ops = {
        "+": program.count("+"),
        "-": program.count("-"),
        "*": program.count("*"),
        "/": program.count("/"),
        "%": program.count("%"),
        "**": program.count("**"),
    }
    tags = []
    total_ops = ops["+"] + ops["-"] + ops["*"] + ops["/"] + ops["%"]
    if ops["**"] > 0:
        tags.append("compound")
    if ops["/"] > 0 or ops["%"] > 0 or has_ratio_words:
        tags.append("ratio")
    if (ops["+"] > 0 or ops["-"] > 0 or ops["*"] > 0) and "ratio" not in tags and "compound" not in tags:
        tags.append("arithmetic")
    if not tags:
        tags.append("extraction")
    if total_ops >= 3 or ops["**"] > 0:
        tags.append("multi_step")
    return tags


def balanced_counts(total: int, labels: list[str]) -> dict[str, int]:
    base = total // len(labels)
    remainder = total % len(labels)
    counts = {label: base for label in labels}
    for label in labels[:remainder]:
        counts[label] += 1
    return counts


def resolve_finmmr_test_image(raw_path: str, image_root: Path) -> Path:
    raw = Path(raw_path)
    if raw.exists():
        return raw
    if "/datasets/FinMMR/images/" in raw_path:
        suffix = raw_path.split("/datasets/FinMMR/images/", 1)[1].replace("/", "\\")
        return image_root / Path(suffix)
    return image_root / raw.name


def normalize_finmmr_item(item: dict, split: str, args: argparse.Namespace) -> dict:
    images = []
    for raw_path in item.get("images", []):
        src = resolve_finmmr_test_image(raw_path, args.finmmr_image_root)
        dest_rel = Path("images") / args.images_subdir / split / src.name
        dest_abs = ROOT / dest_rel
        images.append({
            "src": src,
            "dest_rel": dest_rel.as_posix(),
            "dest_abs": dest_abs,
        })
    answer = item.get("answer")
    if answer is None and "ground_truth" in item:
        answer = item["ground_truth"]
    answer_type = classify_answer_type(answer, item.get("options"))
    return {
        "id": f"finmmr_{item['question_id']}",
        "source": "finmmr_test",
        "image_paths": [img["dest_rel"] for img in images],
        "question": item.get("question", "").strip(),
        "context": item.get("context", "") or "",
        "options": item.get("options") or None,
        "gold_answer": str(answer).strip(),
        "answer_type": answer_type,
        "calc_type": derive_calc_type_finmmr(item.get("statistics")),
        "decimal_places": get_decimal_places(answer),
        "metadata": {
            "split": split,
            "source_dataset": "FinMMR_test",
            "original_question_id": item.get("question_id"),
            "source_id": item.get("source_id"),
            "grade": item.get("grade"),
            "language": item.get("language"),
            "raw_images": item.get("images", []),
        },
        "_image_jobs": images,
    }


def load_finmmr_targets(args: argparse.Namespace) -> tuple[list[dict], dict[str, int]]:
    files = {
        "easy": args.test_root / "finmmr_easy_test.json",
        "medium": args.test_root / "finmmr_medium_test.json",
        "hard": args.test_root / "finmmr_hard_test.json",
    }
    counts = balanced_counts(args.target_count, list(files.keys()))
    rng = random.Random(args.seed)
    normalized = []
    for split, path in files.items():
        rows = read_json(path)
        picked = rng.sample(rows, counts[split])
        for item in picked:
            normalized.append(normalize_finmmr_item(item, split, args))
    normalized.sort(key=lambda row: row["id"])
    return normalized, counts


def normalize_bizbench_item(item: dict, idx: int) -> dict:
    answer = item.get("answer")
    answer_type = classify_answer_type(answer, item.get("options"))
    task = str(item.get("task") or "")
    question = (item.get("question") or "").strip()
    program = item.get("program")
    return {
        "id": f"bizbench_test_{idx}",
        "source": "bizbench_test",
        "image_paths": [],
        "question": question,
        "context": item.get("context") or "",
        "options": item.get("options") or None,
        "gold_answer": str(answer).strip(),
        "answer_type": answer_type,
        "calc_type": derive_calc_type_bizbench(program, task, question, answer_type),
        "decimal_places": get_decimal_places(answer),
        "metadata": {
            "source_dataset": "BizBench_test",
            "task": task,
            "context_type": item.get("context_type"),
            "program": program,
        },
        "_image_jobs": [],
    }


def load_bizbench_targets(args: argparse.Namespace) -> tuple[list[dict], dict[str, int]]:
    test_file = args.test_file or (args.test_root / "bizbench_test.json")
    rows = read_json(test_file)
    if args.target_count and args.target_count < len(rows):
        rng = random.Random(args.seed)
        rows = rng.sample(rows, args.target_count)
    normalized = [normalize_bizbench_item(item, idx) for idx, item in enumerate(rows)]
    normalized.sort(key=lambda row: row["id"])
    task_counts = Counter(item.get("task", "") for item in rows)
    return normalized, dict(sorted(task_counts.items()))


def load_targets(args: argparse.Namespace) -> tuple[list[dict], dict[str, int]]:
    if args.dataset_name == "finmmr":
        return load_finmmr_targets(args)
    if args.dataset_name == "bizbench":
        return load_bizbench_targets(args)
    raise ValueError(f"Unsupported dataset_name: {args.dataset_name}")


def load_label_status(path: Path) -> dict[str, dict]:
    out = {}
    for row in read_jsonl(path):
        rid = row.get("id")
        if rid:
            out[rid] = {
                "status": (row.get("status") or "").strip().lower(),
                "judge_reason": row.get("judge_reason", ""),
            }
    return out


def join_example_pool(
    cases_path: Path,
    trajectories_path: Path,
    dataset_name: str,
    labels_map: dict[str, dict],
    only_success_trajectory: bool,
) -> list[dict]:
    cases = {row["id"]: row for row in read_jsonl(cases_path)}
    trajectories = {row["id"]: row for row in read_jsonl(trajectories_path)}
    pool = []
    for case_id, traj in trajectories.items():
        case = cases.get(case_id)
        if not case:
            continue
        if case.get("source") != dataset_name:
            continue
        if not traj.get("trajectory") or traj["trajectory"].startswith("ERROR:"):
            continue
        label = labels_map.get(case_id, {})
        status = (label.get("status") or "unknown").lower()
        if only_success_trajectory and status != "success":
            continue
        pool.append({
            "id": case_id,
            "source": case.get("source"),
            "image_paths": case.get("image_paths", []),
            "question": case.get("question", ""),
            "context": case.get("context", ""),
            "options": case.get("options"),
            "gold_answer": case.get("gold_answer", ""),
            "answer_type": case.get("answer_type", ""),
            "calc_type": case.get("calc_type", []),
            "decimal_places": case.get("decimal_places"),
            "metadata": case.get("metadata", {}),
            "trajectory": traj.get("trajectory", ""),
            "predicted": traj.get("predicted", ""),
            "status": status,
            "judge_reason": label.get("judge_reason", ""),
        })
    pool.sort(key=lambda row: row["id"])
    return pool


def build_idf(pool: list[dict]) -> dict[str, float]:
    df = Counter()
    for item in pool:
        for tok in tokenize(item.get("question", "")):
            df[tok] += 1
    total = max(len(pool), 1)
    return {tok: math.log((1 + total) / (1 + count)) + 1.0 for tok, count in df.items()}


def score_example(target: dict, example: dict, idf: dict[str, float]) -> float:
    target_tokens = tokenize(target.get("question", ""))
    example_tokens = tokenize(example.get("question", ""))
    overlap = target_tokens & example_tokens
    lexical = sum(idf.get(tok, 1.0) for tok in overlap)
    calc_overlap = len(set(target.get("calc_type", [])) & set(example.get("calc_type", [])))
    shared_answer_type = 1.0 if target.get("answer_type") == example.get("answer_type") else 0.0
    target_task = (target.get("metadata") or {}).get("task", "")
    example_task = (example.get("metadata") or {}).get("task", "")
    shared_task = 1.0 if target_task and target_task == example_task else 0.0
    return lexical + 0.75 * calc_overlap + 0.5 * shared_answer_type + 0.75 * shared_task


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def is_usable_failure_example(example: dict) -> bool:
    pred = str(example.get("predicted", "")).strip()
    if not pred:
        return False
    if len(pred) > 120:
        return False
    at = example.get("answer_type")
    if at == "numerical":
        return parse_float(pred) is not None
    return True


def pick_examples(target: dict, pool: list[dict], idf: dict[str, float], shots: int, failure_shots: int, leakage_check: str) -> tuple[list[dict], list[dict]]:
    def is_leak(example: dict) -> bool:
        if leakage_check == "none":
            return False
        same_q = normalize_question(example.get("question", "")) == normalize_question(target.get("question", ""))
        if leakage_check == "question":
            return same_q
        same_a = normalize_answer(example.get("gold_answer", "")) == normalize_answer(target.get("gold_answer", ""))
        return same_q and same_a

    answer_type = target.get("answer_type")
    usable = [item for item in pool if not is_leak(item) and item.get("answer_type") == answer_type]
    success_rows = [item for item in usable if item.get("status") == "success"]
    failure_rows = [item for item in usable if item.get("status") == "failure" and is_usable_failure_example(item)]

    if len(success_rows) < shots:
        fallback = [item for item in pool if not is_leak(item)]
        success_rows = [item for item in fallback if item.get("status") == "success"] or fallback
    success_ranked = sorted(success_rows, key=lambda item: (-score_example(target, item, idf), item["id"]))
    selected_success = success_ranked[:shots]

    selected_failure: list[dict] = []
    if failure_shots > 0:
        if len(failure_rows) < failure_shots:
            fallback = [item for item in pool if not is_leak(item) and item.get("status") == "failure" and is_usable_failure_example(item)]
            failure_rows = fallback
        failure_ranked = sorted(failure_rows, key=lambda item: (-score_example(target, item, idf), item["id"]))
        selected_failure = failure_ranked[:failure_shots]

    return selected_success, selected_failure


def copy_target_images(targets: list[dict], dry_run: bool):
    for row in targets:
        for job in row.pop("_image_jobs"):
            if not job["src"].exists():
                raise FileNotFoundError(f"Missing target image: {job['src']}")
            if dry_run:
                continue
            job["dest_abs"].parent.mkdir(parents=True, exist_ok=True)
            if not job["dest_abs"].exists():
                shutil.copy2(job["src"], job["dest_abs"])


def main():
    args = parse_args()
    if args.failure_shots > 0 and args.only_success_trajectory:
        raise ValueError("--failure-shots > 0 requires not using --only-success-trajectory")

    targets, counts = load_targets(args)
    labels_map = load_label_status(args.labels_path)
    pool = join_example_pool(
        args.cases_path,
        args.trajectories_path,
        args.dataset_name,
        labels_map,
        args.only_success_trajectory,
    )
    idf = build_idf(pool)

    manifests = []
    for target in targets:
        selected_success, selected_failure = pick_examples(
            target,
            pool,
            idf,
            args.shots,
            args.failure_shots,
            args.leakage_check,
        )
        all_selected = selected_success + selected_failure
        manifests.append({
            "target_id": target["id"],
            "target_source": target["source"],
            "answer_type": target["answer_type"],
            "calc_type": target["calc_type"],
            "gold_answer": target["gold_answer"],
            "decimal_places": target["decimal_places"],
            "example_ids": [row["id"] for row in selected_success],
            "failure_example_ids": [row["id"] for row in selected_failure],
            "selection_scores": {
                row["id"]: round(score_example(target, row, idf), 4) for row in all_selected
            },
        })

    status_counts = Counter(row.get("status", "unknown") for row in pool)
    summary = {
        "dataset_name": args.dataset_name,
        "target_count": len(targets),
        "shots": args.shots,
        "failure_shots": args.failure_shots,
        "seed": args.seed,
        "target_source": f"{args.dataset_name}_test",
        "split_counts": counts,
        "example_pool_size": len(pool),
        "example_pool_status": dict(status_counts),
        "only_success_trajectory": args.only_success_trajectory,
        "leakage_check": args.leakage_check,
    }

    count_tag = len(targets)
    targets_path = args.output_dir / f"targets_{args.dataset_name}_{count_tag}.jsonl"
    pool_path = args.output_dir / f"example_pool_{args.dataset_name}_with_trajectories.jsonl"
    manifest_path = args.output_dir / f"manifest_{args.dataset_name}_{count_tag}_{args.shots}s_{args.failure_shots}f.jsonl"
    summary_path = args.output_dir / "prepare_summary.json"

    print(f"Selected {len(targets)} targets from {args.dataset_name} test: {counts}")
    print(f"Joined example pool with trajectories: {len(pool)} items, status={dict(status_counts)}")
    print(f"Will write targets to: {targets_path}")
    print(f"Will write example pool to: {pool_path}")
    print(f"Will write manifest to: {manifest_path}")

    if args.dry_run:
        print("Dry-run complete. No files written.")
        return

    if args.copy_images:
        copy_target_images(targets, dry_run=False)
    else:
        for row in targets:
            row.pop("_image_jobs", None)

    write_jsonl(targets_path, targets)
    write_jsonl(pool_path, pool)
    write_jsonl(manifest_path, manifests)
    write_json(summary_path, summary)
    print("Preparation complete.")


if __name__ == "__main__":
    main()
