#!/usr/bin/env python3
"""
Prepare a 100-question A/B test set for answer-only vs answer+trajectory few-shot prompts.

Usage:
    python prepare_ab_test_data.py
    python prepare_ab_test_data.py --target-count 100 --shots 3
    python prepare_ab_test_data.py --dry-run
"""

import argparse
import json
import math
import random
import re
import shutil
from collections import Counter
from pathlib import Path

try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:
    _NUMPY_OK = False

ROOT = Path(__file__).resolve().parent

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "the", "to", "was", "what", "when",
    "which", "with", "would", "year", "years",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare normalized 100-question A/B test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--test-root", type=Path, default=ROOT.parent / "datasets" / "test",
                   help="directory containing held-out test datasets")
    p.add_argument("--finmmr-image-root", type=Path,
                   default=ROOT.parent / "datasets" / "FinMMR" / "images",
                   help="directory containing FinMMR test images")
    p.add_argument("--cases-path", type=Path, default=ROOT / "data" / "cases.jsonl",
                   help="existing few-shot case pool")
    p.add_argument("--trajectories-path", type=Path,
                   default=ROOT / "data" / "trajectories.jsonl",
                   help="existing trajectory pool")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data" / "ab_test",
                   help="output directory for normalized targets and manifests")
    p.add_argument("--images-subdir", default="test_targets/finmmr",
                   help="subdir under ReasoningRag/images for copied target images")
    p.add_argument("--target-count", type=int, default=100,
                   help="number of held-out target questions")
    p.add_argument("--shots", type=int, default=3,
                   help="examples per target question")
    p.add_argument("--seed", type=int, default=20260318,
                   help="random seed for sampling")
    p.add_argument("--copy-images", action="store_true", default=True,
                   help="copy selected target images into ReasoningRag/images")
    p.add_argument("--dry-run", action="store_true",
                   help="show planned outputs without writing files")
    p.add_argument("--retrieval", choices=["idf", "faiss"], default="idf",
                   help="example selection: idf (lexical) or faiss (NV-Embed-v2 semantic, A1 experiment)")
    p.add_argument("--embed-model-path", type=Path, default=ROOT / "NV-Embed-v2",
                   help="NV-Embed-v2 model path (only used when --retrieval faiss)")
    p.add_argument("--embed-batch-size", type=int, default=8,
                   help="encoding batch size for NV-Embed-v2")
    return p.parse_args()


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
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


def join_example_pool(cases_path: Path, trajectories_path: Path) -> list[dict]:
    cases = {row["id"]: row for row in read_jsonl(cases_path)}
    trajectories = {row["id"]: row for row in read_jsonl(trajectories_path)}
    pool = []
    for case_id, traj in trajectories.items():
        case = cases.get(case_id)
        if not case:
            continue
        if case.get("source") != "finmmr":
            continue
        if not traj.get("trajectory") or traj["trajectory"].startswith("ERROR:"):
            continue
        pool.append({
            "id": case_id,
            "source": case.get("source"),
            "image_paths": case.get("image_paths", []),
            "question": case.get("question", ""),
            "options": case.get("options"),
            "gold_answer": case.get("gold_answer", ""),
            "answer_type": case.get("answer_type", ""),
            "calc_type": case.get("calc_type", []),
            "decimal_places": case.get("decimal_places"),
            "trajectory": traj.get("trajectory", ""),
            "predicted": traj.get("predicted", ""),
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
    return lexical + 0.75 * calc_overlap + 0.5 * shared_answer_type


def build_pool_faiss_index(pool: list[dict], model_path: Path, batch_size: int):
    """Encode pool questions with NV-Embed-v2 and return (faiss_index, model)."""
    if not _NUMPY_OK:
        raise RuntimeError("numpy is required for --retrieval faiss")
    import faiss
    from step4_build_index import load_nvembed, encode_queries
    print(f"Loading NV-Embed-v2 from {model_path} …")
    model, device = load_nvembed(model_path)
    questions = [item.get("question", "") for item in pool]
    print(f"  Encoding {len(questions)} pool questions on {device} …")
    vecs = encode_queries(model, questions, batch_size=batch_size, max_length=512)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    print(f"  Pool FAISS index built: {index.ntotal} vectors, dim={vecs.shape[1]}")
    return index, model


def pick_examples_faiss(
    target: dict, pool: list[dict], faiss_index, model, batch_size: int, shots: int
) -> list[dict]:
    """Semantic example selection via NV-Embed-v2 + answer_type/calc_type bonus."""
    from step4_build_index import encode_queries
    target_vec = encode_queries(model, [target.get("question", "")], batch_size=1, max_length=512)
    k = min(shots * 6, faiss_index.ntotal)
    scores, ids = faiss_index.search(target_vec, k)
    answer_type = target.get("answer_type")
    calc_types = set(target.get("calc_type", []))
    candidates = []
    for row_id, score in zip(ids[0], scores[0]):
        item = pool[int(row_id)]
        bonus = (0.1 if item.get("answer_type") == answer_type else 0.0)
        bonus += (0.05 if calc_types & set(item.get("calc_type", [])) else 0.0)
        candidates.append((item, float(score) + bonus))
    candidates.sort(key=lambda x: -x[1])
    return [item for item, _ in candidates[:shots]]


def pick_examples(target: dict, pool: list[dict], idf: dict[str, float], shots: int) -> list[dict]:
    answer_type = target.get("answer_type")
    calc_types = set(target.get("calc_type", []))

    candidates = [item for item in pool if item.get("answer_type") == answer_type]
    strict = [item for item in candidates if calc_types & set(item.get("calc_type", []))]
    chosen_pool = strict if len(strict) >= shots else candidates
    if len(chosen_pool) < shots:
        chosen_pool = pool

    ranked = sorted(
        chosen_pool,
        key=lambda item: (-score_example(target, item, idf), item["id"]),
    )
    return ranked[:shots]


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
    targets, counts = load_finmmr_targets(args)
    pool = join_example_pool(args.cases_path, args.trajectories_path)

    if args.retrieval == "faiss":
        faiss_index, embed_model = build_pool_faiss_index(
            pool, args.embed_model_path, args.embed_batch_size
        )
        idf = None
    else:
        idf = build_idf(pool)
        faiss_index = embed_model = None

    tag = f"_faiss" if args.retrieval == "faiss" else ""
    manifests = []
    for target in targets:
        if args.retrieval == "faiss":
            selected = pick_examples_faiss(
                target, pool, faiss_index, embed_model, args.embed_batch_size, args.shots
            )
            scores_dict: dict = {}
        else:
            selected = pick_examples(target, pool, idf, args.shots)
            scores_dict = {row["id"]: round(score_example(target, row, idf), 4) for row in selected}
        manifests.append({
            "target_id": target["id"],
            "target_source": target["source"],
            "answer_type": target["answer_type"],
            "calc_type": target["calc_type"],
            "gold_answer": target["gold_answer"],
            "decimal_places": target["decimal_places"],
            "example_ids": [row["id"] for row in selected],
            "selection_scores": scores_dict,
            "retrieval": args.retrieval,
        })

    summary = {
        "target_count": len(targets),
        "shots": args.shots,
        "seed": args.seed,
        "retrieval": args.retrieval,
        "target_source": "finmmr_test",
        "split_counts": counts,
        "example_pool_size": len(pool),
    }

    targets_path = args.output_dir / "targets_finmmr_100.jsonl"
    pool_path = args.output_dir / "example_pool_finmmr_with_trajectories.jsonl"
    manifest_path = args.output_dir / f"manifest_finmmr_100_{args.shots}shot{tag}.jsonl"
    summary_path = args.output_dir / f"prepare_summary{tag}.json"

    print(f"Selected {len(targets)} targets from FinMMR test: {counts}")
    print(f"Joined example pool with trajectories: {len(pool)} items")
    print(f"Will write targets to: {targets_path}")
    print(f"Will write example pool to: {pool_path}")
    print(f"Will write manifest to: {manifest_path}")

    if args.dry_run:
        print("Dry-run complete. No files written.")
        return

    copy_target_images(targets, dry_run=False)
    write_jsonl(targets_path, targets)
    write_jsonl(pool_path, pool)
    write_jsonl(manifest_path, manifests)
    write_json(summary_path, summary)
    print("Preparation complete.")


if __name__ == "__main__":
    main()
