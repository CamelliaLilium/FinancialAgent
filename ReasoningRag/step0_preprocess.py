#!/usr/bin/env python3
"""
Step 0 — Preprocess FinMMR + FinMME + FinTMM + BizBench into unified cases.jsonl.

Usage:
    python step0_preprocess.py                          # full run with defaults
    python step0_preprocess.py --skip-images            # skip FinMME base64 extraction
    python step0_preprocess.py --raw-dir /data/raw      # custom raw directory
    python step0_preprocess.py -h                       # show all options
"""

import argparse
import base64
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Argument parser — every tuneable parameter lives here
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess FinMMR + FinMME + FinTMM + BizBench -> unified cases.jsonl",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Directories ──
    p.add_argument("--raw-dir", type=Path, default=ROOT / "raw",
                   help="directory containing raw dataset JSONs")
    p.add_argument("--images-dir", type=Path, default=ROOT / "images",
                   help="root directory for extracted/resolved images")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data",
                   help="directory for output cases.jsonl")
    p.add_argument("--output-file", default="cases.jsonl",
                   help="output filename inside --output-dir")

    # ── Image subdirectories (relative to --images-dir) ──
    p.add_argument("--finmmr-image-subdir", default="finmmr",
                   help="subdir under images-dir for FinMMR images")
    p.add_argument("--finmme-image-subdir", default="finmme",
                   help="subdir under images-dir for FinMME images")
    p.add_argument("--fintmm-image-subdir", default="fintmm",
                   help="subdir under images-dir for FinTMM chart images")

    # ── FinMMR ──
    p.add_argument("--finmmr-splits", nargs="+",
                   default=["finmmr_easy_validation.json",
                            "finmmr_medium_validation.json",
                            "finmmr_hard_validation.json"],
                   help="FinMMR split filenames inside --raw-dir")

    # ── FinMME ──
    p.add_argument("--finmme-file", default="finmme_train.json",
                   help="FinMME JSON filename inside --raw-dir")
    p.add_argument("--skip-images", action="store_true",
                   help="skip FinMME base64 extraction (use existing files)")

    # ── FinTMM ──
    p.add_argument("--fintmm-file", default="fintmm_train.json",
                   help="FinTMM JSON filename inside --raw-dir")
    p.add_argument("--fintmm-data-subdir", default="fintmm_data",
                   help="subdir under --raw-dir for FinTMM auxiliary data")
    p.add_argument("--news-text-truncate", type=int, default=500,
                   help="max characters to keep from News text")

    # ── BizBench ──
    p.add_argument("--bizbench-file", default="bizbench_train.json",
                   help="BizBench JSON filename inside --raw-dir")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_decimal_places(gold: str) -> int:
    if "." in gold:
        return len(gold.rstrip("0").split(".")[1]) or 0
    return 0


def _parse_float(s: str) -> float | None:
    cleaned = s.strip().replace(",", "").replace("$", "").replace("%", "").replace("¥", "")
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def classify_answer_type_fintmm(answer: str) -> str:
    a = answer.strip().lower()
    if a in ("yes", "no", "true", "false"):
        return "boolean"
    if _parse_float(answer) is not None:
        return "numerical"
    return "free_text"


def classify_answer_type_generic(answer: str, options=None) -> str:
    a = (answer or "").strip()
    lowered = a.lower()
    if lowered in ("yes", "no", "true", "false"):
        return "boolean"
    if _parse_float(a) is not None:
        return "numerical"
    if options and len(a) == 1 and a.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return "mcq"
    return "free_text"


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


def derive_calc_type_finmme(question_type: str, unit: str) -> list[str]:
    if question_type in ("single_choice", "multiple_choice"):
        return ["reasoning"]
    ratio_keywords = ("%", "percent", "growth", "ratio", "rate", "margin",
                      "yield", "roe", "roa", "p/e", "p/b", "p/s")
    if any(kw in (unit or "").lower() for kw in ratio_keywords):
        return ["ratio"]
    return ["arithmetic"]


def derive_calc_type_program(program: str | None, task: str, question: str, answer_type: str) -> list[str]:
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


# ---------------------------------------------------------------------------
# FinTMM context builder
# ---------------------------------------------------------------------------

class FinTMMContextBuilder:
    """Resolves FinTMM source references to structured text context and image paths."""

    def __init__(self, args: argparse.Namespace):
        self._sp: dict[str, dict] = {}
        self._ft: dict[str, dict] = {}
        self._news: dict[str, dict] = {}
        self._loaded = False
        self._args = args

    def _load(self):
        if self._loaded:
            return
        data_dir = self._args.raw_dir / self._args.fintmm_data_subdir
        for fname, target in [
            ("StockPrice.json", self._sp),
            ("FinancialTable.json", self._ft),
            ("News.json", self._news),
        ]:
            with open(data_dir / fname, encoding="utf-8") as f:
                for item in json.load(f):
                    uid = item.get("uuid", "")
                    if uid:
                        target[uid] = item
        print(f"  FinTMM lookup tables loaded: SP={len(self._sp)}, FT={len(self._ft)}, News={len(self._news)}")
        self._loaded = True

    def build(self, sources: list[str]) -> tuple[str, list[str]]:
        self._load()
        a = self._args
        img_prefix = f"images/{a.fintmm_image_subdir}"
        fintmm_image_dir = a.images_dir / a.fintmm_image_subdir
        ctx: list[str] = []
        imgs: list[str] = []

        for src in sources:
            if src.startswith("Chart_"):
                fname = src.replace("Chart_", "") + ".png"
                if (fintmm_image_dir / fname).is_file():
                    imgs.append(f"{img_prefix}/{fname}")

            elif src.startswith("StockPrice-"):
                item = self._sp.get(src)
                if item:
                    ctx.append(f"{item['Company']}({item['Symbol']}) {item['Date']}: "
                               f"{item['indicator_name']} = {item['indicator_value']} {item['unit']}")
                    self._try_add_kline(item, imgs, fintmm_image_dir, img_prefix)

            elif src.startswith("FinancialTable-"):
                item = self._ft.get(src)
                if item:
                    ctx.append(f"{item['Company']}({item['Symbol']}) {item['Date']}: "
                               f"{item['indicator_name']} = {item['indicator_value']} {item['unit']}")

            elif src.startswith("News-"):
                item = self._news.get(src)
                if item:
                    text = item.get("Text", "")[:a.news_text_truncate]
                    ctx.append(f"[{item.get('Date', '')}] {item.get('Company', '')}: {text}")

        return "\n".join(ctx), imgs

    @staticmethod
    def _try_add_kline(item: dict, image_paths: list[str],
                       fintmm_image_dir: Path, img_prefix: str):
        try:
            d = datetime.strptime(item["Date"], "%Y-%m-%d")
            fname = f"{item['Symbol']}_{d.year}-W{d.isocalendar()[1]}_Kline.png"
            if (fintmm_image_dir / fname).is_file():
                rel = f"{img_prefix}/{fname}"
                if rel not in image_paths:
                    image_paths.append(rel)
        except (KeyError, ValueError):
            pass


# ---------------------------------------------------------------------------
# FinMME image extraction
# ---------------------------------------------------------------------------

def extract_finmme_images(data: list[dict], args: argparse.Namespace) -> dict[int, str]:
    finmme_dir = args.images_dir / args.finmme_image_subdir
    prefix = f"images/{args.finmme_image_subdir}"
    mapping: dict[int, str] = {}

    if args.skip_images:
        for item in data:
            fname = item.get("image", {}).get("path", f"image_{item['id']:06d}.bin")
            if (ROOT / f"{prefix}/{fname}").is_file():
                mapping[item["id"]] = f"{prefix}/{fname}"
        print(f"  FinMME images (skipped extraction): {len(mapping)} already on disk")
        return mapping

    finmme_dir.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(data):
        b64 = item.get("image", {}).get("bytes", "")
        fname = item.get("image", {}).get("path", f"image_{item['id']:06d}.bin")
        if not b64:
            continue
        out_path = finmme_dir / fname
        if not out_path.exists():
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(b64))
        mapping[item["id"]] = f"{prefix}/{fname}"
        if (i + 1) % 2000 == 0:
            print(f"  FinMME images extracted: {i + 1}/{len(data)}")
    print(f"  FinMME images extracted: {len(mapping)}/{len(data)}")
    return mapping


# ---------------------------------------------------------------------------
# Per-dataset processors
# ---------------------------------------------------------------------------

def process_finmmr(args: argparse.Namespace) -> list[dict]:
    img_prefix = f"images/{args.finmmr_image_subdir}"
    records = []
    for name in args.finmmr_splits:
        path = args.raw_dir / name
        if not path.exists():
            print(f"  [SKIP] {name}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            q = item.get("question", "").strip()
            if not q:
                continue
            qid = item.get("question_id", "")
            gold_str = str(item.get("ground_truth"))
            image_paths = []
            for img in item.get("images", []):
                if not img:
                    continue
                rel = f"{img_prefix}/{os.path.basename(img)}"
                if (ROOT / rel).is_file():
                    image_paths.append(rel)
            records.append({
                "id": f"finmmr_{qid}",
                "source": "finmmr",
                "image_paths": image_paths,
                "question": q,
                "context": item.get("context") or "",
                "options": None,
                "gold_answer": gold_str,
                "answer_type": "numerical",
                "calc_type": derive_calc_type_finmmr(item.get("statistics")),
                "decimal_places": get_decimal_places(gold_str),
                "metadata": {
                    "grade": item.get("grade"),
                    "language": item.get("language"),
                    "source_dataset": "FinMMR",
                    "difficulty": item.get("difficulty"),
                    "program": item.get("program") or item.get("python_solution"),
                    "tolerance": None,
                },
            })
    print(f"  FinMMR: {len(records)} records")
    return records


def process_finmme(args: argparse.Namespace) -> list[dict]:
    path = args.raw_dir / args.finmme_file
    if not path.exists():
        print(f"  [SKIP] {args.finmme_file}")
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    img_map = extract_finmme_images(data, args)
    records = []
    for item in data:
        q = item.get("question_text", "").strip()
        if not q:
            continue
        rid = item["id"]
        qt = item.get("question_type", "")
        options = item.get("options", "") or None
        answer = item.get("answer", "")
        answer_type = "mcq" if qt in ("single_choice", "multiple_choice") else "numerical"
        context_parts = []
        if item.get("verified_caption"):
            context_parts.append(item["verified_caption"])
        if item.get("related_sentences"):
            context_parts.append(item["related_sentences"])
        gold_str = str(answer)
        records.append({
            "id": f"finmme_{rid}",
            "source": "finmme",
            "image_paths": [img_map[rid]] if rid in img_map else [],
            "question": q,
            "context": "\n".join(context_parts),
            "options": options,
            "gold_answer": gold_str,
            "answer_type": answer_type,
            "calc_type": derive_calc_type_finmme(qt, item.get("unit", "")),
            "decimal_places": get_decimal_places(gold_str) if answer_type == "numerical" else None,
            "metadata": {
                "source_dataset": "FinMME",
                "question_type": qt,
                "unit": item.get("unit"),
                "tolerance": item.get("tolerance"),
            },
        })
    print(f"  FinMME: {len(records)} records")
    return records


def process_fintmm(args: argparse.Namespace) -> list[dict]:
    path = args.raw_dir / args.fintmm_file
    if not path.exists():
        print(f"  [SKIP] {args.fintmm_file}")
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    ctx_builder = FinTMMContextBuilder(args)
    records = []
    for item in data:
        q = item.get("question", "").strip()
        if not q:
            continue
        uuid = item.get("uuid", "")
        answers_list = item.get("answers", [])
        if not answers_list:
            continue
        first_ans = answers_list[0]
        answer_text = str(first_ans.get("answer", ""))
        context, image_paths = ctx_builder.build(first_ans.get("source", []))
        answer_type = classify_answer_type_fintmm(answer_text)
        gold_str = answer_text.strip()
        subtask = item.get("subtask", [])
        if isinstance(subtask, str):
            subtask = [subtask]
        records.append({
            "id": f"fintmm_{uuid}",
            "source": "fintmm",
            "image_paths": image_paths,
            "question": q,
            "context": context,
            "options": None,
            "gold_answer": gold_str,
            "answer_type": answer_type,
            "calc_type": subtask,
            "decimal_places": get_decimal_places(gold_str) if answer_type == "numerical" else None,
            "metadata": {
                "source_dataset": "FinTMM",
                "type": item.get("type"),
                "subtask": subtask,
                "explanation": first_ans.get("explanation", ""),
                "tolerance": None,
            },
        })
    print(f"  FinTMM: {len(records)} records")
    return records


def process_bizbench(args: argparse.Namespace) -> list[dict]:
    path = args.raw_dir / args.bizbench_file
    if not path.exists():
        print(f"  [SKIP] {args.bizbench_file}")
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for idx, item in enumerate(data):
        question = (item.get("question") or "").strip()
        if not question:
            continue

        answer = str(item.get("answer", "")).strip()
        options = item.get("options")
        answer_type = classify_answer_type_generic(answer, options)
        program = item.get("program")
        task = str(item.get("task") or "")

        records.append({
            "id": f"bizbench_train_{idx}",
            "source": "bizbench",
            "image_paths": [],
            "question": question,
            "context": item.get("context") or "",
            "options": options,
            "gold_answer": answer,
            "answer_type": answer_type,
            "calc_type": derive_calc_type_program(program, task, question, answer_type),
            "decimal_places": get_decimal_places(answer) if answer_type == "numerical" else None,
            "metadata": {
                "source_dataset": "BizBench",
                "task": task,
                "context_type": item.get("context_type"),
                "program": program,
                "tolerance": None,
            },
        })

    print(f"  BizBench: {len(records)} records")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 60)
    print("Step 0: Preprocessing datasets → cases.jsonl")
    print("=" * 60)

    print("\n[1/4] Processing FinMMR...")
    finmmr = process_finmmr(args)
    print("\n[2/4] Processing FinMME...")
    finmme = process_finmme(args)
    print("\n[3/4] Processing FinTMM...")
    fintmm = process_fintmm(args)
    print("\n[4/4] Processing BizBench...")
    bizbench = process_bizbench(args)

    all_records = finmmr + finmme + fintmm + bizbench

    before = len(all_records)
    all_records = [r for r in all_records if "test" not in r["id"].lower()]
    rejected = before - len(all_records)
    if rejected:
        print(f"\n[SAFETY] Rejected {rejected} records containing 'test' in id")

    seen: set[str] = set()
    deduped = []
    for r in all_records:
        if r["id"] not in seen:
            seen.add(r["id"])
            deduped.append(r)
    if len(deduped) < len(all_records):
        print(f"[DEDUP] Removed {len(all_records) - len(deduped)} duplicate ids")
    all_records = deduped

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / args.output_file
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    src_counts = Counter(r["source"] for r in all_records)
    at_counts = Counter(r["answer_type"] for r in all_records)
    img_counts = sum(1 for r in all_records if r["image_paths"])
    print(f"\n{'=' * 60}")
    print(f"Output: {out_path}")
    print(f"Total records: {len(all_records)}")
    print(f"By source:      {dict(src_counts)}")
    print(f"By answer_type: {dict(at_counts)}")
    print(f"With images:    {img_counts}/{len(all_records)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
