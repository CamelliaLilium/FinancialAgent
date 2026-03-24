#!/usr/bin/env python3
"""
Minimal experience retriever.

Stage 1: filter by answer_type + calc_type on index_meta.json
Stage 2: semantic search over FAISS and return top-1 matching experience
"""

import argparse
import json
from pathlib import Path

import faiss

from step4_build_index import QUERY_INSTRUCTION, encode_queries, encode_queries_hash, load_nvembed


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrieve top-1 experience with Stage1 filter + semantic search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--bank-file", default="reasoningbank.json")
    p.add_argument("--index-file", default="index.faiss")
    p.add_argument("--index-meta-file", default="index_meta.json")
    p.add_argument("--embed-model-path", type=Path, default=ROOT / "NV-Embed-v2")
    p.add_argument("--query", required=True)
    p.add_argument("--answer-type", default="")
    p.add_argument("--calc-type", nargs="*", default=[])
    p.add_argument("--embed-backend", choices=["auto", "nvembed", "hash"], default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--search-k", type=int, default=200)
    return p.parse_args()


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_backend(requested: str, index_meta: dict) -> str:
    if requested != "auto":
        return requested
    return "hash" if index_meta.get("embedding_model") == "hash-v1" else "nvembed"


def stage1_filter(rows: list[dict], answer_type: str, calc_types: set[str]) -> set[int]:
    allowed = set()
    for row in rows:
        if answer_type and row.get("answer_type") != answer_type:
            continue
        row_calc = set(row.get("calc_type", []))
        if calc_types and not (calc_types & row_calc):
            continue
        allowed.add(int(row["faiss_row"]))
    return allowed


def encode_query(query: str, backend: str, model_path: Path, batch_size: int, max_length: int):
    if backend == "hash":
        return encode_queries_hash([query])
    model, device = load_nvembed(model_path)
    print(f"NV-Embed-v2 loaded on {device}")
    return encode_queries(model, [query], batch_size=batch_size, max_length=max_length)


def main() -> None:
    args = parse_args()
    bank = read_json(args.data_dir / args.bank_file)
    index_meta = read_json(args.data_dir / args.index_meta_file)
    index = faiss.read_index(str(args.data_dir / args.index_file))

    backend = resolve_backend(args.embed_backend, index_meta)
    allowed = stage1_filter(index_meta["rows"], args.answer_type, set(args.calc_type))
    if not allowed:
        allowed = {int(r["faiss_row"]) for r in index_meta["rows"]}

    query_vec = encode_query(args.query, backend, args.embed_model_path, args.batch_size, args.max_length)
    search_k = min(max(1, args.search_k), index.ntotal)
    scores, ids = index.search(query_vec, search_k)

    hit_row = None
    hit_score = None
    for score, row_id in zip(scores[0], ids[0]):
        row_id = int(row_id)
        if row_id in allowed:
            hit_row = row_id
            hit_score = float(score)
            break

    result = {
        "query": args.query,
        "embed_backend": backend,
        "allowed_count": len(allowed),
        "top1_row": hit_row,
        "top1_score": hit_score,
        "top1_experience": bank[hit_row] if hit_row is not None else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
