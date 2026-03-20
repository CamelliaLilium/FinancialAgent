#!/usr/bin/env python3
"""
Step 4 - Build experience-level ReasoningBank and FAISS index.

Usage:
    python step4_build_index.py
    python step4_build_index.py --embed-model-path NV-Embed-v2 --batch-size 8
"""

import argparse
import json
import re
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel


ROOT = Path(__file__).resolve().parent
QUERY_INSTRUCTION = "Given a financial question, retrieve relevant passages that answer the query"
CORPUS_INSTRUCTION = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build experience-level bank + FAISS index with NV-Embed-v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--cases-file", default="cases.jsonl")
    p.add_argument("--trajectories-file", default="trajectories.jsonl")
    p.add_argument("--labels-file", default="labels.jsonl")
    p.add_argument("--items-file", default="items.jsonl")
    p.add_argument("--bank-file", default="reasoningbank.json")
    p.add_argument("--index-file", default="index.faiss")
    p.add_argument("--index-meta-file", default="index_meta.json")
    p.add_argument("--embed-model-path", type=Path, default=ROOT / "NV-Embed-v2")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def make_experience_id(source_question_id: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", source_question_id)
    return f"exp_{clean}"


def build_reasoningbank(
    cases: dict[str, dict],
    trajectories: dict[str, dict],
    labels: dict[str, dict],
    items_by_source: dict[str, list[dict]],
) -> list[dict]:
    experiences = []

    ids = sorted(set(cases) & set(trajectories) & set(labels))
    for cid in ids:
        if "test" in cid.lower():
            continue

        case = cases[cid]
        traj = trajectories[cid]
        label = labels[cid]
        calc_type = case.get("calc_type") if isinstance(case.get("calc_type"), list) else []

        metadata = case.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        raw_items = items_by_source.get(cid, [])
        memory_items = []
        for item in raw_items:
            memory_items.append(
                {
                    "item_id": item.get("item_id", ""),
                    "source_question_id": item.get("source_question_id", cid),
                    "source": item.get("source", ""),
                    "memory_type": item.get("memory_type", ""),
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "content": item.get("content", ""),
                    "failure_type": item.get("failure_type"),
                }
            )

        exp = {
            "experience_id": make_experience_id(cid),
            "source_question_id": cid,
            "query": case.get("question", ""),
            "trajectory": traj.get("trajectory", ""),
            "status": label.get("status", "failure"),
            "judge_source": label.get("judge_source", ""),
            "metadata": {
                "source": case.get("source", ""),
                "answer_type": case.get("answer_type", ""),
                "calc_type": calc_type,
                "decimal_places": case.get("decimal_places"),
                "tolerance": metadata.get("tolerance"),
            },
            "memory_items": memory_items,
        }
        experiences.append(exp)

    return experiences


def load_nvembed(model_path: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True, torch_dtype=dtype)
    if device == "cuda":
        model = model.half()
    model = model.to(device)
    model.eval()
    return model, device


def encode_queries(model, texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, instruction=QUERY_INSTRUCTION, max_length=max_length)
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().float().numpy()
        emb = emb.astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        emb = emb / norms
        vectors.append(emb)
        print(f"Encoded {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.vstack(vectors)


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def main() -> None:
    args = parse_args()

    cases_path = args.data_dir / args.cases_file
    traj_path = args.data_dir / args.trajectories_file
    labels_path = args.data_dir / args.labels_file
    items_path = args.data_dir / args.items_file

    bank_path = args.data_dir / args.bank_file
    index_path = args.data_dir / args.index_file
    index_meta_path = args.data_dir / args.index_meta_file

    cases = {r["id"]: r for r in read_jsonl(cases_path)}
    trajectories = {r["id"]: r for r in read_jsonl(traj_path)}
    labels = {r["id"]: r for r in read_jsonl(labels_path)}
    items = read_jsonl(items_path) if items_path.exists() else []

    items_by_source: dict[str, list[dict]] = {}
    for row in items:
        sid = row.get("source_question_id")
        if not sid:
            continue
        items_by_source.setdefault(sid, []).append(row)

    experiences = build_reasoningbank(cases, trajectories, labels, items_by_source)
    if not experiences:
        raise RuntimeError("No experiences to index. Check inputs of Step 2/3.")

    print(f"Built experiences: {len(experiences)}")
    write_json(bank_path, experiences)

    model, device = load_nvembed(args.embed_model_path)
    print(f"NV-Embed-v2 loaded on {device}")

    queries = [exp["query"] for exp in experiences]
    vectors = encode_queries(model, queries, batch_size=args.batch_size, max_length=args.max_length)

    index = build_faiss_index(vectors)
    faiss.write_index(index, str(index_path))

    rows = []
    for i, exp in enumerate(experiences):
        rows.append(
            {
                "faiss_row": i,
                "experience_id": exp["experience_id"],
                "source_question_id": exp["source_question_id"],
                "answer_type": exp["metadata"].get("answer_type", ""),
                "calc_type": exp["metadata"].get("calc_type", []),
            }
        )

    index_meta = {
        "version": "v1",
        "retrieval_unit": "experience",
        "embedding_model": "NV-Embed-v2",
        "query_instruction": QUERY_INSTRUCTION,
        "corpus_instruction": CORPUS_INSTRUCTION,
        "vector_dim": int(vectors.shape[1]),
        "metric": "inner_product",
        "normalized": True,
        "total_experiences": len(experiences),
        "rows": rows,
    }
    write_json(index_meta_path, index_meta)

    print("Done.")
    print(f"ReasoningBank: {bank_path}")
    print(f"FAISS index: {index_path}")
    print(f"Index meta:  {index_meta_path}")


if __name__ == "__main__":
    main()
