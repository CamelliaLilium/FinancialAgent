#!/usr/bin/env python3
"""
RAG 效果对比测试 v2 — 多模态版

从 FinMMR 测试集抽取题目，对比 LLM 在有/无 RAG 条件下的表现。
直接调用 SiliconFlow API (openai 兼容，支持 VL 多模态)，不走 agent 框架。

与 v1 的区别：
  - LLM 接收完整的 context 文本 + 原始图片（多模态输入）
  - context 中的 <image N> 占位符替换为实际图片
  - RAG 示例仍为纯文本 few-shot

测试矩阵：
  A-3 / A-5 / A-7 : 有 RAG (top_k=3/5/7) + context + images + question
  C              : 无 RAG, context + images + question
"""

import base64
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────────────────
API_BASE = "https://api.siliconflow.cn/v1"
API_KEY = "sk-tjpwtwbvgdxacmsbarjyksjlcmtgmvgobpumifwyqzhvlpab"
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
MAX_TOKENS = 4096
TEMPERATURE = 0.0

DATASET_DIR = Path(__file__).parent / "Dataset" / "finmmr"
RAG_PERSIST_DIR = Path(__file__).resolve().parent.parent.parent / "rag" / "chroma_db"
RAG_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "rag" / "models" / "NV-Embed-v2"
)

TOP_K_VALUES = [3, 5, 7]

SELECTED_INDICES = {
    "finmmr_easy_test.json": [0, 5],
    "finmmr_medium_test.json": [0, 10],
    "finmmr_hard_test.json": [0],
}

SYSTEM_MSG = (
    "You are a financial analyst. Solve the given question step by step. "
    "Show your reasoning, then give the final numeric answer."
)


# ── 图片工具 ──────────────────────────────────────────────────────
def load_image_b64(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def make_image_content(b64: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}"},
    }


# ── RAG 格式化 (与 benchmark 脚本一致) ────────────────────────────
def format_rag_example(rec: Dict[str, Any]) -> str:
    q = rec.get("question", "")
    meta = rec.get("metadata", rec)
    ans = meta.get("ground_truth") or meta.get("answer", "")
    py_sol = meta.get("python_solution", "") or meta.get("program", "")
    if py_sol:
        lines = [
            l.strip()
            for l in str(py_sol).split("\n")
            if l.strip() and not l.strip().startswith("#") and "def " not in l
        ]
        steps = []
        for i, line in enumerate(lines[:6], 1):
            if "=" in line or "return " in line:
                steps.append(f"Step {i}: {line}")
        reasoning = (
            " ".join(steps) + f" The answer is {ans}."
            if steps
            else f"The answer is {ans}."
        )
    else:
        reasoning = f"The answer is {ans}."
    return f"Q: {q}\nA: {reasoning}"


def build_rag_text(rag_examples: List[Dict]) -> str:
    if not rag_examples:
        return ""
    blocks = [format_rag_example(ex) for ex in rag_examples]
    return (
        "Reference examples (similar questions with step-by-step reasoning):\n\n"
        + "\n\n".join(blocks)
        + "\n\n---\n\nYour task:\n"
    )


# ── 多模态消息构建 ────────────────────────────────────────────────
def build_multimodal_content(
    question: str,
    context: str,
    image_paths: List[str],
    rag_text: str,
) -> List[dict]:
    """
    构建 openai 多模态 content 数组。

    结构：
      [RAG 文本] → [context 中 <image N> 替换为图片] → [剩余图片] → [question]
    """
    content: List[dict] = []

    # 1) RAG 参考示例（纯文本）
    if rag_text:
        content.append({"type": "text", "text": rag_text})

    # 预加载所有图片 base64
    loaded_images: Dict[int, str] = {}
    img_load_errors: List[str] = []
    for idx, img_path in enumerate(image_paths):
        b64 = load_image_b64(img_path)
        if b64:
            loaded_images[idx + 1] = b64
        else:
            img_load_errors.append(f"image {idx+1}: {img_path} NOT FOUND")

    used_image_ids = set()

    # 2) Context 处理
    if context.strip():
        import re as _re

        placeholder_pattern = _re.compile(r"<image\s+(\d+)>")
        parts = placeholder_pattern.split(context)
        # parts 交替为 [text, img_num, text, img_num, ...]
        for i, part in enumerate(parts):
            if i % 2 == 0:
                text_part = part.strip()
                if text_part:
                    content.append(
                        {"type": "text", "text": f"Financial data context:\n{text_part}"}
                        if i == 0 and not rag_text
                        else {"type": "text", "text": text_part}
                    )
            else:
                img_num = int(part)
                used_image_ids.add(img_num)
                if img_num in loaded_images:
                    content.append(make_image_content(loaded_images[img_num]))
                else:
                    content.append(
                        {"type": "text", "text": f"[Image {img_num} unavailable]"}
                    )

    # 3) 追加未被 context 引用的剩余图片
    for img_num in sorted(loaded_images.keys()):
        if img_num not in used_image_ids:
            content.append(make_image_content(loaded_images[img_num]))

    # 4) 如果没有 context 也没有图片，记录
    if not context.strip() and not loaded_images:
        content.append({"type": "text", "text": "[No context or images provided]"})

    # 5) Question
    content.append({"type": "text", "text": f"Question: {question}"})

    return content, img_load_errors


# ── LLM 调用 (多模态) ────────────────────────────────────────────
def call_llm_multimodal(content: List[dict], client: OpenAI) -> Dict[str, Any]:
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": content},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return {
            "output": text,
            "latency_s": round(time.time() - t0, 2),
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
        }
    except Exception as e:
        return {
            "output": f"[ERROR] {e}",
            "latency_s": round(time.time() - t0, 2),
            "input_tokens": 0,
            "output_tokens": 0,
        }


# ── 答案提取与评判 ────────────────────────────────────────────────
def extract_last_number(text: str) -> Optional[float]:
    """提取 LLM 输出中最后出现的数值（通常是最终答案）"""
    if not text:
        return None
    raw = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    for item in reversed(raw):
        try:
            v = float(item.replace(",", ""))
            if abs(v) > 1e8:
                continue
            return v
        except Exception:
            continue
    return None


def judge(predicted: Optional[float], gold: Any) -> str:
    if predicted is None:
        return "failed"
    try:
        gold_f = float(gold)
    except (ValueError, TypeError):
        return "failed"
    if abs(gold_f) < 1e-9:
        return "correct" if abs(predicted) < 1e-6 else "failed"
    rel_err = abs(predicted - gold_f) / max(abs(gold_f), 1e-9)
    if rel_err < 0.005:
        return "correct"
    if rel_err < 0.05:
        return "close"
    if rel_err < 0.3:
        return "understood"
    return "failed"


# ── 主流程 ─────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("RAG 效果对比测试 v2 — 多模态版 (context + images)")
    print(f"Model: {MODEL}  |  API: {API_BASE}")
    print(f"RAG index: {RAG_PERSIST_DIR}")
    print("=" * 80)

    # 加载 retriever
    import importlib.util

    retriever_path = (
        Path(__file__).resolve().parent.parent.parent / "rag" / "retriever.py"
    )
    spec = importlib.util.spec_from_file_location("_rag_retriever", str(retriever_path))
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    CaseLibraryRetriever = _mod.CaseLibraryRetriever

    retriever = CaseLibraryRetriever(
        persist_dir=str(RAG_PERSIST_DIR), model_path=RAG_MODEL_PATH
    )

    client = OpenAI(base_url=API_BASE, api_key=API_KEY)

    # 加载测试题目
    test_samples: List[Dict[str, Any]] = []
    for filename, indices in SELECTED_INDICES.items():
        path = DATASET_DIR / filename
        if not path.exists():
            print(f"[WARN] {path} not found, skipping")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for idx in indices:
            if idx < len(data):
                sample = data[idx]
                sample["_source_file"] = filename
                sample["_index"] = idx
                diff = (
                    "easy"
                    if "easy" in filename
                    else ("medium" if "medium" in filename else "hard")
                )
                sample["_difficulty"] = diff
                test_samples.append(sample)

    print(f"\n共选取 {len(test_samples)} 道测试题\n")

    # 打印每道题的 context / images 概况
    for si, sample in enumerate(test_samples):
        qid = sample.get("question_id", "?")
        ctx = sample.get("context", "")
        imgs = sample.get("images", [])
        print(
            f"  [{qid}] context={len(ctx)}chars, images={len(imgs)}张"
            + (f" — {[Path(p).name for p in imgs]}" if imgs else "")
        )

    all_results: List[Dict[str, Any]] = []
    conditions = [("C", 0)] + [(f"A-{k}", k) for k in TOP_K_VALUES]

    for si, sample in enumerate(test_samples):
        q = sample.get("question", "")
        gold = sample.get("ground_truth", "")
        diff = sample["_difficulty"]
        qid = sample.get("question_id", f"idx_{sample['_index']}")
        context = sample.get("context", "")
        image_paths = sample.get("images", [])

        print(f"\n{'─' * 70}")
        print(f"[题目 {si + 1}/{len(test_samples)}] {diff.upper()} | id={qid}")
        print(f"Q: {q[:200]}{'...' if len(q) > 200 else ''}")
        print(f"Gold: {gold}")
        print(f"Context: {len(context)} chars | Images: {len(image_paths)}张")

        for cond_name, top_k in conditions:
            # RAG 检索
            if top_k > 0:
                print(f"  [{cond_name}] retrieving top_k={top_k} ...", flush=True)
                examples = retriever.retrieve(q, top_k=top_k, source="finmmr")
                print(
                    f"  [{cond_name}] got {len(examples)} examples, building prompt ...",
                    flush=True,
                )
            else:
                examples = []
                print(f"  [{cond_name}] no RAG, building prompt ...", flush=True)

            rag_text = build_rag_text(examples)

            # 构建多模态 content
            content, img_errors = build_multimodal_content(
                question=q,
                context=context,
                image_paths=image_paths,
                rag_text=rag_text,
            )

            if img_errors:
                for err in img_errors:
                    print(f"  [WARN] {err}")

            n_imgs = sum(1 for c in content if c.get("type") == "image_url")
            n_texts = sum(1 for c in content if c.get("type") == "text")
            print(
                f"  [{cond_name}] content: {n_texts} text + {n_imgs} images, calling LLM ...",
                flush=True,
            )

            result = call_llm_multimodal(content, client)

            predicted_num = extract_last_number(result["output"])
            verdict = judge(predicted_num, gold)

            row = {
                "question_id": qid,
                "difficulty": diff,
                "condition": cond_name,
                "top_k": top_k,
                "gold": gold,
                "predicted": predicted_num,
                "verdict": verdict,
                "latency_s": result["latency_s"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "output_text": result["output"],
                "n_images": n_imgs,
                "img_errors": img_errors,
                "question": q,
            }
            all_results.append(row)

            mark = {"correct": "✓", "close": "≈", "understood": "△", "failed": "✗"}
            print(
                f"  {cond_name:5s} | pred={predicted_num} | {mark.get(verdict, '?')} {verdict:10s} "
                f"| {result['latency_s']}s | in={result['input_tokens']} out={result['output_tokens']}"
            )

    # ── 汇总表 ──────────────────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("汇总对比表 (v2 多模态: context + images)")
    print(f"{'=' * 80}")
    header = f"{'题目ID':<30s} {'难度':<8s} {'Gold':<15s}"
    for cond_name, _ in conditions:
        header += f" {cond_name:>8s}"
    print(header)
    print("─" * len(header))

    for si, sample in enumerate(test_samples):
        qid = sample.get("question_id", f"idx_{sample['_index']}")
        diff = sample["_difficulty"]
        gold = sample.get("ground_truth", "")
        row_str = f"{str(qid):<30s} {diff:<8s} {str(gold):<15s}"
        for cond_name, _ in conditions:
            r = next(
                (
                    x
                    for x in all_results
                    if x["question_id"] == qid and x["condition"] == cond_name
                ),
                None,
            )
            if r:
                mark = {"correct": "✓", "close": "≈", "understood": "△", "failed": "✗"}
                cell = f"{mark.get(r['verdict'], '?')}{r['predicted']}"
                row_str += f" {cell:>8s}"
            else:
                row_str += f" {'N/A':>8s}"
        print(row_str)

    # verdict 统计
    print(f"\n{'─' * 60}")
    print("verdict 统计:")
    for cond_name, _ in conditions:
        rows = [r for r in all_results if r["condition"] == cond_name]
        counts = {}
        for r in rows:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        print(f"  {cond_name:5s}: {counts}")

    # ── 输出详细解题过程 ──────────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("各题目各条件的 LLM 完整输出")
    print(f"{'=' * 80}")

    for si, sample in enumerate(test_samples):
        qid = sample.get("question_id", f"idx_{sample['_index']}")
        q = sample.get("question", "")
        gold = sample.get("ground_truth", "")
        diff = sample["_difficulty"]

        print(f"\n{'━' * 70}")
        print(f"题目 {si + 1}: [{diff.upper()}] {qid}")
        print(f"Q: {q}")
        print(f"Gold Answer: {gold}")

        for cond_name, _ in conditions:
            r = next(
                (
                    x
                    for x in all_results
                    if x["question_id"] == qid and x["condition"] == cond_name
                ),
                None,
            )
            if r:
                mark = {"correct": "✓", "close": "≈", "understood": "△", "failed": "✗"}
                print(f"\n  ── {cond_name} ({mark.get(r['verdict'], '?')} {r['verdict']}) ──")
                print(
                    f"  Predicted: {r['predicted']}  |  Tokens: in={r['input_tokens']} out={r['output_tokens']}"
                )
                if r.get("img_errors"):
                    print(f"  Image errors: {r['img_errors']}")
                print(f"  LLM Output:")
                for line in r["output_text"].split("\n"):
                    print(f"    {line}")

    # ── 保存 JSON 结果 ────────────────────────────────────────────
    out_path = Path(__file__).parent / "test_rag_effect_results_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n\n详细结果已保存至: {out_path}")


if __name__ == "__main__":
    main()
