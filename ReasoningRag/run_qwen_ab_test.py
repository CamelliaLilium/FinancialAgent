#!/usr/bin/env python3
"""
Run paired A/B few-shot evaluation for answer-only vs answer+trajectory prompts.

Uses DashScope OpenAI-compatible API for Qwen3-VL-8B-Instruct. API key is read
from QWEN_API_KEY in .env (ReasoningRag/.env); add it manually.

Usage:
    python run_qwen_ab_test.py --backend dry-run
    python run_qwen_ab_test.py --backend openai
"""

import argparse
import base64
import io
import json
import os
import random
import re
import time
from pathlib import Path

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from PIL import Image
except ImportError:
    Image = None

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

SYSTEM_PROMPT = (
    "You are a financial reasoning expert. Study the worked examples carefully, "
    "then solve the target problem. Use the examples as references for method, "
    "but answer the target question independently. You must end with exactly one line: "
    "**Final Answer:** <your answer>"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run paired A/B few-shot test for answer-only vs answer+trajectory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--targets-path", type=Path,
                   default=ROOT / "data" / "ab_test" / "targets_finmmr_100.jsonl",
                   help="normalized target set")
    p.add_argument("--example-pool-path", type=Path,
                   default=ROOT / "data" / "ab_test" / "example_pool_finmmr_with_trajectories.jsonl",
                   help="example pool with trajectories")
    p.add_argument("--manifest-path", type=Path,
                   default=ROOT / "data" / "ab_test" / "manifest_finmmr_100_3shot.jsonl",
                   help="paired example selection manifest")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "data" / "ab_test" / "results",
                   help="directory for per-mode results and summaries")
    p.add_argument("--model", default="qwen3-vl-8b-instruct",
                   help="model name passed to the backend")
    p.add_argument("--backend", choices=["openai", "dry-run"], default="dry-run",
                   help="inference backend")
    p.add_argument("--api-base", default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                   help="OpenAI-compatible API base URL (DashScope)")
    p.add_argument("--mode", choices=["baseline", "treatment", "both"], default="both",
                   help="which branch of the A/B test to run")
    p.add_argument("--limit", type=int, default=0,
                   help="run only the first N targets from the manifest")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="generation temperature")
    p.add_argument("--top-p", type=float, default=1.0,
                   help="generation top_p")
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="maximum output tokens")
    p.add_argument("--trajectory-char-limit", type=int, default=4000,
                   help="truncate each example trajectory to this many characters (0 disables)")
    p.add_argument("--example-max-images", type=int, default=1,
                   help="maximum images to include per example (0 keeps all)")
    p.add_argument("--target-max-images", type=int, default=0,
                   help="maximum images to include for the target problem (0 keeps all)")
    p.add_argument("--image-max-side", type=int, default=768,
                   help="resize images so the longest side is at most this many pixels (0 disables)")
    p.add_argument("--image-jpeg-quality", type=int, default=85,
                   help="JPEG quality when re-encoding images")
    p.add_argument("--image-detail", choices=["auto", "low", "high"], default="auto",
                   help="detail level sent for each image_url block")
    p.add_argument("--request-retries", type=int, default=3,
                   help="retry count for transient API failures")
    p.add_argument("--retry-base-delay", type=float, default=2.0,
                   help="initial retry backoff delay in seconds")
    p.add_argument("--retry-max-delay", type=float, default=30.0,
                   help="maximum retry backoff delay in seconds")
    p.add_argument("--resume", action="store_true",
                   help="append only missing targets for each mode")
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


def get_openai_client(args: argparse.Namespace):
    if OpenAI is None:
        raise RuntimeError("Missing dependency `openai`. Install it with `pip install openai`.")

    api_key = os.getenv("QWEN_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing API key. Set QWEN_API_KEY in .env (ReasoningRag/.env).")

    return OpenAI(api_key=api_key, base_url=args.api_base)


def probe_openai_backend(args: argparse.Namespace):
    if args.backend != "openai":
        return

    client = get_openai_client(args)
    try:
        models = client.models.list()
    except Exception as e:
        raise RuntimeError(
            "Cannot reach the OpenAI-compatible backend.\n"
            f"- endpoint: {args.api_base.rstrip('/')}/models\n"
            f"- cause: {e}\n"
            "- if you want DashScope, use `--api-base https://dashscope.aliyuncs.com/compatible-mode/v1`"
        ) from e

    model_ids = [item.id for item in getattr(models, "data", []) if getattr(item, "id", None)]
    print(f"Connected to backend: {args.api_base.rstrip('/')}/models")
    if model_ids:
        preview = ", ".join(model_ids[:8])
        print(f"Available models: {preview}{' ...' if len(model_ids) > 8 else ''}")
        if args.model not in model_ids:
            print(
                "Warning: requested model id is not listed by backend. "
                f"Requested `{args.model}`."
            )


def extract_final_answer(text: str) -> str:
    match = re.search(r"\*\*Final Answer:\*\*\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().splitlines()[0].rstrip(".")
    match = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().splitlines()[0].rstrip(".")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def clean_trajectory(text: str, char_limit: int) -> str:
    cleaned = re.sub(r"\*\*Final Answer:\*\*.*", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    cleaned = re.sub(r"\\boxed\{([^}]+)\}", r"\1", cleaned)
    if char_limit > 0 and len(cleaned) > char_limit:
        cleaned = cleaned[:char_limit].rstrip() + "\n[Trajectory truncated]"
    return cleaned


def parse_float(value: str) -> float | None:
    cleaned = (value or "").strip().replace(",", "")
    cleaned = cleaned.replace("$", "").replace("%", "")
    cleaned = cleaned.replace("USD", "").replace("TWD", "")
    cleaned = cleaned.replace("usd", "").replace("twd", "")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
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
    gold = record.get("gold_answer", "")
    answer_type = record.get("answer_type")
    if answer_type == "numerical":
        gold_num = parse_float(str(gold))
        pred_num = parse_float(predicted)
        return gold_num is not None and pred_num is not None and abs(gold_num - pred_num) <= 1e-9
    if answer_type == "mcq":
        return normalize_mcq(str(gold)) == normalize_mcq(predicted)
    if answer_type == "boolean":
        return normalize_boolean(str(gold)) == normalize_boolean(predicted)
    return str(gold).strip().lower() == predicted.strip().lower()


def encode_image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def select_image_paths(paths: list[str], max_images: int) -> list[str]:
    if max_images and max_images > 0:
        return paths[:max_images]
    return paths


def encode_optimized_image_data_url(path: Path, args: argparse.Namespace, cache: dict[tuple[str, int, int], str]) -> str:
    cache_key = (str(path), args.image_max_side, args.image_jpeg_quality)
    if cache_key in cache:
        return cache[cache_key]

    if args.image_max_side <= 0 or Image is None:
        data_url = encode_image_data_url(path)
        cache[cache_key] = data_url
        return data_url

    with Image.open(path) as img:
        img = img.convert("RGBA")
        if max(img.size) > args.image_max_side:
            img.thumbnail((args.image_max_side, args.image_max_side))
        background = Image.new("RGB", img.size, "white")  # type: ignore[arg-type]
        background.paste(img, mask=img.split()[-1])
        buf = io.BytesIO()
        background.save(buf, format="JPEG", quality=args.image_jpeg_quality, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    data_url = f"data:image/jpeg;base64,{encoded}"
    cache[cache_key] = data_url
    return data_url


def build_image_block(url: str, detail: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {
            "url": url,
            "detail": detail,
        },
    }


def build_user_content(
    target: dict,
    examples: list[dict],
    mode: str,
    trajectory_char_limit: int,
    args: argparse.Namespace,
    image_cache: dict[tuple[str, int, int], str],
) -> list[dict]:
    content = []
    content.append({
        "type": "text",
        "text": (
            "You will see three worked examples, then one target problem. "
            "Each example contains its question, image, and gold answer. "
            "In the treatment setting, each example also contains a trajectory."
        ),
    })

    for idx, example in enumerate(examples, start=1):
        text = [
            f"=== Example {idx} ===",
            f"Question: {example['question']}",
        ]
        if example.get("options"):
            text.append(f"Options: {example['options']}")
        text.append(f"Answer: {example['gold_answer']}")
        if mode == "treatment":
            trajectory = clean_trajectory(example.get("trajectory", ""), trajectory_char_limit)
            text.append("Trajectory:")
            text.append(trajectory)
        content.append({"type": "text", "text": "\n".join(text)})
        for rel_path in select_image_paths(example.get("image_paths", []), args.example_max_images):
            image_path = ROOT / rel_path
            if not image_path.exists():
                raise FileNotFoundError(f"Missing example image: {image_path}")
            content.append(build_image_block(
                encode_optimized_image_data_url(image_path, args, image_cache),
                args.image_detail,
            ))

    target_text = [
        "=== Target Problem ===",
        f"Question: {target['question']}",
    ]
    if target.get("options"):
        target_text.append(f"Options: {target['options']}")
    target_text.append("Solve the target problem. Do not repeat the examples. End with `**Final Answer:** <answer>`.")
    content.append({"type": "text", "text": "\n".join(target_text)})
    for rel_path in select_image_paths(target.get("image_paths", []), args.target_max_images):
        image_path = ROOT / rel_path
        if not image_path.exists():
            raise FileNotFoundError(f"Missing target image: {image_path}")
        content.append(build_image_block(
            encode_optimized_image_data_url(image_path, args, image_cache),
            args.image_detail,
        ))
    return content


def with_image_detail(user_content: list[dict], detail: str) -> list[dict]:
    if not detail:
        return user_content

    updated = []
    for item in user_content:
        if item.get("type") != "image_url":
            updated.append(item)
            continue
        image_url = dict(item.get("image_url", {}))
        image_url["detail"] = detail
        updated.append({"type": "image_url", "image_url": image_url})
    return updated


def build_chat_payload(args: argparse.Namespace, user_content: list[dict]) -> dict:
    return {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }


def compute_retry_delay(args: argparse.Namespace, attempt: int) -> float:
    base = min(args.retry_max_delay, args.retry_base_delay * (2 ** max(0, attempt - 1)))
    jitter = base * random.uniform(0.1, 0.3)
    return min(args.retry_max_delay, base + jitter)


def extract_trace_id(headers) -> str:
    if headers is None:
        return ""
    return headers.get("x-siliconcloud-trace-id", "") or headers.get("x-request-id", "") or ""


def parse_openai_exception(err: Exception) -> tuple[int | None, str, str]:
    status_code = getattr(err, "status_code", None)
    detail = str(err)
    body = getattr(err, "body", None)
    if body is not None:
        detail = str(body)
    response = getattr(err, "response", None)
    trace_id = extract_trace_id(getattr(response, "headers", None))
    return status_code, detail, trace_id


def format_api_error(prefix: str, detail: str, trace_id: str) -> str:
    message = f"{prefix}: {detail}"
    if trace_id:
        message += f" [trace_id={trace_id}]"
    return message


def call_openai_chat(args: argparse.Namespace, user_content: list[dict]) -> tuple[str, str]:
    client = get_openai_client(args)
    last_error = None
    last_trace_id = ""
    for attempt in range(1, args.request_retries + 1):
        detail = args.image_detail
        if attempt > 1 and detail != "low":
            detail = "low"

        payload = build_chat_payload(args, with_image_detail(user_content, detail))
        try:
            completion = client.chat.completions.create(**payload, timeout=600)
            last_trace_id = getattr(completion, "_request_id", "") or last_trace_id
            message = completion.choices[0].message.content
            if isinstance(message, str):
                return message, last_trace_id
            if isinstance(message, list):
                parts = []
                for item in message:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                return "\n".join(parts).strip(), last_trace_id
            return str(message), last_trace_id
        except Exception as e:
            status_code, detail, trace_id = parse_openai_exception(e)
            last_trace_id = trace_id or last_trace_id
            prefix = f"HTTP {status_code}" if status_code is not None else "Request failed"
            last_error = RuntimeError(format_api_error(prefix, detail, trace_id))
            if status_code is not None and status_code >= 500 and attempt < args.request_retries:
                time.sleep(compute_retry_delay(args, attempt))
                continue
            if status_code is None and attempt < args.request_retries:
                time.sleep(compute_retry_delay(args, attempt))
                continue
            raise last_error from e
    else:
        raise last_error or RuntimeError("Unknown request failure")

def load_completed_ids(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    for row in read_jsonl(path):
        done.add(row["target_id"])
    return done


def run_mode(mode: str, args: argparse.Namespace, targets: dict[str, dict], examples: dict[str, dict], manifests: list[dict]) -> Path:
    output_path = args.output_dir / f"{mode}_results.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_ids(output_path) if args.resume else set()
    image_cache: dict[tuple[str, int, int], str] = {}

    pending = [row for row in manifests if row["target_id"] not in completed]
    print(f"Running {mode}: {len(pending)} pending / {len(manifests)} total")

    open_mode = "a" if args.resume else "w"
    with open(output_path, open_mode, encoding="utf-8") as out:
        for idx, manifest in enumerate(pending, start=1):
            target = targets[manifest["target_id"]]
            selected_examples = [examples[eid] for eid in manifest["example_ids"]]
            started = time.time()
            error = ""
            trace_id = ""
            try:
                user_content = build_user_content(
                    target,
                    selected_examples,
                    mode,
                    args.trajectory_char_limit,
                    args,
                    image_cache,
                )
                if args.backend == "dry-run":
                    response_text = "**Final Answer:** DRY_RUN"
                else:
                    response_text, trace_id = call_openai_chat(args, user_content)
                predicted = extract_final_answer(response_text)
                correct = is_correct(target, predicted)
            except Exception as e:
                response_text = ""
                predicted = ""
                correct = False
                error = str(e)
                match = re.search(r"\[trace_id=([^\]]+)\]", error)
                if match:
                    trace_id = match.group(1)
            record = {
                "mode": mode,
                "target_id": target["id"],
                "example_ids": manifest["example_ids"],
                "gold_answer": target["gold_answer"],
                "answer_type": target["answer_type"],
                "predicted": predicted,
                "correct": correct,
                "latency_sec": round(time.time() - started, 3),
                "response_text": response_text,
                "error": error,
                "trace_id": trace_id,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            suffix = f" | error={error[:120]}" if error else ""
            print(f"  [{idx}/{len(pending)}] {target['id']} -> {predicted!r} | correct={correct}{suffix}")
    return output_path


def summarize(mode_to_path: dict[str, Path], args: argparse.Namespace):
    if not {"baseline", "treatment"}.issubset(mode_to_path):
        return
    baseline_rows = {row["target_id"]: row for row in read_jsonl(mode_to_path["baseline"])}
    treatment_rows = {row["target_id"]: row for row in read_jsonl(mode_to_path["treatment"])}
    common_ids = sorted(set(baseline_rows) & set(treatment_rows))
    if not common_ids:
        return

    improved = []
    regressed = []
    same_correct = 0
    same_wrong = 0
    for target_id in common_ids:
        b = baseline_rows[target_id]["correct"]
        t = treatment_rows[target_id]["correct"]
        if not b and t:
            improved.append(target_id)
        elif b and not t:
            regressed.append(target_id)
        elif b and t:
            same_correct += 1
        else:
            same_wrong += 1

    baseline_acc = sum(1 for row in baseline_rows.values() if row["correct"]) / len(common_ids)
    treatment_acc = sum(1 for row in treatment_rows.values() if row["correct"]) / len(common_ids)
    summary = {
        "paired_count": len(common_ids),
        "baseline_accuracy": round(baseline_acc, 4),
        "treatment_accuracy": round(treatment_acc, 4),
        "delta": round(treatment_acc - baseline_acc, 4),
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "unchanged_correct_count": same_correct,
        "unchanged_wrong_count": same_wrong,
        "improved_ids": improved[:20],
        "regressed_ids": regressed[:20],
        "model": args.model,
        "backend": args.backend,
    }
    write_json(args.output_dir / "summary.json", summary)
    lines = [
        "# A/B Few-shot Summary",
        "",
        f"- paired_count: {summary['paired_count']}",
        f"- baseline_accuracy: {summary['baseline_accuracy']}",
        f"- treatment_accuracy: {summary['treatment_accuracy']}",
        f"- delta: {summary['delta']}",
        f"- improved_count: {summary['improved_count']}",
        f"- regressed_count: {summary['regressed_count']}",
        f"- unchanged_correct_count: {summary['unchanged_correct_count']}",
        f"- unchanged_wrong_count: {summary['unchanged_wrong_count']}",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    if Image is None and args.image_max_side > 0:
        print("Warning: Pillow is not installed; image resizing/compression is disabled.")
    probe_openai_backend(args)
    targets = {row["id"]: row for row in read_jsonl(args.targets_path)}
    examples = {row["id"]: row for row in read_jsonl(args.example_pool_path)}
    manifests = read_jsonl(args.manifest_path)
    if args.limit:
        manifests = manifests[:args.limit]

    modes = [args.mode] if args.mode in {"baseline", "treatment"} else ["baseline", "treatment"]
    mode_to_path = {}
    for mode in modes:
        mode_to_path[mode] = run_mode(mode, args, targets, examples, manifests)
    summarize(mode_to_path, args)


if __name__ == "__main__":
    main()
