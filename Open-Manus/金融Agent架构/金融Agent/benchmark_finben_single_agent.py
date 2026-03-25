import argparse
import asyncio
import json
import logging
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agent.finance import FinanceAgent
from app.agent.manus import Manus
from app.config import config
from app.logger import logger


LABELS = {"hawkish", "dovish", "neutral"}


class _ListLogHandler(logging.Handler):
    def __init__(self, collector: List[str]) -> None:
        super().__init__()
        self.collector = collector

    def emit(self, record: logging.LogRecord) -> None:
        self.collector.append(self.format(record))


def attach_log_capture(log_collector: List[str]):
    """
    Attach a temporary sink/handler to capture logs.
    Supports both loguru logger and stdlib logger fallback.
    Returns a detach callback.
    """
    if hasattr(logger, "add") and hasattr(logger, "remove"):
        sink_id = logger.add(
            lambda msg: log_collector.append(str(msg).rstrip()), level="INFO"
        )

        def _detach():
            logger.remove(sink_id)

        return _detach

    handler = _ListLogHandler(log_collector)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    def _detach():
        logger.removeHandler(handler)

    return _detach


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark single-agent baseline on FinBen test set."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Dataset/Finben/finben_test.json",
        help="Path to FinBen test json file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark artifacts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of samples (0 means full dataset).",
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Start index in dataset."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per sample in seconds.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="manus",
        choices=["manus", "finance"],
        help="Which single agent to benchmark.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array.")
    return data


def extract_label(text: str) -> Dict[str, Optional[str]]:
    """
    Extract classification label from model output.
    Returns both first and last mention for auditability.
    """
    if not text:
        return {"first": None, "last": None, "final": None}

    cleaned = text.strip().lower()
    if cleaned in LABELS:
        return {"first": cleaned, "last": cleaned, "final": cleaned}

    matches = re.findall(r"\b(hawkish|dovish|neutral)\b", cleaned)
    if not matches:
        return {"first": None, "last": None, "final": None}

    # For instruction "return only label", first mention is usually most faithful.
    first = matches[0]
    last = matches[-1]
    return {"first": first, "last": last, "final": first}


def parse_tool_calls(log_lines: List[str]) -> Dict[str, Any]:
    tool_counter: Counter = Counter()
    arg_counter: Counter = Counter()
    repeated_signatures: Dict[str, int] = {}
    has_rate_limit = False
    has_tool_error = False
    has_repeat_guard = False

    for line in log_lines:
        if "Tools being prepared:" in line:
            match = re.search(r"Tools being prepared:\s*(\[.*\])", line)
            if match:
                tools_literal = match.group(1)
                for tool in re.findall(r"'([^']+)'", tools_literal):
                    tool_counter[tool] += 1

        if "Tool arguments:" in line:
            args = line.split("Tool arguments:", 1)[1].strip()
            arg_counter[args] += 1

        if "Rate limit exceeded" in line or "TPM limit reached" in line:
            has_rate_limit = True

        if "Error:" in line or "encountered a problem" in line:
            has_tool_error = True

        if "Detected repeated identical tool call" in line:
            has_repeat_guard = True

    for arg, cnt in arg_counter.items():
        if cnt > 1:
            repeated_signatures[arg] = cnt

    return {
        "tool_counter": dict(tool_counter),
        "repeated_tool_args": repeated_signatures,
        "has_rate_limit": has_rate_limit,
        "has_tool_error": has_tool_error,
        "has_repeat_guard": has_repeat_guard,
    }


def compute_macro_f1(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_label = {}
    for label in sorted(LABELS):
        tp = sum(
            1 for r in records if r["gold"] == label and r["predicted"] == label
        )
        fp = sum(
            1 for r in records if r["gold"] != label and r["predicted"] == label
        )
        fn = sum(
            1 for r in records if r["gold"] == label and r["predicted"] != label
        )
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": sum(1 for r in records if r["gold"] == label),
        }

    macro_f1 = sum(v["f1"] for v in per_label.values()) / len(LABELS)
    return {"macro_f1": round(macro_f1, 6), "per_label": per_label}


def classify_failure(
    sample: Dict[str, Any],
    predicted: Optional[str],
    gold: str,
    status: str,
    tool_info: Dict[str, Any],
    label_extract_info: Dict[str, Optional[str]],
) -> List[str]:
    tags: List[str] = []
    query = sample.get("query", "")
    is_fomc_classification = (
        "HAWKISH" in query and "DOVISH" in query and "NEUTRAL" in query
    )

    if status == "timeout":
        tags.append("timeout")
    if status == "runtime_error":
        tags.append("runtime_error")
    if tool_info["has_rate_limit"]:
        tags.append("rate_limit")
    if tool_info["has_repeat_guard"]:
        tags.append("repeated_tool_loop")
    if tool_info["has_tool_error"]:
        tags.append("tool_execution_error")
    if tool_info["repeated_tool_args"]:
        tags.append("same_args_repeated")
    if is_fomc_classification and tool_info["tool_counter"].get("web_search", 0) > 0:
        tags.append("likely_unnecessary_web_search")
    if predicted is None:
        tags.append("output_format_error")
    elif predicted not in LABELS:
        tags.append("invalid_label")
    elif predicted != gold:
        tags.append("reasoning_or_prompt_misalignment")

    if (
        label_extract_info.get("first")
        and label_extract_info.get("last")
        and label_extract_info["first"] != label_extract_info["last"]
    ):
        tags.append("label_extraction_ambiguity")

    if not tags:
        tags.append("correct")
    return tags


def create_agent(agent_name: str):
    if agent_name == "manus":
        return Manus()
    if agent_name == "finance":
        return FinanceAgent()
    raise ValueError(f"Unsupported agent: {agent_name}")


async def run_single_sample(
    sample: Dict[str, Any], timeout_s: int, agent_name: str
) -> Dict[str, Any]:
    agent = create_agent(agent_name)
    prompt = sample.get("query", "")

    start = time.time()
    logs: List[str] = []
    detach_logs = attach_log_capture(logs)

    status = "ok"
    output_text = ""
    err_msg = ""

    try:
        output_text = await asyncio.wait_for(agent.run(prompt), timeout=timeout_s)
    except asyncio.TimeoutError:
        status = "timeout"
        err_msg = f"Timeout after {timeout_s}s"
    except Exception as e:  # noqa: BLE001
        status = "runtime_error"
        err_msg = str(e)
    finally:
        detach_logs()

    elapsed = time.time() - start
    gold = str(sample.get("answer", "")).strip().lower()
    label_extract_info = extract_label(output_text)
    predicted = label_extract_info["final"]
    tool_info = parse_tool_calls(logs)
    tags = classify_failure(
        sample=sample,
        predicted=predicted,
        gold=gold,
        status=status,
        tool_info=tool_info,
        label_extract_info=label_extract_info,
    )
    is_correct = predicted == gold and status == "ok"

    return {
        "id": sample.get("id"),
        "gold": gold,
        "predicted": predicted,
        "predicted_first_label": label_extract_info["first"],
        "predicted_last_label": label_extract_info["last"],
        "is_correct": is_correct,
        "status": status,
        "error_message": err_msg,
        "elapsed_seconds": round(elapsed, 3),
        "error_tags": tags,
        "tool_usage": tool_info["tool_counter"],
        "repeated_tool_args": tool_info["repeated_tool_args"],
        "query": prompt,
        "model_output": output_text,
        "logs": logs,
    }


def aggregate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    correct = sum(1 for r in records if r["is_correct"])
    accuracy = (correct / total) if total else 0.0

    confusion = defaultdict(lambda: defaultdict(int))
    error_tags = Counter()
    tool_counter = Counter()
    elapsed_total = 0.0
    first_last_mismatch = 0

    for r in records:
        g = r["gold"] or "unknown"
        p = r["predicted"] or "none"
        confusion[g][p] += 1
        for tag in r["error_tags"]:
            error_tags[tag] += 1
        for t, c in r["tool_usage"].items():
            tool_counter[t] += c
        elapsed_total += r["elapsed_seconds"]
        if (
            r.get("predicted_first_label")
            and r.get("predicted_last_label")
            and r["predicted_first_label"] != r["predicted_last_label"]
        ):
            first_last_mismatch += 1

    macro = compute_macro_f1(records)
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 6),
        "macro_f1": macro["macro_f1"],
        "per_label": macro["per_label"],
        "avg_elapsed_seconds": round(elapsed_total / total, 3) if total else 0.0,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "error_tag_counts": dict(error_tags),
        "tool_usage_counts": dict(tool_counter),
        "label_extraction_first_last_mismatch": first_last_mismatch,
    }


def write_outputs(
    output_root: Path,
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    pred_path = output_root / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for r in records:
            log_file = logs_dir / f"{r['id']}.log"
            with log_file.open("w", encoding="utf-8") as lf:
                lf.write("\n".join(r["logs"]))
            row = {k: v for k, v in r.items() if k != "logs"}
            row["log_file"] = str(log_file)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    failure_cases = [r for r in records if not r["is_correct"] or r["status"] != "ok"]
    failure_path = output_root / "failure_cases.jsonl"
    with failure_path.open("w", encoding="utf-8") as f:
        for r in failure_cases:
            primary_tag = next((t for t in r["error_tags"] if t != "correct"), "unknown")
            row = {
                "id": r["id"],
                "gold": r["gold"],
                "predicted": r["predicted"],
                "status": r["status"],
                "primary_error_tag": primary_tag,
                "all_error_tags": r["error_tags"],
                "tool_usage": r["tool_usage"],
                "repeated_tool_args": r["repeated_tool_args"],
                "elapsed_seconds": r["elapsed_seconds"],
                "query": r["query"],
                "model_output": r["model_output"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    report_path = output_root / "error_analysis.md"
    top_errors = sorted(
        [(k, v) for k, v in summary["error_tag_counts"].items() if k != "correct"],
        key=lambda x: x[1],
        reverse=True,
    )
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# FinBen Single-Agent Benchmark Report\n\n")
        f.write(f"- Agent: `{args.agent}`\n")
        f.write(f"- Total samples: {summary['total']}\n")
        f.write(f"- Correct: {summary['correct']}\n")
        f.write(f"- Accuracy: {summary['accuracy']:.4f}\n")
        f.write(f"- Macro-F1: {summary['macro_f1']:.4f}\n")
        f.write(f"- Avg latency/sample (s): {summary['avg_elapsed_seconds']}\n")
        f.write(
            "- Label extraction first/last mismatch count: "
            f"{summary['label_extraction_first_last_mismatch']}\n\n"
        )

        f.write("## Error Breakdown\n\n")
        if top_errors:
            for tag, cnt in top_errors:
                f.write(f"- {tag}: {cnt}\n")
        else:
            f.write("- No errors detected.\n")

        f.write("\n## Tool Usage\n\n")
        for tool, cnt in sorted(
            summary["tool_usage_counts"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            f.write(f"- {tool}: {cnt}\n")

        f.write("\n## Confusion Matrix\n\n")
        f.write("```json\n")
        f.write(json.dumps(summary["confusion_matrix"], ensure_ascii=False, indent=2))
        f.write("\n```\n")

    review_path = output_root / "scientific_assessment.md"
    with review_path.open("w", encoding="utf-8") as f:
        f.write("# Single-Agent Base Benchmark Scientific Assessment\n\n")
        f.write("## 1) Experimental Setup\n\n")
        f.write(f"- Agent: `{args.agent}`\n")
        f.write(f"- Dataset: `{args.dataset}`\n")
        f.write(f"- Sample range: offset={args.offset}, limit={args.limit}\n")
        f.write(f"- Timeout per sample: {args.timeout}s\n")
        f.write("- Architecture mode: single-agent direct execution (no PlanningFlow)\n")

        f.write("\n## 2) Core Metrics\n\n")
        f.write(f"- Accuracy: {summary['accuracy']:.4f}\n")
        f.write(f"- Macro-F1: {summary['macro_f1']:.4f}\n")
        f.write(f"- Correct/Total: {summary['correct']}/{summary['total']}\n")
        f.write(f"- Avg latency/sample: {summary['avg_elapsed_seconds']}s\n")

        f.write("\n## 3) Failure Taxonomy Evidence\n\n")
        if summary["error_tag_counts"]:
            for tag, cnt in sorted(
                summary["error_tag_counts"].items(), key=lambda x: x[1], reverse=True
            ):
                if tag == "correct":
                    continue
                f.write(f"- {tag}: {cnt}\n")
        else:
            f.write("- No failure tags.\n")

        f.write("\n## 4) Reproducibility Checklist\n\n")
        f.write("- Keep model/config fixed across baseline comparisons.\n")
        f.write("- Persist full artifacts: predictions, failure_cases, logs, summary.\n")
        f.write("- Run multiple trials and report confidence intervals.\n")


async def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        candidate_project = Path.cwd() / dataset_path
        candidate_workspace = Path(config.workspace_root) / dataset_path
        if candidate_project.exists():
            dataset_path = candidate_project
        else:
            dataset_path = candidate_workspace
    data = load_dataset(dataset_path)

    start = max(args.offset, 0)
    end = len(data) if args.limit <= 0 else min(len(data), start + args.limit)
    subset = data[start:end]
    if not subset:
        raise ValueError("No samples selected. Check --offset and --limit.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) / f"finben_single_agent_{args.agent}_{run_id}"
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root

    logger.info(f"Benchmark samples: {len(subset)}")
    logger.info(f"Single agent: {args.agent}")
    logger.info(f"Output directory: {output_root}")

    records: List[Dict[str, Any]] = []
    for i, sample in enumerate(subset, start=1):
        logger.info(f"[{i}/{len(subset)}] Running sample: {sample.get('id')}")
        record = await run_single_sample(
            sample=sample,
            timeout_s=args.timeout,
            agent_name=args.agent,
        )
        records.append(record)
        logger.info(
            f"[{i}/{len(subset)}] done id={record['id']} "
            f"pred={record['predicted']} gold={record['gold']} correct={record['is_correct']}"
        )

    summary = aggregate_metrics(records)
    write_outputs(output_root=output_root, records=records, summary=summary, args=args)

    logger.info("Single-agent benchmark completed.")
    logger.info(
        f"Accuracy={summary['accuracy']:.4f}, Macro-F1={summary['macro_f1']:.4f}, "
        f"Correct={summary['correct']}/{summary['total']}"
    )
    logger.info(f"Artifacts saved to: {output_root}")


if __name__ == "__main__":
    asyncio.run(main())
