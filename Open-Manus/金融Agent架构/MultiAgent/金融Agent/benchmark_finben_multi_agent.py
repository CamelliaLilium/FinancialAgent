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
from app.agent.planning import PlanningAgent
from app.config import config
from app.flow.flow_factory import FlowFactory, FlowType
from app.logger import logger
from app.tool import WorkflowStateTool


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
        sink_id = logger.add(lambda msg: log_collector.append(str(msg).rstrip()), level="INFO")

        def _detach():
            logger.remove(sink_id)

        return _detach

    handler = _ListLogHandler(log_collector)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    def _detach():
        logger.removeHandler(handler)

    return _detach


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark multi-agent architecture on FinBen test set."
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
        "--force-planning",
        action="store_true",
        help="Always enable planning agent regardless of config.",
    )
    parser.add_argument(
        "--disable-planning",
        action="store_true",
        help="Disable planning agent for ablation.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array.")
    return data


def extract_label(text: str) -> Optional[str]:
    if not text:
        return None
    matches = re.findall(r"\b(hawkish|dovish|neutral)\b", text.lower())
    if not matches:
        return None
    # Use the last label mention because many outputs include explanations first.
    return matches[-1]


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


def classify_failure(
    sample: Dict[str, Any],
    predicted: Optional[str],
    gold: str,
    status: str,
    tool_info: Dict[str, Any],
) -> List[str]:
    tags: List[str] = []
    query = sample.get("query", "")
    is_fomc_classification = "HAWKISH" in query and "DOVISH" in query and "NEUTRAL" in query

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

    if not tags:
        tags.append("correct")
    return tags


async def run_single_sample(
    sample: Dict[str, Any],
    timeout_s: int,
    use_planning_agent: bool,
) -> Dict[str, Any]:
    workflow_state_tool = WorkflowStateTool()
    agents = {
        "finance": FinanceAgent(),
    }
    executor_keys = ["finance"]
    if use_planning_agent:
        agents["planning"] = PlanningAgent(workflow_state_tool=workflow_state_tool)

    primary_key = "planning" if "planning" in agents else "finance"
    flow = FlowFactory.create_flow(
        flow_type=FlowType.PLANNING,
        agents=agents,
        primary_agent_key=primary_key,
        executors=executor_keys,
        workflow_state_tool=workflow_state_tool,
    )

    prompt = sample.get("query", "")
    start = time.time()
    logs: List[str] = []
    detach_logs = attach_log_capture(logs)
    status = "ok"
    output_text = ""
    err_msg = ""

    try:
        output_text = await asyncio.wait_for(flow.execute(prompt), timeout=timeout_s)
    except asyncio.TimeoutError:
        status = "timeout"
        err_msg = f"Timeout after {timeout_s}s"
    except Exception as e:  # noqa: BLE001
        status = "runtime_error"
        err_msg = str(e)
    finally:
        detach_logs()

    elapsed = time.time() - start
    predicted = extract_label(output_text)
    gold = str(sample.get("answer", "")).strip().lower()
    tool_info = parse_tool_calls(logs)
    tags = classify_failure(sample, predicted, gold, status, tool_info)
    is_correct = predicted == gold and status == "ok"

    step_outputs = getattr(flow, "step_outputs", {})

    return {
        "id": sample.get("id"),
        "gold": gold,
        "predicted": predicted,
        "is_correct": is_correct,
        "status": status,
        "error_message": err_msg,
        "elapsed_seconds": round(elapsed, 3),
        "error_tags": tags,
        "tool_usage": tool_info["tool_counter"],
        "repeated_tool_args": tool_info["repeated_tool_args"],
        "query": prompt,
        "model_output": output_text,
        "step_outputs": step_outputs,
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

    for r in records:
        g = r["gold"] or "unknown"
        p = r["predicted"] or "none"
        confusion[g][p] += 1
        for tag in r["error_tags"]:
            error_tags[tag] += 1
        for t, c in r["tool_usage"].items():
            tool_counter[t] += c
        elapsed_total += r["elapsed_seconds"]

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 6),
        "avg_elapsed_seconds": round(elapsed_total / total, 3) if total else 0.0,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "error_tag_counts": dict(error_tags),
        "tool_usage_counts": dict(tool_counter),
    }


def write_outputs(
    output_root: Path, records: List[Dict[str, Any]], summary: Dict[str, Any], args: argparse.Namespace
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
        f.write("# FinBen Multi-Agent Benchmark Report\n\n")
        f.write(f"- Total samples: {summary['total']}\n")
        f.write(f"- Correct: {summary['correct']}\n")
        f.write(f"- Accuracy: {summary['accuracy']:.4f}\n")
        f.write(f"- Avg latency/sample (s): {summary['avg_elapsed_seconds']}\n\n")

        f.write("## Error Breakdown\n\n")
        if top_errors:
            for tag, cnt in top_errors:
                f.write(f"- {tag}: {cnt}\n")
        else:
            f.write("- No errors detected.\n")

        f.write("\n## Tool Usage\n\n")
        for tool, cnt in sorted(summary["tool_usage_counts"].items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {tool}: {cnt}\n")

        f.write("\n## Confusion Matrix\n\n")
        f.write("```json\n")
        f.write(json.dumps(summary["confusion_matrix"], ensure_ascii=False, indent=2))
        f.write("\n```\n")

        f.write("\n## Research-Oriented Failure Interpretation\n\n")
        f.write("- `same_args_repeated` / `repeated_tool_loop`: tool policy or step-boundary control is weak.\n")
        f.write("- `likely_unnecessary_web_search`: prompt/tool routing mismatch for in-context solvable tasks.\n")
        f.write("- `output_format_error`: output contract under-specified (classification label not strictly enforced).\n")
        f.write("- `reasoning_or_prompt_misalignment`: model reasoning drift or prompt instruction ambiguity.\n")
        f.write("- `tool_execution_error`: tool reliability gap or missing task-fit tools.\n")

    # Export failure cases for direct research inspection.
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

    # Scientific assessment scaffold for rigorous base-benchmark interpretation.
    review_path = output_root / "scientific_assessment.md"
    with review_path.open("w", encoding="utf-8") as f:
        f.write("# Base Benchmark Scientific Assessment\n\n")
        f.write("## 1) Experimental Setup\n\n")
        f.write(f"- Dataset: `{args.dataset}`\n")
        f.write(f"- Sample range: offset={args.offset}, limit={args.limit}\n")
        f.write(f"- Timeout per sample: {args.timeout}s\n")
        f.write(f"- Planning enabled: {not args.disable_planning or args.force_planning}\n")
        f.write("- Agent architecture: PlanningFlow + executors(finance)\n")

        f.write("\n## 2) Core Metrics\n\n")
        f.write(f"- Accuracy: {summary['accuracy']:.4f}\n")
        f.write(f"- Correct/Total: {summary['correct']}/{summary['total']}\n")
        f.write(f"- Avg latency/sample: {summary['avg_elapsed_seconds']}s\n")

        f.write("\n## 3) Failure Taxonomy Evidence\n\n")
        if summary["error_tag_counts"]:
            for tag, cnt in sorted(summary["error_tag_counts"].items(), key=lambda x: x[1], reverse=True):
                if tag == "correct":
                    continue
                f.write(f"- {tag}: {cnt}\n")
        else:
            f.write("- No failure tags.\n")

        f.write("\n## 4) Internal Validity\n\n")
        f.write("- Strength: raw logs, tool args, and predictions are all persisted for auditability.\n")
        f.write("- Risk: online model stochasticity and external API latency can introduce run-to-run variance.\n")
        f.write("- Mitigation: run repeated seeds/splits and report mean/std confidence interval.\n")

        f.write("\n## 5) External Validity\n\n")
        f.write("- FinBen may emphasize specific financial-language distributions.\n")
        f.write("- Transfer to downstream financial tasks (report parsing, ratio analysis, QA) should be verified separately.\n")

        f.write("\n## 6) Construct Validity\n\n")
        f.write("- Current endpoint metric is label accuracy.\n")
        f.write("- Multi-agent quality should also include process metrics: tool efficiency, loop rate, and failure mode composition.\n")

        f.write("\n## 7) Reproducibility Checklist\n\n")
        f.write("- Keep config/model/version fixed during base benchmark runs.\n")
        f.write("- Store full artifacts: predictions.jsonl, logs/, summary.json, failure_cases.jsonl.\n")
        f.write("- Run multiple trials and report confidence intervals.\n")


async def main() -> None:
    args = parse_args()

    if args.force_planning and args.disable_planning:
        raise ValueError("Cannot use --force-planning and --disable-planning together.")

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        # Prefer project root relative path, then fallback to workspace_root.
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

    if args.force_planning:
        use_planning = True
    elif args.disable_planning:
        use_planning = False
    else:
        use_planning = bool(config.run_flow_config.use_planning_agent)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) / f"finben_multi_agent_{run_id}"
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root

    logger.info(f"Benchmark samples: {len(subset)}")
    logger.info(f"Use planning agent: {use_planning}")
    logger.info(f"Output directory: {output_root}")

    records: List[Dict[str, Any]] = []
    for i, sample in enumerate(subset, start=1):
        logger.info(f"[{i}/{len(subset)}] Running sample: {sample.get('id')}")
        record = await run_single_sample(
            sample=sample,
            timeout_s=args.timeout,
            use_planning_agent=use_planning,
        )
        records.append(record)
        logger.info(
            f"[{i}/{len(subset)}] done id={record['id']} "
            f"pred={record['predicted']} gold={record['gold']} correct={record['is_correct']}"
        )

    summary = aggregate_metrics(records)
    write_outputs(output_root=output_root, records=records, summary=summary, args=args)

    logger.info("Benchmark completed.")
    logger.info(
        f"Accuracy={summary['accuracy']:.4f}, Correct={summary['correct']}/{summary['total']}"
    )
    logger.info(f"Artifacts saved to: {output_root}")


if __name__ == "__main__":
    asyncio.run(main())
