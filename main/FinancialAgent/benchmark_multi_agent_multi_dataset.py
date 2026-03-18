import argparse
import asyncio
import base64
import json
import logging
import math
import mimetypes
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure rag module is importable (rag/ is at workspace root)
_workspace_root = Path(__file__).resolve().parents[3]  # .../autodl-tmp
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from app.agent.finance import FinanceAgent
from app.agent.manus import Manus
from app.agent.planning import PlanningAgent
from app.config import config
from app.flow.flow_factory import FlowFactory, FlowType
from app.llm import LLM
from app.logger import logger
from app.schema import Message
from app.tool import WorkflowStateTool

DATASET_PRESETS: Dict[str, List[str]] = {
    "all": [
        "Dataset/bizbench_test",
        "Dataset/finmmr",
        "Dataset/flarez-confinqa_test",
    ],
    "bizbench": ["Dataset/bizbench_test"],
    "finmmr": ["Dataset/finmmr"],
    "flarez": ["Dataset/flarez-confinqa_test"],
}

# When --dataset-root points to datasets/test/, use these file names
DATASET_FILES_FROM_ROOT: Dict[str, List[str]] = {
    "all": [
        "bizbench_test.json",
        "finmmr_easy_test.json",
        "finmmr_medium_test.json",
        "finmmr_hard_test.json",
        "flare-convfinqa_test.json",
    ],
    "bizbench": ["bizbench_test.json"],
    "finmmr": ["finmmr_easy_test.json", "finmmr_medium_test.json", "finmmr_hard_test.json"],
    "flarez": ["flare-convfinqa_test.json"],
}


class _ListLogHandler(logging.Handler):
    def __init__(self, collector: List[str]) -> None:
        super().__init__()
        self.collector = collector

    def emit(self, record: logging.LogRecord) -> None:
        self.collector.append(self.format(record))


def attach_log_capture(log_collector: List[str]):
    """Attach a temporary logger sink and return a detach callback."""
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
        description="Run multi-agent benchmark across selected datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "bizbench", "finmmr", "flarez"],
        help="Choose one dataset preset or run all presets.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save all benchmark artifacts.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per sample in seconds.",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=0,
        help="Limit samples per dataset file (0 means full dataset).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start index for every dataset file.",
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
    parser.add_argument(
        "--image-root",
        type=str,
        default="",
        help="Optional local root for remapping dataset image paths (for multimodal datasets).",
    )
    parser.add_argument(
        "--multimodal-mode",
        type=str,
        default="best_effort",
        choices=["off", "best_effort", "strict"],
        help=(
            "off: ignore images; best_effort: use vision extraction when possible; "
            "strict: image samples fail when vision input is unavailable."
        ),
    )
    parser.add_argument(
        "--vision-timeout",
        type=int,
        default=120,
        help="Timeout (seconds) for image-to-text extraction per sample.",
    )
    parser.add_argument(
        "--max-images-per-sample",
        type=int,
        default=1,
        help="Maximum number of images to load per sample.",
    )
    parser.add_argument(
        "--numeric-extract-strategy",
        type=str,
        default="any",
        choices=["any", "first", "last"],
        help=(
            "Numeric scoring strategy: any/first/last candidate in model output. "
            "Use 'any' for robust research evaluation."
        ),
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=5,
        help="Number of RAG examples to inject (0 = disable RAG). Default: 5.",
    )
    parser.add_argument(
        "--rag-persist-dir",
        type=str,
        default="",
        help="Chroma persist dir for RAG. Default: rag/chroma_db relative to workspace.",
    )
    parser.add_argument(
        "--cot-enabled",
        action="store_true",
        dest="cot_enabled",
        default=True,
        help="Enable CoT few-shot for FinMMR (default: True).",
    )
    parser.add_argument(
        "--no-cot",
        action="store_false",
        dest="cot_enabled",
        help="Disable CoT few-shot for FinMMR (ablation).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="",
        help="Root dir for datasets (e.g. datasets/test). When set, resolves files from DATASET_FILES_FROM_ROOT.",
    )
    return parser.parse_args()


def resolve_dataset_inputs(dataset_choice: str) -> List[str]:
    if dataset_choice not in DATASET_PRESETS:
        raise ValueError(f"Unsupported dataset preset: {dataset_choice}")
    return DATASET_PRESETS[dataset_choice]


def resolve_dataset_files(
    dataset_inputs: List[str],
    dataset_root: Optional[Path] = None,
    dataset_choice: str = "all",
) -> List[Path]:
    files: List[Path] = []

    if dataset_root and dataset_root.exists():
        file_names = DATASET_FILES_FROM_ROOT.get(dataset_choice, DATASET_FILES_FROM_ROOT["all"])
        for name in file_names:
            p = dataset_root / name
            if p.exists() and p.suffix.lower() == ".json":
                files.append(p)
        if files:
            return sorted(files)

    for raw in dataset_inputs:
        p = Path(raw)
        if not p.is_absolute():
            project_candidate = Path.cwd() / p
            workspace_candidate = Path(config.workspace_root) / p
            if project_candidate.exists():
                p = project_candidate
            elif workspace_candidate.exists():
                p = workspace_candidate
        if p.is_file() and p.suffix.lower() == ".json":
            files.append(p)
            continue
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
            continue
        raise FileNotFoundError(f"Dataset path not found: {raw}")
    if not files:
        raise ValueError("No json dataset files resolved.")
    return files


def resolve_image_path(
    image_ref: str, dataset_file: Path, image_root: Optional[Path]
) -> Optional[Path]:
    """Resolve a dataset image reference to a local readable path."""
    ref = (image_ref or "").strip()
    if not ref:
        return None

    direct = Path(ref)
    if direct.exists() and direct.is_file():
        return direct

    candidates: List[Path] = []
    # Try relative to dataset directory.
    candidates.append(dataset_file.parent / ref)
    # Try common images folder.
    candidates.append(dataset_file.parent / "images" / Path(ref).name)

    if image_root:
        candidates.append(image_root / Path(ref).name)
        # Preserve trailing path after ".../images/" when available.
        match = re.search(r"images[/\\](.+)$", ref)
        if match:
            candidates.append(image_root / match.group(1))

    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand
    return None


def image_file_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _format_rag_example(rec: Dict[str, Any]) -> str:
    """Format a single RAG example as CoT-style Q/A block."""
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
        reasoning = " ".join(steps) + f" The answer is {ans}." if steps else f"The answer is {ans}."
    else:
        reasoning = f"The answer is {ans}."
    return f"Q: {q}\nA: {reasoning}"


def _retrieve_rag_examples(
    query: str,
    top_k: int,
    persist_dir: Optional[Path],
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve similar examples from RAG case library."""
    if top_k <= 0:
        return []
    try:
        from rag.retriever import get_retriever
    except ImportError:
        logger.warning("RAG retriever not found; skipping RAG injection.")
        return []
    persist_path = str(persist_dir) if persist_dir else None
    try:
        r = get_retriever(persist_path) if persist_path else get_retriever()
        return r.retrieve(query, top_k=top_k, source=source)
    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}")
        return []


def _load_cot_fewshot_finmmr(rag_dir: Path, count: int = 6) -> List[Dict[str, Any]]:
    """Load fixed FinMMR CoT few-shot examples when RAG is disabled."""
    examples = []
    for name in ["finmmr_easy_validation.json", "finmmr_medium_validation.json"]:
        p = rag_dir / name
        if not p.exists():
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data[: count - len(examples)]:
                if len(examples) >= count:
                    break
                examples.append({"question": item.get("question"), "metadata": item})
        except Exception as e:
            logger.warning(f"Failed to load {p}: {e}")
        if len(examples) >= count:
            break
    return examples


async def extract_visual_evidence(
    question: str, image_paths: List[Path], timeout_s: int
) -> str:
    """
    Use multimodal LLM to convert chart/table images into concise textual evidence.
    """
    llm = LLM()
    image_payload = [{"url": image_file_to_data_url(p)} for p in image_paths]
    prompt = (
        "You are extracting evidence from financial images for a downstream QA agent.\n"
        "Task:\n"
        "1) Extract key table values and row/column headers relevant to the question.\n"
        "2) Preserve units and years.\n"
        "3) If needed, provide intermediate arithmetic setup but no long explanations.\n"
        "4) If details are uncertain due to image quality, explicitly mark uncertainty.\n\n"
        f"Question:\n{question}\n"
    )
    return await asyncio.wait_for(
        llm.ask_with_images(
            messages=[Message.user_message(prompt)],
            images=image_payload,
            stream=False,
        ),
        timeout=timeout_s,
    )


async def build_augmented_prompt(
    sample: Dict[str, Any],
    dataset_file: Path,
    base_prompt: str,
    image_root: Optional[Path],
    multimodal_mode: str,
    vision_timeout_s: int,
    max_images_per_sample: int,
    rag_top_k: int = 0,
    rag_persist_dir: Optional[Path] = None,
    cot_enabled: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build final prompt with optional RAG examples, CoT few-shot, and multimodal evidence extraction.
    """
    meta: Dict[str, Any] = {
        "has_image_refs": False,
        "image_ref_count": 0,
        "resolved_image_count": 0,
        "unresolved_image_refs": [],
        "used_vision_extraction": False,
        "vision_status": "not_applicable",
        "vision_error": "",
        "vision_latency_seconds": 0.0,
    }

    prompt_prefix = ""
    is_finmmr = "finmmr" in str(dataset_file).lower()

    # RAG injection
    if rag_top_k > 0:
        source_filter = "finmmr" if is_finmmr else None
        examples = _retrieve_rag_examples(
            query=base_prompt,
            top_k=rag_top_k,
            persist_dir=rag_persist_dir,
            source=source_filter,
        )
        if examples:
            blocks = [_format_rag_example(ex) for ex in examples]
            prompt_prefix = (
                "Reference examples (similar questions with step-by-step reasoning):\n\n"
                + "\n\n".join(blocks)
                + "\n\n---\n\nYour task:\n"
            )

    # CoT few-shot for FinMMR when RAG is disabled
    elif cot_enabled and is_finmmr:
        rag_dir = _workspace_root / "rag"
        if rag_dir.exists():
            cot_examples = _load_cot_fewshot_finmmr(rag_dir, count=6)
            if cot_examples:
                blocks = [_format_rag_example(ex) for ex in cot_examples]
                prompt_prefix = (
                    "Few-shot examples (Chain-of-Thought format):\n\n"
                    + "\n\n".join(blocks)
                    + "\n\n---\n\nYour task:\n"
                )

    if prompt_prefix:
        base_prompt = prompt_prefix + base_prompt

    image_refs_raw = sample.get("images", []) or []
    image_refs = [str(x) for x in image_refs_raw if x]
    if max_images_per_sample > 0:
        image_refs = image_refs[:max_images_per_sample]

    meta["has_image_refs"] = bool(image_refs)
    meta["image_ref_count"] = len(image_refs)

    if not image_refs or multimodal_mode == "off":
        if image_refs and multimodal_mode == "off":
            meta["vision_status"] = "disabled"
        return base_prompt, meta

    resolved: List[Path] = []
    unresolved: List[str] = []
    for ref in image_refs:
        p = resolve_image_path(ref, dataset_file=dataset_file, image_root=image_root)
        if p is None:
            unresolved.append(ref)
        else:
            resolved.append(p)

    meta["resolved_image_count"] = len(resolved)
    meta["unresolved_image_refs"] = unresolved
    if not resolved:
        meta["vision_status"] = "no_resolved_images"
        return base_prompt, meta

    started = time.time()
    try:
        vision_text = await extract_visual_evidence(
            question=base_prompt,
            image_paths=resolved,
            timeout_s=vision_timeout_s,
        )
        meta["vision_latency_seconds"] = round(time.time() - started, 3)
        meta["used_vision_extraction"] = True
        meta["vision_status"] = "ok"
        augmented_prompt = (
            f"{base_prompt}\n\n"
            "VISUAL EVIDENCE (extracted from attached financial images):\n"
            f"{vision_text}\n\n"
            "Use the visual evidence above together with the original prompt. "
            "If conflicts exist, prioritize directly extracted visual numbers."
        )
        return augmented_prompt, meta
    except Exception as e:  # noqa: BLE001
        meta["vision_latency_seconds"] = round(time.time() - started, 3)
        meta["used_vision_extraction"] = True
        meta["vision_status"] = "failed"
        meta["vision_error"] = str(e)
        return base_prompt, meta


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
                for tool in re.findall(r"'([^']+)'", match.group(1)):
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


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def normalize_code_text(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip())


def extract_numeric_candidates(text: str) -> List[float]:
    if not text:
        return []
    raw = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    vals: List[float] = []
    for item in raw:
        try:
            vals.append(float(item.replace(",", "")))
        except Exception:
            continue
    return vals


def extract_first_number(text: str) -> Optional[float]:
    vals = extract_numeric_candidates(text)
    return vals[0] if vals else None


def infer_prompt(sample: Dict[str, Any]) -> str:
    return str(sample.get("query") or sample.get("question") or "").strip()


def infer_gold(sample: Dict[str, Any]) -> Any:
    if "answer" in sample:
        return sample["answer"]
    if "ground_truth" in sample:
        return sample["ground_truth"]
    return ""


def infer_sample_id(sample: Dict[str, Any], idx: int) -> str:
    for key in ("id", "question_id", "source_id"):
        if key in sample and sample[key]:
            return str(sample[key])
    return f"sample_{idx}"


def is_numeric_like(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return True
    if value is None:
        return False
    if isinstance(value, str):
        return extract_first_number(value) is not None
    return False


def evaluate_prediction(
    sample: Dict[str, Any], model_output: str, numeric_extract_strategy: str = "any"
) -> Dict[str, Any]:
    gold_raw = infer_gold(sample)
    task_name = str(sample.get("task", "")).strip().lower()
    pred_text = (model_output or "").strip()

    mode = "text"
    gold_num: Optional[float] = None
    pred_num: Optional[float] = None
    pred_num_first: Optional[float] = None
    pred_num_last: Optional[float] = None
    pred_num_any: Optional[float] = None
    pred_num_candidates: List[float] = []
    numeric_extract_strategy_used = ""
    is_correct = False
    score_detail = ""

    if task_name == "formulaeval":
        mode = "code"
        gold_code = normalize_code_text(str(gold_raw))
        pred_code = normalize_code_text(pred_text)
        is_correct = bool(gold_code) and gold_code in pred_code
        score_detail = "code_substring_match"
    elif is_numeric_like(gold_raw):
        mode = "numeric"
        gold_num = (
            float(gold_raw)
            if isinstance(gold_raw, (int, float))
            else extract_first_number(str(gold_raw))
        )
        pred_num_candidates = extract_numeric_candidates(pred_text)
        pred_num_first = pred_num_candidates[0] if pred_num_candidates else None
        pred_num_last = pred_num_candidates[-1] if pred_num_candidates else None
        numeric_extract_strategy_used = numeric_extract_strategy

        if gold_num is not None and pred_num_candidates:
            tol = max(1e-4, 1e-3 * abs(gold_num))
            numeric_matches = [
                v for v in pred_num_candidates if abs(v - gold_num) <= tol
            ]
            pred_num_any = numeric_matches[0] if numeric_matches else None

            if numeric_extract_strategy == "first":
                pred_num = pred_num_first
                is_correct = pred_num is not None and abs(pred_num - gold_num) <= tol
            elif numeric_extract_strategy == "last":
                pred_num = pred_num_last
                is_correct = pred_num is not None and abs(pred_num - gold_num) <= tol
            else:  # any
                pred_num = pred_num_any if pred_num_any is not None else pred_num_first
                is_correct = pred_num_any is not None
            score_detail = (
                f"abs_tol={tol:.6g}; strategy={numeric_extract_strategy}; "
                f"candidates={len(pred_num_candidates)}"
            )
    else:
        gold_text = normalize_text(str(gold_raw))
        pred_norm = normalize_text(pred_text)
        is_correct = bool(gold_text) and (gold_text == pred_norm)
        score_detail = "normalized_exact_match"

    return {
        "mode": mode,
        "gold_raw": gold_raw,
        "pred_text": pred_text,
        "gold_num": gold_num,
        "pred_num": pred_num,
        "pred_num_first": pred_num_first,
        "pred_num_last": pred_num_last,
        "pred_num_any": pred_num_any,
        "pred_num_candidates_count": len(pred_num_candidates),
        "numeric_extract_strategy_used": numeric_extract_strategy_used,
        "is_correct": is_correct,
        "score_detail": score_detail,
    }


def classify_failure(
    status: str,
    eval_info: Dict[str, Any],
    tool_info: Dict[str, Any],
    output_text: str,
    multimodal_meta: Dict[str, Any],
) -> List[str]:
    tags: List[str] = []
    if status == "timeout":
        tags.append("timeout")
    if status == "runtime_error":
        tags.append("runtime_error")
    if not (output_text or "").strip():
        tags.append("empty_output")
    if tool_info["has_rate_limit"]:
        tags.append("rate_limit")
    if tool_info["has_repeat_guard"]:
        tags.append("repeated_tool_loop")
    if tool_info["has_tool_error"]:
        tags.append("tool_execution_error")
    if tool_info["repeated_tool_args"]:
        tags.append("same_args_repeated")
    if multimodal_meta.get("has_image_refs") and multimodal_meta.get(
        "resolved_image_count", 0
    ) == 0:
        tags.append("multimodal_input_missing")
    if multimodal_meta.get("vision_status") == "failed":
        tags.append("multimodal_vision_failure")

    if status == "ok" and not eval_info["is_correct"]:
        if eval_info["mode"] == "numeric":
            if eval_info["pred_num"] is None:
                tags.append("prediction_parse_failure")
            else:
                tags.append("numeric_mismatch")
        elif eval_info["mode"] == "code":
            tags.append("code_mismatch")
        else:
            tags.append("text_mismatch")

    if not tags:
        tags.append("correct")
    return tags


async def run_single_sample(
    sample: Dict[str, Any],
    idx: int,
    timeout_s: int,
    use_planning_agent: bool,
    dataset_file: Path,
    image_root: Optional[Path],
    multimodal_mode: str,
    vision_timeout_s: int,
    max_images_per_sample: int,
    numeric_extract_strategy: str,
    rag_top_k: int = 0,
    rag_persist_dir: Optional[Path] = None,
    cot_enabled: bool = True,
) -> Dict[str, Any]:
    workflow_state_tool = WorkflowStateTool()
    agents = {
        "manus": Manus(),
        "finance": FinanceAgent(),
    }
    executor_keys = ["manus", "finance"]
    if use_planning_agent:
        agents["planning"] = PlanningAgent(workflow_state_tool=workflow_state_tool)

    primary_key = "planning" if "planning" in agents else "manus"
    flow = FlowFactory.create_flow(
        flow_type=FlowType.PLANNING,
        agents=agents,
        primary_agent_key=primary_key,
        executors=executor_keys,
        workflow_state_tool=workflow_state_tool,
    )

    prompt_raw = infer_prompt(sample)
    prompt, multimodal_meta = await build_augmented_prompt(
        sample=sample,
        dataset_file=dataset_file,
        base_prompt=prompt_raw,
        image_root=image_root,
        multimodal_mode=multimodal_mode,
        vision_timeout_s=vision_timeout_s,
        max_images_per_sample=max_images_per_sample,
        rag_top_k=rag_top_k,
        rag_persist_dir=rag_persist_dir,
        cot_enabled=cot_enabled,
    )
    sample_id = infer_sample_id(sample, idx)
    start = time.time()
    logs: List[str] = []
    detach_logs = attach_log_capture(logs)
    status = "ok"
    output_text = ""
    err_msg = ""

    try:
        strict_failed = (
            multimodal_mode == "strict"
            and multimodal_meta.get("has_image_refs")
            and (
                multimodal_meta.get("resolved_image_count", 0) == 0
                or multimodal_meta.get("vision_status") == "failed"
            )
        )
        if strict_failed:
            status = "runtime_error"
            err_msg = (
                "Strict multimodal mode failure: "
                f"resolved_images={multimodal_meta.get('resolved_image_count', 0)}, "
                f"vision_status={multimodal_meta.get('vision_status')}, "
                f"vision_error={multimodal_meta.get('vision_error', '')}"
            )
        else:
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
    eval_info = evaluate_prediction(
        sample=sample,
        model_output=output_text,
        numeric_extract_strategy=numeric_extract_strategy,
    )
    tool_info = parse_tool_calls(logs)
    tags = classify_failure(
        status=status,
        eval_info=eval_info,
        tool_info=tool_info,
        output_text=output_text,
        multimodal_meta=multimodal_meta,
    )
    is_correct = eval_info["is_correct"] and status == "ok"
    llm_call_count = sum(1 for line in logs if "Token usage:" in line)
    zero_tool_rounds = sum(1 for line in logs if "selected 0 tools to use" in line)
    blocked_marks = sum(
        1 for line in logs if "Marked step" in line and "as blocked" in line
    )

    return {
        "id": sample_id,
        "status": status,
        "error_message": err_msg,
        "elapsed_seconds": round(elapsed, 3),
        "error_tags": tags,
        "tool_usage": tool_info["tool_counter"],
        "repeated_tool_args": tool_info["repeated_tool_args"],
        "query": prompt,
        "gold": eval_info["gold_raw"],
        "predicted": eval_info["pred_text"],
        "eval_mode": eval_info["mode"],
        "gold_num": eval_info["gold_num"],
        "pred_num": eval_info["pred_num"],
        "pred_num_first": eval_info["pred_num_first"],
        "pred_num_last": eval_info["pred_num_last"],
        "pred_num_any": eval_info["pred_num_any"],
        "pred_num_candidates_count": eval_info["pred_num_candidates_count"],
        "numeric_extract_strategy_used": eval_info["numeric_extract_strategy_used"],
        "score_detail": eval_info["score_detail"],
        "is_correct": is_correct,
        "llm_call_count": llm_call_count,
        "zero_tool_rounds": zero_tool_rounds,
        "blocked_marks": blocked_marks,
        "logs": logs,
        "multimodal_meta": multimodal_meta,
        "workflow_final_answer_source": "planning_flow._synthesize_user_final_answer",
        "llm_model": config.llm["default"].model,
    }


def aggregate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    correct = sum(1 for r in records if r["is_correct"])
    accuracy = (correct / total) if total else 0.0
    status_counter = Counter(r["status"] for r in records)
    error_tags = Counter()
    tool_counter = Counter()
    elapsed = [float(r["elapsed_seconds"]) for r in records]
    numeric_abs_errors: List[float] = []
    llm_calls: List[int] = []
    zero_tool_rounds: List[int] = []
    blocked_marks: List[int] = []
    parse_ambiguity_count = 0
    mm_samples = 0
    mm_resolved_samples = 0
    mm_vision_ok = 0
    mm_vision_failed = 0

    for r in records:
        for tag in r["error_tags"]:
            error_tags[tag] += 1
        for t, c in r["tool_usage"].items():
            tool_counter[t] += c
        llm_calls.append(int(r.get("llm_call_count", 0)))
        zero_tool_rounds.append(int(r.get("zero_tool_rounds", 0)))
        blocked_marks.append(int(r.get("blocked_marks", 0)))
        if (
            r["eval_mode"] == "numeric"
            and r["gold_num"] is not None
            and r["pred_num"] is not None
        ):
            numeric_abs_errors.append(abs(float(r["pred_num"]) - float(r["gold_num"])))
        if (
            r.get("eval_mode") == "numeric"
            and r.get("pred_num_candidates_count", 0) > 1
            and r.get("pred_num_first") != r.get("pred_num_last")
        ):
            parse_ambiguity_count += 1
        mm = r.get("multimodal_meta", {})
        if mm.get("has_image_refs"):
            mm_samples += 1
            if mm.get("resolved_image_count", 0) > 0:
                mm_resolved_samples += 1
            if mm.get("vision_status") == "ok":
                mm_vision_ok += 1
            if mm.get("vision_status") == "failed":
                mm_vision_failed += 1

    elapsed_sorted = sorted(elapsed)

    def _pct(values: List[float], p: float) -> float:
        if not values:
            return 0.0
        k = (len(values) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return round(values[int(k)], 3)
        d0 = values[f] * (c - k)
        d1 = values[c] * (k - f)
        return round(d0 + d1, 3)

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 6),
        "status_counts": dict(status_counter),
        "avg_elapsed_seconds": round(sum(elapsed) / total, 3) if total else 0.0,
        "p50_elapsed_seconds": _pct(elapsed_sorted, 0.5),
        "p95_elapsed_seconds": _pct(elapsed_sorted, 0.95),
        "error_tag_counts": dict(error_tags),
        "tool_usage_counts": dict(tool_counter),
        "numeric_mae": round(sum(numeric_abs_errors) / len(numeric_abs_errors), 6)
        if numeric_abs_errors
        else None,
        "numeric_eval_samples": len(numeric_abs_errors),
        "avg_llm_call_count": round(sum(llm_calls) / len(llm_calls), 3)
        if llm_calls
        else 0.0,
        "avg_zero_tool_rounds": round(
            sum(zero_tool_rounds) / len(zero_tool_rounds), 3
        )
        if zero_tool_rounds
        else 0.0,
        "avg_blocked_marks": round(sum(blocked_marks) / len(blocked_marks), 3)
        if blocked_marks
        else 0.0,
        "numeric_parse_ambiguity_count": parse_ambiguity_count,
        "multimodal_samples": mm_samples,
        "multimodal_resolved_samples": mm_resolved_samples,
        "multimodal_resolution_rate": round(mm_resolved_samples / mm_samples, 6)
        if mm_samples
        else None,
        "multimodal_vision_ok": mm_vision_ok,
        "multimodal_vision_failed": mm_vision_failed,
    }


def write_dataset_outputs(
    output_root: Path,
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
    dataset_meta: Dict[str, Any],
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
            row = {k: v for k, v in r.items() if k != "logs"}
            row["primary_error_tag"] = primary_tag
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_meta": dataset_meta,
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    report_path = output_root / "error_analysis.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Multi-Agent Dataset Report\n\n")
        f.write(f"- Dataset file: `{dataset_meta['dataset_file']}`\n")
        f.write(f"- LLM model: `{dataset_meta['llm_model']}`\n")
        f.write(f"- Planning enabled: `{dataset_meta['planning_enabled']}`\n")
        f.write(f"- Multimodal mode: `{dataset_meta['multimodal_mode']}`\n")
        f.write(f"- Total samples: {summary['total']}\n")
        f.write(f"- Correct: {summary['correct']}\n")
        f.write(f"- Accuracy: {summary['accuracy']:.4f}\n")
        f.write(f"- Avg latency/sample (s): {summary['avg_elapsed_seconds']}\n")
        f.write(f"- P50 latency/sample (s): {summary['p50_elapsed_seconds']}\n")
        f.write(f"- P95 latency/sample (s): {summary['p95_elapsed_seconds']}\n")
        f.write(f"- Avg LLM calls/sample: {summary['avg_llm_call_count']}\n")
        f.write(f"- Avg zero-tool rounds/sample: {summary['avg_zero_tool_rounds']}\n")
        f.write(f"- Avg blocked marks/sample: {summary['avg_blocked_marks']}\n")
        f.write(f"- Multimodal samples: {summary['multimodal_samples']}\n")
        f.write(
            f"- Multimodal resolved samples: {summary['multimodal_resolved_samples']}\n"
        )
        f.write(
            f"- Multimodal resolution rate: {summary['multimodal_resolution_rate']}\n"
        )
        f.write(f"- Vision extraction ok: {summary['multimodal_vision_ok']}\n")
        f.write(f"- Vision extraction failed: {summary['multimodal_vision_failed']}\n")
        if summary["numeric_mae"] is not None:
            f.write(f"- Numeric MAE: {summary['numeric_mae']}\n")
            f.write(f"- Numeric eval samples: {summary['numeric_eval_samples']}\n")
            f.write(
                f"- Numeric parse ambiguity count: {summary['numeric_parse_ambiguity_count']}\n"
            )
        f.write("\n## Status Breakdown\n\n")
        for k, v in sorted(summary["status_counts"].items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {k}: {v}\n")
        f.write("\n## Error Tag Breakdown\n\n")
        for k, v in sorted(
            summary["error_tag_counts"].items(), key=lambda x: x[1], reverse=True
        ):
            f.write(f"- {k}: {v}\n")
        f.write("\n## Tool Usage\n\n")
        for k, v in sorted(
            summary["tool_usage_counts"].items(), key=lambda x: x[1], reverse=True
        ):
            f.write(f"- {k}: {v}\n")
        f.write("\n## Research Notes\n\n")
        f.write("- `timeout` / `runtime_error`: stability and serving-layer risks.\n")
        f.write("- `repeated_tool_loop` / `same_args_repeated`: workflow-control inefficiency.\n")
        f.write("- `prediction_parse_failure`: output contract mismatch (answer extraction risk).\n")
        f.write("- `multimodal_input_missing`: image path resolution gap or data mounting issue.\n")
        f.write("- `multimodal_vision_failure`: vision extraction call failed or timed out.\n")
        f.write("- `numeric_mismatch` / `code_mismatch` / `text_mismatch`: reasoning or grounding error.\n")
        f.write("- `numeric_mae` helps quantify magnitude, not only correctness rate.\n")


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a JSON array: {path}")
    return data


async def run_dataset_job_async(
    dataset_file: Path,
    output_root: Path,
    timeout_s: int,
    offset: int,
    limit_per_dataset: int,
    use_planning_agent: bool,
    image_root: Optional[Path],
    multimodal_mode: str,
    vision_timeout_s: int,
    max_images_per_sample: int,
    numeric_extract_strategy: str,
    rag_top_k: int = 0,
    rag_persist_dir: Optional[Path] = None,
    cot_enabled: bool = True,
) -> Dict[str, Any]:
    data = load_dataset(dataset_file)
    start = max(offset, 0)
    end = len(data) if limit_per_dataset <= 0 else min(len(data), start + limit_per_dataset)
    subset = data[start:end]
    if not subset:
        raise ValueError(f"No samples selected for dataset: {dataset_file}")

    records: List[Dict[str, Any]] = []
    logger.info(f"Dataset {dataset_file.name}: running {len(subset)} samples")
    for i, sample in enumerate(subset, start=1):
        logger.info(f"[{dataset_file.name}] [{i}/{len(subset)}] sample")
        rec = await run_single_sample(
            sample=sample,
            idx=start + i - 1,
            timeout_s=timeout_s,
            use_planning_agent=use_planning_agent,
            dataset_file=dataset_file,
            image_root=image_root,
            multimodal_mode=multimodal_mode,
            vision_timeout_s=vision_timeout_s,
            max_images_per_sample=max_images_per_sample,
            numeric_extract_strategy=numeric_extract_strategy,
            rag_top_k=rag_top_k,
            rag_persist_dir=rag_persist_dir,
            cot_enabled=cot_enabled,
        )
        records.append(rec)

    summary = aggregate_metrics(records)
    dataset_meta = {
        "dataset_file": str(dataset_file),
        "selected_samples": len(subset),
        "offset": offset,
        "limit_per_dataset": limit_per_dataset,
        "planning_enabled": use_planning_agent,
        "multimodal_mode": multimodal_mode,
        "image_root": str(image_root) if image_root else "",
        "numeric_extract_strategy": numeric_extract_strategy,
        "rag_top_k": rag_top_k,
        "rag_persist_dir": str(rag_persist_dir) if rag_persist_dir else "",
        "cot_enabled": cot_enabled,
        "llm_model": config.llm["default"].model,
        "workflow_final_answer_source": "planning_flow._synthesize_user_final_answer",
    }
    write_dataset_outputs(
        output_root=output_root,
        records=records,
        summary=summary,
        dataset_meta=dataset_meta,
    )
    return {
        "dataset_file": str(dataset_file),
        "output_dir": str(output_root),
        "summary": summary,
        "dataset_meta": dataset_meta,
    }


def _safe_name(path: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", path.stem)


def run_dataset_job_sync(job: Dict[str, Any]) -> Dict[str, Any]:
    dataset_file = Path(job["dataset_file"])
    output_root = Path(job["output_root"])
    image_root_raw = str(job.get("image_root", "")).strip()
    image_root = Path(image_root_raw) if image_root_raw else None
    rag_persist_raw = str(job.get("rag_persist_dir", "")).strip()
    rag_persist_dir = Path(rag_persist_raw) if rag_persist_raw else None
    return asyncio.run(
        run_dataset_job_async(
            dataset_file=dataset_file,
            output_root=output_root,
            timeout_s=int(job["timeout_s"]),
            offset=int(job["offset"]),
            limit_per_dataset=int(job["limit_per_dataset"]),
            use_planning_agent=bool(job["use_planning_agent"]),
            image_root=image_root,
            multimodal_mode=str(job.get("multimodal_mode", "best_effort")),
            vision_timeout_s=int(job.get("vision_timeout_s", 120)),
            max_images_per_sample=int(job.get("max_images_per_sample", 1)),
            numeric_extract_strategy=str(job.get("numeric_extract_strategy", "any")),
            rag_top_k=int(job.get("rag_top_k", 0)),
            rag_persist_dir=rag_persist_dir,
            cot_enabled=bool(job.get("cot_enabled", True)),
        )
    )


def write_global_outputs(output_root: Path, run_meta: Dict[str, Any], job_results: List[Dict[str, Any]]):
    summary_rows = []
    overall_total = 0
    overall_correct = 0
    for item in job_results:
        s = item["summary"]
        row = {
            "dataset_file": item["dataset_file"],
            "output_dir": item["output_dir"],
            "total": s["total"],
            "correct": s["correct"],
            "accuracy": s["accuracy"],
            "avg_elapsed_seconds": s["avg_elapsed_seconds"],
            "p95_elapsed_seconds": s["p95_elapsed_seconds"],
            "numeric_mae": s["numeric_mae"],
            "multimodal_resolution_rate": s["multimodal_resolution_rate"],
            "multimodal_vision_failed": s["multimodal_vision_failed"],
        }
        overall_total += s["total"]
        overall_correct += s["correct"]
        summary_rows.append(row)

    global_summary = {
        "run_meta": run_meta,
        "overall": {
            "total": overall_total,
            "correct": overall_correct,
            "accuracy": round(overall_correct / overall_total, 6) if overall_total else 0.0,
        },
        "datasets": summary_rows,
    }

    with (output_root / "global_summary.json").open("w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)

    with (output_root / "global_report.md").open("w", encoding="utf-8") as f:
        f.write("# Multi-Dataset Multi-Agent Benchmark Report\n\n")
        f.write(f"- Model: `{run_meta['llm_model']}`\n")
        f.write(f"- Planning enabled: `{run_meta['planning_enabled']}`\n")
        f.write(f"- Dataset selection: `{run_meta['dataset_selection']}`\n")
        f.write(f"- Overall accuracy: {global_summary['overall']['accuracy']:.4f}\n")
        f.write(
            f"- Overall correct/total: {global_summary['overall']['correct']}/{global_summary['overall']['total']}\n\n"
        )
        f.write("## Dataset Comparison\n\n")
        for row in summary_rows:
            f.write(
                f"- `{row['dataset_file']}`: acc={row['accuracy']:.4f}, "
                f"correct={row['correct']}/{row['total']}, "
                f"avg={row['avg_elapsed_seconds']}s, p95={row['p95_elapsed_seconds']}s, "
                f"numeric_mae={row['numeric_mae']}, "
                f"mm_resolve_rate={row['multimodal_resolution_rate']}, "
                f"mm_vision_failed={row['multimodal_vision_failed']}\n"
            )
        f.write("\n## Scientific Guidance\n\n")
        f.write("- 先按数据集分别分析错因分布，避免被总体均值掩盖。\n")
        f.write("- 对同一数据集至少重复 3 次运行，报告均值与标准差。\n")
        f.write("- 对失败样本进行分层抽样（数值、代码、文本）做人工复核。\n")
        f.write("- 对高延迟样本追踪 tool 调用链，识别可裁剪步骤。\n")


def main() -> None:
    args = parse_args()
    if args.force_planning and args.disable_planning:
        raise ValueError("Cannot use --force-planning and --disable-planning together.")

    dataset_inputs = resolve_dataset_inputs(args.dataset)
    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    if dataset_root and not dataset_root.is_absolute():
        dataset_root = (Path.cwd() / dataset_root).resolve()
    dataset_files = resolve_dataset_files(
        dataset_inputs,
        dataset_root=dataset_root,
        dataset_choice=args.dataset,
    )

    if args.force_planning:
        use_planning = True
    elif args.disable_planning:
        use_planning = False
    else:
        use_planning = bool(config.run_flow_config.use_planning_agent)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) / f"multi_agent_multi_dataset_{run_id}"
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    image_root = Path(args.image_root) if args.image_root else None
    if image_root and not image_root.is_absolute():
        image_root = (Path.cwd() / image_root).resolve()

    rag_persist_dir = Path(args.rag_persist_dir) if args.rag_persist_dir else (_workspace_root / "rag" / "chroma_db")
    cot_enabled = getattr(args, "cot_enabled", True)

    run_meta = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "llm_model": config.llm["default"].model,
        "planning_enabled": use_planning,
        "multimodal_mode": args.multimodal_mode,
        "numeric_extract_strategy": args.numeric_extract_strategy,
        "image_root": str(image_root) if image_root else "",
        "vision_timeout_s": args.vision_timeout,
        "max_images_per_sample": args.max_images_per_sample,
        "timeout_s": args.timeout,
        "offset": args.offset,
        "limit_per_dataset": args.limit_per_dataset,
        "dataset_selection": args.dataset,
        "datasets": [str(p) for p in dataset_files],
        "rag_top_k": args.rag_top_k,
        "rag_persist_dir": str(rag_persist_dir),
        "cot_enabled": cot_enabled,
        "workflow_final_answer_source": "planning_flow._synthesize_user_final_answer",
    }
    with (output_root / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    jobs = []
    for dataset_file in dataset_files:
        dataset_out = output_root / f"{dataset_file.parent.name}__{_safe_name(dataset_file)}"
        jobs.append(
            {
                "dataset_file": str(dataset_file),
                "output_root": str(dataset_out),
                "timeout_s": args.timeout,
                "offset": args.offset,
                "limit_per_dataset": args.limit_per_dataset,
                "use_planning_agent": use_planning,
                "image_root": str(image_root) if image_root else "",
                "multimodal_mode": args.multimodal_mode,
                "vision_timeout_s": args.vision_timeout,
                "max_images_per_sample": args.max_images_per_sample,
                "numeric_extract_strategy": args.numeric_extract_strategy,
                "rag_top_k": args.rag_top_k,
                "rag_persist_dir": str(rag_persist_dir),
                "cot_enabled": cot_enabled,
            }
        )

    logger.info(f"Resolved dataset files: {len(jobs)}")
    logger.info(f"Model: {config.llm['default'].model}")
    logger.info(f"Planning enabled: {use_planning}")
    logger.info(f"Multimodal mode: {args.multimodal_mode}")
    logger.info(f"Image root: {image_root}")
    logger.info(f"Output root: {output_root}")
    logger.info("Dataset execution mode: sequential")

    job_results: List[Dict[str, Any]] = []
    for i, job in enumerate(jobs, start=1):
        logger.info(f"[{i}/{len(jobs)}] Running dataset: {job['dataset_file']}")
        result = run_dataset_job_sync(job)
        job_results.append(result)
        s = result["summary"]
        logger.info(
            f"[{i}/{len(jobs)}] Completed {result['dataset_file']} | "
            f"acc={s['accuracy']:.4f} ({s['correct']}/{s['total']}) | "
            f"out={result['output_dir']}"
        )

    write_global_outputs(output_root=output_root, run_meta=run_meta, job_results=job_results)
    logger.info("All dataset jobs completed.")
    logger.info(f"Global artifacts saved to: {output_root}")


if __name__ == "__main__":
    main()
