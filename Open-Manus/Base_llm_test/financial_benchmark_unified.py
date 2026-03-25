#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金融LLM基准测试 - 统一框架（支持实时API + Batch API）

支持两种模式:
  - realtime: 同步API，适合冒烟测试，可加 --limit N 快速验证
  - batch: OpenAI Batch API，适合大规模测试，约50%成本节省

使用方法:
    # 冒烟测试：仅跑前10个样本，实时反馈
    python financial_benchmark_unified.py --full --mode realtime --limit 10

    # 完整实时测试（与原有逻辑一致）
    python financial_benchmark_unified.py --full --mode realtime

    # Batch模式 - 阶段1：提交任务
    python financial_benchmark_unified.py --full --mode batch --batch-phase submit

    # Batch模式 - 阶段2：轮询并下载结果（需 OpenAI 官方 API）
    python financial_benchmark_unified.py --full --mode batch --batch-phase fetch --batch-id <batch_id>

注意: Batch API 仅支持 OpenAI 官方 API (api.openai.com)，第三方如 Silicon Flow 不支持。
      Batch 模式需设置 OPENAI_API_KEY，且 base_url 为 https://api.openai.com/v1
"""

import json
import os
import re
import time
import argparse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# 复用原有框架
from financial_benchmark_framework_final import (
    FinancialLLMBenchmarkFinal,
    _model_to_tag,
)


class FinancialLLMBenchmarkUnified(FinancialLLMBenchmarkFinal):
    """扩展版：支持 realtime + batch 双模式"""

    def __init__(self, *args, batch_dir: str = "./batch_benchmark", **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_dir = batch_dir
        os.makedirs(os.path.join(batch_dir, "input"), exist_ok=True)
        os.makedirs(os.path.join(batch_dir, "output"), exist_ok=True)

    def _check_batch_api_support(self) -> bool:
        """检查 base_url 是否支持 Batch API（仅 OpenAI 官方）"""
        url = (self.base_url or "").lower()
        return "api.openai.com" in url or "openai.com/v1" in url

    # ==================== 数据集：构建 Batch 请求 ====================

    def _build_messages_for_item(
        self, dataset_key: str, item: dict, idx: int
    ) -> Tuple[Optional[List[dict]], Optional[dict]]:
        """
        为单个样本构建 OpenAI messages，用于 Batch API。
        返回 (messages, metadata)，metadata 用于后续评估。
        若样本无效返回 (None, None)。
        """
        if dataset_key == "finben":
            return self._build_finben_batch_item(item, idx)
        if dataset_key == "bizbench":
            return self._build_bizbench_batch_item(item, idx)
        if dataset_key.startswith("finmmr_"):
            return self._build_finmmr_batch_item(item, idx, dataset_key)
        if dataset_key == "convfinqa":
            return self._build_convfinqa_batch_item(item, idx)
        return None, None

    def _build_finben_batch_item(
        self, item: dict, idx: int
    ) -> Tuple[Optional[List[dict]], Optional[dict]]:
        query = item.get("query")
        answer_raw = item.get("answer")
        if not isinstance(query, str) or not isinstance(answer_raw, str):
            return None, None
        prompt = f"""{query}

请只回答: HAWKISH, DOVISH, 或 NEUTRAL"""
        messages = [{"role": "user", "content": prompt}]
        metadata = {"item_id": item.get("id", f"finben_{idx}"), "answer": answer_raw.lower().strip()}
        return messages, metadata

    def _build_bizbench_batch_item(
        self, item: dict, idx: int
    ) -> Tuple[Optional[List[dict]], Optional[dict]]:
        question = item.get("question")
        answer_raw = item.get("answer")
        task = item.get("task", "")
        if not isinstance(question, str) or answer_raw is None:
            return None, None
        prompt = self._build_bizbench_prompt(
            question=question,
            task=task,
            context=item.get("context"),
            options=item.get("options"),
        )
        image_paths = self._resolve_image_paths(question, item.get("context"))
        content: Any = prompt
        if self.enable_vision and image_paths:
            content = [{"type": "text", "text": prompt}]
            for path in image_paths:
                data_url = self._image_to_data_url(path)
                if data_url:
                    content.append({"type": "image_url", "image_url": {"url": data_url}})
        messages = [{"role": "user", "content": content}]
        metadata = {
            "item_id": f"bizbench_{idx}",
            "answer": str(answer_raw).strip(),
            "task": task,
            "options": item.get("options"),
        }
        return messages, metadata

    def _build_finmmr_batch_item(
        self, item: dict, idx: int, dataset_key: str
    ) -> Tuple[Optional[List[dict]], Optional[dict]]:
        question = item.get("question")
        ground_truth_raw = item.get("ground_truth")
        if not isinstance(question, str):
            return None, None
        try:
            ground_truth = float(ground_truth_raw)
        except Exception:
            return None, None
        context = item.get("context", "")
        prompt = f"""你是一个金融分析专家。请根据以下信息回答问题，只输出最终的数值答案。

问题: {question}

上下文信息:
{context}

请只输出最终的数值答案，不要包含任何解释或单位。"""
        image_refs = []
        image_refs.extend(item.get("images") or [])
        image_refs.extend(item.get("ground_images") or [])
        image_paths = self._resolve_image_paths(question, context, extra_refs=image_refs)
        content: Any = prompt
        if self.enable_vision and image_paths:
            content = [{"type": "text", "text": prompt}]
            for path in image_paths:
                data_url = self._image_to_data_url(path)
                if data_url:
                    content.append({"type": "image_url", "image_url": {"url": data_url}})
        messages = [{"role": "user", "content": content}]
        metadata = {
            "item_id": item.get("question_id", f"finmmr_{dataset_key}_{idx}"),
            "ground_truth": ground_truth,
        }
        return messages, metadata

    def _build_convfinqa_batch_item(
        self, item: dict, idx: int
    ) -> Tuple[Optional[List[dict]], Optional[dict]]:
        query = item.get("query")
        answer_raw = item.get("answer")
        if not isinstance(query, str) or answer_raw is None:
            return None, None
        prompt = f"""{query}

请只输出最终的数值答案，不要包含任何解释、单位或计算过程。"""
        messages = [{"role": "user", "content": prompt}]
        metadata = {"item_id": item.get("id", f"convfinqa_{idx}"), "answer": str(answer_raw).strip()}
        return messages, metadata

    # ==================== Batch 提交 ====================

    def batch_submit_dataset(
        self,
        dataset_key: str,
        test_file: str,
        limit: Optional[int] = None,
    ) -> Optional[str]:
        """
        准备并提交 Batch 任务。返回 batch_id，失败返回 None。
        """
        if not self._check_batch_api_support():
            print("  错误: Batch API 仅支持 OpenAI 官方 API (api.openai.com)，当前 base_url 不支持。")
            return None

        with open(test_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if limit:
            data = data[:limit]
        total = len(data)

        batch_items: List[Tuple[int, List[dict], dict]] = []
        for idx, item in enumerate(data):
            messages, metadata = self._build_messages_for_item(dataset_key, item, idx)
            if messages is not None and metadata is not None:
                batch_items.append((idx, messages, metadata))

        if not batch_items:
            print(f"  无有效样本可提交。")
            return None

        model_tag = _model_to_tag(self.model)
        prefix = f"{dataset_key}_{model_tag}"
        input_path = os.path.join(self.batch_dir, "input", f"{prefix}_batch.jsonl")
        meta_path = os.path.join(self.batch_dir, "input", f"{prefix}_metadata.json")

        with open(input_path, "w", encoding="utf-8") as f:
            for i, (orig_idx, messages, _) in enumerate(batch_items):
                req = {
                    "custom_id": str(i),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.0,
                        "max_tokens": 2048,
                    },
                }
                f.write(json.dumps(req, ensure_ascii=False) + "\n")

        meta = {
            "dataset_key": dataset_key,
            "total": len(batch_items),
            "items": [m for _, _, m in batch_items],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"  已生成 {len(batch_items)} 条请求 -> {input_path}")

        try:
            with open(input_path, "rb") as f:
                file_obj = self.client.files.create(file=f, purpose="batch")
            batch = self.client.batches.create(
                input_file_id=file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": prefix},
            )
            batch_id = batch.id
            print(f"  Batch 已提交: {batch_id}")
            print(f"  可在 https://platform.openai.com/batches 查看状态")
            return batch_id
        except Exception as e:
            print(f"  提交失败: {e}")
            return None

    # ==================== Batch 拉取与评估 ====================

    def _format_batch_response(self, resp: Optional[dict]) -> Optional[str]:
        """从 Batch 单条响应中提取 content。Batch 输出格式: {response: {status_code, body: {...}}}"""
        if not resp:
            return None
        try:
            r = resp.get("response", {})
            if r.get("status_code") != 200:
                return None
            body = r.get("body", {})
            choices = body.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content")
                return content.strip() if content else None
        except Exception:
            pass
        return None

    def batch_fetch_and_evaluate(
        self,
        dataset_key: str,
        test_file: str,
        batch_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        轮询 Batch 完成，下载结果并评估。返回该数据集的 results 字典。
        """
        if not self._check_batch_api_support():
            print("  错误: Batch API 仅支持 OpenAI 官方 API。")
            return None

        model_tag = _model_to_tag(self.model)
        prefix = f"{dataset_key}_{model_tag}"
        meta_path = os.path.join(self.batch_dir, "input", f"{prefix}_metadata.json")
        output_path = os.path.join(self.batch_dir, "output", f"{prefix}_{batch_id}_results.jsonl")

        if not os.path.exists(meta_path):
            print(f"  元数据文件不存在: {meta_path}，请先执行 submit。")
            return None

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        items_meta = meta["items"]

        print(f"  轮询 Batch {batch_id} ...")
        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status
            print(f"    状态: {status}")
            if status == "completed":
                break
            if status in ("failed", "cancelled", "expired"):
                print(f"  Batch 异常结束: {status}")
                return None
            time.sleep(30)

        content_obj = self.client.files.content(batch.output_file_id)
        raw = content_obj.read() if hasattr(content_obj, "read") else content_obj
        content = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        results_list = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                results_list.append(json.loads(line))

        # 按 custom_id 排序并评估
        results_by_id = {r["custom_id"]: r for r in results_list}
        correct = 0
        evaluated_total = 0
        api_errors = 0
        predictions = []
        ground_truths = []
        error_cases = []
        logs = []
        task_metrics: Dict[str, Dict[str, int]] = {}

        for i, m in enumerate(items_meta):
            r = results_by_id.get(str(i))
            content = self._format_batch_response(r) if r else None
            item_id = m.get("item_id", f"item_{i}")

            if content is None:
                api_errors += 1
                task_name = self._normalize_task_name(m.get("task")) if dataset_key == "bizbench" else ""
                if task_name:
                    self._update_task_metric(task_metrics, task_name, "api_errors")
                logs.append(
                    self._build_error_log_entry(
                        dataset=dataset_key,
                        item_id=item_id,
                        error_type="api_error",
                        response="",
                    )
                )
                continue

            if dataset_key == "finben":
                pred = self._extract_finben_answer(content)
                truth = m["answer"]
                is_correct = pred == truth
            elif dataset_key == "bizbench":
                pred = self._extract_bizbench_answer(
                    content, task=m.get("task"), options=m.get("options")
                )
                truth = m["answer"]
                is_correct = self._compare_bizbench_answer(
                    pred, truth, task=m.get("task"), options=m.get("options")
                )
                task_name = self._normalize_task_name(m.get("task"))
                self._update_task_metric(task_metrics, task_name, "total")
                self._update_task_metric(task_metrics, task_name, "evaluated_total")
                if is_correct:
                    self._update_task_metric(task_metrics, task_name, "correct")
            elif dataset_key.startswith("finmmr_"):
                pred = self._extract_finmmr_answer(content)
                truth = m["ground_truth"]
                is_correct = self._compare_numeric(pred, truth, tolerance=0.002)
            elif dataset_key == "convfinqa":
                pred = self._extract_convfinqa_answer(content)
                truth = m["answer"]
                is_correct = self._compare_convfinqa_answer(pred, truth)
            else:
                is_correct = False
                pred = content

            evaluated_total += 1
            if is_correct:
                correct += 1
            else:
                ec = {"id": item_id, "prediction": pred, "ground_truth": truth}
                if dataset_key == "bizbench":
                    ec["task"] = m.get("task", "")
                error_cases.append(ec)
            predictions.append(pred)
            ground_truths.append(truth)
            log_entry = {
                "dataset": dataset_key,
                "id": item_id,
                "prompt": "",
                "response": content,
                "prediction": pred,
                "ground_truth": truth,
                "correct": is_correct,
            }
            if dataset_key == "bizbench":
                log_entry["task"] = m.get("task", "")
            logs.append(log_entry)

        accuracy = correct / evaluated_total * 100 if evaluated_total else 0
        result = {
            "dataset": dataset_key,
            "total": len(items_meta),
            "evaluated_total": evaluated_total,
            "api_errors": api_errors,
            "data_errors": 0,
            "correct": correct,
            "accuracy": accuracy,
            "predictions": predictions,
            "ground_truths": ground_truths,
            "error_cases": error_cases,
        }
        if task_metrics:
            result["task_metrics"] = task_metrics
        self.results[dataset_key] = result
        for log in logs:
            self.logs.append(log)

        print(f"  {dataset_key}: 正确 {correct}/{evaluated_total}, 准确率 {accuracy:.2f}%")
        return result

    def run_dataset(
        self,
        dataset_key: str,
        test_file: str,
        mode: str = "realtime",
        limit: Optional[int] = None,
        resume: bool = False,
        batch_size: int = 50,
        batch_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        统一入口：根据 mode 调用 realtime 或 batch。
        """
        if mode == "batch":
            if batch_id:
                return self.batch_fetch_and_evaluate(dataset_key, test_file, batch_id)
            return {"batch_id": self.batch_submit_dataset(dataset_key, test_file, limit)}

        # realtime
        if dataset_key == "finben":
            return self._run_finben_realtime(test_file, limit, resume, batch_size)
        if dataset_key == "bizbench":
            return self._run_bizbench_realtime(test_file, limit, resume, batch_size)
        if dataset_key in ("finmmr_easy", "finmmr_medium", "finmmr_hard"):
            return self._run_finmmr_realtime(test_file, limit, resume, batch_size)
        if dataset_key == "convfinqa":
            return self._run_convfinqa_realtime(test_file, limit, resume, batch_size)
        return None

    # _run_*_realtime 由下方 _run_*_realtime_impl 实现（支持 limit 通过临时文件）


def _create_limited_test_file(data_path: str, limit: int) -> str:
    """创建限制样本数的临时测试文件"""
    import tempfile
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = data[:limit]
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


# 重写 run_dataset 中的 realtime 调用，传入 limit
def _apply_limit_to_test_file(test_file: str, limit: Optional[int]) -> str:
    if not limit:
        return test_file
    return _create_limited_test_file(test_file, limit)


# 修正：基类 test_* 没有 limit 参数，需要在调用前先生成限制后的文件
FinancialLLMBenchmarkUnified._create_limited_test_file = staticmethod(_create_limited_test_file)
FinancialLLMBenchmarkUnified._apply_limit_to_test_file = staticmethod(_apply_limit_to_test_file)


def _run_finben_realtime_impl(self, test_file, limit, resume, batch_size):
    path = self._apply_limit_to_test_file(test_file, limit)
    try:
        return self.test_finben_full(path, resume=resume, batch_size=batch_size)
    finally:
        if limit and path != test_file and os.path.exists(path):
            os.unlink(path)


def _run_bizbench_realtime_impl(self, test_file, limit, resume, batch_size):
    path = self._apply_limit_to_test_file(test_file, limit)
    try:
        return self.test_bizbench_full(path, resume=resume, batch_size=batch_size)
    finally:
        if limit and path != test_file and os.path.exists(path):
            os.unlink(path)


def _run_finmmr_realtime_impl(self, test_file, limit, resume, batch_size):
    path = self._apply_limit_to_test_file(test_file, limit)
    try:
        return self.test_finmmr_full(path, resume=resume, batch_size=batch_size)
    finally:
        if limit and path != test_file and os.path.exists(path):
            os.unlink(path)


def _run_convfinqa_realtime_impl(self, test_file, limit, resume, batch_size):
    path = self._apply_limit_to_test_file(test_file, limit)
    try:
        return self.test_convfinqa_full(path, resume=resume, batch_size=batch_size)
    finally:
        if limit and path != test_file and os.path.exists(path):
            os.unlink(path)


FinancialLLMBenchmarkUnified._run_finben_realtime = _run_finben_realtime_impl
FinancialLLMBenchmarkUnified._run_bizbench_realtime = _run_bizbench_realtime_impl
FinancialLLMBenchmarkUnified._run_finmmr_realtime = _run_finmmr_realtime_impl
FinancialLLMBenchmarkUnified._run_convfinqa_realtime = _run_convfinqa_realtime_impl


def main():
    """主程序：支持 realtime / batch 双模式"""
    parser = argparse.ArgumentParser(
        description="金融LLM基准测试 - 统一框架（实时API + Batch API）"
    )
    parser.add_argument("--full", action="store_true", help="执行完整测试")
    parser.add_argument("--resume", action="store_true", help="从断点继续（仅 realtime）")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=[
            "all",
            "finben",
            "bizbench",
            "finmmr_easy",
            "finmmr_medium",
            "finmmr_hard",
            "convfinqa",
        ],
        help="指定数据集",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="realtime",
        choices=["realtime", "batch"],
        help="realtime=同步API实时反馈; batch=OpenAI Batch API 节省成本",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="冒烟测试：仅跑前 N 个样本（realtime 实时反馈，batch 限制提交量）",
    )
    parser.add_argument(
        "--batch-phase",
        type=str,
        choices=["submit", "fetch"],
        help="Batch 模式阶段: submit=提交任务, fetch=轮询并评估",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Batch 模式 fetch 时指定 batch_id（单数据集时使用）",
    )
    parser.add_argument(
        "--batch-ids-file",
        type=str,
        default="./batch_benchmark/batch_ids.json",
        help="多数据集时 submit 保存的 batch_id 映射文件",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="检查点保存间隔（realtime）")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--base-url", type=str, default="https://api.siliconflow.cn/v1")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--checkpoint-root", type=str, default="./checkpoints")
    parser.add_argument("--results-root", type=str, default="./results_full")
    parser.add_argument("--image-root", type=str, default="")
    parser.add_argument("--enable-vision", action="store_true")
    parser.add_argument("--disable-vision", action="store_true")
    parser.add_argument("--api-timeout", type=int, default=120)

    args = parser.parse_args()

    if args.dataset == "all" and not args.full:
        print("错误: --dataset=all 时需加 --full")
        return
    if args.mode == "batch" and not args.batch_phase:
        print("错误: Batch 模式需指定 --batch-phase submit 或 fetch")
        return

    API_KEY = (
        args.api_key
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("SILICONFLOW_API_KEY", "")
    ).strip()
    if not API_KEY:
        print("错误: 未设置 API Key（--api-key 或环境变量）")
        return

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_TAG = _model_to_tag(args.model)
    CHECKPOINT_DIR = os.path.join(args.checkpoint_root, MODEL_TAG)
    RESULTS_DIR = os.path.join(args.results_root, MODEL_TAG)
    default_image_root = os.path.join(BASE_DIR, "Dataset", "FinMMR-main")
    IMAGE_ROOT = args.image_root.strip() or default_image_root
    auto_vision = any(k in args.model.lower() for k in ("vl", "omni", "vision"))
    ENABLE_VISION = args.enable_vision or (auto_vision and not args.disable_vision)

    DATASETS = {
        "finben": os.path.join(BASE_DIR, "Dataset", "FinBen", "finben_test.json"),
        "bizbench": os.path.join(BASE_DIR, "Dataset", "bizbench_test", "bizbench_test.json"),
        "finmmr_easy": os.path.join(BASE_DIR, "Dataset", "finmmr", "finmmr_easy_test.json"),
        "finmmr_medium": os.path.join(BASE_DIR, "Dataset", "finmmr", "finmmr_medium_test.json"),
        "finmmr_hard": os.path.join(BASE_DIR, "Dataset", "finmmr", "finmmr_hard_test.json"),
        "convfinqa": os.path.join(
            BASE_DIR, "Dataset", "flarez-confinqa_test", "flare-convfinqa_test.json"
        ),
    }

    benchmark = FinancialLLMBenchmarkUnified(
        model=args.model,
        base_url=args.base_url,
        api_key=API_KEY,
        checkpoint_dir=CHECKPOINT_DIR,
        image_root=IMAGE_ROOT,
        enable_vision=ENABLE_VISION,
        api_timeout=args.api_timeout,
    )

    to_run = (
        [args.dataset]
        if args.dataset != "all"
        else ["finben", "bizbench", "finmmr_easy", "finmmr_medium", "finmmr_hard", "convfinqa"]
    )

    print("=" * 70)
    print("金融LLM基准测试 - 统一框架")
    print("=" * 70)
    print(f"模式: {args.mode}" + (f" (phase={args.batch_phase})" if args.mode == "batch" else ""))
    print(f"数据集: {to_run}")
    if args.limit:
        print(f"限制: 前 {args.limit} 样本（冒烟测试）")
    print("=" * 70)

    if args.mode == "batch":
        if args.batch_phase == "submit":
            batch_ids = {}
            for dk in to_run:
                path = DATASETS.get(dk)
                if not path or not os.path.exists(path):
                    print(f"  跳过 {dk}: 文件不存在")
                    continue
                bid = benchmark.batch_submit_dataset(dk, path, limit=args.limit)
                if bid:
                    batch_ids[dk] = bid
            os.makedirs(os.path.dirname(args.batch_ids_file) or ".", exist_ok=True)
            with open(args.batch_ids_file, "w", encoding="utf-8") as f:
                json.dump(batch_ids, f, ensure_ascii=False, indent=2)
            print(f"\nBatch ID 已保存: {args.batch_ids_file}")
            print("稍后使用 --batch-phase fetch 拉取结果")
        else:  # fetch
            if args.batch_id:
                # 单数据集
                dk = to_run[0]
                path = DATASETS.get(dk, "")
                benchmark.batch_fetch_and_evaluate(dk, path, args.batch_id)
            else:
                # 多数据集：从文件读取
                if not os.path.exists(args.batch_ids_file):
                    print(f"错误: 未找到 {args.batch_ids_file}，请先执行 submit")
                    return
                with open(args.batch_ids_file, "r", encoding="utf-8") as f:
                    batch_ids = json.load(f)
                for dk in to_run:
                    bid = batch_ids.get(dk)
                    if not bid:
                        continue
                    path = DATASETS.get(dk, "")
                    benchmark.batch_fetch_and_evaluate(dk, path, bid)
            benchmark.save_results(output_dir=RESULTS_DIR)
            print("\nBatch 评估完成，结果已保存")
    else:
        # realtime
        for dk in to_run:
            path = DATASETS.get(dk)
            if not path or not os.path.exists(path):
                print(f"  跳过 {dk}: 文件不存在")
                continue
            benchmark.run_dataset(
                dk,
                path,
                mode="realtime",
                limit=args.limit,
                resume=args.resume,
                batch_size=args.batch_size,
            )
        benchmark.save_results(output_dir=RESULTS_DIR)
        print("\n实时测试完成")


if __name__ == "__main__":
    main()
