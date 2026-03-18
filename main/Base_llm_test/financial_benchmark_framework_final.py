#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

支持数据集: FinBen, BizBench, FinMMR, FLARE-ConvFinQA
模型: Qwen/Qwen3-8B via Silicon Flow API

完整数据集规模:
- FinBen: 496 样本
- BizBench: 4,673 样本  
- FinMMR Easy: 1,200 样本
- FinMMR Medium: 1,200 样本
- FinMMR Hard: 1,000 样本
- ConvFinQA: 1,490 样本
- 总计: 10,059 样本

使用方法:
    # 测试完整数据集
    python financial_benchmark_framework_final.py --full

    # 从断点继续测试
    python financial_benchmark_framework_final.py --resume

    # 测试指定数据集
    python financial_benchmark_framework_final.py --dataset finben --full
"""

import json
import os
import re
import time
import argparse
import base64
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from openai import OpenAI


def _model_to_tag(model_name: str) -> str:
    """将模型名转为安全目录/文件名"""
    tag = re.sub(r'[^a-zA-Z0-9._-]+', '_', model_name.strip())
    tag = tag.strip('._-')
    return tag or "unknown_model"


class FinancialLLMBenchmarkFinal:
    """金融LLM基准测试框架 - 最终修复版"""

    def __init__(self, model: str, base_url: str, api_key: str, 
                 checkpoint_dir: str = "./checkpoints",
                 image_root: Optional[str] = None,
                 enable_vision: bool = False,
                 api_timeout: int = 120):
        """
        初始化测试框架

        Args:
            model: 模型名称
            base_url: API基础URL
            api_key: API密钥
            checkpoint_dir: 检查点保存目录
            image_root: 图片资源根目录（用于VL模型）
            enable_vision: 是否启用图片输入
            api_timeout: 单次API调用超时时间（秒）
        """
        self.model = model
        self.base_url = base_url.rstrip()  # 移除末尾空格
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        self.checkpoint_dir = checkpoint_dir
        self.image_root = image_root
        self.enable_vision = enable_vision
        self.api_timeout = api_timeout
        self.results = {}
        self.logs = []

        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _extract_image_refs(self, text: str) -> List[str]:
        """从文本中提取可能的图片文件引用"""
        if not isinstance(text, str) or not text:
            return []
        refs = re.findall(r'([A-Za-z0-9_./\\-]+\.(?:png|jpg|jpeg|webp))', text, flags=re.IGNORECASE)
        # 去重并保持顺序
        seen = set()
        ordered = []
        for ref in refs:
            norm = ref.strip()
            if norm and norm not in seen:
                seen.add(norm)
                ordered.append(norm)
        return ordered

    def _resolve_image_paths(self, *texts: Any, extra_refs: Optional[List[str]] = None) -> List[str]:
        """从多个文本字段解析并定位本地图片路径"""
        refs: List[str] = []
        for text in texts:
            refs.extend(self._extract_image_refs(str(text) if text is not None else ""))
        if extra_refs:
            refs.extend([str(r) for r in extra_refs if r])

        if not refs:
            return []

        candidates: List[str] = []
        for ref in refs:
            ref = str(ref).strip()
            basename = os.path.basename(ref)
            if os.path.isabs(ref):
                candidates.append(ref)
            if self.image_root:
                candidates.append(os.path.join(self.image_root, ref))
                candidates.append(os.path.join(self.image_root, basename))
                candidates.append(os.path.join(self.image_root, 'images', basename))
                candidates.append(os.path.join(self.image_root, 'MultiFinance', 'images', basename))

        existing: List[str] = []
        seen = set()
        for path in candidates:
            norm = os.path.normpath(path)
            if os.path.exists(norm) and norm not in seen:
                seen.add(norm)
                existing.append(norm)
        return existing

    def _image_to_data_url(self, image_path: str) -> Optional[str]:
        """将本地图片编码为data URL"""
        try:
            ext = os.path.splitext(image_path)[1].lower().replace('.', '')
            mime = 'jpeg' if ext in ('jpg', 'jpeg') else ext
            if mime not in ('png', 'jpeg', 'webp'):
                return None
            with open(image_path, 'rb') as f:
                data = base64.b64encode(f.read()).decode('utf-8')
            return f"data:image/{mime};base64,{data}"
        except Exception:
            return None

    def call_model(self, prompt: str, temperature: float = 0.0, 
                   max_tokens: int = 2048, retry: int = 3,
                   image_paths: Optional[List[str]] = None) -> Optional[str]:
        """调用LLM模型（带重试机制）"""
        for attempt in range(retry):
            try:
                message_content: Any = prompt
                if self.enable_vision and image_paths:
                    content_parts = [{"type": "text", "text": prompt}]
                    for image_path in image_paths:
                        data_url = self._image_to_data_url(image_path)
                        if data_url:
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": data_url}
                            })
                    if len(content_parts) > 1:
                        message_content = content_parts
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": message_content}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.api_timeout
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"  API调用错误 (尝试 {attempt+1}/{retry}): {str(e)}")
                if attempt < retry - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    return None

    def _build_error_log_entry(self, dataset: str, item_id: str, error_type: str,
                               prompt: str = "", response: str = "",
                               prediction: Any = "", ground_truth: Any = "") -> Dict[str, Any]:
        """构建错误日志（用于区分API/数据错误与模型错误）"""
        return {
            'dataset': dataset,
            'id': item_id,
            'prompt': prompt,
            'response': response,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'correct': False,
            'error_type': error_type
        }

    def _normalize_task_name(self, task: Any) -> str:
        """标准化BizBench任务名"""
        if task is None:
            return 'unknown'
        task_name = str(task).strip()
        return task_name if task_name else 'unknown'

    def _init_task_metric(self) -> Dict[str, int]:
        """初始化单个任务统计项"""
        return {
            'total': 0,
            'evaluated_total': 0,
            'correct': 0,
            'api_errors': 0,
            'data_errors': 0
        }

    def _update_task_metric(self, task_metrics: Dict[str, Dict[str, int]],
                            task_name: str, field: str, inc: int = 1):
        """更新BizBench任务统计"""
        metric = task_metrics.setdefault(task_name, self._init_task_metric())
        metric[field] = metric.get(field, 0) + inc

    def _rebuild_bizbench_task_metrics(self, logs: List[dict]) -> Dict[str, Dict[str, int]]:
        """从checkpoint日志重建BizBench任务统计"""
        task_metrics: Dict[str, Dict[str, int]] = {}
        for log in logs:
            task_name = self._normalize_task_name(log.get('task'))
            self._update_task_metric(task_metrics, task_name, 'total')
            error_type = log.get('error_type')
            if error_type == 'api_error':
                self._update_task_metric(task_metrics, task_name, 'api_errors')
                continue
            if error_type == 'data_error':
                self._update_task_metric(task_metrics, task_name, 'data_errors')
                continue
            self._update_task_metric(task_metrics, task_name, 'evaluated_total')
            if log.get('correct'):
                self._update_task_metric(task_metrics, task_name, 'correct')
        return task_metrics

    def save_checkpoint(self, dataset_name: str, processed_ids: set, 
                        current_results: dict, current_logs: list):
        """保存检查点"""
        checkpoint = {
            'dataset': dataset_name,
            'processed_ids': list(processed_ids),
            'results': current_results,
            'logs': current_logs,
            'timestamp': datetime.now().isoformat()
        }
        checkpoint_file = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_{dataset_name}.json"
        )
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        print(f"  检查点已保存: {checkpoint_file}")

    def load_checkpoint(self, dataset_name: str) -> Optional[dict]:
        """加载检查点"""
        checkpoint_file = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{dataset_name}.json"
        )
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _merge_checkpoint_logs(self, dataset_name: str, checkpoint_logs: List[dict]):
        """将检查点日志合并到self.logs，按(dataset, id)去重覆盖"""
        if not checkpoint_logs:
            return

        for cp_log in checkpoint_logs:
            log_entry = dict(cp_log)
            log_entry['dataset'] = dataset_name
            log_id = log_entry.get('id')
            existing_idx = next(
                (i for i, log in enumerate(self.logs)
                 if log.get('dataset') == dataset_name and log.get('id') == log_id),
                None
            )
            if existing_idx is not None:
                self.logs[existing_idx] = log_entry
            else:
                self.logs.append(log_entry)

    # ==================== FinBen完整测试 ====================

    def test_finben_full(self, test_file: str, resume: bool = False,
                         batch_size: int = 50) -> Dict[str, Any]:
        """
        完整测试FinBen数据集

        Args:
            test_file: 测试文件路径
            resume: 是否从断点继续
            batch_size: 每多少样本保存一次检查点
        """
        print(f"\n{'='*70}")
        print(f"完整测试 FinBen 数据集")
        print(f"{'='*70}")

        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total = len(data)
        print(f"  加载了 {total} 个样本")

        # 初始化变量
        processed_ids = set()
        correct = 0
        evaluated_total = 0
        api_errors = 0
        data_errors = 0
        predictions = []
        ground_truths = []
        error_cases = []

        # 尝试加载检查点
        if resume:
            checkpoint = self.load_checkpoint('finben')
            if checkpoint:
                processed_ids = set(checkpoint['processed_ids'])
                checkpoint_results = checkpoint.get('results', {})
                correct = checkpoint_results.get('correct', 0)
                evaluated_total = checkpoint_results.get(
                    'evaluated_total',
                    max(0, len(processed_ids) - checkpoint_results.get('api_errors', 0) - checkpoint_results.get('data_errors', 0))
                )
                api_errors = checkpoint_results.get('api_errors', 0)
                data_errors = checkpoint_results.get('data_errors', 0)
                checkpoint_logs = checkpoint.get('logs', [])
                self._merge_checkpoint_logs('finben', checkpoint_logs)
                predictions = [log['prediction'] for log in checkpoint_logs
                               if log.get('error_type') not in ('api_error', 'data_error')]
                ground_truths = [log['ground_truth'] for log in checkpoint_logs
                                 if log.get('error_type') not in ('api_error', 'data_error')]
                # 从检查点日志重建 error_cases
                error_cases = [
                    {
                        'id': log['id'],
                        'query': '',
                        'prediction': log['prediction'],
                        'ground_truth': log['ground_truth'],
                        'raw_response': ''
                    }
                    for log in checkpoint_logs
                    if log.get('error_type') not in ('api_error', 'data_error') and not log['correct']
                ]
                print(f"  从检查点恢复，已处理 {len(processed_ids)}/{total} 样本，正确数: {correct}")

        start_time = time.time()

        for idx, item in enumerate(data):
            item_id = item.get('id', f"finben_{idx}")

            # 跳过已处理的样本
            if item_id in processed_ids:
                continue

            query = item.get('query')
            answer_raw = item.get('answer')
            if not isinstance(query, str) or not isinstance(answer_raw, str):
                data_errors += 1
                processed_ids.add(item_id)
                self.logs.append(self._build_error_log_entry(
                    dataset='finben',
                    item_id=item_id,
                    error_type='data_error',
                    prediction='',
                    ground_truth=str(answer_raw) if answer_raw is not None else ''
                ))
                continue
            answer = answer_raw.lower().strip()

            prompt = f"""{query}

请只回答: HAWKISH, DOVISH, 或 NEUTRAL"""

            response = self.call_model(prompt, temperature=0.0)
            if response is None:
                api_errors += 1
                processed_ids.add(item_id)
                self.logs.append(self._build_error_log_entry(
                    dataset='finben',
                    item_id=item_id,
                    error_type='api_error',
                    prompt=prompt,
                    response=''
                ))
                continue
            pred = self._extract_finben_answer(response)

            is_correct = pred == answer
            evaluated_total += 1
            if is_correct:
                correct += 1
            else:
                error_cases.append({
                    'id': item_id,
                    'query': query[:200],
                    'prediction': pred,
                    'ground_truth': answer,
                    'raw_response': response[:500]
                })

            predictions.append(pred)
            ground_truths.append(answer)
            processed_ids.add(item_id)

            # 添加到日志（避免重复）
            log_entry = {
                'dataset': 'finben',
                'id': item_id,
                'prompt': prompt,
                'response': response,
                'prediction': pred,
                'ground_truth': answer,
                'correct': is_correct
            }
            # 检查是否已存在相同id的日志，避免重复
            existing_idx = next((i for i, log in enumerate(self.logs) 
                                if log.get('id') == item_id and log.get('dataset') == 'finben'), None)
            if existing_idx is not None:
                self.logs[existing_idx] = log_entry
            else:
                self.logs.append(log_entry)

            # 使用 len(processed_ids) 而不是 idx 来显示进度
            if len(processed_ids) % batch_size == 0 and len(processed_ids) > 0:
                elapsed = time.time() - start_time
                speed = len(processed_ids) / elapsed * 60 if elapsed > 0 else 0
                current_acc = correct / evaluated_total * 100 if evaluated_total else 0
                print(f"  进度: {len(processed_ids)}/{total} | "
                      f"准确率: {current_acc:.2f}% | "
                      f"速度: {speed:.1f}样本/分钟")
                # 保存检查点（使用当前数据集的日志，避免重复）
                dataset_logs = [log for log in self.logs 
                               if log.get('dataset') == 'finben']
                self.save_checkpoint('finben', processed_ids, 
                    {
                        'correct': correct,
                        'total': len(processed_ids),
                        'evaluated_total': evaluated_total,
                        'api_errors': api_errors,
                        'data_errors': data_errors
                    }, 
                    dataset_logs)

        # 最终保存
        accuracy = correct / evaluated_total * 100 if evaluated_total else 0

        results = {
            'dataset': 'finben',
            'total': len(processed_ids),
            'evaluated_total': evaluated_total,
            'api_errors': api_errors,
            'data_errors': data_errors,
            'correct': correct,
            'accuracy': accuracy,
            'predictions': predictions,
            'ground_truths': ground_truths,
            'error_cases': error_cases
        }

        self.results['finben'] = results

        print(f"\nFinBen 完整测试结果:")
        print(f"  总样本数: {len(processed_ids)}")
        print(f"  正确数: {correct}")
        print(f"  准确率: {accuracy:.2f}%")
        print(f"  耗时: {(time.time()-start_time)/60:.2f}分钟")

        return results

    def _extract_finben_answer(self, response: str) -> str:
        """提取FinBen答案"""
        response = response.lower().strip()
        # 使用单词边界匹配，避免匹配到包含这些词的其他单词
        if re.search(r'\bhawkish\b', response):
            return 'hawkish'
        elif re.search(r'\bdovish\b', response):
            return 'dovish'
        elif re.search(r'\bneutral\b', response):
            return 'neutral'
        return ''

    # ==================== BizBench完整测试 ====================

    def test_bizbench_full(self, test_file: str, resume: bool = False,
                           batch_size: int = 100) -> Dict[str, Any]:
        """完整测试BizBench数据集 (4,673样本)"""
        print(f"\n{'='*70}")
        print(f"完整测试 BizBench 数据集")
        print(f"{'='*70}")

        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total = len(data)
        print(f"  加载了 {total} 个样本")

        # 初始化变量
        processed_ids = set()
        correct = 0
        evaluated_total = 0
        api_errors = 0
        data_errors = 0
        task_metrics: Dict[str, Dict[str, int]] = {}
        predictions = []
        ground_truths = []
        error_cases = []

        # 尝试加载检查点
        if resume:
            checkpoint = self.load_checkpoint('bizbench')
            if checkpoint:
                processed_ids = set(checkpoint['processed_ids'])
                checkpoint_results = checkpoint.get('results', {})
                correct = checkpoint_results.get('correct', 0)
                evaluated_total = checkpoint_results.get(
                    'evaluated_total',
                    max(0, len(processed_ids) - checkpoint_results.get('api_errors', 0) - checkpoint_results.get('data_errors', 0))
                )
                api_errors = checkpoint_results.get('api_errors', 0)
                data_errors = checkpoint_results.get('data_errors', 0)
                checkpoint_logs = checkpoint.get('logs', [])
                task_metrics = checkpoint_results.get('task_metrics', {}) or {}
                if not task_metrics:
                    task_metrics = self._rebuild_bizbench_task_metrics(checkpoint_logs)
                self._merge_checkpoint_logs('bizbench', checkpoint_logs)
                predictions = [log['prediction'] for log in checkpoint_logs
                               if log.get('error_type') not in ('api_error', 'data_error')]
                ground_truths = [log['ground_truth'] for log in checkpoint_logs
                                 if log.get('error_type') not in ('api_error', 'data_error')]
                error_cases = [
                    {
                        'id': log['id'],
                        'prediction': log['prediction'][:200] if isinstance(log['prediction'], str) else log['prediction'],
                        'ground_truth': log['ground_truth'][:200] if isinstance(log['ground_truth'], str) else log['ground_truth']
                    }
                    for log in checkpoint_logs
                    if log.get('error_type') not in ('api_error', 'data_error') and not log['correct']
                ]
                print(f"  从检查点恢复，已处理 {len(processed_ids)}/{total} 样本，正确数: {correct}")

        start_time = time.time()

        for idx, item in enumerate(data):
            item_id = f"bizbench_{idx}"

            if item_id in processed_ids:
                continue

            question = item.get('question')
            answer_raw = item.get('answer')
            task = item.get('task', '')
            task_name = self._normalize_task_name(task)
            self._update_task_metric(task_metrics, task_name, 'total')
            if not isinstance(question, str) or answer_raw is None:
                data_errors += 1
                processed_ids.add(item_id)
                error_log = self._build_error_log_entry(
                    dataset='bizbench',
                    item_id=item_id,
                    error_type='data_error',
                    prediction='',
                    ground_truth=str(answer_raw) if answer_raw is not None else ''
                )
                error_log['task'] = task_name
                self.logs.append(error_log)
                self._update_task_metric(task_metrics, task_name, 'data_errors')
                continue
            answer = str(answer_raw).strip()
            context = item.get('context')
            options = item.get('options')

            prompt = self._build_bizbench_prompt(
                question=question,
                task=task,
                context=context,
                options=options
            )

            image_paths = self._resolve_image_paths(question, context)
            response = self.call_model(
                prompt,
                temperature=0.0,
                max_tokens=512,
                image_paths=image_paths
            )
            if response is None:
                api_errors += 1
                processed_ids.add(item_id)
                error_log = self._build_error_log_entry(
                    dataset='bizbench',
                    item_id=item_id,
                    error_type='api_error',
                    prompt=prompt,
                    response=''
                )
                error_log['task'] = task_name
                self.logs.append(error_log)
                self._update_task_metric(task_metrics, task_name, 'api_errors')
                continue
            pred = self._extract_bizbench_answer(response, task=task, options=options)

            is_correct = self._compare_bizbench_answer(
                pred=pred,
                truth=answer,
                task=task,
                options=options
            )
            evaluated_total += 1
            self._update_task_metric(task_metrics, task_name, 'evaluated_total')

            if is_correct:
                correct += 1
                self._update_task_metric(task_metrics, task_name, 'correct')
            else:
                error_cases.append({
                    'id': item_id,
                    'task': task_name,
                    'prediction': pred[:200],
                    'ground_truth': answer[:200]
                })

            predictions.append(pred)
            ground_truths.append(answer)
            processed_ids.add(item_id)

            # 添加到日志（避免重复）
            log_entry = {
                'dataset': 'bizbench',
                'id': item_id,
                'task': task,
                'image_paths': image_paths,
                'prompt': prompt,
                'response': response,
                'prediction': pred,
                'ground_truth': answer,
                'correct': is_correct
            }
            existing_idx = next((i for i, log in enumerate(self.logs) 
                                if log.get('id') == item_id and log.get('dataset') == 'bizbench'), None)
            if existing_idx is not None:
                self.logs[existing_idx] = log_entry
            else:
                self.logs.append(log_entry)

            if len(processed_ids) % batch_size == 0 and len(processed_ids) > 0:
                elapsed = time.time() - start_time
                speed = len(processed_ids) / elapsed * 60 if elapsed > 0 else 0
                current_acc = correct / evaluated_total * 100 if evaluated_total else 0
                remaining_time = (total - len(processed_ids)) / speed if speed > 0 else 0
                print(f"  进度: {len(processed_ids)}/{total} | "
                      f"准确率: {current_acc:.2f}% | "
                      f"速度: {speed:.1f}样本/分钟 | "
                      f"预计剩余: {remaining_time:.1f}分钟")
                dataset_logs = [log for log in self.logs 
                               if log.get('dataset') == 'bizbench']
                self.save_checkpoint('bizbench', processed_ids,
                    {
                        'correct': correct,
                        'total': len(processed_ids),
                        'evaluated_total': evaluated_total,
                        'api_errors': api_errors,
                        'data_errors': data_errors,
                        'task_metrics': task_metrics
                    },
                    dataset_logs)

        accuracy = correct / evaluated_total * 100 if evaluated_total else 0

        results = {
            'dataset': 'bizbench',
            'total': len(processed_ids),
            'evaluated_total': evaluated_total,
            'api_errors': api_errors,
            'data_errors': data_errors,
            'task_metrics': task_metrics,
            'correct': correct,
            'accuracy': accuracy,
            'predictions': predictions,
            'ground_truths': ground_truths,
            'error_cases': error_cases
        }

        self.results['bizbench'] = results

        print(f"\nBizBench 完整测试结果:")
        print(f"  总样本数: {len(processed_ids)}")
        print(f"  正确数: {correct}")
        print(f"  准确率: {accuracy:.2f}%")
        print(f"  有效评估样本: {evaluated_total} | API错误: {api_errors} | 数据错误: {data_errors}")
        if task_metrics:
            print("  分任务准确率:")
            for task_name, stat in sorted(task_metrics.items(), key=lambda x: x[0]):
                t_eval = stat.get('evaluated_total', 0)
                t_acc = (stat.get('correct', 0) / t_eval * 100) if t_eval else 0.0
                print(f"    - {task_name}: {stat.get('correct', 0)}/{t_eval} ({t_acc:.2f}%), "
                      f"API错误={stat.get('api_errors', 0)}, 数据错误={stat.get('data_errors', 0)}")
        print(f"  耗时: {(time.time()-start_time)/60:.2f}分钟")

        return results

    def _build_bizbench_prompt(self, question: str, task: str,
                               context: Optional[Any], options: Optional[Any]) -> str:
        """按BizBench子任务构造prompt"""
        task = (task or '').strip()
        context_text = ""
        if context not in (None, ''):
            if isinstance(context, (dict, list)):
                context_text = json.dumps(context, ensure_ascii=False)
            else:
                context_text = str(context)

        if task == 'FormulaEval':
            return f"""你是一个金融编程专家。请根据给定的Python类定义，补全缺失的方法实现。

{question}

请只输出补全的代码，不要输出任何解释。"""

        if isinstance(options, list) and options:
            option_lines = [f"{i}. {str(opt)}" for i, opt in enumerate(options)]
            options_text = "\n".join(option_lines)
            context_block = f"\n上下文信息:\n{context_text}\n" if context_text else "\n"
            return f"""你是一个金融问答助手。请根据题目和选项作答。{context_block}
问题: {question}
选项:
{options_text}

请只输出一个选项编号（从0开始），不要输出解释。"""

        context_block = f"\n上下文信息:\n{context_text}\n" if context_text else "\n"
        return f"""你是一个金融问答助手。请根据问题给出最终答案。{context_block}
问题: {question}

请只输出最终答案，不要输出解释。"""

    def _extract_bizbench_answer(self, response: str, task: str = '',
                                 options: Optional[Any] = None) -> str:
        """提取BizBench答案（按任务类型）"""
        response = re.sub(r'```python', '', response, flags=re.IGNORECASE)
        response = re.sub(r'```', '', response)
        clean = response.strip()

        if task == 'FormulaEval':
            return clean

        if isinstance(options, list) and options:
            lines = [line.strip() for line in clean.splitlines() if line.strip()]
            first = lines[0] if lines else clean
            first = re.sub(r'^[\(（\[]?([A-Za-z])[\)）\].:、\s]*$', r'\1', first)
            first = first.strip()

            num_match = re.search(r'-?\d+', first)
            if num_match:
                return num_match.group()

            letter_match = re.search(r'\b([A-Za-z])\b', first)
            if letter_match:
                idx = ord(letter_match.group(1).upper()) - ord('A')
                if 0 <= idx < len(options):
                    return str(idx)
            return first

        numeric_tasks = {'SEC-NUM', 'ConvFinQA', 'TAT-QA', 'CodeTAT-QA', 'CodeFinQA', 'FinCode'}
        if task in numeric_tasks:
            matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', clean)
            if matches:
                return matches[-1]

        # 非代码任务，优先取第一行（避免解释污染）
        lines = [line.strip() for line in clean.splitlines() if line.strip()]
        return lines[0] if lines else clean

    def _compare_bizbench_answer(self, pred: str, truth: str, task: str = '',
                                 options: Optional[Any] = None) -> bool:
        """按BizBench子任务比较答案"""
        if task == 'FormulaEval':
            return self._compare_code(pred, truth)

        if isinstance(options, list) and options:
            pred_norm = pred.strip().lower()
            truth_norm = truth.strip().lower()
            if pred_norm == truth_norm:
                return True

            # 支持模型输出选项文本，与标准答案编号互转后比较
            try:
                truth_idx = int(float(truth_norm))
                if 0 <= truth_idx < len(options):
                    option_text = str(options[truth_idx]).strip().lower()
                    return pred_norm == option_text
            except Exception:
                pass
            return False

        # 数值任务使用统一数值比较；SEC-NUM趋向精确匹配
        numeric_tasks = {'SEC-NUM', 'ConvFinQA', 'TAT-QA', 'CodeTAT-QA', 'CodeFinQA', 'FinCode'}
        if task in numeric_tasks:
            if task == 'SEC-NUM':
                return self._compare_numeric_string(pred, truth, tolerance=0.0, absolute_tolerance=1e-9)
            return self._compare_numeric_string(pred, truth, tolerance=0.002, absolute_tolerance=0.01)

        # 其余任务默认文本匹配
        return pred.strip().lower() == truth.strip().lower()

    def _compare_numeric_string(self, pred: str, truth: str,
                                tolerance: float = 0.002,
                                absolute_tolerance: float = 0.01) -> bool:
        """将字符串数值按容忍度比较"""
        try:
            pred_clean = str(pred).replace(',', '').replace('%', '').replace('$', '').strip()
            truth_clean = str(truth).replace(',', '').replace('%', '').replace('$', '').strip()
            pred_num = float(pred_clean)
            truth_num = float(truth_clean)
            if truth_num != 0:
                return abs(pred_num - truth_num) / abs(truth_num) <= tolerance
            return abs(pred_num - truth_num) <= absolute_tolerance
        except Exception:
            return str(pred).strip().lower() == str(truth).strip().lower()

    def _compare_code(self, pred: str, truth: str) -> bool:
        """比较两段代码是否等价"""
        def normalize_code(code):
            # 使用原始字符串避免转义警告
            code = re.sub(r'\s+', ' ', code)
            code = re.sub(r'#.*', '', code)
            code = code.replace("'", '"')
            code = code.strip()
            return code.lower()

        norm_pred = normalize_code(pred)
        norm_truth = normalize_code(truth)

        if norm_pred == norm_truth:
            return True

        pred_expr = re.search(r'return\s+(.+)', norm_pred)
        truth_expr = re.search(r'return\s+(.+)', norm_truth)

        if pred_expr and truth_expr:
            return pred_expr.group(1).strip() == truth_expr.group(1).strip()

        return False

    # ==================== FinMMR完整测试 ====================

    def test_finmmr_full(self, test_file: str, resume: bool = False,
                         batch_size: int = 50) -> Dict[str, Any]:
        """完整测试FinMMR数据集"""
        dataset_name = os.path.basename(test_file).replace('.json', '')

        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total = len(data)
        difficulty = dataset_name.split('_')[1] if '_' in dataset_name else 'unknown'

        print(f"\n{'='*70}")
        print(f"完整测试 FinMMR-{difficulty.upper()} 数据集 (共{total}样本)")
        print(f"{'='*70}")

        # 初始化变量
        processed_ids = set()
        correct = 0
        evaluated_total = 0
        api_errors = 0
        data_errors = 0
        predictions = []
        ground_truths = []
        error_cases = []
        image_required_samples = 0
        image_attached_samples = 0
        image_missing_samples = 0

        # 尝试加载检查点
        if resume:
            checkpoint = self.load_checkpoint(dataset_name)
            if checkpoint:
                processed_ids = set(checkpoint['processed_ids'])
                checkpoint_results = checkpoint.get('results', {})
                correct = checkpoint_results.get('correct', 0)
                evaluated_total = checkpoint_results.get(
                    'evaluated_total',
                    max(0, len(processed_ids) - checkpoint_results.get('api_errors', 0) - checkpoint_results.get('data_errors', 0))
                )
                api_errors = checkpoint_results.get('api_errors', 0)
                data_errors = checkpoint_results.get('data_errors', 0)
                image_required_samples = checkpoint_results.get('image_required_samples', 0)
                image_attached_samples = checkpoint_results.get('image_attached_samples', 0)
                image_missing_samples = checkpoint_results.get('image_missing_samples', 0)
                checkpoint_logs = checkpoint.get('logs', [])
                self._merge_checkpoint_logs(dataset_name, checkpoint_logs)
                predictions = [log['prediction'] for log in checkpoint_logs
                               if log.get('error_type') not in ('api_error', 'data_error')]
                ground_truths = [log['ground_truth'] for log in checkpoint_logs
                                 if log.get('error_type') not in ('api_error', 'data_error')]
                error_cases = [
                    {
                        'id': log['id'],
                        'question': '',
                        'prediction': log['prediction'],
                        'ground_truth': log['ground_truth']
                    }
                    for log in checkpoint_logs
                    if log.get('error_type') not in ('api_error', 'data_error') and not log['correct']
                ]
                print(f"  从检查点恢复，已处理 {len(processed_ids)}/{total} 样本，正确数: {correct}")

        start_time = time.time()

        for idx, item in enumerate(data):
            item_id = item.get('question_id', f"finmmr_{difficulty}_{idx}")

            if item_id in processed_ids:
                continue

            question = item.get('question')
            ground_truth_raw = item.get('ground_truth')
            if not isinstance(question, str):
                data_errors += 1
                processed_ids.add(item_id)
                self.logs.append(self._build_error_log_entry(
                    dataset=dataset_name,
                    item_id=item_id,
                    error_type='data_error',
                    ground_truth=str(ground_truth_raw) if ground_truth_raw is not None else ''
                ))
                continue
            try:
                ground_truth = float(ground_truth_raw)
            except Exception:
                data_errors += 1
                processed_ids.add(item_id)
                self.logs.append(self._build_error_log_entry(
                    dataset=dataset_name,
                    item_id=item_id,
                    error_type='data_error',
                    ground_truth=str(ground_truth_raw) if ground_truth_raw is not None else ''
                ))
                continue
            context = item.get('context', '')
            image_refs = []
            image_refs.extend(item.get('images') or [])
            image_refs.extend(item.get('ground_images') or [])
            if image_refs:
                image_required_samples += 1
            image_paths = self._resolve_image_paths(question, context, extra_refs=image_refs)
            if image_refs:
                if image_paths:
                    image_attached_samples += 1
                else:
                    image_missing_samples += 1

            prompt = f"""你是一个金融分析专家。请根据以下信息回答问题，只输出最终的数值答案。

问题: {question}

上下文信息:
{context}

请只输出最终的数值答案，不要包含任何解释或单位。"""

            response = self.call_model(
                prompt,
                temperature=0.0,
                max_tokens=256,
                image_paths=image_paths
            )
            if response is None:
                api_errors += 1
                processed_ids.add(item_id)
                self.logs.append(self._build_error_log_entry(
                    dataset=dataset_name,
                    item_id=item_id,
                    error_type='api_error',
                    prompt=prompt
                ))
                continue
            pred = self._extract_finmmr_answer(response)

            is_correct = self._compare_numeric(pred, ground_truth, tolerance=0.002)
            evaluated_total += 1

            if is_correct:
                correct += 1
            else:
                error_cases.append({
                    'id': item_id,
                    'question': question[:200],
                    'prediction': pred,
                    'ground_truth': ground_truth
                })

            predictions.append(pred)
            ground_truths.append(ground_truth)
            processed_ids.add(item_id)

            # 添加到日志（避免重复）
            log_entry = {
                'dataset': dataset_name,
                'id': item_id,
                'image_refs_count': len(image_refs),
                'image_paths': image_paths,
                'prompt': prompt,
                'response': response,
                'prediction': pred,
                'ground_truth': ground_truth,
                'correct': is_correct
            }
            existing_idx = next((i for i, log in enumerate(self.logs) 
                                if log.get('id') == item_id and log.get('dataset') == dataset_name), None)
            if existing_idx is not None:
                self.logs[existing_idx] = log_entry
            else:
                self.logs.append(log_entry)

            if len(processed_ids) % batch_size == 0 and len(processed_ids) > 0:
                elapsed = time.time() - start_time
                speed = len(processed_ids) / elapsed * 60 if elapsed > 0 else 0
                current_acc = correct / evaluated_total * 100 if evaluated_total else 0
                print(f"  进度: {len(processed_ids)}/{total} | "
                      f"准确率: {current_acc:.2f}% | "
                      f"速度: {speed:.1f}样本/分钟")
                dataset_logs = [log for log in self.logs 
                               if log.get('dataset') == dataset_name]
                self.save_checkpoint(dataset_name, processed_ids,
                    {
                        'correct': correct,
                        'total': len(processed_ids),
                        'evaluated_total': evaluated_total,
                        'api_errors': api_errors,
                        'data_errors': data_errors,
                        'image_required_samples': image_required_samples,
                        'image_attached_samples': image_attached_samples,
                        'image_missing_samples': image_missing_samples
                    },
                    dataset_logs)

        accuracy = correct / evaluated_total * 100 if evaluated_total else 0

        results = {
            'dataset': dataset_name,
            'total': len(processed_ids),
            'evaluated_total': evaluated_total,
            'api_errors': api_errors,
            'data_errors': data_errors,
            'image_required_samples': image_required_samples,
            'image_attached_samples': image_attached_samples,
            'image_missing_samples': image_missing_samples,
            'correct': correct,
            'accuracy': accuracy,
            'predictions': predictions,
            'ground_truths': ground_truths,
            'error_cases': error_cases
        }

        self.results[dataset_name] = results

        print(f"\n{dataset_name} 完整测试结果:")
        print(f"  总样本数: {len(processed_ids)}")
        print(f"  正确数: {correct}")
        print(f"  准确率: {accuracy:.2f}%")
        if image_required_samples > 0:
            attach_rate = image_attached_samples / image_required_samples * 100
            print(f"  图像样本: 需要{image_required_samples} | 命中{image_attached_samples} "
                  f"| 缺失{image_missing_samples} | 命中率{attach_rate:.2f}%")
        print(f"  耗时: {(time.time()-start_time)/60:.2f}分钟")

        return results

    def _extract_finmmr_answer(self, response: str) -> Optional[float]:
        """提取FinMMR数值答案"""
        response = re.sub(r'(answer is|the answer|answer:|结果是|答案：|答案为)', '', 
                         response, flags=re.IGNORECASE)
        response = response.strip()

        # 兼容整数/小数/科学计数法，优先取最后一个数值
        matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', response)
        if matches:
            try:
                return float(matches[-1])
            except:
                return None
        return None

    def _compare_numeric(self, pred: Optional[float], truth: float, 
                        tolerance: float = 0.002) -> bool:
        """比较两个数值是否在容忍范围内"""
        if pred is None:
            return False

        if truth != 0:
            relative_error = abs(pred - truth) / abs(truth)
            return relative_error <= tolerance
        else:
            return abs(pred - truth) <= 0.01

    # ==================== ConvFinQA完整测试 ====================

    def test_convfinqa_full(self, test_file: str, resume: bool = False,
                            batch_size: int = 50) -> Dict[str, Any]:
        """完整测试ConvFinQA数据集 (1,490样本)"""
        print(f"\n{'='*70}")
        print(f"完整测试 FLARE-ConvFinQA 数据集")
        print(f"{'='*70}")

        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total = len(data)
        print(f"  加载了 {total} 个样本")

        # 初始化变量
        processed_ids = set()
        correct = 0
        evaluated_total = 0
        api_errors = 0
        data_errors = 0
        predictions = []
        ground_truths = []
        error_cases = []

        # 尝试加载检查点
        if resume:
            checkpoint = self.load_checkpoint('convfinqa')
            if checkpoint:
                processed_ids = set(checkpoint['processed_ids'])
                checkpoint_results = checkpoint.get('results', {})
                correct = checkpoint_results.get('correct', 0)
                evaluated_total = checkpoint_results.get(
                    'evaluated_total',
                    max(0, len(processed_ids) - checkpoint_results.get('api_errors', 0) - checkpoint_results.get('data_errors', 0))
                )
                api_errors = checkpoint_results.get('api_errors', 0)
                data_errors = checkpoint_results.get('data_errors', 0)
                checkpoint_logs = checkpoint.get('logs', [])
                self._merge_checkpoint_logs('convfinqa', checkpoint_logs)
                predictions = [log['prediction'] for log in checkpoint_logs
                               if log.get('error_type') not in ('api_error', 'data_error')]
                ground_truths = [log['ground_truth'] for log in checkpoint_logs
                                 if log.get('error_type') not in ('api_error', 'data_error')]
                error_cases = [
                    {
                        'id': log['id'],
                        'prediction': log['prediction'],
                        'ground_truth': log['ground_truth']
                    }
                    for log in checkpoint_logs
                    if log.get('error_type') not in ('api_error', 'data_error') and not log['correct']
                ]
                print(f"  从检查点恢复，已处理 {len(processed_ids)}/{total} 样本，正确数: {correct}")

        start_time = time.time()

        for idx, item in enumerate(data):
            item_id = item.get('id', f"convfinqa_{idx}")

            if item_id in processed_ids:
                continue

            query = item.get('query')
            answer_raw = item.get('answer')
            if not isinstance(query, str) or answer_raw is None:
                data_errors += 1
                processed_ids.add(item_id)
                self.logs.append(self._build_error_log_entry(
                    dataset='convfinqa',
                    item_id=item_id,
                    error_type='data_error',
                    ground_truth=str(answer_raw) if answer_raw is not None else ''
                ))
                continue
            answer = str(answer_raw).strip()

            prompt = f"""{query}

请只输出最终的数值答案，不要包含任何解释、单位或计算过程。"""

            response = self.call_model(prompt, temperature=0.0, max_tokens=256)
            if response is None:
                api_errors += 1
                processed_ids.add(item_id)
                self.logs.append(self._build_error_log_entry(
                    dataset='convfinqa',
                    item_id=item_id,
                    error_type='api_error',
                    prompt=prompt
                ))
                continue
            pred = self._extract_convfinqa_answer(response)

            is_correct = self._compare_convfinqa_answer(pred, answer)
            evaluated_total += 1

            if is_correct:
                correct += 1
            else:
                error_cases.append({
                    'id': item_id,
                    'prediction': pred,
                    'ground_truth': answer
                })

            predictions.append(pred)
            ground_truths.append(answer)
            processed_ids.add(item_id)

            # 添加到日志（避免重复）
            log_entry = {
                'dataset': 'convfinqa',
                'id': item_id,
                'prompt': prompt,
                'response': response,
                'prediction': pred,
                'ground_truth': answer,
                'correct': is_correct
            }
            existing_idx = next((i for i, log in enumerate(self.logs) 
                                if log.get('id') == item_id and log.get('dataset') == 'convfinqa'), None)
            if existing_idx is not None:
                self.logs[existing_idx] = log_entry
            else:
                self.logs.append(log_entry)

            if len(processed_ids) % batch_size == 0 and len(processed_ids) > 0:
                elapsed = time.time() - start_time
                speed = len(processed_ids) / elapsed * 60 if elapsed > 0 else 0
                current_acc = correct / evaluated_total * 100 if evaluated_total else 0
                print(f"  进度: {len(processed_ids)}/{total} | "
                      f"准确率: {current_acc:.2f}% | "
                      f"速度: {speed:.1f}样本/分钟")
                dataset_logs = [log for log in self.logs 
                               if log.get('dataset') == 'convfinqa']
                self.save_checkpoint('convfinqa', processed_ids,
                    {
                        'correct': correct,
                        'total': len(processed_ids),
                        'evaluated_total': evaluated_total,
                        'api_errors': api_errors,
                        'data_errors': data_errors
                    },
                    dataset_logs)

        accuracy = correct / evaluated_total * 100 if evaluated_total else 0

        results = {
            'dataset': 'convfinqa',
            'total': len(processed_ids),
            'evaluated_total': evaluated_total,
            'api_errors': api_errors,
            'data_errors': data_errors,
            'correct': correct,
            'accuracy': accuracy,
            'predictions': predictions,
            'ground_truths': ground_truths,
            'error_cases': error_cases
        }

        self.results['convfinqa'] = results

        print(f"\nFLARE-ConvFinQA 完整测试结果:")
        print(f"  总样本数: {len(processed_ids)}")
        print(f"  正确数: {correct}")
        print(f"  准确率: {accuracy:.2f}%")
        print(f"  耗时: {(time.time()-start_time)/60:.2f}分钟")

        return results

    def _extract_convfinqa_answer(self, response: str) -> str:
        """提取ConvFinQA答案"""
        response = re.sub(r'(answer is|the answer|answer:|结果是|答案：|答案为)', '', 
                         response, flags=re.IGNORECASE)
        response = response.strip()

        # 与FinMMR保持一致：优先取最后一个数值
        matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', response)
        if matches:
            return matches[-1]
        return response.strip()

    def _compare_convfinqa_answer(self, pred: str, truth: str) -> bool:
        """
        比较ConvFinQA答案 - 使用与FinMMR一致的0.2%容忍度
        参考论文：FinMMR使用0.2%相对误差容忍度
        """
        try:
            # 清理数值字符串（新增：处理$符号）
            pred_clean = pred.replace(',', '').replace('%', '').replace('$', '').strip()
            truth_clean = truth.replace(',', '').replace('%', '').replace('$', '').strip()
            
            pred_num = float(pred_clean)
            truth_num = float(truth_clean)

            # 使用0.2%相对误差容忍度（与FinMMR一致）
            tolerance = 0.002
            
            if truth_num != 0:
                relative_error = abs(pred_num - truth_num) / abs(truth_num)
                return relative_error <= tolerance
            else:
                return abs(pred_num - truth_num) <= 0.01
        except:
            return pred.lower().strip() == truth.lower().strip()

    # ==================== 结果保存 ====================

    def save_results(self, output_dir: str = "./results_full",
                     enable_visualization: bool = False,
                     compare_with_file: Optional[str] = None):
        """保存所有测试结果，并可选生成可视化与模型对比"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_tag = _model_to_tag(self.model)

        results_file = os.path.join(output_dir, f"results_{model_tag}_{timestamp}.json")
        results_payload = {
            "meta": {
                "model": self.model,
                "base_url": self.base_url,
                "timestamp": timestamp
            },
            "datasets": self.results
        }
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_payload, f, ensure_ascii=False, indent=2)

        logs_file = os.path.join(output_dir, f"logs_{model_tag}_{timestamp}.json")
        logs_payload = {
            "meta": {
                "model": self.model,
                "timestamp": timestamp
            },
            "logs": self.logs
        }
        with open(logs_file, 'w', encoding='utf-8') as f:
            json.dump(logs_payload, f, ensure_ascii=False, indent=2)

        report_file = os.path.join(output_dir, f"report_{model_tag}_{timestamp}.txt")
        self._generate_full_report(report_file)

        visualization_files: List[str] = []
        if enable_visualization:
            visualization_files = self._generate_visualizations(output_dir, timestamp, model_tag)

        comparison_files: List[str] = []
        if compare_with_file:
            comparison_files = self._generate_comparison_artifacts(
                compare_with_file=compare_with_file,
                current_results_file=results_file,
                output_dir=output_dir,
                timestamp=timestamp,
                model_tag=model_tag
            )

        print(f"\n完整测试结果已保存到:")
        print(f"  详细结果: {results_file}")
        print(f"  完整日志: {logs_file}")
        print(f"  汇总报告: {report_file}")
        for file_path in visualization_files:
            print(f"  可视化: {file_path}")
        for file_path in comparison_files:
            print(f"  对比分析: {file_path}")

        return results_file, logs_file, report_file

    def _load_results_payload(self, file_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """兼容读取新旧格式结果文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if isinstance(payload, dict) and isinstance(payload.get('datasets'), dict):
            return payload.get('meta', {}), payload['datasets']
        # 兼容旧格式（顶层就是datasets）
        return {}, payload if isinstance(payload, dict) else {}

    def _generate_visualizations(self, output_dir: str, timestamp: str, model_tag: str) -> List[str]:
        """生成结果可视化图表（需要matplotlib）"""
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("  提示: 未安装matplotlib，跳过可视化生成。")
            return []

        files: List[str] = []
        datasets = list(self.results.keys())
        if not datasets:
            return files

        accuracies = [self.results[d].get('accuracy', 0.0) for d in datasets]
        eval_totals = [self.results[d].get('evaluated_total', self.results[d].get('total', 0)) for d in datasets]
        api_errs = [self.results[d].get('api_errors', 0) for d in datasets]
        data_errs = [self.results[d].get('data_errors', 0) for d in datasets]

        fig1 = plt.figure(figsize=(10, 5))
        plt.bar(datasets, accuracies)
        plt.xticks(rotation=30, ha='right')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Dataset Accuracy - {self.model}')
        plt.tight_layout()
        acc_file = os.path.join(output_dir, f"viz_accuracy_{model_tag}_{timestamp}.png")
        fig1.savefig(acc_file, dpi=150)
        plt.close(fig1)
        files.append(acc_file)

        fig2 = plt.figure(figsize=(10, 5))
        x = list(range(len(datasets)))
        plt.bar(x, eval_totals, label='evaluated_total')
        plt.bar(x, api_errs, bottom=eval_totals, label='api_errors')
        stacked_bottom = [eval_totals[i] + api_errs[i] for i in range(len(datasets))]
        plt.bar(x, data_errs, bottom=stacked_bottom, label='data_errors')
        plt.xticks(x, datasets, rotation=30, ha='right')
        plt.ylabel('Count')
        plt.title(f'Evaluation Coverage - {self.model}')
        plt.legend()
        plt.tight_layout()
        cov_file = os.path.join(output_dir, f"viz_coverage_{model_tag}_{timestamp}.png")
        fig2.savefig(cov_file, dpi=150)
        plt.close(fig2)
        files.append(cov_file)

        bizbench = self.results.get('bizbench', {})
        task_metrics = bizbench.get('task_metrics', {})
        if task_metrics:
            task_names = sorted(task_metrics.keys())
            task_acc = []
            for task_name in task_names:
                stat = task_metrics[task_name]
                t_eval = stat.get('evaluated_total', 0)
                task_acc.append((stat.get('correct', 0) / t_eval * 100) if t_eval else 0.0)

            fig3 = plt.figure(figsize=(11, 5))
            plt.bar(task_names, task_acc)
            plt.xticks(rotation=35, ha='right')
            plt.ylabel('Accuracy (%)')
            plt.title(f'BizBench Task Accuracy - {self.model}')
            plt.tight_layout()
            task_file = os.path.join(output_dir, f"viz_bizbench_tasks_{model_tag}_{timestamp}.png")
            fig3.savefig(task_file, dpi=150)
            plt.close(fig3)
            files.append(task_file)

        return files

    def _generate_comparison_artifacts(self, compare_with_file: str, current_results_file: str,
                                       output_dir: str, timestamp: str, model_tag: str) -> List[str]:
        """生成双模型对比文本与图表"""
        if not os.path.exists(compare_with_file):
            print(f"  提示: 对比文件不存在，跳过对比: {compare_with_file}")
            return []

        base_meta, base_datasets = self._load_results_payload(compare_with_file)
        current_meta, current_datasets = self._load_results_payload(current_results_file)
        base_model = base_meta.get('model', 'baseline_model')
        current_model = current_meta.get('model', self.model)

        all_datasets = sorted(set(base_datasets.keys()) | set(current_datasets.keys()))
        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("双模型对比分析")
        lines.append("=" * 70)
        lines.append(f"基线模型: {base_model}")
        lines.append(f"当前模型: {current_model}")
        lines.append(f"基线结果: {compare_with_file}")
        lines.append(f"当前结果: {current_results_file}")
        lines.append("")

        base_better = 0
        current_better = 0
        for ds in all_datasets:
            b = base_datasets.get(ds, {})
            c = current_datasets.get(ds, {})
            b_acc = float(b.get('accuracy', 0.0))
            c_acc = float(c.get('accuracy', 0.0))
            delta = c_acc - b_acc
            if delta > 1e-12:
                current_better += 1
            elif delta < -1e-12:
                base_better += 1
            lines.append(
                f"{ds}: baseline={b_acc:.2f}%, current={c_acc:.2f}%, delta={delta:+.2f}%"
            )

        lines.append("")
        if current_better > base_better:
            lines.append(f"结论: 当前模型在更多数据集上更优 ({current_better} vs {base_better})。")
        elif base_better > current_better:
            lines.append(f"结论: 基线模型在更多数据集上更优 ({base_better} vs {current_better})。")
        else:
            lines.append("结论: 两模型整体接近，需要结合子任务与业务约束决策。")
        lines.append("建议: 对于含图像信息任务，优先查看多模态相关任务集的对比结果。")

        report_file = os.path.join(output_dir, f"compare_{model_tag}_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")

        files = [report_file]
        try:
            import matplotlib.pyplot as plt
            labels = all_datasets
            x = list(range(len(labels)))
            b_vals = [float(base_datasets.get(ds, {}).get('accuracy', 0.0)) for ds in labels]
            c_vals = [float(current_datasets.get(ds, {}).get('accuracy', 0.0)) for ds in labels]
            width = 0.4
            fig = plt.figure(figsize=(11, 5))
            plt.bar([i - width / 2 for i in x], b_vals, width=width, label=base_model)
            plt.bar([i + width / 2 for i in x], c_vals, width=width, label=current_model)
            plt.xticks(x, labels, rotation=30, ha='right')
            plt.ylabel('Accuracy (%)')
            plt.title('Model Comparison by Dataset')
            plt.legend()
            plt.tight_layout()
            png_file = os.path.join(output_dir, f"compare_{model_tag}_{timestamp}.png")
            fig.savefig(png_file, dpi=150)
            plt.close(fig)
            files.append(png_file)
        except Exception:
            pass

        return files

    def _generate_full_report(self, report_file: str):
        """生成完整测试报告"""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("金融LLM基准测试报告 - 完整数据集\n")
            f.write(f"模型: {self.model}\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")

            total_samples = 0
            total_evaluated = 0
            total_correct = 0
            total_api_errors = 0
            total_data_errors = 0

            for dataset_name, result in self.results.items():
                evaluated_total = result.get('evaluated_total', result.get('total', 0))
                api_errors = result.get('api_errors', 0)
                data_errors = result.get('data_errors', 0)
                f.write(f"【{dataset_name.upper()}】\n")
                f.write(f"  总样本数: {result['total']}\n")
                f.write(f"  有效评估样本: {evaluated_total}\n")
                f.write(f"  API错误: {api_errors}\n")
                f.write(f"  数据错误: {data_errors}\n")
                f.write(f"  正确数: {result['correct']}\n")
                f.write(f"  准确率: {result['accuracy']:.2f}%\n")
                if 'image_required_samples' in result:
                    req = int(result.get('image_required_samples', 0))
                    att = int(result.get('image_attached_samples', 0))
                    miss = int(result.get('image_missing_samples', 0))
                    hit_rate = (att / req * 100.0) if req > 0 else 0.0
                    f.write(f"  图像样本: 需要={req}, 命中={att}, 缺失={miss}, 命中率={hit_rate:.2f}%\n")
                f.write(f"  错误数: {len(result['error_cases'])}\n")
                if dataset_name == 'bizbench' and result.get('task_metrics'):
                    f.write("  分任务结果:\n")
                    for task_name, stat in sorted(result['task_metrics'].items(), key=lambda x: x[0]):
                        t_eval = stat.get('evaluated_total', 0)
                        t_acc = (stat.get('correct', 0) / t_eval * 100) if t_eval else 0.0
                        f.write(f"    - {task_name}: total={stat.get('total', 0)}, "
                                f"eval={t_eval}, correct={stat.get('correct', 0)}, "
                                f"acc={t_acc:.2f}%, api_err={stat.get('api_errors', 0)}, "
                                f"data_err={stat.get('data_errors', 0)}\n")
                f.write("\n")

                total_samples += result['total']
                total_evaluated += evaluated_total
                total_correct += result['correct']
                total_api_errors += api_errors
                total_data_errors += data_errors

            overall_accuracy = total_correct / total_evaluated * 100 if total_evaluated > 0 else 0

            f.write("="*70 + "\n")
            f.write("总体统计\n")
            f.write("="*70 + "\n")
            f.write(f"  总样本数: {total_samples}\n")
            f.write(f"  有效评估样本: {total_evaluated}\n")
            f.write(f"  API错误总数: {total_api_errors}\n")
            f.write(f"  数据错误总数: {total_data_errors}\n")
            f.write(f"  总正确数: {total_correct}\n")
            f.write(f"  总体准确率: {overall_accuracy:.2f}%\n")


def main():
    """主程序"""
    parser = argparse.ArgumentParser(description='金融LLM基准测试 - 最终修复版')
    parser.add_argument('--full', action='store_true', help='测试完整数据集')
    parser.add_argument('--resume', action='store_true', help='从断点继续测试')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['all', 'finben', 'bizbench', 'finmmr_easy', 
                               'finmmr_medium', 'finmmr_hard', 'convfinqa'],
                       help='指定要测试的数据集')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='每多少样本保存一次检查点')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-8B-Instruct',
                       help='要测试的模型名称，例如 Qwen/Qwen3-VL-8B-Thinking')
    parser.add_argument('--base-url', type=str, default='https://api.siliconflow.cn/v1',
                       help='API基础URL')
    parser.add_argument('--api-key', type=str, default='sk-tjpwtwbvgdxacmsbarjyksjlcmtgmvgobpumifwyqzhvlpab',
                       help='API密钥；为空时读取环境变量 SILICONFLOW_API_KEY')
    parser.add_argument('--checkpoint-root', type=str, default='./checkpoints',
                       help='检查点根目录（会自动按模型分子目录）')
    parser.add_argument('--results-root', type=str, default='./results_full',
                       help='结果根目录（会自动按模型分子目录）')
    parser.add_argument('--image-root', type=str, default='',
                       help='图片资源根目录（VL模型可选）')
    parser.add_argument('--enable-vision', action='store_true',
                       help='强制启用图片输入（适用于VL模型）')
    parser.add_argument('--disable-vision', action='store_true',
                       help='强制禁用图片输入（覆盖自动判断）')
    parser.add_argument('--visualize', action='store_true',
                       help='保存测试后自动生成可视化图表')
    parser.add_argument('--compare-with', type=str, default='',
                       help='对比另一份结果JSON文件，自动生成模型对比报告')
    parser.add_argument('--api-timeout', type=int, default=120,
                       help='单次API请求超时时间（秒）')

    args = parser.parse_args()

    if args.dataset == 'all' and not args.full:
        print("参数错误: 当 --dataset=all 时，请显式添加 --full 以避免误触发全量测试。")
        print("示例: python financial_benchmark_framework_final.py --full")
        return
    if args.full and args.dataset != 'all':
        print(f"提示: 你指定了 --dataset {args.dataset}，将仅测试该数据集。")

    # 配置
    MODEL = args.model
    BASE_URL = args.base_url
    API_KEY = args.api_key.strip() or os.getenv("SILICONFLOW_API_KEY", "").strip()
    if not API_KEY:
        print("参数错误: 未提供 API Key。请通过 --api-key 或环境变量 SILICONFLOW_API_KEY 设置。")
        return

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_TAG = _model_to_tag(MODEL)
    CHECKPOINT_DIR = os.path.join(args.checkpoint_root, MODEL_TAG)
    RESULTS_DIR = os.path.join(args.results_root, MODEL_TAG)
    default_image_root = os.path.join(BASE_DIR, 'Dataset', 'FinMMR-main')
    IMAGE_ROOT = args.image_root.strip() if args.image_root.strip() else default_image_root
    auto_vision = any(k in MODEL.lower() for k in ('vl', 'omni', 'vision'))
    if args.enable_vision and args.disable_vision:
        print("参数错误: --enable-vision 与 --disable-vision 不能同时使用。")
        return
    ENABLE_VISION = args.enable_vision or (auto_vision and not args.disable_vision)

    DATASETS = {
        'finben': os.path.join(BASE_DIR, 'Dataset', 'FinBen', 'finben_test.json'),
        'bizbench': os.path.join(BASE_DIR, 'Dataset', 'bizbench_test', 'bizbench_test.json'),
        'finmmr_easy': os.path.join(BASE_DIR, 'Dataset', 'finmmr', 'finmmr_easy_test.json'),
        'finmmr_medium': os.path.join(BASE_DIR, 'Dataset', 'finmmr', 'finmmr_medium_test.json'),
        'finmmr_hard': os.path.join(BASE_DIR, 'Dataset', 'finmmr', 'finmmr_hard_test.json'),
        'convfinqa': os.path.join(BASE_DIR, 'Dataset', 'flarez-confinqa_test', 'flare-convfinqa_test.json')
    }

    # 初始化
    benchmark = FinancialLLMBenchmarkFinal(
        model=MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        checkpoint_dir=CHECKPOINT_DIR,
        image_root=IMAGE_ROOT,
        enable_vision=ENABLE_VISION,
        api_timeout=args.api_timeout
    )

    print("="*70)
    print("金融LLM基准测试 - 多模型实验版")
    print("="*70)
    print(f"模型: {MODEL}")
    print(f"视觉输入: {'启用' if ENABLE_VISION else '禁用'}")
    print(f"图片根目录: {IMAGE_ROOT}")
    print(f"API超时: {args.api_timeout} 秒")
    print(f"检查点目录: {CHECKPOINT_DIR}")
    print(f"结果目录: {RESULTS_DIR}")
    print(f"\n完整数据集规模:")
    print(f"  - FinBen: 496 样本")
    print(f"  - BizBench: 4,673 样本")
    print(f"  - FinMMR Easy: 1,200 样本")
    print(f"  - FinMMR Medium: 1,200 样本")
    print(f"  - FinMMR Hard: 1,000 样本")
    print(f"  - ConvFinQA: 1,490 样本")
    print(f"  - 总计: 10,059 样本")
    print(f"\n按平均2秒/样本计算，预计总耗时约: 5.6 小时")
    print(f"检查点保存间隔: 每 {args.batch_size} 个样本")
    print("="*70)

    # 运行测试
    if args.dataset == 'all' or args.dataset == 'finben':
        benchmark.test_finben_full(DATASETS['finben'], resume=args.resume, 
                                   batch_size=args.batch_size)

    if args.dataset == 'all' or args.dataset == 'bizbench':
        benchmark.test_bizbench_full(DATASETS['bizbench'], resume=args.resume,
                                     batch_size=args.batch_size)

    if args.dataset == 'all' or args.dataset == 'finmmr_easy':
        benchmark.test_finmmr_full(DATASETS['finmmr_easy'], resume=args.resume,
                                   batch_size=args.batch_size)

    if args.dataset == 'all' or args.dataset == 'finmmr_medium':
        benchmark.test_finmmr_full(DATASETS['finmmr_medium'], resume=args.resume,
                                   batch_size=args.batch_size)

    if args.dataset == 'all' or args.dataset == 'finmmr_hard':
        benchmark.test_finmmr_full(DATASETS['finmmr_hard'], resume=args.resume,
                                   batch_size=args.batch_size)

    if args.dataset == 'all' or args.dataset == 'convfinqa':
        benchmark.test_convfinqa_full(DATASETS['convfinqa'], resume=args.resume,
                                      batch_size=args.batch_size)

    # 保存结果
    benchmark.save_results(
        output_dir=RESULTS_DIR,
        enable_visualization=args.visualize,
        compare_with_file=args.compare_with.strip() or None
    )

    print("\n" + "="*70)
    print("完整数据集测试完成!")
    print("="*70)


if __name__ == "__main__":
    main()
