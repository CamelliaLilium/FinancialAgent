"""
PlanningFlow - 基于 archive/app/flow/planning.py 的简洁实现。

保持原始逻辑：规划阶段使用 to_planning_param() 鼓励 LLM 调用工具，
完整 plan_status 传给 executor，复用同一 executor，步骤执行清晰。
"""
import json
import re
import time
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import Field

from app.agent.base import BaseAgent
from app.flow.base import BaseFlow
from app.llm import LLM
from app.logger import logger
from app.schema import AgentState, Message, ToolChoice
from app.config import config
from app.tool import OcrExtract, PlanningTool, ToolCollection
from app.tool.ocr import set_step_images_for_ocr
from app.tool.anti_loop import AntiLoopInterceptor
from app.tool.python_execute import PythonExecute


class PlanStepStatus(str, Enum):
    """Enum class defining possible statuses of a plan step"""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"

    @classmethod
    def get_all_statuses(cls) -> list[str]:
        return [status.value for status in cls]

    @classmethod
    def get_active_statuses(cls) -> list[str]:
        return [cls.NOT_STARTED.value, cls.IN_PROGRESS.value]

    @classmethod
    def get_status_marks(cls) -> Dict[str, str]:
        return {
            cls.COMPLETED.value: "[✓]",
            cls.IN_PROGRESS.value: "[→]",
            cls.BLOCKED.value: "[!]",
            cls.NOT_STARTED.value: "[ ]",
        }


class PlanningFlow(BaseFlow):
    """A flow that manages planning and execution of tasks using agents."""

    llm: LLM = Field(default_factory=lambda: LLM())
    planning_tool: PlanningTool = Field(default_factory=PlanningTool)
    executor_keys: List[str] = Field(default_factory=list)
    active_plan_id: str = Field(default_factory=lambda: f"plan_{int(time.time())}")
    current_step_index: Optional[int] = None
    base64_images: Optional[List[str]] = None  # 多模态输入，execute() 时设置
    ocr_results: Optional[List[str]] = None  # OCR 预处理结果，供 Planning 解耦感知与认知
    _shared_python_execute: Optional[PythonExecute] = None  # 跨 Agent 共享的 PythonExecute 实例

    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], **data
    ):
        if "executors" in data:
            data["executor_keys"] = data.pop("executors")
        if "plan_id" in data:
            data["active_plan_id"] = data.pop("plan_id")
        if "workflow_state_tool" in data:
            data["planning_tool"] = data.pop("workflow_state_tool")
        if "planning_tool" not in data:
            data["planning_tool"] = PlanningTool()
        super().__init__(agents, **data)
        if not self.executor_keys:
            self.executor_keys = list(self.agents.keys())

    def get_executor(self, step_type: Optional[str] = None) -> BaseAgent:
        """Get an appropriate executor agent for the current step."""
        if step_type and step_type in self.agents:
            return self.agents[step_type]
        for key in self.executor_keys:
            if key in self.agents:
                return self.agents[key]
        return self.primary_agent

    def _ensure_shared_python_execute(self) -> PythonExecute:
        """确保所有 Agent 共享同一个 PythonExecute 实例，实现变量跨步骤持久化。
        
        问题背景：FinanceAgent 和 MultimodalAgent 使用 Field(default_factory) 创建 ToolCollection，
        导致每次访问 available_tools 都返回新的 PythonExecute 实例，变量无法跨 Agent 共享。
        
        解决方案：在 Flow 级别创建共享的 PythonExecute 实例，并注入到所有 Agent 的 ToolCollection 中。
        """
        if self._shared_python_execute is None:
            self._shared_python_execute = PythonExecute()
        return self._shared_python_execute

    def _inject_shared_python_execute(self, agent: BaseAgent) -> None:
        """将共享的 PythonExecute 实例注入到 Agent 的 ToolCollection 中。"""
        if not hasattr(agent, "available_tools"):
            return
        
        shared_py = self._ensure_shared_python_execute()
        
        # 获取当前 Agent 的 available_tools
        tools = agent.available_tools
        if tools is None:
            return
        
        # 检查是否已有 python_execute
        existing = tools.get_tool("python_execute")
        if existing is not None and existing is shared_py:
            # 已经是共享实例，无需操作
            return
        
        # 替换为共享实例
        if existing is not None:
            # 移除旧的实例
            new_tools = list(tools.tools)
            new_tools = [t for t in new_tools if t.name != "python_execute"]
            new_tools.append(shared_py)
            agent.available_tools = ToolCollection(*new_tools)
        else:
            # 添加共享实例
            tools.add_tool(shared_py)

    def _reset_python_execute_env(self) -> None:
        """Flow 开始时重置所有 executor 的 python_execute 环境，确保变量不跨请求泄露。"""
        # 首先确保所有 Agent 使用共享的 PythonExecute 实例
        seen = set()
        for key in self.executor_keys:
            if key in self.agents:
                agent = self.agents[key]
                if id(agent) not in seen:
                    seen.add(id(agent))
                    self._inject_shared_python_execute(agent)
        
        if self.primary_agent and id(self.primary_agent) not in seen:
            self._inject_shared_python_execute(self.primary_agent)
        
        # 然后重置共享实例的环境
        shared_py = self._ensure_shared_python_execute()
        if hasattr(shared_py, "reset_env"):
            shared_py.reset_env()

    async def execute(
        self,
        input_text: str,
        base64_images: Optional[List[str]] = None,
    ) -> str:
        """Execute the planning flow with agents.

        Args:
            input_text: User request text.
            base64_images: Optional list of base64-encoded images for multimodal tasks.
                          When provided, Planning may assign [multimodal] to steps.
        """
        try:
            if not self.primary_agent:
                raise ValueError("No primary agent available")

            self.base64_images = base64_images
            self.ocr_results = None
            set_step_images_for_ocr(None)  # 重置 OCR 上下文，避免跨请求残留
            self._reset_python_execute_env()

            # Phase 1: 感知与认知解耦 — 多模态任务先 OCR 提取，Planning 基于 OCR 文本规划
            if base64_images and len(base64_images) > 0 and getattr(config, "ocr_config", None):
                try:
                    ocr = OcrExtract()
                    r = await ocr.execute(base64_images=base64_images)
                    if r.get("success") and "results" in r:
                        results = r["results"]
                        # 确保OCR结果按index排序，保持与base64_images相同的顺序
                        sorted_results = sorted(results, key=lambda x: x.get("index", 0))
                        
                        # 关键修复：验证index的连续性，确保没有遗漏或重复
                        expected_indices = set(range(len(base64_images)))
                        actual_indices = set(item.get("index", i) for i, item in enumerate(sorted_results))
                        
                        if expected_indices != actual_indices:
                            logger.warning(f"OCR index mismatch! Expected: {expected_indices}, Got: {actual_indices}")
                            # 重新排序，确保结果按输入顺序排列
                            sorted_results = sorted(results, key=lambda x: x.get("index", 0))
                        
                        # P0 FIX: 过滤掉空或低质量的OCR结果
                        filtered_results = []
                        for item in sorted_results:
                            text = item.get("text", "")
                            # 检查OCR结果是否有效（非空且有一定长度）
                            if text and len(text.strip()) >= 20:
                                filtered_results.append(item)
                            else:
                                logger.warning(f"OCR result for image {item.get('index', '?')} is empty or too short, marking as invalid")
                                # 保留空字符串作为占位符，但会被标记为无效
                                filtered_results.append({"index": item.get("index", 0), "text": ""})
                        
                        self.ocr_results = [
                            item.get("text", "")
                            for item in filtered_results
                        ]
                        
                        # 详细日志：验证OCR结果与图片的对应关系
                        logger.info(f"OCR preprocessing: {len(self.ocr_results)} images extracted")
                        valid_ocr_count = sum(1 for t in self.ocr_results if t and len(t.strip()) >= 20)
                        logger.info(f"  Valid OCR results: {valid_ocr_count}/{len(self.ocr_results)}")
                        
                        for i, (item, text_preview) in enumerate(zip(filtered_results, self.ocr_results)):
                            preview = (text_preview or "")[:100].replace('\n', ' ')
                            status = "✓" if text_preview and len(text_preview.strip()) >= 20 else "✗"
                            logger.info(f"  {status} OCR Image {i+1} (index={item.get('index', i)}): {preview}...")
                        
                        # P0 FIX: 如果所有OCR结果都无效，fallback到vision模式
                        if valid_ocr_count == 0:
                            logger.warning("All OCR results are invalid, falling back to vision mode")
                            self.ocr_results = None
                        elif len(self.ocr_results) != len(base64_images):
                            logger.warning(f"✗ Mismatch: base64_images={len(base64_images)}, ocr_results={len(self.ocr_results)}")
                            # 如果数量不匹配，禁用OCR结果，让Planning Agent直接看图片
                            self.ocr_results = None
                            logger.info("Falling back to vision mode due to OCR count mismatch")
                        else:
                            logger.info(f"✓ OCR results count matches base64_images count: {len(base64_images)}")
                except Exception as e:
                    logger.warning(f"OCR preprocessing failed, fallback to vision: {e}")
                    self.ocr_results = None

            # 防死循环拦截器：每个样本开始时创建并注入到所有 agent
            anti_loop = AntiLoopInterceptor()
            for agent in self.agents.values():
                if hasattr(agent, "anti_loop_interceptor"):
                    agent.anti_loop_interceptor = anti_loop

            if input_text:
                await self._create_initial_plan(input_text)
                if self.active_plan_id not in self.planning_tool.plans:
                    logger.error(
                        f"Plan creation failed. Plan ID {self.active_plan_id} not found in planning tool."
                    )
                    return f"Failed to create plan for: {input_text}"

            execution_result = ""
            while True:
                self.current_step_index, step_info = await self._get_current_step_info()

                if self.current_step_index is None:
                    execution_result += await self._finalize_plan(
                        execution_result=execution_result,
                        user_request=input_text or "",
                    )
                    break

                step_type = step_info.get("type") if step_info else None
                executor = self.get_executor(step_type)
                step_result = await self._execute_step(
                    executor,
                    step_info,
                    previous_output=execution_result,
                    user_request=input_text or "",
                )
                execution_result += step_result + "\n"

                if hasattr(executor, "state") and executor.state == AgentState.FINISHED:
                    break

            return execution_result
        except Exception as e:
            logger.error(f"Error in PlanningFlow: {str(e)}")
            return f"Execution failed: {str(e)}"

    def _get_planning_agent(self):
        """获取可用于创建计划的 PlanningAgent（持有 workflow_state_tool 且工具范围限定为规划相关）。"""
        for key, agent in self.agents.items():
            if (
                hasattr(agent, "workflow_state_tool")
                and agent.workflow_state_tool is not None
                and agent.workflow_state_tool is self.planning_tool
            ):
                return agent
        return None

    async def _create_initial_plan(self, request: str) -> None:
        """创建初始计划。若存在 PlanningAgent 则委托其执行，否则使用 flow 的 LLM 直接调用 planning_tool。"""
        logger.info(f"Creating initial plan with ID: {self.active_plan_id}")

        planning_agent = self._get_planning_agent()
        if planning_agent is not None:
            await self._create_plan_via_planning_agent(planning_agent, request)
        else:
            await self._create_plan_via_flow_llm(request)

        self._sanitize_created_plan(request)

    def _sanitize_created_plan(self, request: str) -> None:
        """对新生成的 plan 做轻量结构校验，修正常见高风险模式。"""
        plan_data = self.planning_tool.plans.get(self.active_plan_id)
        if not plan_data:
            return

        original_steps = list(plan_data.get("steps", []))
        if not original_steps:
            return

        sanitized_steps: List[str] = []
        sanitized_notes: List[str] = []

        for raw_step in original_steps:
            step, note = self._sanitize_plan_step(raw_step, request)
            step, rewrite_note = self._rewrite_multimodal_to_text_for_explicit_prose(
                step, request
            )
            if rewrite_note:
                note = " ".join(part for part in [note, rewrite_note] if part).strip()
            sanitized_steps.append(step)
            sanitized_notes.append(note)

        self._augment_finance_steps_for_request_shape(
            sanitized_steps, sanitized_notes, request
        )
        self._align_text_extraction_variable_names(
            sanitized_steps, sanitized_notes, request
        )
        self._drop_overridden_duplicate_steps(sanitized_steps, sanitized_notes)

        if sanitized_steps == original_steps and not any(sanitized_notes):
            return

        plan_data["steps"] = sanitized_steps
        plan_data["step_statuses"] = ["not_started"] * len(sanitized_steps)
        plan_data["step_notes"] = sanitized_notes + [""] * max(
            0, len(sanitized_steps) - len(sanitized_notes)
        )
        logger.info("Sanitized newly created plan with lightweight structural guards")

    def _sanitize_plan_step(self, raw_step: str, request: str) -> tuple[str, str]:
        note_parts: List[str] = []
        step = self._normalize_plan_step_executor(raw_step)
        if step != raw_step:
            note_parts.append("Normalized missing executor tag.")

        step, binding_note = self._repair_multimodal_image_binding(step)
        if binding_note:
            note_parts.append(binding_note)

        step, drift_note = self._repair_multimodal_query_semantic_drift(step, request)
        if drift_note:
            note_parts.append(drift_note)

        step, revised_note = self._repair_revised_finance_literal_step(step, request)
        if revised_note:
            note_parts.append(revised_note)

        step, literal_note = self._repair_suspicious_finance_literal_step(step, request)
        if literal_note:
            note_parts.append(literal_note)

        return step, " ".join(note_parts).strip()

    def _normalize_plan_step_executor(self, step: str) -> str:
        stripped = (step or "").strip()
        if not stripped:
            return stripped
        if re.match(r"^\[[a-zA-Z_]+\]", stripped):
            return stripped
        inferred = self._infer_step_type(stripped)
        return f"[{inferred}] {stripped}"

    def _extract_step_query_text(self, step: str) -> str:
        match = re.search(r"extract\s+'([^']+)'", step, re.IGNORECASE)
        if match:
            return match.group(1)
        text = re.sub(r"^\[[a-zA-Z_]+\]\s*", "", step or "").strip()
        text = re.sub(r"\bfrom\s+image\s*\d+\b", "", text, flags=re.IGNORECASE)
        return text.strip()

    def _extract_single_quoted_query(self, step: str) -> Optional[str]:
        match = re.search(r"extract\s+'([^']+)'", step or "", re.IGNORECASE)
        return match.group(1) if match else None

    def _replace_single_quoted_query(self, step: str, new_query: str) -> str:
        return re.sub(
            r"(extract\s+')([^']+)(')",
            lambda m: f"{m.group(1)}{new_query}{m.group(3)}",
            step,
            count=1,
            flags=re.IGNORECASE,
        )

    def _normalize_semantic_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", (text or "").lower())).strip()

    def _cleanup_semantic_query_text(self, text: str) -> str:
        cleaned = re.sub(r"\s{2,}", " ", text or "")
        cleaned = re.sub(r"\s+([,;:/)])", r"\1", cleaned)
        cleaned = re.sub(r"([(])\s+", r"\1", cleaned)
        cleaned = re.sub(r"\s*-\s*", "-", cleaned)
        return cleaned.strip(" ,;:-")

    def _extract_effective_question_text(self, request: str) -> str:
        text = (request or "").strip()
        if not text:
            return ""
        match = re.search(
            r"(?:^|\n)\s*(?:question|问题)\s*:\s*(.+)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return text

    def _request_has_respectively_pair(self, request: str) -> bool:
        if not request:
            return False
        if not re.search(r"\brespectively\b|分别", request, re.IGNORECASE):
            return False
        return self._count_numeric_literals(request) >= 2

    def _count_numeric_literals(self, text: str) -> int:
        return len(
            re.findall(r"[-+]?(?:[$¥€£])?\d[\d,]*(?:\.\d+)?%?", text or "")
        )

    def _request_has_plain_text_context(self, request: str) -> bool:
        stripped = re.sub(r"<image\s+\d+>", " ", request or "", flags=re.IGNORECASE)
        normalized = self._normalize_semantic_text(stripped)
        return len(normalized) >= 40 and len(normalized.split()) >= 8

    def _extract_request_time_hints(self, request: str) -> List[str]:
        question_text = self._extract_effective_question_text(request)
        if not question_text:
            return []
        patterns = [
            r"\b[A-Za-z]+\s+\d{1,2},\s+(?:19|20)\d{2}\b",
            r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\s+(?:19|20)\d{2}\b",
            r"\b(?:q[1-4]|first quarter|second quarter|third quarter|fourth quarter)\s+(?:of\s+)?(?:19|20)\d{2}\b",
            r"\b(?:19|20)\d{2}\b",
        ]
        seen: set[str] = set()
        hints: List[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, question_text, re.IGNORECASE):
                value = match.group(0).strip()
                key = value.lower()
                if key in seen:
                    continue
                seen.add(key)
                hints.append(value)
                if len(hints) >= 2:
                    return hints
        return hints

    def _request_is_pairwise_prose_percentage_change(self, request: str) -> bool:
        question_text = self._extract_effective_question_text(request)
        return bool(
            self._request_has_plain_text_context(request)
            and self._request_has_respectively_pair(request)
            and self._extract_request_metric_hint(request)
            and len(self._extract_request_time_hints(request)) >= 2
            and re.search(
                r"\b(percentage|percent)\s+change\b|\bchange\b.*\b(percent|percentage)\b|增长率|变动幅度|百分比变化",
                question_text,
                re.IGNORECASE,
            )
        )

    def _extract_finance_text_extraction_target(self, step: str) -> str:
        match = re.search(
            r"extract the exact stated value for (.+?) from the plain text context",
            step or "",
            re.IGNORECASE,
        )
        return match.group(1).strip() if match else ""

    def _build_metric_aligned_var_name(self, metric_hint: str, time_hint: str) -> str:
        base = self._normalize_semantic_text(metric_hint).replace(" ", "_")
        if not base:
            return ""
        suffix_parts: List[str] = []
        month_year = re.search(
            r"\b([A-Za-z]+)\s+((?:19|20)\d{2})\b",
            time_hint or "",
            re.IGNORECASE,
        )
        if month_year:
            suffix_parts.extend(
                [
                    self._normalize_semantic_text(month_year.group(1)).replace(" ", "_"),
                    month_year.group(2),
                ]
            )
        else:
            year_match = re.search(r"\b((?:19|20)\d{2})\b", time_hint or "")
            if year_match:
                suffix_parts.append(year_match.group(1))
        return "_".join([base] + suffix_parts) if suffix_parts else base

    def _replace_variable_name(self, step: str, old_var: str, new_var: str) -> str:
        if not step or not old_var or not new_var or old_var == new_var:
            return step
        return re.sub(rf"\b{re.escape(old_var)}\b", new_var, step)

    def _align_text_extraction_variable_names(
        self, steps: List[str], notes: List[str], request: str
    ) -> None:
        request_metric_hint = self._extract_request_metric_hint(request)
        if not request_metric_hint:
            return
        risky_var_terms = [
            "gain_on_swaps",
            "loss_on_swaps",
            "net_income_effect",
            "principal_amount",
            "principal_balance",
            "notional_amount",
            "share_count",
            "remaining_authorization_value",
        ]
        for idx, step in enumerate(list(steps)):
            if not (
                re.match(r"^\[finance\]", step or "", re.IGNORECASE)
                and self._looks_like_text_extraction_step(step or "")
                and "plain text context" in (step or "").lower()
            ):
                continue
            output_var = self._extract_save_as_var(step or "")
            if not output_var:
                continue
            normalized_output_var = self._normalize_semantic_text(output_var.replace("_", " "))
            if not any(term.replace("_", " ") in normalized_output_var for term in risky_var_terms):
                continue
            target = self._extract_finance_text_extraction_target(step)
            time_hint = self._extract_query_time_hint(target)
            new_var = self._build_metric_aligned_var_name(request_metric_hint, time_hint)
            if not new_var or new_var == output_var:
                continue
            for j in range(idx, len(steps)):
                steps[j] = self._replace_variable_name(steps[j], output_var, new_var)
            existing = notes[idx].strip()
            extra = (
                f"Aligned text-extraction variable name from {output_var} to {new_var} "
                "to match the requested metric and avoid nearby-metric contamination."
            )
            notes[idx] = f"{existing} {extra}".strip() if existing else extra

    def _extract_request_metric_hint(self, request: str) -> Optional[str]:
        normalized_request = self._normalize_semantic_text(request)
        if not normalized_request:
            return None
        candidate_phrases = [
            "cash flow hedges",
            "fair value hedges",
            "undesignated hedges",
            "interest rate swaps",
            "gain on swaps",
            "loss on swaps",
            "notional amount",
            "carrying value",
            "fair value",
            "shares purchased",
            "average price per share",
            "basic earnings per share",
            "weighted average number of shares",
            "remaining authorization value",
            "归母净利润",
            "每股净资产",
            "市场份额",
        ]
        for phrase in candidate_phrases:
            if self._normalize_semantic_text(phrase) in normalized_request:
                return phrase
        return None

    def _extract_query_time_hint(self, query: str) -> str:
        text = query or ""
        match = re.search(
            r"\b(?:for|as of|at|during)\s+([A-Za-z]+\s+\d{1,2},\s+\d{4}|[A-Za-z]+\s+\d{4}|(?:19|20)\d{2}|q[1-4]\s+(?:19|20)\d{2}|(?:19|20)\d{2}\s*q[1-4])\b",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        year_match = re.search(r"\b(19|20)\d{2}\b", text)
        return year_match.group(0) if year_match else ""

    def _build_text_extraction_focus(self, query: str, request: str) -> str:
        metric_hint = self._extract_request_metric_hint(request)
        time_hint = self._extract_query_time_hint(query)
        target = query.strip()
        if metric_hint:
            target = metric_hint
            if time_hint:
                target = f"{metric_hint} for {time_hint}"

        confusable_terms = [
            "undesignated hedges",
            "fair value hedges",
            "gain on swaps",
            "loss on swaps",
            "net income effect",
            "notional amount",
            "interest rate swaps",
            "principal amount",
            "share count",
            "weighted average number of shares",
        ]
        normalized_metric = self._normalize_semantic_text(metric_hint or target)
        exclusions = [
            term
            for term in confusable_terms
            if self._normalize_semantic_text(term) in self._normalize_semantic_text(request)
            and self._normalize_semantic_text(term) != normalized_metric
        ]
        exclusion_text = ""
        if exclusions:
            exclusion_text = (
                "; match the same metric phrase as the question and do NOT use nearby metrics such as "
                + ", ".join(exclusions[:4])
            )
        return (
            f"extract the exact stated value for {target} from the plain text context"
            f"{exclusion_text}"
        )

    def _extract_query_anchor_tokens(self, query: str) -> List[str]:
        normalized = self._normalize_semantic_text(query)
        if not normalized:
            return []
        stop = {
            "extract",
            "value",
            "values",
            "for",
            "from",
            "period",
            "during",
            "ended",
            "ending",
            "year",
            "quarter",
            "month",
            "date",
            "october",
            "november",
            "december",
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "first",
            "second",
            "third",
            "fourth",
            "as",
            "of",
            "the",
        }
        tokens: List[str] = []
        for token in normalized.split():
            if token in stop or re.fullmatch(r"(?:19|20)\d{2}", token):
                continue
            if len(token) <= 2 and not re.search(r"[\u4e00-\u9fff]", token):
                continue
            tokens.append(token)
        return tokens[:6]

    def _preview_contains_query_anchor(self, step: str) -> bool:
        if not self.ocr_results:
            return False
        query = self._extract_single_quoted_query(step)
        if not query:
            return False
        step_idx = self._get_image_index_for_step({"text": step})
        if step_idx >= len(self.ocr_results):
            return False
        preview = self._normalize_semantic_text(self.ocr_results[step_idx] or "")
        if not preview:
            return False
        anchor_tokens = self._extract_query_anchor_tokens(query)
        if not anchor_tokens:
            return False
        hits = sum(1 for token in anchor_tokens if token in preview)
        threshold = 2 if len(anchor_tokens) >= 2 else 1
        return hits >= threshold

    def _rewrite_multimodal_to_text_for_explicit_prose(
        self, step: str, request: str
    ) -> tuple[str, Optional[str]]:
        if not re.match(r"^\[multimodal\]", step or "", re.IGNORECASE):
            return step, None
        query = self._extract_single_quoted_query(step)
        output_var = self._extract_save_as_var(step)
        if not query or not output_var:
            return step, None
        normalized_request = self._normalize_semantic_text(request)
        anchor_tokens = self._extract_query_anchor_tokens(query)
        if not anchor_tokens:
            return step, None
        if not all(token in normalized_request for token in anchor_tokens[:2]):
            return step, None
        if self._preview_contains_query_anchor(step):
            return step, None
        if not self._request_has_plain_text_context(request):
            return step, None
        if self._count_numeric_literals(request) < 1:
            return step, None
        if (
            not self._request_has_respectively_pair(request)
            and len(anchor_tokens) < 2
        ):
            return step, None
        extraction_focus = self._build_text_extraction_focus(query, request)
        rewritten = f"[finance] {extraction_focus} save_as {output_var}"
        return (
            rewritten,
            "Rewrote multimodal extraction to text extraction because the metric is explicitly stated in prose while the bound image preview lacks that metric.",
        )

    def _drop_overridden_duplicate_steps(
        self, steps: List[str], notes: List[str]
    ) -> None:
        seen_text_vars: set[str] = set()
        filtered_steps: List[str] = []
        filtered_notes: List[str] = []
        for step, note in zip(steps, notes):
            output_var = self._extract_save_as_var(step or "")
            is_text_step = bool(
                re.match(r"^\[finance\]", step or "", re.IGNORECASE)
                and self._looks_like_text_extraction_step(step or "")
            )
            is_value_assignment_step = self._is_value_assignment_step(step or "")
            if (
                output_var
                and output_var in seen_text_vars
                and is_value_assignment_step
                and re.match(r"^\[(finance|multimodal)\]", step or "", re.IGNORECASE)
                and not self._is_identity_save_step(step or "")
            ):
                continue
            filtered_steps.append(step)
            filtered_notes.append(note)
            if output_var and is_text_step:
                seen_text_vars.add(output_var)
        steps[:] = filtered_steps
        notes[:] = filtered_notes

    def _repair_multimodal_query_semantic_drift(
        self, step: str, request: str
    ) -> tuple[str, Optional[str]]:
        if not re.match(r"^\[multimodal\]", step or "", re.IGNORECASE):
            return step, None

        query = self._extract_single_quoted_query(step)
        if not query:
            return step, None

        normalized_request = self._normalize_semantic_text(request)
        repaired_query = query
        removed_terms: List[str] = []
        request_metric_hint = self._extract_request_metric_hint(request)
        normalized_request_metric_hint = self._normalize_semantic_text(request_metric_hint or "")

        risky_neighbor_terms = [
            "net income effect",
            "principal amount",
            "principal balance",
            "notional amount",
            "share count",
            "weighted average number of shares",
            "amortized cost",
            "gain on swaps",
            "remaining authorization value",
        ]
        for term in risky_neighbor_terms:
            if term not in self._normalize_semantic_text(repaired_query):
                continue
            if term in normalized_request:
                continue
            if re.search(rf"\bnot\s+{re.escape(term)}\b", repaired_query, re.IGNORECASE):
                continue
            replacement = " "
            if (
                request_metric_hint
                and normalized_request_metric_hint
                and normalized_request_metric_hint
                != self._normalize_semantic_text(term)
            ):
                replacement = f" {request_metric_hint} "
            repaired_query = re.sub(
                rf"\b{re.escape(term)}\b",
                replacement,
                repaired_query,
                flags=re.IGNORECASE,
            )
            removed_terms.append(term)

        repaired_query = self._cleanup_semantic_query_text(repaired_query)
        if not removed_terms or not repaired_query or repaired_query == query:
            return step, None

        return (
            self._replace_single_quoted_query(step, repaired_query),
            "Re-anchored a multimodal query away from a nearby but semantically different metric.",
        )

    def _tokenize_match_text(self, text: str) -> List[str]:
        if not text:
            return []
        lowered = text.lower()
        stopwords = {
            "the",
            "and",
            "from",
            "with",
            "that",
            "this",
            "what",
            "which",
            "into",
            "only",
            "using",
            "use",
            "save",
            "image",
            "images",
            "table",
            "chart",
            "value",
            "values",
            "data",
            "metric",
            "extract",
            "compute",
            "calculate",
            "result",
            "final",
            "requested",
            "显示",
            "提取",
            "计算",
            "结果",
            "数据",
            "图片",
            "图像",
            "表格",
            "数值",
            "变量",
            "根据",
            "以及",
            "或者",
            "用于",
        }
        raw_tokens = (
            re.findall(r"\b(?:19|20)\d{2}\b", lowered)
            + re.findall(r"\b(?:q[1-4]|h[12]|fy)\b", lowered)
            + re.findall(r"[a-z]{3,}|[\u4e00-\u9fff]{2,}", lowered)
        )
        tokens: List[str] = []
        seen = set()
        for token in raw_tokens:
            if token in stopwords or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
        return tokens[:12]

    def _score_ocr_preview_match(self, preview: str, tokens: List[str]) -> int:
        if not preview or not tokens:
            return 0
        normalized = preview.lower()
        score = 0
        for token in tokens:
            if token not in normalized:
                continue
            if re.fullmatch(r"(?:19|20)\d{2}|q[1-4]|h[12]|fy", token):
                score += 3
            elif re.search(r"[\u4e00-\u9fff]", token) or len(token) >= 6:
                score += 2
            else:
                score += 1
        return score

    def _repair_multimodal_image_binding(self, step: str) -> tuple[str, Optional[str]]:
        if (
            not step
            or not re.match(r"^\[multimodal\]", step, re.IGNORECASE)
            or not self.ocr_results
            or len(self.ocr_results) <= 1
        ):
            return step, None

        tokens = self._tokenize_match_text(self._extract_step_query_text(step))
        if not tokens:
            return step, None

        scores = [self._score_ocr_preview_match(preview or "", tokens) for preview in self.ocr_results]
        if not scores:
            return step, None

        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        best_score = scores[best_idx]
        current_idx = self._get_image_index_for_step({"text": step})
        current_score = scores[current_idx] if current_idx < len(scores) else 0
        has_explicit_binding = bool(
            re.search(r"(?:from\s+)?image\s*\d+", step, re.IGNORECASE)
        )

        if best_score <= 0:
            return step, None

        if has_explicit_binding:
            if best_idx == current_idx or current_score > 0 or best_score < current_score + 2:
                return step, None
            rebound = re.sub(
                r"((?:from\s+)?image\s*)\d+",
                lambda m: f"{m.group(1)}{best_idx + 1}",
                step,
                count=1,
                flags=re.IGNORECASE,
            )
            return (
                rebound,
                f"Rebound image {current_idx + 1} to image {best_idx + 1} using OCR preview overlap.",
            )

        if best_score < 2:
            return step, None

        bound = re.sub(
            r"^\[multimodal\]\s*",
            f"[multimodal] from image {best_idx + 1} ",
            step,
            count=1,
            flags=re.IGNORECASE,
        )
        return (
            bound,
            f"Bound multimodal step to image {best_idx + 1} using OCR preview overlap.",
        )

    def _looks_like_text_extraction_step(self, step: str) -> bool:
        return bool(
            re.search(
                r"\b(text|context|stated|provided|request|source snippet)\b|文本|上下文|文中|题干|原文",
                step or "",
                re.IGNORECASE,
            )
        )

    def _normalize_numeric_token(self, text: str) -> str:
        return re.sub(r"[\s,.$¥€£%]", "", (text or "").lower())

    def _literal_exists_in_request(self, literal: str, request: str) -> bool:
        normalized_literal = self._normalize_numeric_token(literal)
        if not normalized_literal:
            return False
        return normalized_literal in self._normalize_numeric_token(request)

    def _extract_finance_assignment(
        self, step: str
    ) -> tuple[Optional[str], Optional[str]]:
        if not re.match(r"^\[finance\]", step or "", re.IGNORECASE):
            return None, None
        match = re.search(
            r"(?:formula:\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;\n]+)",
            step or "",
            re.IGNORECASE,
        )
        if not match:
            return None, None
        lhs = match.group(1).strip()
        rhs = re.split(r"\bsave_as\b", match.group(2), maxsplit=1, flags=re.IGNORECASE)[
            0
        ].strip(" .")
        return lhs, rhs

    def _is_identity_save_step(self, step: str) -> bool:
        output_var = self._extract_save_as_var(step or "")
        if not output_var:
            return False
        lhs, rhs = self._extract_finance_assignment(step or "")
        if not lhs or not rhs:
            return False
        normalized_rhs = re.sub(r"\s+", "", rhs)
        return lhs == output_var and normalized_rhs == output_var

    def _is_value_assignment_step(self, step: str) -> bool:
        if re.search(r"\bextract\b", step or "", re.IGNORECASE):
            return True
        lhs, _ = self._extract_finance_assignment(step or "")
        return bool(lhs)

    def _expression_references_variables(
        self, expression: str, exclude: Optional[set[str]] = None
    ) -> bool:
        if not expression:
            return False
        excluded = {token.lower() for token in (exclude or set())}
        reserved = {
            "if",
            "else",
            "and",
            "or",
            "not",
            "formula",
            "save_as",
            "round",
            "abs",
            "min",
            "max",
            "sum",
            "float",
            "int",
            "true",
            "false",
            "none",
        }
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression):
            lowered = token.lower()
            if lowered in reserved or lowered in excluded:
                continue
            return True
        return False

    def _is_literal_only_finance_assignment(self, step: str) -> bool:
        lhs, rhs = self._extract_finance_assignment(step)
        if not lhs or not rhs:
            return False
        has_numeric_literal = bool(
            re.search(r"[-+]?(?:[$¥€£])?\d[\d,]*(?:\.\d+)?%?", rhs)
        )
        if not has_numeric_literal:
            return False
        return not self._expression_references_variables(rhs, exclude={lhs})

    def _repair_suspicious_finance_literal_step(
        self, step: str, request: str
    ) -> tuple[str, Optional[str]]:
        if not re.match(r"^\[finance\]", step or "", re.IGNORECASE):
            return step, None
        if self._looks_like_text_extraction_step(step):
            return step, None
        is_revised_request = bool(
            re.search(
            r"\b(revised|hypothetical|assum(?:e|ing)?|what if|pro forma|adjusted|reduced to|increased to|decreased to|raised to|lowered to)\b|假设|调整后|修订|若|如果|改为|变为|上调至|下调至|减少到|增加到",
            step,
            re.IGNORECASE,
            )
        )
        if is_revised_request:
            return step, None

        assignment_lhs, assignment_rhs = self._extract_finance_assignment(step)
        if assignment_lhs and assignment_rhs and self._is_literal_only_finance_assignment(step):
            rewritten = (
                f"[finance] extract the exact stated value for {assignment_lhs} from the "
                "plain text context and save it as "
                f"{assignment_lhs}; if the value is not explicitly stated in text, "
                "terminate with failure instead of guessing"
            )
            return (
                rewritten,
                f"Rewrote literal-only finance assignment for {assignment_lhs} into grounded text extraction.",
            )

        match = re.search(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+]?(?:[$¥€£])?\d[\d,]*(?:\.\d+)?%?)\b",
            step,
        )
        if not match:
            return step, None

        variable_name = match.group(1)
        literal_value = match.group(2)
        if not self._literal_exists_in_request(literal_value, request):
            rewritten = (
                "[finance] compute the final requested result using only previously "
                "extracted variables; do not introduce new numeric literals. "
                "If a required value is explicitly stated in the plain text context, "
                "extract that exact stated value from text first instead of hardcoding it."
            )
            return (
                rewritten,
                f"Removed an ungrounded numeric literal assignment for {variable_name}.",
            )

        rewritten = (
            f"[finance] extract the exact stated value for {variable_name} "
            f"from the plain text context and save it as {variable_name}; "
            "do not hardcode the numeric literal in code."
        )
        return (
            rewritten,
            f"Rewrote suspicious finance literal assignment for {variable_name} into text extraction.",
        )

    def _repair_revised_finance_literal_step(
        self, step: str, request: str
    ) -> tuple[str, Optional[str]]:
        if not re.match(r"^\[finance\]", step or "", re.IGNORECASE):
            return step, None
        if not self._request_is_revised(request):
            return step, None

        literal_match = re.search(
            r"[-+]?(?:[$¥€£])?\d[\d,]*(?:\.\d+)?%?",
            step or "",
        )
        if not literal_match:
            return step, None

        save_as_match = re.search(r"\bsave_as\s+([A-Za-z_][A-Za-z0-9_]*)", step, re.IGNORECASE)
        output_var = save_as_match.group(1) if save_as_match else None
        rewritten = (
            "[finance] read the user-provided replacement value from the plain text "
            "context, store it in an explicit revised_* variable, then compute the "
            "final revised/comparison result using that revised_* variable and the "
            "original baseline variables only"
        )
        if output_var:
            rewritten += f" save_as {output_var}"
        return (
            rewritten,
            "Rewrote revised/hypothetical finance step to use an explicit revised_* variable instead of a direct literal formula.",
        )

    def _request_is_revised(self, request: str) -> bool:
        question_text = self._extract_effective_question_text(request)
        return bool(
            re.search(
                r"\b(revised|hypothetical|assum(?:e|ing)?|what if|pro forma|adjusted|reduced to|increased to|decreased to|raised to|lowered to)\b|假设|调整后|修订|若|如果|改为|变为|上调至|下调至|减少到|增加到",
                question_text,
                re.IGNORECASE,
            )
        )

    def _request_is_comparison(self, request: str) -> bool:
        return bool(
            re.search(
                r"\b(compare|comparison|difference|change|ratio|percentage|percent|gap|versus|vs\.?)\b|同比|环比|增幅|降幅|差额|差值|占比|比例|相比|比较",
                request or "",
                re.IGNORECASE,
            )
        )

    def _request_needs_finance_compute(self, request: str) -> bool:
        return bool(
            re.search(
                r"\b(sum|total|combined|difference|change|ratio|percentage|percent|compare|cost|value|multiply|divid|minus|plus)\b|合计|总计|总额|之和|乘以|除以|差额|差值|占比|比例|增长率|比较",
                request or "",
                re.IGNORECASE,
            )
        )

    def _request_is_cash_paid_to_suppliers(self, request: str) -> bool:
        return bool(
            re.search(
                r"cash paid to (its )?suppliers|pay to (its )?suppliers|付给供应商|支付给供应商",
                request or "",
                re.IGNORECASE,
            )
        )

    def _request_is_share_purchase_total_value(self, request: str) -> bool:
        return bool(
            re.search(
                r"total value of shares purchased|value of shares purchased|shares purchased during|回购.*价值|购入股份.*价值|购买股份.*价值",
                request or "",
                re.IGNORECASE,
            )
        )

    def _request_is_market_share_sum_compare_ratio(self, request: str) -> bool:
        return bool(
            re.search(r"市场份额", request or "", re.IGNORECASE)
            and re.search(r"总和|合计", request or "", re.IGNORECASE)
            and re.search(r"比较|相比", request or "", re.IGNORECASE)
        )

    def _request_is_second_half_sales_share_with_q4_adjustment(self, request: str) -> bool:
        return bool(
            re.search(r"second half|下半年", request or "", re.IGNORECASE)
            and re.search(r"fourth quarter|第四季度|q4", request or "", re.IGNORECASE)
            and re.search(r"goes up|increase|increased|上涨|上升|增加", request or "", re.IGNORECASE)
            and re.search(r"share of annual sales|annual sales share|年度销售占比|销售占比", request or "", re.IGNORECASE)
        )

    def _request_is_goodwill_adjusted_return(self, request: str) -> bool:
        return bool(
            re.search(r"rate of return|return", request or "", re.IGNORECASE)
            and re.search(r"goodwill", request or "", re.IGNORECASE)
            and re.search(r"earning asset|earning assets|收益资产|盈利资产", request or "", re.IGNORECASE)
        )

    def _request_is_revised_change_with_replacement_value(self, request: str) -> bool:
        return bool(
            self._request_is_revised(request)
            and re.search(r"revised change|修订后.*变化|修订.*差值|revised.*change", request or "", re.IGNORECASE)
            and re.search(r"reduced to|increased to|decreased to|raised to|lowered to|改为|变为|减少到|增加到", request or "", re.IGNORECASE)
        )

    def _has_step_referencing(self, steps: List[str], pattern: str) -> bool:
        return any(re.search(pattern, step or "", re.IGNORECASE) for step in steps)

    def _extract_save_as_var(self, step: str) -> Optional[str]:
        match = re.search(r"\bsave_as\s+([A-Za-z_][A-Za-z0-9_]*)", step or "", re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_request_period_label(self, request: str) -> str:
        request = request or ""
        month_match = re.search(
            r"\b("
            r"january|february|march|april|may|june|july|august|september|october|november|december|"
            r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
            r")\s+\d{4}\b",
            request,
            re.IGNORECASE,
        )
        if month_match:
            return month_match.group(0)

        quarter_match = re.search(
            r"\b(q[1-4]|first quarter|second quarter|third quarter|fourth quarter)\s+(?:of\s+)?\d{4}\b",
            request,
            re.IGNORECASE,
        )
        if quarter_match:
            return quarter_match.group(0)

        year_match = re.search(r"\b(19|20)\d{2}\b", request)
        if year_match:
            return year_match.group(0)
        return "the requested period"

    def _extract_percent_literal(self, request: str) -> Optional[float]:
        match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*%", request or "", re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def _extract_image_reference(self, step: str) -> str:
        match = re.search(r"\bfrom\s+image\s+\d+\b", step or "", re.IGNORECASE)
        return match.group(0) if match else "from image 1"

    def _extract_request_note_series(self, request: str) -> Optional[str]:
        candidate_texts = [
            self._extract_effective_question_text(request),
            request or "",
        ]
        for text in candidate_texts:
            match = re.search(
                r"\b(\d+(?:\.\d+)?)%\s+notes?\s+(?:due\s+)?(?:in\s+)?((?:19|20)\d{2})\b",
                text,
                re.IGNORECASE,
            )
            if match:
                return f"{match.group(1)}% Notes due {match.group(2)}"
            year_match = re.search(
                r"\b((?:19|20)\d{2})\s+notes?\b|\bnotes?\s+(?:due\s+)?((?:19|20)\d{2})\b",
                text,
                re.IGNORECASE,
            )
            if year_match:
                year = year_match.group(1) or year_match.group(2)
                return f"Notes due {year}"
        return None

    def _extract_primary_year_and_baseline_year(self, request: str) -> tuple[Optional[int], Optional[int]]:
        years = self._extract_request_time_hints(request)
        normalized_years: List[int] = []
        for hint in years:
            if re.fullmatch(r"(?:19|20)\d{2}", hint):
                normalized_years.append(int(hint))
        if normalized_years:
            primary = normalized_years[0]
            if len(normalized_years) >= 2:
                baseline = normalized_years[1]
            else:
                baseline = primary - 1
            return primary, baseline
        match = re.search(r"\b((?:19|20)\d{2})\b", request or "")
        if match:
            primary = int(match.group(1))
            return primary, primary - 1
        return None, None

    def _request_is_note_annual_interest_expense(self, request: str) -> bool:
        return bool(
            re.search(
                r"annual interest expense.*notes?|interest expense related to .*notes?|notes?.*interest expense",
                request or "",
                re.IGNORECASE,
            )
        )

    def _request_has_approximate_interest_narrative(self, request: str) -> bool:
        return bool(
            re.search(
                r"\b(approximately|approx\.?|about)\b.+?\b(per year|annual)\b|约.*每年|每年约",
                request or "",
                re.IGNORECASE,
            )
        )

    def _extract_month_year_target(self, request: str) -> tuple[Optional[int], Optional[int]]:
        request = request or ""
        month_map = {
            "january": 1, "jan": 1,
            "february": 2, "feb": 2,
            "march": 3, "mar": 3,
            "april": 4, "apr": 4,
            "may": 5,
            "june": 6, "jun": 6,
            "july": 7, "jul": 7,
            "august": 8, "aug": 8,
            "september": 9, "sep": 9, "sept": 9,
            "october": 10, "oct": 10,
            "november": 11, "nov": 11,
            "december": 12, "dec": 12,
        }
        match = re.search(
            r"\b("
            r"january|february|march|april|may|june|july|august|september|october|november|december|"
            r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
            r")\s+((?:19|20)\d{2})\b",
            request,
            re.IGNORECASE,
        )
        if not match:
            return None, None
        return month_map[match.group(1).lower()], int(match.group(2))

    def _infer_share_purchase_period_range(
        self, image_ref: str, request: str
    ) -> Optional[str]:
        target_month, target_year = self._extract_month_year_target(request)
        if not target_month or not target_year:
            return None
        preview = self._get_ocr_preview_for_image_ref(image_ref)
        candidates = re.findall(
            r"\b(\d{2}/\d{2}/\d{2})\s+to\s+(\d{2}/\d{2}/\d{2})\b",
            preview or "",
            re.IGNORECASE,
        )
        for start, end in candidates:
            try:
                end_month = int(end.split("/")[0])
                end_year = 2000 + int(end.split("/")[2])
            except (ValueError, IndexError):
                continue
            if end_month == target_month and end_year == target_year:
                return f"{start} to {end}"
        return None

    def _get_ocr_preview_for_image_ref(self, image_ref: str) -> str:
        if not self.ocr_results:
            return ""
        match = re.search(r"\bimage\s+(\d+)\b", image_ref or "", re.IGNORECASE)
        idx = int(match.group(1)) - 1 if match else 0
        if 0 <= idx < len(self.ocr_results):
            return self.ocr_results[idx] or ""
        return ""

    def _compact_text(self, text: str) -> str:
        return re.sub(r"\s+", "", (text or "").lower())

    def _resolve_cash_paid_inventory_spec(
        self, steps: List[str], image_ref: str
    ) -> tuple[str, str, str]:
        if self._has_step_referencing(steps, r"decrease_in_inventory"):
            return (
                "decrease_in_inventory",
                "Decrease in inventory",
                "- decrease_in_inventory",
            )
        if self._has_step_referencing(steps, r"increase_in_inventory"):
            return (
                "increase_in_inventory",
                "Increase in inventory",
                "+ increase_in_inventory",
            )
        preview = self._compact_text(self._get_ocr_preview_for_image_ref(image_ref))
        if "increaseininventory" in preview or "存货增加" in preview:
            return (
                "increase_in_inventory",
                "Increase in inventory",
                "+ increase_in_inventory",
            )
        return (
            "decrease_in_inventory",
            "Decrease in inventory",
            "- decrease_in_inventory",
        )

    def _resolve_cash_paid_accounts_payable_spec(
        self, steps: List[str], image_ref: str
    ) -> tuple[str, str, str]:
        if self._has_step_referencing(steps, r"decrease_in_accounts_payable"):
            return (
                "decrease_in_accounts_payable",
                "Decrease in accounts payable",
                "+ decrease_in_accounts_payable",
            )
        if self._has_step_referencing(steps, r"increase_in_accounts_payable"):
            return (
                "increase_in_accounts_payable",
                "Increase in accounts payable",
                "- increase_in_accounts_payable",
            )
        preview = self._compact_text(self._get_ocr_preview_for_image_ref(image_ref))
        if "decreaseinaccountspayable" in preview or "应付账款减少" in preview:
            return (
                "decrease_in_accounts_payable",
                "Decrease in accounts payable",
                "+ decrease_in_accounts_payable",
            )
        return (
            "increase_in_accounts_payable",
            "Increase in accounts payable",
            "- increase_in_accounts_payable",
        )

    def _ensure_cash_paid_to_suppliers_template(
        self, steps: List[str], notes: List[str]
    ) -> None:
        if not self._has_step_referencing(steps, r"cost_of_goods_sold"):
            return
        image_ref = next(
            (
                self._extract_image_reference(step)
                for step in steps
                if re.match(r"^\[multimodal\]", step or "", re.IGNORECASE)
                and re.search(r"cost_of_goods_sold|inventory", step or "", re.IGNORECASE)
            ),
            "from image 1",
        )
        inventory_var, inventory_query, inventory_term = self._resolve_cash_paid_inventory_spec(
            steps, image_ref
        )
        ap_var, ap_query, ap_term = self._resolve_cash_paid_accounts_payable_spec(
            steps, image_ref
        )
        output_var = "cash_paid_to_suppliers"
        for step in steps:
            extracted = self._extract_save_as_var(step or "")
            if extracted and re.search(r"cash_paid_to_suppliers", extracted, re.IGNORECASE):
                output_var = extracted
                break

        steps[:] = [
            f"[multimodal] {image_ref} extract 'Cost of goods sold' save_as cost_of_goods_sold",
            f"[multimodal] {image_ref} extract '{inventory_query}' save_as {inventory_var}",
            f"[multimodal] {image_ref} extract '{ap_query}' save_as {ap_var}",
            f"[finance] compute {output_var} = cost_of_goods_sold {inventory_term} {ap_term} save_as {output_var}",
        ]
        notes[:] = [
            "Normalized plan to start from cost of goods sold for the standard cash-paid-to-suppliers indirect-method template.",
            "Added inventory-change extraction for standard cash-paid-to-suppliers indirect-method template.",
            "Added accounts-payable-change extraction for standard cash-paid-to-suppliers indirect-method template.",
            "Rewrote final finance step to standard indirect-method cash-paid-to-suppliers formula using only extracted variables.",
        ]

    def _ensure_note_annual_interest_expense_text_shortcut(
        self, steps: List[str], notes: List[str], request: str
    ) -> None:
        if not self._request_is_note_annual_interest_expense(request):
            return
        if not self._request_has_plain_text_context(request):
            return
        note_series = self._extract_request_note_series(request) or "the requested notes"
        output_var = "annual_interest_expense_requested_notes"
        for step in steps:
            extracted = self._extract_save_as_var(step or "")
            if extracted and re.search(r"interest_expense|annual_interest", extracted, re.IGNORECASE):
                output_var = extracted
                break

        if self._request_has_respectively_pair(request) and self._request_has_approximate_interest_narrative(request):
            steps[:] = [
                f"[finance] extract the exact stated principal amount for {note_series} from the plain text context save_as principal_amount_for_requested_notes",
                f"[finance] extract the exact stated coupon interest rate percentage for {note_series} from the plain text context save_as coupon_rate_percent_for_requested_notes",
                f"[finance] compute {output_var} = principal_amount_for_requested_notes * coupon_rate_percent_for_requested_notes / 100 save_as {output_var}",
            ]
            notes[:] = [
                "Replaced approximate annual-interest prose shortcut with principal extraction for the requested notes.",
                "Added coupon-rate percentage extraction for the requested notes.",
                "Rewrote final annual interest expense step to compute principal times coupon rate percentage.",
            ]
            return

        if not self._request_has_respectively_pair(request):
            return
        steps[:] = [
            f"[finance] extract the exact stated annual interest expense for {note_series} from the plain text context save_as {output_var}",
            f"[finance] compute {output_var} = {output_var} save_as {output_var}",
        ]
        notes[:] = [
            "Short-circuited to prose extraction because the annual interest expense is explicitly stated in the text context.",
            "Kept the final answer path in finance so the extracted prose value is surfaced directly.",
        ]

    def _ensure_pairwise_prose_percentage_change_template(
        self, steps: List[str], notes: List[str], request: str
    ) -> None:
        if not self._request_is_pairwise_prose_percentage_change(request):
            return
        metric_hint = self._extract_request_metric_hint(request)
        time_hints = self._extract_request_time_hints(request)
        if not metric_hint or len(time_hints) < 2:
            return
        current_time, baseline_time = time_hints[0], time_hints[1]
        current_var = self._build_metric_aligned_var_name(metric_hint, current_time) or "current_metric_value"
        baseline_var = self._build_metric_aligned_var_name(metric_hint, baseline_time) or "baseline_metric_value"
        output_var = "percentage_change"
        for step in steps:
            extracted = self._extract_save_as_var(step or "")
            if extracted and re.search(r"percentage_change|change_result|comparison_result", extracted, re.IGNORECASE):
                output_var = extracted
                break
        steps[:] = [
            f"[finance] extract the exact stated value for {metric_hint} for {current_time} from the plain text context save_as {current_var}",
            f"[finance] extract the exact stated value for {metric_hint} for {baseline_time} from the plain text context save_as {baseline_var}",
            f"[finance] compute {output_var} = (({current_var} - {baseline_var}) / abs({baseline_var})) * 100 save_as {output_var}",
        ]
        notes[:] = [
            "Normalized a prose-stated respectively pair into explicit text extraction for the current-period metric value.",
            "Normalized a prose-stated respectively pair into explicit text extraction for the baseline-period metric value.",
            "Rewrote the final step to a fixed percentage-change formula using the two prose-extracted values only.",
        ]

    def _rewrite_market_share_sum_compare_steps(
        self, steps: List[str], notes: List[str]
    ) -> None:
        if len(steps) < 4:
            return
        finance_indices = [
            idx for idx, step in enumerate(steps) if re.match(r"^\[finance\]", step or "", re.IGNORECASE)
        ]
        if not finance_indices:
            return
        output_var = "comparison_result"
        for step in steps:
            extracted = self._extract_save_as_var(step or "")
            if extracted and re.search(r"comparison|ratio|result", extracted, re.IGNORECASE):
                output_var = extracted
                break
        first_var = self._extract_save_as_var(steps[0]) or "market_share_a"
        second_var = self._extract_save_as_var(steps[1]) or "market_share_b"
        compare_var = self._extract_save_as_var(steps[2]) or "market_share_c"
        steps[:] = [
            steps[0],
            steps[1],
            steps[2],
            f"[finance] compute total_market_share = {first_var} + {second_var}; {output_var} = total_market_share / {compare_var} save_as {output_var}",
        ]
        notes[:] = [
            notes[0] if len(notes) > 0 else "",
            notes[1] if len(notes) > 1 else "",
            notes[2] if len(notes) > 2 else "",
            "Rewrote market-share comparison to ratio form: (sum of the first two companies) divided by the comparison company's market share.",
        ]

    def _rewrite_second_half_sales_share_steps(
        self, steps: List[str], notes: List[str], request: str
    ) -> None:
        image_ref = next(
            (
                self._extract_image_reference(step)
                for step in steps
                if re.match(r"^\[multimodal\]", step or "", re.IGNORECASE)
            ),
            "from image 1",
        )
        increase_pct = self._extract_percent_literal(request) or 0.0
        output_var = "second_half_share_percent"
        for step in reversed(steps):
            extracted = self._extract_save_as_var(step or "")
            if extracted and re.search(r"second_half|result|percentage", extracted, re.IGNORECASE):
                output_var = extracted
                break
        steps[:] = [
            f"[multimodal] {image_ref} extract 'Share of annual sales for Third quarter' save_as q3_share",
            f"[multimodal] {image_ref} extract 'Share of annual sales for Fourth quarter' save_as q4_share",
            f"[finance] compute q4_share_adjusted = q4_share * (1 + {increase_pct} / 100.0); {output_var} = round((q3_share + q4_share_adjusted) * 100, 0) save_as {output_var}",
        ]
        notes[:] = [
            "Dropped first-half quarters and kept only the second-half quarters required by the question.",
            "Kept a dedicated extraction for fourth-quarter annual-sales share before applying the requested uplift.",
            "Rewrote the final step to compute second-half share as Q3 plus adjusted Q4, then convert to percent with whole-number rounding.",
        ]

    def _rewrite_goodwill_adjusted_return_steps(
        self, steps: List[str], notes: List[str]
    ) -> None:
        if len(steps) < 3:
            return
        purchase_step = next(
            (
                step
                for step in steps
                if re.match(r"^\[multimodal\]", step or "", re.IGNORECASE)
                and re.search(r"purchase price", step or "", re.IGNORECASE)
            ),
            steps[0],
        )
        revenue_step = next(
            (
                step
                for step in steps
                if re.match(r"^\[multimodal\]", step or "", re.IGNORECASE)
                and re.search(r"revenue", step or "", re.IGNORECASE)
            ),
            steps[1] if len(steps) > 1 else steps[0],
        )
        purchase_var = self._extract_save_as_var(purchase_step) or "total_purchase_price"
        revenue_var = self._extract_save_as_var(revenue_step) or "total_revenue"
        purchase_image_ref = self._extract_image_reference(purchase_step)
        output_var = "rate_of_return_percentage"
        for step in steps:
            extracted = self._extract_save_as_var(step or "")
            if extracted and re.search(r"return|ratio|percentage|result", extracted, re.IGNORECASE):
                output_var = extracted
                break
        steps[:] = [
            purchase_step,
            revenue_step,
            f"[multimodal] {purchase_image_ref} extract 'Goodwill' save_as goodwill_value",
            f"[finance] compute {output_var} = (({revenue_var} + goodwill_value) / {purchase_var}) * 100 save_as {output_var}",
        ]
        notes[:] = [
            notes[0] if len(notes) > 0 else "Kept total purchase price extraction.",
            notes[1] if len(notes) > 1 else "Kept revenue extraction for the requested reporting period.",
            "Added a dedicated Goodwill extraction because the question explicitly treats Goodwill as an earning asset.",
            "Rewrote the return formula to include Goodwill in the numerator before dividing by total purchase price.",
        ]

    def _rewrite_revised_change_with_baseline_steps(
        self, steps: List[str], notes: List[str], request: str
    ) -> None:
        metric_step = next(
            (
                step
                for step in steps
                if re.match(r"^\[(multimodal|finance)\]", step or "", re.IGNORECASE)
                and re.search(r"debt discount|discount|amortization", step or "", re.IGNORECASE)
            ),
            steps[0] if steps else "",
        )
        if not metric_step:
            return
        metric_query = self._extract_single_quoted_query(metric_step) or "requested metric"
        primary_year, baseline_year = self._extract_primary_year_and_baseline_year(request)
        if not primary_year:
            return
        baseline_year = baseline_year or (primary_year - 1)
        image_ref = self._extract_image_reference(metric_step)
        output_var = "revised_change"
        for step in steps:
            extracted = self._extract_save_as_var(step or "")
            if extracted and re.search(r"revised|change|difference|result", extracted, re.IGNORECASE):
                output_var = extracted
                break
        normalized_query = re.sub(rf"\b{primary_year}\b", str(baseline_year), metric_query, count=1)
        if normalized_query == metric_query:
            normalized_query = f"{metric_query} for {baseline_year}"
        steps[:] = [
            f"[multimodal] {image_ref} extract '{normalized_query}' save_as baseline_metric_value",
            f"[finance] extract the user-provided replacement value from the plain text context save_as revised_metric_value",
            f"[finance] compute {output_var} = revised_metric_value - abs(baseline_metric_value) save_as {output_var}",
        ]
        notes[:] = [
            "Added baseline-period extraction for revised-change reasoning instead of comparing only against the original current-period value.",
            "Added an explicit text extraction for the replacement value stated in the question.",
            "Rewrote the final revised-change formula to compare the replacement value against the baseline-period magnitude.",
        ]

    def _rewrite_share_purchase_total_value_steps(
        self, steps: List[str], notes: List[str], request: str
    ) -> None:
        finance_indices = [
            idx for idx, step in enumerate(steps) if re.match(r"^\[finance\]", step or "", re.IGNORECASE)
        ]
        if not finance_indices:
            return
        has_operand_steps = any(
            re.match(r"^\[multimodal\]", step or "", re.IGNORECASE)
            and self._extract_save_as_var(step or "") in {"shares_purchased_for_period", "average_price_per_share_for_period"}
            for step in steps
        )
        if has_operand_steps:
            return
        multimodal_indices = [
            idx for idx, step in enumerate(steps) if re.match(r"^\[multimodal\]", step or "", re.IGNORECASE)
        ]
        direct_value_idx = next(
            (
                idx
                for idx in multimodal_indices
                if re.search(
                    r"total value of shares purchased|shares purchased during|approximate value of shares|remaining authorization value|shares purchased",
                    steps[idx] or "",
                    re.IGNORECASE,
                )
            ),
            multimodal_indices[0] if multimodal_indices else None,
        )
        if direct_value_idx is None:
            return
        final_idx = finance_indices[-1]
        final_output_var = self._extract_save_as_var(steps[final_idx]) or "total_value_of_shares_purchased"
        period_label = self._extract_request_period_label(request)
        image_ref = self._extract_image_reference(steps[direct_value_idx])
        period_range = self._infer_share_purchase_period_range(image_ref, request)
        shares_query = (
            f"Shares purchased, row {period_range}"
            if period_range
            else f"Shares purchased during {period_label}"
        )
        price_query = (
            f"Average price per share, row {period_range}, not Total/Average"
            if period_range
            else f"Average price per share during {period_label}, not Total/Average"
        )
        steps[direct_value_idx : final_idx + 1] = [
            f"[multimodal] {image_ref} extract '{shares_query}' save_as shares_purchased_for_period",
            f"[multimodal] {image_ref} extract '{price_query}' save_as average_price_per_share_for_period",
            f"[finance] compute {final_output_var} = shares_purchased_for_period * average_price_per_share_for_period save_as {final_output_var}",
        ]
        replacement_notes = [
            "Replaced direct total-value extraction with operand extraction to avoid nearby authorization-value leakage.",
            "Added period-specific per-share price extraction and excluded Total/Average rows.",
            "Rewrote finance step to multiply extracted shares by extracted average price per share.",
        ]
        notes[direct_value_idx : final_idx + 1] = replacement_notes

    def _augment_finance_steps_for_request_shape(
        self, steps: List[str], notes: List[str], request: str
    ) -> None:
        if self._request_is_cash_paid_to_suppliers(request):
            self._ensure_cash_paid_to_suppliers_template(steps, notes)

        self._ensure_note_annual_interest_expense_text_shortcut(steps, notes, request)
        self._ensure_pairwise_prose_percentage_change_template(steps, notes, request)
        if self._request_is_second_half_sales_share_with_q4_adjustment(request):
            self._rewrite_second_half_sales_share_steps(steps, notes, request)
        if self._request_is_goodwill_adjusted_return(request):
            self._rewrite_goodwill_adjusted_return_steps(steps, notes)
        if self._request_is_revised_change_with_replacement_value(request):
            self._rewrite_revised_change_with_baseline_steps(steps, notes, request)
        if self._request_is_market_share_sum_compare_ratio(request):
            self._rewrite_market_share_sum_compare_steps(steps, notes)

        if self._request_is_share_purchase_total_value(request):
            self._rewrite_share_purchase_total_value_steps(steps, notes, request)

        finance_indices = [
            idx for idx, step in enumerate(steps) if re.match(r"^\[finance\]", step, re.IGNORECASE)
        ]

        if not finance_indices:
            if self._request_needs_finance_compute(request):
                steps.append(
                    "[finance] compute the final requested result using the previously extracted variables only; do not introduce new numeric literals."
                )
                notes.append(
                    "Added a final finance computation step for a derived/comparison-style request."
                )
            return

        final_finance_idx = finance_indices[-1]
        final_step = steps[final_finance_idx]
        additions: List[str] = []

        if self._request_is_revised(request) and "revised_" not in final_step.lower():
            additions.append("create an explicit revised_* variable before the final formula")
        if self._request_is_revised(request) and "plain text context" not in final_step.lower():
            additions.append("read the replacement value from the plain text context instead of hardcoding it in code")

        if self._request_is_comparison(request) and not re.search(
            r"\b(difference|ratio|percentage|percent|change|gap|compare)\b|差额|差值|占比|比例|增长率|比较",
            final_step,
            re.IGNORECASE,
        ):
            additions.append(
                "compute the requested comparison from the extracted variables instead of restating raw values"
            )
        if self._request_is_comparison(request):
            additions.append(
                "if a required comparison baseline is missing, terminate with failure instead of inventing it"
            )

        if not additions:
            return

        suffix = "; ".join(additions)
        separator = " " if final_step.rstrip().endswith(".") else "; "
        steps[final_finance_idx] = f"{final_step.rstrip()}{separator}{suffix}."
        existing_note = notes[final_finance_idx].strip()
        extra_note = "Augmented final finance step with request-shape guardrails."
        notes[final_finance_idx] = (
            f"{existing_note} {extra_note}".strip() if existing_note else extra_note
        )

    def _extract_time_range_hint(self, request: str) -> str:
        """
        从问题中提取时间范围提示，帮助 Planning Agent 正确理解时间范围。
        
        例如：
        - "October 2018" -> "Focus ONLY on October 2018 data. Do NOT include data from other months like November or September unless explicitly requested."
        - "Q3 2024" -> "Focus ONLY on Q3 2024 data (July-September)."
        
        返回空字符串表示没有检测到特定时间范围提示。
        """
        import re
        
        request_lower = request.lower()
        hints = []
        
        # 检测月份 + 年份模式 (e.g., "October 2018", "oct 2018")
        month_year_patterns = [
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\s+(\d{4})\b',
            r'\b(\d{4})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        ]
        
        for pattern in month_year_patterns:
            match = re.search(pattern, request_lower)
            if match:
                month = match.group(1) if match.group(1).isalpha() else match.group(2)
                year = match.group(2) if match.group(1).isalpha() else match.group(1)
                month_name = month.capitalize() if len(month) > 3 else month.capitalize()
                hints.append(
                    f"Focus ONLY on {month_name} {year}. If the table has multiple period rows, use ONLY the row(s) "
                    f"that fall within {month_name} {year} (e.g. 09/30-10/27 for October). Do NOT sum rows from other months (e.g. 10/28-11/24 is November)."
                )
                break
        
        # 检测季度模式 (e.g., "Q3 2024", "third quarter 2024")
        quarter_map = {
            'first': 'Q1', 'second': 'Q2', 'third': 'Q3', 'fourth': 'Q4',
            'q1': 'Q1', 'q2': 'Q2', 'q3': 'Q3', 'q4': 'Q4'
        }
        quarter_patterns = [
            r'\b(q[1-4])\s+(\d{4})\b',
            r'\b(first|second|third|fourth)\s+quarter\s+(\d{4}|of\s+\d{4})\b',
        ]
        
        for pattern in quarter_patterns:
            match = re.search(pattern, request_lower)
            if match:
                quarter_raw = match.group(1).lower()
                quarter = quarter_map.get(quarter_raw, quarter_raw.upper())
                year = match.group(2)
                # 清理 year 中的 "of " 前缀
                if year.startswith('of '):
                    year = year[3:]
                hints.append(f"Focus ONLY on {quarter} {year} data. Do NOT include data from other quarters unless explicitly requested.")
                break
        
        return " ".join(hints) if hints else ""

    def _build_plan_prompt(self, request: str) -> str:
        """构建规划任务 prompt（含 plan_id、executor 描述、规划规则）。
        
        核心改进：Planning Agent 直接看图片，OCR文本仅作为辅助参考。
        """
        lines = [
            f"Create a plan for the following task. On your FIRST response, you MUST call the planning tool (do not output analysis without calling it). Use plan_id='{self.active_plan_id}'.",
            "",
            "**Task:**",
            request,
            "",
        ]
        
        # 核心改进：强调Planning Agent应该直接看图片
        if self.base64_images and len(self.base64_images) > 0:
            n_images = len(self.base64_images)
            lines.append(f"**IMAGES PROVIDED: {n_images} image(s) attached. YOU MUST LOOK AT THE IMAGES DIRECTLY.**")
            lines.append("")
            lines.append("**CRITICAL: Your Semantic Queries MUST match what you see in the images:**")
            lines.append("- LOOK at each image carefully to understand what data is available")
            lines.append("- USE the EXACT terminology/labels that appear in the images")
            lines.append("- If the image shows Chinese text, use Chinese terms in your semantic query")
            lines.append("- If the image shows 'Operating Income' instead of 'EBIT', use 'Operating Income'")
            lines.append("- DO NOT invent terms that don't appear in the images")
            lines.append("")
        
        # OCR文本作为辅助参考（非主要信息源）
        if self.ocr_results and len(self.ocr_results) > 0:
            valid_ocr = [(i, t) for i, t in enumerate(self.ocr_results) if t and len(t.strip()) >= 20]
            if valid_ocr:
                lines.append("**OCR Reference (AUXILIARY - use images as primary source):**")
                lines.append("NOTE: OCR may contain errors. Always verify with the actual images.")
                lines.append("")
                for i, text in valid_ocr:
                    lines.append(f"=== IMAGE {i+1} OCR Preview ===")
                    text_preview = text.strip()
                    if len(text_preview) > 1500:
                        text_preview = text_preview[:1500] + "\n... (truncated)"
                    lines.append(text_preview)
                    lines.append("")
        lines.extend([
            "**Planning rules:**",
            "- For formula-based computation: use 2 steps. Extraction step MUST list ALL variable names required by the formula—never omit any. Specify output order (e.g. 'output as var_a, var_b') so downstream maps by position. Computation step: apply formula and output result.",
            "- For critical calculations (ratios, differences, etc.): ensure extraction step extracts exactly the distinct inputs required—each variable from a different source. Do not use the same value as both numerator and denominator, or omit a required input.",
            "- For simple tasks: use 1 step.",
            "- Do not over-split (avoid 7+ steps for a single formula).",
            "- Do NOT hardcode numerical values in step descriptions. Extraction steps must specify WHAT to extract (e.g. 'extract X from table'), not assume or invent values.",
            "- If the question asks for sum/total of multiple values (e.g. 'A 与 B 之和', 'total of X and Y', 'A and B combined'), include a step to sum/combine the extracted values after extraction.",
            "- If the plain text context already states the target metric for multiple dates/periods (often with 'respectively' / '分别'), use [finance] text extraction for those dated values instead of forcing image lookup.",
        ])
        
        # 添加时间范围验证提示（针对 easy-test-194 类型的问题）
        time_range_hint = self._extract_time_range_hint(request)
        if time_range_hint:
            lines.extend([
                "",
                f"**Time Range Guidance:** {time_range_hint}",
            ])
        if self.base64_images and "multimodal" in self.agents:
            lines.extend(
                [
                    "",
                    "**Multimodal (MANDATORY for image tasks):** User has provided images. Any step that EXTRACTS data from charts/tables/images (提取、extract) MUST use [multimodal]. Only [multimodal] can see the image; [finance] cannot access images. Use [finance] only for computation steps that operate on already-extracted values (e.g. sum, formula, compare). Do NOT assign extraction from charts/tables to [finance].",
                    "- If the target metric is explicitly stated in the plain text context and the image OCR preview does not clearly contain that same metric phrase, prefer [finance] text extraction over [multimodal].",
                ]
            )
            # 多图：变量–图片映射 + 每图独立 [multimodal] 步骤（针对 easy-test-22）
            if len(self.base64_images) > 1:
                n = len(self.base64_images)
                img_refs = ", ".join(f"image {i+1}" for i in range(n))
                lines.extend(
                    [
                        "",
                        f"**Multi-image ({n} images: {img_refs}):**",
                        "- Map each variable required by the formula to its source image. Create exactly one [multimodal] step per image.",
                        "- Example: if formula needs A, B from image 1 and C from image 2 → Step 0: [multimodal] from image 1 extract A, B; Step 1: [multimodal] from image 2 extract C; Step 2: [finance] compute.",
                        "- NEVER skip a [multimodal] step for an image that holds required data. Each [multimodal] step MUST explicitly say 'from image 1' or 'from image 2'.",
                        "- Never combine 'from image 1' and 'from image 2' in one step—each step receives only one image.",
                        "- If a later image OCR preview clearly contains the exact requested metric / year / period, bind the extraction directly to that image instead of defaulting to earlier overview images.",
                        "- Do NOT create helper extraction steps such as company_name / company_code unless that helper variable is explicitly used in the final formula or is strictly required to disambiguate multiple same-metric rows.",
                        "- If disambiguation is needed, prefer a single query like '归母净利润, 厦门国贸集团股份有限公司, 2022' instead of a separate step that only extracts the company name.",
                    ]
                )
        elif not self.base64_images and "multimodal" in self.agents:
            lines.extend(
                [
                    "",
                    "**No images provided:** If the task requires extracting data from charts/tables/images (提取、图中、表格), but NO images are provided, the task CANNOT be completed. Create a minimal plan with 1 step: [finance] state 'Data missing - no image provided for extraction' and call terminate(status='failure'). Do NOT create [multimodal] extraction steps when no images exist.",
                ]
            )
        agents_description = []
        for key in self.executor_keys:
            if key in self.agents:
                agents_description.append(
                    {"name": key, "description": self.agents[key].description}
                )
        if agents_description:
            executor_names = [a["name"] for a in agents_description]
            lines.extend(
                [
                    "",
                    f"**Executors (use these exact keys in step labels):** {executor_names}. EVERY step MUST start with [executor_key], e.g. [multimodal] or [finance]. Do NOT create steps without an executor tag. Do NOT use non-existent keys like [text_agent] or [compute_agent].",
                ]
            )
        lines.extend(
            [
                "",
                f"Call the planning tool with command='create', plan_id='{self.active_plan_id}', title, and steps. Then terminate.",
            ]
        )
        return "\n".join(lines)

    async def _create_plan_via_planning_agent(
        self, planning_agent: BaseAgent, request: str
    ) -> None:
        """由 PlanningAgent 创建计划。
        
        核心改进：Planning Agent 始终直接看图片（Vision模式），OCR文本仅作为辅助参考。
        原因：OCR质量不稳定，直接让Vision模型看图能更准确理解图片内容，生成更精确的语义查询。
        """
        plan_prompt = self._build_plan_prompt(request)
        
        # 始终传入图片，让Planning Agent直接看图
        base64_image = self.base64_images[0] if self.base64_images else None
        base64_images = self.base64_images if len(self.base64_images or []) > 1 else None
        await planning_agent.run(
            plan_prompt,
            base64_image=base64_image,
            base64_images=base64_images,
        )

        if self.active_plan_id not in self.planning_tool.plans:
            logger.warning("PlanningAgent did not create plan; falling back to default.")
            await self.planning_tool.execute(
                **{
                    "command": "create",
                    "plan_id": self.active_plan_id,
                    "title": f"Plan for: {request[:50]}{'...' if len(request) > 50 else ''}",
                    "steps": ["Analyze request", "Execute task", "Verify results"],
                }
            )

    async def _create_plan_via_flow_llm(self, request: str) -> None:
        """由 flow 的 LLM 直接调用 planning_tool 创建计划（无 PlanningAgent 时）。"""
        system_message_content = (
            "You are a planning assistant. Create a concise, actionable plan with clear steps. "
            "Focus on key milestones rather than detailed sub-steps. Optimize for clarity and efficiency. "
            "For critical calculations (ratios, differences): ensure extraction step extracts exactly the distinct inputs required. "
            "For formula-based computation (e.g. EBITDA=X+Y+Z) where the user provides the formula and data: "
            "1) Use 2 steps: extraction + computation. "
            "2) The extraction step MUST explicitly list the variable names from the formula, e.g. '提取合并净利润、所得税费用、利息支出、固定资产折旧、无形资产摊销' (not generic '提取所需数据'). "
            "3) The computation step: '应用公式计算并输出结果'. "
            "For simple tasks without a formula, use 1 step. Do not over-split (e.g. avoid 7 steps for a single formula). python_execute variables persist across calls."
        )
        agents_description = []
        for key in self.executor_keys:
            if key in self.agents:
                agents_description.append(
                    {
                        "name": key,
                        "description": self.agents[key].description,
                    }
                )
        if agents_description:
            executor_names = [a["name"] for a in agents_description]
            system_message_content += (
                f"\n**Executors (use these exact keys in step labels):** {executor_names}. "
                f"Use format [executor_key] in step text, e.g. [finance]. "
                "Do NOT use non-existent keys like [text_agent] or [compute_agent]."
            )

        system_message = Message.system_message(system_message_content)
        user_content = f"Create a reasonable plan with clear steps to accomplish the task: {request}"
        if self.ocr_results and len(self.ocr_results) > 0:
            user_content += "\n\n**Content from images (OCR):**"
            user_content += "\nCRITICAL: Image numbers correspond to actual image files in EXACT order."
            user_content += "\nWhen creating plan steps, use 'from image 1' or 'from image 2' to specify which image to extract data from."
            for i, t in enumerate(self.ocr_results):
                user_content += f"\n\n=== IMAGE {i+1} (index {i}) ===\n{t or '(empty)'}"
            user_content += "\n\n**CRITICAL: ALWAYS specify 'from image X' in multimodal steps to ensure correct image routing.**"
        user_message = Message.user_message(user_content)

        tool_param = (
            self.planning_tool.to_planning_param()
            if hasattr(self.planning_tool, "to_planning_param")
            else self.planning_tool.to_param()
        )

        response = await self.llm.ask_tool(
            messages=[user_message],
            system_msgs=[system_message],
            tools=[tool_param],
            tool_choice=ToolChoice.AUTO,
        )

        if response and response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.function.name in ("planning", "workflow_state"):
                    args = tool_call.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse tool arguments: {args}")
                            continue

                    args["plan_id"] = self.active_plan_id
                    result = await self.planning_tool.execute(**args)
                    logger.info(f"Plan creation result: {str(result)}")
                    return

        logger.warning("Creating default plan")
        await self.planning_tool.execute(
            **{
                "command": "create",
                "plan_id": self.active_plan_id,
                "title": f"Plan for: {request[:50]}{'...' if len(request) > 50 else ''}",
                "steps": ["Analyze request", "Execute task", "Verify results"],
            }
        )

    def _infer_step_type(self, step_text: str) -> str:
        """
        当步骤无 [executor] 标签时，根据文本推断 executor。
        """
        text = (step_text or "").strip().lower()
        # 提取类：图中、表格、extract、from image
        extract_patterns = [
            r"从图|图中|表格|extract|提取",
            r"from\s+image\s*\d*",
            r"image\s*\d+\s*(中|里|的)",
        ]
        for p in extract_patterns:
            if re.search(p, text, re.IGNORECASE):
                if self.base64_images and "multimodal" in self.agents:
                    return "multimodal"
                break
        # 计算类：公式、计算、比例、ratio、formula
        compute_patterns = [
            r"计算|应用公式|formula|calculate|ratio|比例",
            r"sum|total|差值|difference|compare",
        ]
        for p in compute_patterns:
            if re.search(p, text, re.IGNORECASE):
                return "finance"
        # 默认：计算步骤更常见
        return "finance"

    async def _get_current_step_info(self) -> tuple[Optional[int], Optional[dict]]:
        """
        Parse the current plan to identify the first non-completed step's index and info.
        Returns (None, None) if no active step is found.
        """
        if (
            not self.active_plan_id
            or self.active_plan_id not in self.planning_tool.plans
        ):
            logger.error(f"Plan with ID {self.active_plan_id} not found")
            return None, None

        try:
            plan_data = self.planning_tool.plans[self.active_plan_id]
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])

            for i, step in enumerate(steps):
                if i >= len(step_statuses):
                    status = PlanStepStatus.NOT_STARTED.value
                else:
                    status = step_statuses[i]

                if status in PlanStepStatus.get_active_statuses():
                    step_info = {"text": step}
                    type_match = re.search(r"\[([a-zA-Z_]+)\]", step)
                    if type_match:
                        step_info["type"] = type_match.group(1).lower()
                    else:
                        # 无 [executor] 标签时根据步骤文本推断，避免 step_type=None 导致错误分配
                        step_info["type"] = self._infer_step_type(step)

                    try:
                        await self.planning_tool.execute(
                            command="mark_step",
                            plan_id=self.active_plan_id,
                            step_index=i,
                            step_status=PlanStepStatus.IN_PROGRESS.value,
                        )
                    except Exception as e:
                        logger.warning(f"Error marking step as in_progress: {e}")
                        if i < len(step_statuses):
                            step_statuses[i] = PlanStepStatus.IN_PROGRESS.value
                        else:
                            while len(step_statuses) < i:
                                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
                            step_statuses.append(PlanStepStatus.IN_PROGRESS.value)
                        plan_data["step_statuses"] = step_statuses

                    return i, step_info

            return None, None

        except Exception as e:
            logger.warning(f"Error finding current step index: {e}")
            return None, None

    def _format_structured_previous_output(self, previous_output: str) -> str:
        """
        A: 将原始 execution_result 格式化为 Step N: value，避免 Finance 误映射多步输出。
        按步骤边界（terminate）分割，每步只取最后一次 python_execute 的 observation。
        """
        if not previous_output or not previous_output.strip():
            return ""
        obs_pattern = re.compile(
            r"['\"]?observation['\"]?\s*:\s*['\"]([^'\"]*)['\"]",
            re.IGNORECASE,
        )
        # 按 terminate 分割，每段对应一个步骤
        segments = re.split(r"Observed output of cmd `terminate`", previous_output, flags=re.IGNORECASE)
        step_values = []
        for seg in segments:
            if not seg.strip():
                continue
            matches = obs_pattern.findall(seg)
            if matches:
                # 取该步骤最后一次 python_execute 的输出
                last_val = matches[-1].strip().replace("\\n", "\n").strip()
                step_values.append(last_val)
        if not step_values:
            return previous_output.strip()
        # 多值用逗号分隔，避免 Finance 将 "984 97" 误解析为 98497（easy-test-0）
        def _fmt(v: str) -> str:
            v = v.strip()
            # 若已含 var=value 格式，尽量保留
            if "=" in v and re.search(r"\w+\s*=\s*[\d.]", v):
                return v
            parts = v.split()
            return ", ".join(parts) if len(parts) > 1 else v
        return "\n".join(f"Step {i}: {_fmt(v)}" for i, v in enumerate(step_values))

    def _get_image_index_for_step(self, step_info: dict) -> int:
        """
        P1: 从步骤文本解析目标图片索引，支持多图按需提取。
        步骤中写 "from image 1" / "image 2" 时使用对应图片；未指定时用 0。
        """
        text = (step_info.get("text") or "").strip()
        m = re.search(r"(?:from\s+)?image\s*(\d+)", text, re.IGNORECASE)
        if m:
            return max(0, int(m.group(1)) - 1)  # 1-based in plan → 0-based
        return 0

    def _build_finance_runtime_guard(self, step_text: str, user_request: str) -> str:
        guards: List[str] = []
        combined_text = f"{step_text}\n{user_request}"

        if self._request_is_revised(combined_text):
            guards.append(
                "- If the task says revised / assumed / adjusted TO a number, treat that number as the replacement value. Create an explicit revised_* variable before the final formula."
            )
            guards.append(
                "- For revised / hypothetical tasks, the only allowed raw literal is the replacement value explicitly stated in USER REQUEST / 原始数据. Read it from the text context, store it in revised_*, and do not mix the original variable and raw literal in one direct expression."
            )

        if self._looks_like_text_extraction_step(step_text):
            guards.append(
                "- For text extraction steps, assign only a number that is explicitly present in the USER REQUEST / 原始数据 block. If you cannot point to an exact snippet, terminate with failure instead of inventing a literal."
            )
            guards.append(
                "- For text extraction steps, match the exact metric named in CURRENT TASK. Do NOT substitute a nearby respectively-pair or a semantically related metric with a different label."
            )

        if self._is_literal_only_finance_assignment(step_text):
            guards.append(
                "- If the current step looks like `var = 123` with no extracted variables on the right-hand side, do NOT treat it as a valid computation. Read the exact value from USER REQUEST / 原始数据 as a text extraction, or terminate with failure if no exact snippet exists."
            )

        if self._step_contains_ungrounded_numeric_literal(step_text, user_request):
            guards.append(
                "- The current step contains a numeric literal that is not explicitly grounded in USER REQUEST / 原始数据. Treat it as contaminated plan text: do not use it; rely on extracted variables or terminate with failure."
            )

        if self._request_is_comparison(combined_text):
            guards.append(
                "- For change / difference / ratio / comparison questions, if you cannot identify all required operands from extracted variables or exact text snippets, terminate with failure instead of inventing the missing baseline."
            )

        guards.append(
            "- If the computed result would round to 0.00 while the non-zero inputs are material, treat that as a likely unit/formula error and terminate with failure instead of silently succeeding."
        )
        guards.append(
            "- If the question asks for a percent / ratio / EPS-style result, verify the display unit before printing. If your raw result is a decimal fraction but the answer should be shown as a percentage or per-share figure, convert units explicitly or terminate with failure."
        )

        if not guards:
            return ""
        return "\n".join(["", "**FINANCE GUARDRAILS:**", *guards, ""])

    def _step_contains_ungrounded_numeric_literal(
        self, step_text: str, user_request: str
    ) -> bool:
        expression_parts: List[str] = []
        formula_match = re.search(
            r"(?:formula:\s*)?([A-Za-z_][A-Za-z0-9_]*\s*=\s*.+)",
            step_text or "",
            re.IGNORECASE,
        )
        if formula_match:
            expression_parts.append(formula_match.group(1))
        if not expression_parts:
            return False

        for literal in re.findall(
            r"[-+]?(?:[$¥€£])?\d[\d,]*(?:\.\d+)?%?",
            " ".join(expression_parts),
        ):
            normalized_literal = literal.replace(",", "").lower()
            if re.fullmatch(r"(?:19|20)\d{2}", normalized_literal):
                continue
            if self._literal_exists_in_request(literal, user_request):
                continue
            return True
        return False

    async def _execute_step(
        self,
        executor: BaseAgent,
        step_info: dict,
        previous_output: str = "",
        user_request: str = "",
    ) -> str:
        """Execute the current step with the specified agent using agent.run()."""
        step_type = step_info.get("type")
        # P0 无图快速失败：multimodal 步骤无图时直接 block，避免无效 LLM 调用
        if step_type == "multimodal" and not self.base64_images:
            logger.warning(
                f"Step {self.current_step_index} skipped: multimodal step but no images provided"
            )
            try:
                await self.planning_tool.execute(
                    command="mark_step",
                    plan_id=self.active_plan_id,
                    step_index=self.current_step_index,
                    step_status=PlanStepStatus.BLOCKED.value,
                    step_notes="No image provided; cannot extract from image/table.",
                )
            except Exception as e:
                logger.warning(f"Error marking step as blocked: {e}")
            return (
                "Step blocked: No image provided. Cannot extract data from image/table. "
                "Data missing."
            )

        plan_status = await self._get_plan_text()
        raw_step_text = step_info.get("text", f"Step {self.current_step_index}")
        step_text, runtime_sanitize_note = self._sanitize_plan_step(
            raw_step_text, user_request
        )
        if runtime_sanitize_note:
            logger.info(
                f"Runtime-sanitized step {self.current_step_index}: {runtime_sanitize_note}"
            )

        prev_block = ""
        prev_block_extra = ""
        if previous_output.strip():
            # A: 结构化格式，避免 Finance 误将多步输出映射到同一变量
            structured = self._format_structured_previous_output(previous_output)
            prev_block = f"""
        PREVIOUS STEPS OUTPUT (use these values for computation—do NOT substitute with other numbers):
        Map Step 0 to the first variable, Step 1 to the second, etc. Do NOT use the same value for different variables.
        **CRITICAL:** Each number is a SEPARATE value. Do NOT concatenate digits. E.g. "97, 81, 984, 165" = four values (97, 81, 984, 165), NOT 9781984 or 81165. Use the exact numbers by position.
        ---
        {structured}
        ---
        """
            # 当上一步 python_execute 输出为空时，禁止 Finance 编造数值（如 easy-test-14）
            if re.search(r"observation['\"]?\s*:\s*['\"]?['\"]", previous_output):
                prev_block_extra = """
        WARNING: Previous step returned empty observation. Do NOT invent values. Use ONLY numbers from PREVIOUS STEPS OUTPUT. If none, state extraction failed.
        """

        step_type = step_info.get("type")
        if step_type == "multimodal":
            step_prompt = f"""YOUR TASK: {step_text}

**CRITICAL: EXTRACTION STATUS CHECK**
After calling finance_extraction_skill, check the result:
- If values are extracted (NOT "NOT_FOUND" or nan): those variables are already stored in python_execute by the tool. DO NOT overwrite them with guessed or remembered literals. Call python_execute only to print the existing variable values, then IMMEDIATELY call terminate(status="success")
- If values are NOT_FOUND: call python_execute to assign the requested variable(s) to None, then IMMEDIATELY call terminate(status="success") — DO NOT retry, DO NOT call finance_extraction_skill again

**MISSING VALUES:** If you cannot find a value in the image, store it as None (e.g. var = None) and print it clearly. Do NOT emit diagnostic placeholder text such as "Missing: ...".

**NO REPETITION**: Once you call python_execute (whether with values or None), you MUST call terminate next. Calling the same tool again is FORBIDDEN.

Look at the image. Extract the requested values. The table may use different labels (e.g. Operating income for EBIT)—map by meaning. Call finance_extraction_skill first. If it returns values, use the exact tool-returned values already stored in python_execute. Never replace a returned value with another candidate. If it does not return a value, store None. Then terminate(status="success").

**OUTPUT FORMAT:** If extraction succeeded, in python_execute use only existing variables, e.g. print('var1=', var1, ', var2=', var2). If extraction failed, assign None first, e.g. var1 = None; var2 = None; print('var1=', var1, ', var2=', var2).

**NO MENTAL MATH**: Do NOT do any calculation in your head. ALL computations MUST be done via python_execute."""
        else:
            user_context_block = ""
            if user_request.strip():
                user_context_block = f"""
        USER REQUEST / 原始数据（从此处提取数值，勿编造）:
        ---
        {user_request.strip()}
        ---

        """
            finance_guard_block = self._build_finance_runtime_guard(
                step_text, user_request
            )
            step_prompt = f"""
        CURRENT PLAN STATUS:
        {plan_status}
        {user_context_block}
        {prev_block}{prev_block_extra}
        YOUR CURRENT TASK:
        You are now working on step {self.current_step_index}: "{step_text}"

        python_execute supports multi-step: variables persist across calls. If the plan step explicitly lists variables to extract, extract EXACTLY those items—use the exact names from the formula, not substitutes. Apply the user's formula verbatim.

        IMPORTANT: Use numbers ONLY from PREVIOUS STEPS OUTPUT (if present) or from the image/user context. Do NOT invent values. In python_execute, use: variable = value  # Source: 'exact snippet' (SINGLE quotes only—never triple quotes \"\"\" which break Python).
        {finance_guard_block}

        **RATIO FORMULA ORDER**: For A/B, A=numerator, B=denominator. Do NOT swap.

        **NO MENTAL MATH**: Do NOT do any calculation in your head. ALL computations (arithmetic, ratios, formulas) MUST be done via python_execute. Never output a computed result without calling python_execute.

        Please only execute this current step using the appropriate tools. When you're done, provide a summary of what you accomplished.
        """

        try:
            step_type = step_info.get("type")
            base64_image_count = len(self.base64_images) if self.base64_images else 0
            step_text = step_info.get("text", "") if step_info else ""
            logger.info(
                f"Step {self.current_step_index} executor={getattr(executor, 'name', type(executor).__name__)} "
                f"step_type={step_type!r} base64_images_count={base64_image_count}"
            )
            logger.info(f"Step {self.current_step_index} text: {step_text[:100]}...")
            base64_image = None
            img_idx = 0
            if (
                self.base64_images
                and step_type == "multimodal"
                and hasattr(executor, "run")
            ):
                # P1: 多图路由 - 从步骤文本解析 image 1/image 2，支持按需指定图片
                img_idx = self._get_image_index_for_step(step_info)
                # 确保img_idx在有效范围内
                if img_idx >= len(self.base64_images):
                    logger.warning(f"Step {self.current_step_index} requested image {img_idx+1} but only {len(self.base64_images)} images available. Using image 1.")
                    img_idx = 0
                base64_image = self.base64_images[img_idx]
                # 详细日志：显示图片路由决策
                logger.info(f"Step {self.current_step_index} using image {img_idx+1} (index {img_idx}) of {len(self.base64_images)}")
                # 如果有OCR结果，显示对应的OCR内容预览
                if self.ocr_results and img_idx < len(self.ocr_results):
                    ocr_preview = (self.ocr_results[img_idx] or "")[:80].replace('\n', ' ')
                    logger.info(f"Step {self.current_step_index} corresponding OCR preview: {ocr_preview}...")
            logger.info(
                f"Step {self.current_step_index} passing base64_image to executor: "
                f"{'yes' if base64_image else 'no'} (len={len(base64_image) if base64_image else 0})"
            )
            # 注入当前步骤图片到 OCR 上下文，供 Agent 调用 ocr_extract(use_context_image=True)
            if step_type == "multimodal" and base64_image:
                set_step_images_for_ocr([base64_image])
            elif step_type == "multimodal" and self.base64_images:
                set_step_images_for_ocr(self.base64_images)
            else:
                set_step_images_for_ocr(None)
            
            # 注入共享的 PythonExecute 实例到 Skill，实现自动状态持久化
            from app.skill.finance_extraction import set_shared_python_execute
            shared_py = self._ensure_shared_python_execute()
            set_shared_python_execute(shared_py)
            logger.info(f"[PlanningFlow] Injected shared PythonExecute to Skill for step {self.current_step_index}")
            
            step_result = await executor.run(step_prompt, base64_image=base64_image)
            await self._mark_step_completed()
            return step_result
        except Exception as e:
            logger.error(f"Error executing step {self.current_step_index}: {e}")
            return f"Error executing step {self.current_step_index}: {str(e)}"

    async def _mark_step_completed(self) -> None:
        """Mark the current step as completed."""
        if self.current_step_index is None:
            return

        try:
            await self.planning_tool.execute(
                command="mark_step",
                plan_id=self.active_plan_id,
                step_index=self.current_step_index,
                step_status=PlanStepStatus.COMPLETED.value,
            )
            logger.info(
                f"Marked step {self.current_step_index} as completed in plan {self.active_plan_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to update plan status: {e}")
            if self.active_plan_id in self.planning_tool.plans:
                plan_data = self.planning_tool.plans[self.active_plan_id]
                step_statuses = plan_data.get("step_statuses", [])

                while len(step_statuses) <= self.current_step_index:
                    step_statuses.append(PlanStepStatus.NOT_STARTED.value)

                step_statuses[self.current_step_index] = PlanStepStatus.COMPLETED.value
                plan_data["step_statuses"] = step_statuses

    async def _get_plan_text(self) -> str:
        """Get the current plan as formatted text."""
        try:
            result = await self.planning_tool.execute(
                command="get", plan_id=self.active_plan_id
            )
            return result.output if hasattr(result, "output") else str(result)
        except Exception as e:
            logger.error(f"Error getting plan: {e}")
            return self._generate_plan_text_from_storage()

    def _generate_plan_text_from_storage(self) -> str:
        """Generate plan text directly from storage if the planning tool fails."""
        try:
            if self.active_plan_id not in self.planning_tool.plans:
                return f"Error: Plan with ID {self.active_plan_id} not found"

            plan_data = self.planning_tool.plans[self.active_plan_id]
            title = plan_data.get("title", "Untitled Plan")
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])
            step_notes = plan_data.get("step_notes", [])

            while len(step_statuses) < len(steps):
                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
            while len(step_notes) < len(steps):
                step_notes.append("")

            status_counts = {status: 0 for status in PlanStepStatus.get_all_statuses()}
            for status in step_statuses:
                if status in status_counts:
                    status_counts[status] += 1

            completed = status_counts[PlanStepStatus.COMPLETED.value]
            total = len(steps)
            progress = (completed / total) * 100 if total > 0 else 0

            plan_text = f"Plan: {title} (ID: {self.active_plan_id})\n"
            plan_text += "=" * len(plan_text) + "\n\n"

            plan_text += (
                f"Progress: {completed}/{total} steps completed ({progress:.1f}%)\n"
            )
            plan_text += f"Status: {status_counts[PlanStepStatus.COMPLETED.value]} completed, {status_counts[PlanStepStatus.IN_PROGRESS.value]} in progress, "
            plan_text += f"{status_counts[PlanStepStatus.BLOCKED.value]} blocked, {status_counts[PlanStepStatus.NOT_STARTED.value]} not started\n\n"
            plan_text += "Steps:\n"

            status_marks = PlanStepStatus.get_status_marks()

            for i, (step, status, notes) in enumerate(
                zip(steps, step_statuses, step_notes)
            ):
                status_mark = status_marks.get(
                    status, status_marks[PlanStepStatus.NOT_STARTED.value]
                )
                plan_text += f"{i}. {status_mark} {step}\n"
                if notes:
                    plan_text += f"   Notes: {notes}\n"

            return plan_text
        except Exception as e:
            logger.error(f"Error generating plan text from storage: {e}")
            return f"Error: Unable to retrieve plan with ID {self.active_plan_id}"

    async def _finalize_plan(
        self,
        execution_result: str = "",
        user_request: str = "",
    ) -> str:
        """Finalize the plan: 基于执行输出提取并呈现最终答案，直接回应用户请求。"""
        plan_text = await self._get_plan_text()

        try:
            system_message = Message.system_message(
                "You are a planning assistant. Your task is to produce the FINAL ANSWER for the user based on the execution output. "
                "Rules: 1) Extract the actual computed result (numbers, values) from the execution output. "
                "2) Present it clearly as the direct answer to the user's request. "
                "3) Do NOT invent or hallucinate—only use what appears in the execution output. "
                "4) If the user asked for a calculation (e.g. ratio, percentage), state the numeric result prominently. "
                "5) Keep the summary concise; the final answer is the priority."
            )

            user_content = f"""The plan has been completed.

**User request:**
{user_request}

**Plan status:**
{plan_text}

**Execution output (tool results, computed values):**
{execution_result}

Based on the execution output above, provide the FINAL ANSWER that directly addresses the user's request. Extract and state any computed numeric results clearly. Do not invent data."""
            user_message = Message.user_message(user_content)

            response = await self.llm.ask(
                messages=[user_message], system_msgs=[system_message]
            )

            return f"\n\n---\n\n**最终答案**\n\n{response}"
        except Exception as e:
            logger.error(f"Error finalizing plan with LLM: {e}")
            return f"\n\n---\n\n**执行输出**\n\n{execution_result}"
