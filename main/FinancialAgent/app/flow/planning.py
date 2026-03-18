import json
import re
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from pydantic import Field

from app.agent.base import BaseAgent
from app.agent.planning import PlanningAgent
from app.flow.base import BaseFlow
from app.llm import LLM
from app.logger import logger
from app.prompt.planning import SYSTEM_PROMPT
from app.schema import AgentState, Message
from app.tool import WorkflowStateTool


class PlanStepStatus(str, Enum):
    """Enum class defining possible statuses of a plan step"""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"

    @classmethod
    def get_all_statuses(cls) -> list[str]:
        """Return a list of all possible step status values"""
        return [status.value for status in cls]

    @classmethod
    def get_active_statuses(cls) -> list[str]:
        """Return a list of values representing active statuses (not started or in progress)"""
        return [cls.NOT_STARTED.value, cls.IN_PROGRESS.value]

    @classmethod
    def get_status_marks(cls) -> Dict[str, str]:
        """Return a mapping of statuses to their marker symbols"""
        return {
            cls.COMPLETED.value: "[✓]",
            cls.IN_PROGRESS.value: "[→]",
            cls.BLOCKED.value: "[!]",
            cls.NOT_STARTED.value: "[ ]",
        }


class PlanningFlow(BaseFlow):
    """A flow that manages planning and execution of tasks using agents."""

    llm: LLM = Field(default_factory=lambda: LLM())
    workflow_state_tool: WorkflowStateTool = Field(default_factory=WorkflowStateTool)
    executor_keys: List[str] = Field(default_factory=list)
    active_plan_id: str = Field(default_factory=lambda: f"plan_{int(time.time())}")
    current_step_index: Optional[int] = None
    original_request: str = ""
    step_outputs: Dict[int, str] = Field(default_factory=dict)
    max_agent_steps_per_plan_step: int = 6
    step_output_history_window: int = 4
    allowed_workflow_state_commands: List[str] = Field(
        default_factory=lambda: ["create", "get", "mark_step"]
    )

    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], **data
    ):
        # Set executor keys before super().__init__
        if "executors" in data:
            data["executor_keys"] = data.pop("executors")

        # Set plan ID if provided
        if "plan_id" in data:
            data["active_plan_id"] = data.pop("plan_id")

        # Backward-compatible injection key: planning_tool
        if "planning_tool" in data and "workflow_state_tool" not in data:
            data["workflow_state_tool"] = data.pop("planning_tool")

        # Initialize workflow-state tool if not provided
        if "workflow_state_tool" not in data:
            data["workflow_state_tool"] = WorkflowStateTool()

        # Call parent's init with the processed data
        super().__init__(agents, **data)

        # Set executor_keys to all agent keys if not specified
        if not self.executor_keys:
            self.executor_keys = list(self.agents.keys())

    def get_executor(self, step_type: Optional[str] = None) -> BaseAgent:
        """
        Get an appropriate executor agent for the current step.
        Can be extended to select agents based on step type/requirements.
        """
        # If step type is provided and matches an agent key, use that agent
        if step_type and step_type in self.agents:
            return self.agents[step_type]

        # Otherwise use the first available executor or fall back to primary agent
        for key in self.executor_keys:
            if key in self.agents:
                return self.agents[key]

        # Fallback to primary agent
        return self.primary_agent

    def _get_planner_agent(self) -> Optional[PlanningAgent]:
        """Return planning agent if it exists in current flow."""
        planner = self.agents.get("planning")
        if isinstance(planner, PlanningAgent):
            return planner
        return None

    @staticmethod
    def _extract_json_payload(text: str) -> Optional[dict]:
        """Extract JSON object from plain text or fenced code block."""
        if not text:
            return None

        candidates: List[str] = []

        fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced_match:
            candidates.append(fenced_match.group(1))

        direct_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if direct_match:
            candidates.append(direct_match.group(1))

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                continue

        return None

    @staticmethod
    def _normalize_plan_steps(
        steps: List[str], allowed_executors: List[str]
    ) -> List[str]:
        """
        Normalize step list:
        - trim blanks
        - deduplicate exact duplicates while preserving order
        - ensure executor hints use allowed keys when present
        """
        seen = set()
        normalized: List[str] = []

        for raw_step in steps:
            if not isinstance(raw_step, str):
                continue

            step = raw_step.strip()
            if not step:
                continue

            # If executor hint exists but is invalid, remove the hint.
            type_match = re.match(r"^\[([a-zA-Z_]+)\]\s*(.*)$", step)
            if type_match:
                hint = type_match.group(1).lower()
                rest = type_match.group(2).strip()
                if hint in allowed_executors and rest:
                    step = f"[{hint}] {rest}"
                else:
                    step = rest

            if step not in seen:
                seen.add(step)
                normalized.append(step)

        return normalized

    @staticmethod
    def _bound_step_count(steps: List[str], min_steps: int = 1, max_steps: int = 6) -> List[str]:
        """Bound step count to avoid over-fragmentation for simple tasks."""
        if not steps:
            return []
        bounded = steps[:max_steps]
        if len(bounded) < min_steps:
            return bounded + ["Execute task"]
        return bounded

    async def _draft_plan_with_planner(
        self, planner: PlanningAgent, request: str, available_executors: List[str]
    ) -> Optional[Tuple[str, List[str]]]:
        """
        Phase A (plan synthesis): planner drafts a plan in JSON without calling tools.
        """
        system_message = Message.system_message(
            planner.system_prompt
            + "\nYou are in plan-synthesis phase."
            "\nDo NOT call tools."
            "\nOutput ONLY JSON with keys: title (string), steps (string array)."
        )
        user_message = Message.user_message(
            (
                "Create a concise, non-overlapping execution plan.\n"
                "Rules:\n"
                "1) Prefer minimal sufficient decomposition.\n"
                "2) For simple tasks, keep steps between 1 and 3.\n"
                "3) Add executor hints only when useful: [agent_key] step text.\n"
                f"4) Allowed executor keys: {available_executors}.\n"
                "5) Do NOT include markdown fences unless needed.\n\n"
                f"Task:\n{request}"
            )
        )

        draft = await planner.llm.ask(
            messages=[user_message], system_msgs=[system_message], stream=False
        )
        if not isinstance(draft, str):
            return None

        payload = self._extract_json_payload(draft)
        if not payload:
            return None

        title = str(payload.get("title", "")).strip()
        raw_steps = payload.get("steps", [])
        if not isinstance(raw_steps, list):
            return None

        steps = self._normalize_plan_steps(raw_steps, available_executors)
        steps = self._bound_step_count(steps, min_steps=1, max_steps=6)

        if not title:
            title = f"Plan for: {request[:50]}{'...' if len(request) > 50 else ''}"
        if not steps:
            return None

        return title, steps

    async def _commit_workflow_state(self, **kwargs):
        """
        Single commit gateway for workflow state updates in PlanningFlow.
        """
        command = kwargs.get("command")
        if command not in self.allowed_workflow_state_commands:
            raise ValueError(
                f"Unsupported workflow_state command in flow commit gateway: {command}"
            )
        return await self.workflow_state_tool.execute(**kwargs)

    async def _draft_plan_with_flow_llm(
        self, request: str, available_executors: List[str]
    ) -> Optional[Tuple[str, List[str]]]:
        """
        Fallback plan synthesis using flow LLM only (no tool-calls in Phase A).
        """
        system_message = Message.system_message(
            SYSTEM_PROMPT.format(directory="planning-flow")
            + "\nYou are in plan-synthesis phase."
            "\nDo NOT call tools."
            "\nOutput ONLY JSON with keys: title (string), steps (string array)."
        )
        user_message = Message.user_message(
            (
                "Create a concise, non-overlapping execution plan.\n"
                "Rules:\n"
                "1) Prefer minimal sufficient decomposition.\n"
                "2) For simple tasks, keep steps between 1 and 3.\n"
                "3) Add executor hints only when useful: [agent_key] step text.\n"
                f"4) Allowed executor keys: {available_executors}.\n"
                "5) Do NOT include markdown fences unless needed.\n\n"
                f"Task:\n{request}"
            )
        )

        draft = await self.llm.ask(
            messages=[user_message], system_msgs=[system_message], stream=False
        )
        if not isinstance(draft, str):
            return None

        payload = self._extract_json_payload(draft)
        if not payload:
            return None

        title = str(payload.get("title", "")).strip()
        raw_steps = payload.get("steps", [])
        if not isinstance(raw_steps, list):
            return None

        steps = self._normalize_plan_steps(raw_steps, allowed_executors=available_executors)
        steps = self._bound_step_count(steps, min_steps=1, max_steps=6)

        if not title:
            title = f"Plan for: {request[:50]}{'...' if len(request) > 50 else ''}"
        if not steps:
            return None

        return title, steps

    def _reset_plan_progress_if_needed(self) -> None:
        """
        Keep initial plan in NOT_STARTED state after planning phase.

        The planner should only create/update steps during `_create_initial_plan`,
        not execute business steps by marking them completed.
        """
        if self.active_plan_id not in self.workflow_state_tool.plans:
            return

        plan_data = self.workflow_state_tool.plans[self.active_plan_id]
        steps = plan_data.get("steps", [])
        step_statuses = plan_data.get("step_statuses", [])
        step_notes = plan_data.get("step_notes", [])

        if len(step_statuses) != len(steps) or any(
            status != PlanStepStatus.NOT_STARTED.value for status in step_statuses
        ):
            plan_data["step_statuses"] = [PlanStepStatus.NOT_STARTED.value] * len(steps)
            logger.warning(
                "Planner produced progressed step statuses during plan creation; "
                "resetting all steps to not_started."
            )

        if len(step_notes) < len(steps):
            plan_data["step_notes"] = step_notes + [""] * (len(steps) - len(step_notes))

    async def execute(self, input_text: str) -> str:
        """Execute the planning flow with agents."""
        try:
            if not self.primary_agent:
                raise ValueError("No primary agent available")

            self.original_request = input_text or ""
            self.step_outputs.clear()

            # Create initial plan if input provided
            if input_text:
                await self._create_initial_plan(input_text)

                # Verify plan was created successfully
                if self.active_plan_id not in self.workflow_state_tool.plans:
                    logger.error(
                        f"Plan creation failed. Plan ID {self.active_plan_id} not found in workflow_state tool."
                    )
                    return f"Failed to create plan for: {input_text}"

            result = ""
            while True:
                # Get current step to execute
                self.current_step_index, step_info = await self._get_current_step_info()

                # Exit if no more steps or plan completed
                if self.current_step_index is None:
                    break

                # Execute current step with appropriate agent
                step_type = step_info.get("type") if step_info else None
                executor = self.get_executor(step_type)
                step_result = await self._execute_step(executor, step_info)
                result += step_result + "\n"

            final_answer = await self._synthesize_user_final_answer(
                original_request=input_text,
                execution_log=result,
            )
            return final_answer or result
        except Exception as e:
            logger.error(f"Error in PlanningFlow: {str(e)}")
            return f"Execution failed: {str(e)}"

    async def _synthesize_user_final_answer(
        self, original_request: str, execution_log: str
    ) -> str:
        """Create a user-facing final answer after flow execution."""
        if not original_request:
            return ""
        try:
            system_message = Message.system_message(
                "You generate a final answer for the user after a multi-agent execution."
            )
            user_message = Message.user_message(
                (
                    "Based on the completed plan execution, provide the final response to the user.\n\n"
                    f"Original request:\n{original_request}\n\n"
                    f"Execution details:\n{execution_log}\n\n"
                    "Requirements:\n"
                    "1. Answer the user request directly first.\n"
                    "2. Keep it concise but complete.\n"
                    "3. Mention key evidence or caveats where needed.\n"
                )
            )
            response = await self.llm.ask(
                messages=[user_message], system_msgs=[system_message], stream=False
            )
            return response.strip() if isinstance(response, str) else ""
        except Exception as e:
            logger.warning(f"Failed to synthesize final flow answer: {e}")
            return ""

    async def _create_initial_plan(self, request: str) -> None:
        """Create an initial plan based on the request using flow-level synthesis and workflow_state persistence."""
        logger.info(f"Creating initial plan with ID: {self.active_plan_id}")

        # Phase A: plan synthesis (no workflow-state tool calls).
        # Prefer dedicated planning agent when available.
        planner = self._get_planner_agent()
        if planner:
            available_executors = [
                key for key in self.executor_keys if key in self.agents and key != "planning"
            ]
            try:
                draft = await self._draft_plan_with_planner(
                    planner=planner,
                    request=request,
                    available_executors=available_executors,
                )
                if draft:
                    title, steps = draft
                    await self._commit_workflow_state(
                        command="create",
                        plan_id=self.active_plan_id,
                        title=title,
                        steps=steps,
                    )
                    self._reset_plan_progress_if_needed()
                    return
                logger.warning("Planning agent draft failed, falling back to flow LLM draft.")
            except Exception as e:
                logger.warning(f"Planning agent failed, falling back to flow LLM: {e}")
        # Fallback Phase A synthesis using flow LLM (still no tool-calls by LLM).
        available_executors = [
            key for key in self.executor_keys if key in self.agents and key != "planning"
        ]
        flow_draft = await self._draft_plan_with_flow_llm(
            request=request, available_executors=available_executors
        )
        if flow_draft:
            title, steps = flow_draft
            await self._commit_workflow_state(
                command="create",
                plan_id=self.active_plan_id,
                title=title,
                steps=steps,
            )
            self._reset_plan_progress_if_needed()
            return

        # If execution reached here, create a default plan
        logger.warning("Creating default plan")

        # Create default plan using the ToolCollection
        await self._commit_workflow_state(
            **{
                "command": "create",
                "plan_id": self.active_plan_id,
                "title": f"Plan for: {request[:50]}{'...' if len(request) > 50 else ''}",
                "steps": ["Analyze request", "Execute task", "Verify results"],
            }
        )
        self._reset_plan_progress_if_needed()

    async def _get_current_step_info(self) -> tuple[Optional[int], Optional[dict]]:
        """
        Parse the current plan to identify the first non-completed step's index and info.
        Returns (None, None) if no active step is found.
        """
        if (
            not self.active_plan_id
            or self.active_plan_id not in self.workflow_state_tool.plans
        ):
            logger.error(f"Plan with ID {self.active_plan_id} not found")
            return None, None

        try:
            # Direct access to plan data from workflow-state storage
            plan_data = self.workflow_state_tool.plans[self.active_plan_id]
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])

            # Find first non-completed step
            for i, step in enumerate(steps):
                if i >= len(step_statuses):
                    status = PlanStepStatus.NOT_STARTED.value
                else:
                    status = step_statuses[i]

                if status in PlanStepStatus.get_active_statuses():
                    # Extract step type/category if available
                    step_info = {"text": step}

                    # Try to extract step type from the text (e.g., [search] or [CODE])
                    type_match = re.search(r"\[([a-zA-Z_]+)\]", step)
                    if type_match:
                        step_info["type"] = type_match.group(1).lower()

                    # Mark current step as in_progress
                    try:
                        await self._commit_workflow_state(
                            command="mark_step",
                            plan_id=self.active_plan_id,
                            step_index=i,
                            step_status=PlanStepStatus.IN_PROGRESS.value,
                        )
                    except Exception as e:
                        logger.warning(f"Error marking step as in_progress: {e}")

                    return i, step_info

            return None, None  # No active step found

        except Exception as e:
            logger.warning(f"Error finding current step index: {e}")
            return None, None

    async def _execute_step(self, executor: BaseAgent, step_info: dict) -> str:
        """Execute the current step with the specified agent using agent.run()."""
        # Reuse agent instances safely: clear previous interaction residue per plan step.
        executor.state = AgentState.IDLE
        executor.current_step = 0
        if hasattr(executor, "memory") and hasattr(executor.memory, "clear"):
            executor.memory.clear()

        step_text = step_info.get("text", f"Step {self.current_step_index}")
        previous_outputs = self._get_previous_step_outputs_text()

        # Create a prompt for the agent to execute the current step
        step_prompt = f"""
        ORIGINAL USER DATA (highest priority source):
        {self.original_request}

        PREVIOUS STEP OUTPUTS (reuse these, do not recompute if already available):
        {previous_outputs}

        YOUR CURRENT TASK:
        You are now working on step {self.current_step_index}: "{step_text}"

        Rules:
        1) Use the ORIGINAL USER DATA first. Do not fabricate or replace user-provided values.
        2) For pure calculation steps, use python_execute directly.
        3) Do not rely on ad-hoc external retrieval in this streamlined setup; prioritize provided data and local computation.
        4) If the numbers are already sufficient, finish this step quickly and conclude.
        5) IMPORTANT: Only complete this single assigned step. Do not execute future plan steps.
        6) Reuse values from PREVIOUS STEP OUTPUTS whenever possible instead of recalculating from raw data.
        7) After finishing this step, call terminate with status='success' immediately.

        Please only execute this current step using the appropriate tools. When you're done, provide a summary of what you accomplished.
        """

        # Use agent.run() to execute the step
        original_max_steps = executor.max_steps
        executor.max_steps = min(original_max_steps, self.max_agent_steps_per_plan_step)
        try:
            step_result = await executor.run(step_prompt)
            if self.current_step_index is not None:
                self.step_outputs[self.current_step_index] = step_result

            # Avoid false completion when executor is stuck or reaches local max steps.
            has_tool_observation = "Observed output of cmd" in step_result
            valid_no_tool_completion = self._is_valid_no_tool_completion(step_result)
            if (
                "Terminated: Reached max steps" in step_result
                or "Detected repeated identical tool call" in step_result
                or (not has_tool_observation and not valid_no_tool_completion)
            ):
                await self._mark_step_blocked(
                    "Executor did not complete the step effectively (loop/max steps)."
                )
            else:
                # Mark the step as completed after successful execution
                await self._mark_step_completed(self._build_step_note(step_result))

            return step_result
        except Exception as e:
            logger.error(f"Error executing step {self.current_step_index}: {e}")
            await self._mark_step_blocked(f"Execution error: {str(e)}")
            return f"Error executing step {self.current_step_index}: {str(e)}"
        finally:
            executor.max_steps = original_max_steps

    @staticmethod
    def _is_valid_no_tool_completion(step_result: str) -> bool:
        """
        Accept no-tool textual completion when it is substantive and not a loop marker.
        """
        if not step_result:
            return False
        text = step_result.strip()
        if not text:
            return False
        lowered = text.lower()
        invalid_markers = [
            "thinking complete - no action needed",
            "no content or commands to execute",
            "terminated: reached max steps",
            "detected repeated identical tool call",
            "error:",
        ]
        if any(marker in lowered for marker in invalid_markers):
            return False
        # Minimal substantive-length gate to avoid accepting trivial filler text.
        return len(text) >= 20

    async def _mark_step_blocked(self, notes: str) -> None:
        """Mark the current step as blocked with notes."""
        if self.current_step_index is None:
            return

        try:
            await self._commit_workflow_state(
                command="mark_step",
                plan_id=self.active_plan_id,
                step_index=self.current_step_index,
                step_status=PlanStepStatus.BLOCKED.value,
                step_notes=notes,
            )
            logger.warning(
                f"Marked step {self.current_step_index} as blocked in plan {self.active_plan_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to mark blocked status for step: {e}")

    async def _mark_step_completed(self, step_notes: str = "") -> None:
        """Mark the current step as completed."""
        if self.current_step_index is None:
            return

        try:
            # Mark the step as completed
            await self._commit_workflow_state(
                command="mark_step",
                plan_id=self.active_plan_id,
                step_index=self.current_step_index,
                step_status=PlanStepStatus.COMPLETED.value,
                step_notes=step_notes,
            )
            logger.info(
                f"Marked step {self.current_step_index} as completed in plan {self.active_plan_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to update plan status: {e}")

    def _build_step_note(self, step_result: str) -> str:
        """Build a compact step note for traceability and downstream reuse."""
        normalized = " ".join(step_result.split())
        return normalized[:280]

    def _get_previous_step_outputs_text(self) -> str:
        """Return recent prior step outputs to support non-redundant execution."""
        if self.current_step_index is None or not self.step_outputs:
            return "None"

        start = max(0, self.current_step_index - self.step_output_history_window)
        chunks: List[str] = []
        for i in range(start, self.current_step_index):
            output = self.step_outputs.get(i)
            if output:
                chunks.append(f"Step {i} output:\n{output[:1200]}")

        return "\n\n".join(chunks) if chunks else "None"

    async def _get_plan_text(self) -> str:
        """Get the current plan as formatted text."""
        try:
            result = await self._commit_workflow_state(
                command="get", plan_id=self.active_plan_id
            )
            return result.output if hasattr(result, "output") else str(result)
        except Exception as e:
            logger.error(f"Error getting plan: {e}")
            return self._generate_plan_text_from_storage()

    def _generate_plan_text_from_storage(self) -> str:
        """Generate plan text directly from storage if the workflow_state tool fails."""
        try:
            if self.active_plan_id not in self.workflow_state_tool.plans:
                return f"Error: Plan with ID {self.active_plan_id} not found"

            plan_data = self.workflow_state_tool.plans[self.active_plan_id]
            title = plan_data.get("title", "Untitled Plan")
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])
            step_notes = plan_data.get("step_notes", [])

            # Ensure step_statuses and step_notes match the number of steps
            while len(step_statuses) < len(steps):
                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
            while len(step_notes) < len(steps):
                step_notes.append("")

            # Count steps by status
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
                # Use status marks to indicate step status
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

