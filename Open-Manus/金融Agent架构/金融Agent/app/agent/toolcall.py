import asyncio
import json
import re
from pathlib import Path
from typing import Any, List, Optional, Union

from pydantic import Field

from app.agent.react import ReActAgent
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import TOOL_CHOICE_TYPE, AgentState, Message, ToolCall, ToolChoice
from app.tool import Terminate, ToolCollection


TOOL_CALL_REQUIRED = "Tool calls required but none provided"


class ToolCallAgent(ReActAgent):
    """Base agent class for handling tool/function calls with enhanced abstraction"""

    name: str = "toolcall"
    description: str = "an agent that can execute tool calls."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    # Keep base default minimal to avoid noisy tool space.
    # Concrete agents should explicitly inject domain tools.
    available_tools: ToolCollection = ToolCollection(Terminate())
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    tool_calls: List[ToolCall] = Field(default_factory=list)
    _current_base64_image: Optional[str] = None
    _last_tool_signature: Optional[str] = None
    _same_tool_call_count: int = 0
    _consecutive_no_tool_rounds: int = 0
    _last_no_tool_content: str = ""

    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None
    max_identical_tool_calls: int = 3
    max_consecutive_no_tool_rounds: int = 2
    auto_finish_on_no_tool_content: bool = True
    compute_no_tool_retry_rounds: int = 1
    enforce_str_replace_editor_evidence_guard: bool = True
    # 是否在 run() 结束后自动生成最终答复（给 flow 场景可按需关闭）。
    enable_final_answer: bool = True

    async def _synthesize_final_answer(self) -> Optional[str]:
        """
        Build a user-facing final answer from execution memory.

        This is used after the tool loop ends so the caller always receives
        a clear final response rather than raw step logs only.
        """
        if not self.messages:
            return None

        # The first user message is treated as the original request.
        original_request = next(
            (
                msg.content
                for msg in self.messages
                if msg.role == "user" and msg.content
            ),
            None,
        )
        if not original_request:
            return None

        try:
            final_prompt = Message.user_message(
                (
                    "Please provide the final answer to the original user request based on the completed "
                    "tool interactions and observations.\n\n"
                    f"Original request:\n{original_request}\n\n"
                    "Requirements:\n"
                    "1. Give a direct final answer first.\n"
                    "2. Include key evidence or numbers if available.\n"
                    "3. Mention uncertainty briefly if any.\n"
                    "4. Do not mention internal chain-of-thought.\n"
                )
            )
            summary = await self.llm.ask(
                messages=self.messages + [final_prompt],
                system_msgs=[
                    Message.system_message(
                        "You are generating the final user-facing answer for a completed task."
                    )
                ],
                stream=False,
            )
            return summary.strip() if isinstance(summary, str) and summary.strip() else None
        except Exception as e:
            logger.warning(f"Failed to synthesize final answer: {e}")
            return None

    async def think(self) -> bool:
        """Process current state and decide next actions using tools"""
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        try:
            # Get response with tool options
            response = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=(
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
            )
        except ValueError:
            raise
        except Exception as e:
            # Check if this is a RetryError containing TokenLimitExceeded
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                logger.error(
                    f"🚨 Token limit error (from RetryError): {token_limit_error}"
                )
                self.memory.add_message(
                    Message.assistant_message(
                        f"Maximum token limit reached, cannot continue execution: {str(token_limit_error)}"
                    )
                )
                self.state = AgentState.FINISHED
                return False
            raise

        self.tool_calls = tool_calls = (
            response.tool_calls if response and response.tool_calls else []
        )
        content = response.content if response and response.content else ""

        # Log response info
        logger.info(f"✨ {self.name}'s thoughts: {content}")
        logger.info(
            f"🛠️ {self.name} selected {len(tool_calls) if tool_calls else 0} tools to use"
        )
        if tool_calls:
            logger.info(
                f"🧰 Tools being prepared: {[call.function.name for call in tool_calls]}"
            )
            logger.info(f"🔧 Tool arguments: {tool_calls[0].function.arguments}")

        try:
            if response is None:
                raise RuntimeError("No response received from the LLM")

            # Handle different tool_choices modes
            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if content:
                    self.memory.add_message(Message.assistant_message(content))
                    return True
                return False

            # Create and add assistant message
            assistant_msg = (
                Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
                if self.tool_calls
                else Message.assistant_message(content)
            )
            self.memory.add_message(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # Will be handled in act()

            # For 'auto' mode, continue with content if no commands but content exists
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return self._handle_no_tool_round(content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"Error encountered while processing: {str(e)}"
                )
            )
            return False

    def _handle_no_tool_round(self, content: str) -> bool:
        """
        Handle no-tool rounds in AUTO mode.

        Goals:
        1) Allow legitimate no-tool completion for text-only tasks.
        2) Prevent no-tool idle spinning across repeated rounds.
        """
        normalized = " ".join((content or "").split())

        # Empty content in no-tool mode is treated as no-progress; stop quickly.
        if not normalized:
            self._consecutive_no_tool_rounds += 1
            if self._consecutive_no_tool_rounds >= 1:
                logger.warning(
                    f"{self.name}: no-tool empty response detected; finishing to avoid idle loop."
                )
                self.state = AgentState.FINISHED
                return False
            return True

        # Track repeated no-tool textual outputs.
        if normalized == self._last_no_tool_content:
            self._consecutive_no_tool_rounds += 1
        else:
            self._consecutive_no_tool_rounds = 1
            self._last_no_tool_content = normalized

        # Common case: textual answer without tools is already sufficient.
        if self.auto_finish_on_no_tool_content:
            # For compute-like contexts, allow one extra no-tool retry round before
            # finalizing, so the model can reconsider deterministic computation tools
            # without hard-coding any specific tool name.
            if (
                self._is_compute_context()
                and self._consecutive_no_tool_rounds <= self.compute_no_tool_retry_rounds
            ):
                self.memory.add_message(
                    Message.assistant_message(
                        "Computation-like task detected. Re-check whether a deterministic "
                        "computation tool is needed for reliable, auditable numeric output "
                        "before finalizing."
                    )
                )
                return True
            self.state = AgentState.FINISHED
            return False

        # Fallback guard if auto-finish is disabled.
        if self._consecutive_no_tool_rounds >= self.max_consecutive_no_tool_rounds:
            logger.warning(
                f"{self.name}: repeated no-tool rounds reached "
                f"{self._consecutive_no_tool_rounds}; finishing to avoid idle loop."
            )
            self.state = AgentState.FINISHED
            return False

        return True

    def _is_compute_context(self) -> bool:
        """
        Heuristic check for computation-heavy context.
        This is intentionally generic to keep extensibility and avoid hard-coded tool routing.
        """
        user_text = "\n".join(
            [msg.content for msg in self.messages if msg.role == "user" and msg.content]
        ).lower()
        if not user_text:
            return False

        compute_cues = [
            "calculate",
            "ratio",
            "percentage",
            "growth",
            "increase",
            "decrease",
            "sum",
            "total",
            "average",
            "formula",
            "roa",
            "roe",
            "cagr",
            "同比",
            "环比",
            "增长率",
            "占比",
            "均值",
            "合计",
            "差值",
            "计算",
        ]
        return any(cue in user_text for cue in compute_cues)

    async def act(self) -> str:
        """Execute tool calls and handle their results"""
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)

            # Return last message content if no tool calls
            return self.messages[-1].content or "No content or commands to execute"

        results = []
        for command in self.tool_calls:
            # Reset base64_image for each tool call
            self._current_base64_image = None

            result = await self.execute_tool(command)

            if self.max_observe:
                result = result[: self.max_observe]

            logger.info(
                f"🎯 Tool '{command.function.name}' completed its mission! Result: {result}"
            )

            # Add tool response to memory
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name,
                base64_image=self._current_base64_image,
            )
            self.memory.add_message(tool_msg)
            results.append(result)

        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        """Execute a single tool call with robust error handling"""
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        if self.available_tools.get_tool(name) is None:
            return f"Error: Unknown tool '{name}'"

        try:
            # Parse arguments
            args = json.loads(command.function.arguments or "{}")

            # Guard str_replace_editor with evidence-based file checks.
            blocked_reason = self._validate_str_replace_editor_call(name=name, args=args)
            if blocked_reason:
                return f"Error: {blocked_reason}"

            # Guard against repeated identical tool calls that burn tokens.
            tool_signature = f"{name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
            if tool_signature == self._last_tool_signature:
                self._same_tool_call_count += 1
            else:
                self._last_tool_signature = tool_signature
                self._same_tool_call_count = 1

            if self._same_tool_call_count > self.max_identical_tool_calls:
                return (
                    "Error: Detected repeated identical tool call. "
                    "Stop repeating the same query and switch strategy "
                    "(e.g., use another tool, refine query, or terminate with current confidence)."
                )

            # Execute the tool
            logger.info(f"🔧 Activating tool: '{name}'...")
            result = await self.available_tools.execute(name=name, tool_input=args)

            # Handle special tools
            await self._handle_special_tool(name=name, result=result)

            # Check if result is a ToolResult with base64_image
            if hasattr(result, "base64_image") and result.base64_image:
                # Store the base64_image for later use in tool_message
                self._current_base64_image = result.base64_image

            # Format result for display (standard case)
            observation = (
                f"Observed output of cmd `{name}` executed:\n{str(result)}"
                if result
                else f"Cmd `{name}` completed with no output"
            )

            return observation
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(
                f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{command.function.arguments}"
            )
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.exception(error_msg)
            return f"Error: {error_msg}"

    def _validate_str_replace_editor_call(
        self, name: str, args: dict[str, Any]
    ) -> Optional[str]:
        """
        Evidence gate for str_replace_editor:
        1) require explicit absolute path argument
        2) require user/context text to contain concrete file evidence
        3) require path existence for non-create commands
        """
        if name != "str_replace_editor" or not self.enforce_str_replace_editor_evidence_guard:
            return None

        path_value = str(args.get("path", "")).strip()
        command = str(args.get("command", "")).strip()
        if not path_value:
            return (
                "str_replace_editor blocked: missing `path`. "
                "Only call this tool when a concrete file path is provided."
            )

        path_obj = Path(path_value)
        if not path_obj.is_absolute():
            return (
                "str_replace_editor blocked: `path` must be absolute and verifiable."
            )

        # Evidence from initial user/request context.
        context_text = "\n".join(
            [msg.content for msg in self.messages if msg.role == "user" and msg.content]
        )
        has_explicit_file_evidence = self._contains_file_evidence(
            context_text=context_text, path_value=path_value
        )
        if not has_explicit_file_evidence:
            return (
                "str_replace_editor blocked: no explicit file evidence found in request context. "
                "If context already has sufficient information, continue without file editing."
            )

        if command and command != "create" and not path_obj.exists():
            return (
                f"str_replace_editor blocked: target path does not exist for command `{command}`: {path_value}"
            )

        if command == "create":
            parent = path_obj.parent
            if not parent.exists():
                return (
                    f"str_replace_editor blocked: parent directory does not exist for create: {parent}"
                )

        return None

    @staticmethod
    def _contains_file_evidence(context_text: str, path_value: str) -> bool:
        """Check whether user/context includes concrete file evidence."""
        if not context_text:
            return False

        lowered = context_text.lower()
        path_lower = path_value.lower()
        name_lower = Path(path_value).name.lower()

        # Strongest evidence: exact path appears in context.
        if path_lower in lowered:
            return True

        # Secondary evidence: absolute path patterns or explicit filename mention.
        abs_path_pattern = r"([a-zA-Z]:\\[^\\\n]+(?:\\[^\\\n]+)*)|(/[^ \n]+)+"
        if re.search(abs_path_pattern, context_text) and name_lower in lowered:
            return True

        # Explicit file-like token mention.
        file_token_pattern = r"\b[\w\-.]+\.(txt|md|csv|json|xlsx|xls|pdf|docx?)\b"
        if re.search(file_token_pattern, lowered) and name_lower in lowered:
            return True

        return False

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """Handle special tool execution and state changes"""
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # Set agent state to finished
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """Check if tool name is in special tools list"""
        return name.lower() in [n.lower() for n in self.special_tool_names]

    async def cleanup(self):
        """Clean up resources used by the agent's tools."""
        logger.info(f"🧹 Cleaning up resources for agent '{self.name}'...")
        for tool_name, tool_instance in self.available_tools.tool_map.items():
            if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(
                tool_instance.cleanup
            ):
                try:
                    logger.debug(f"🧼 Cleaning up tool: {tool_name}")
                    await tool_instance.cleanup()
                except Exception as e:
                    logger.error(
                        f"🚨 Error cleaning up tool '{tool_name}': {e}", exc_info=True
                    )
        logger.info(f"✨ Cleanup complete for agent '{self.name}'.")

    async def run(self, request: Optional[str] = None) -> str:
        """Run the agent with cleanup when done."""
        # Reset per-run duplicate-call guard state.
        self._last_tool_signature = None
        self._same_tool_call_count = 0
        self._consecutive_no_tool_rounds = 0
        self._last_no_tool_content = ""
        try:
            execution_log = await super().run(request)
            final_answer = (
                await self._synthesize_final_answer()
                if self.enable_final_answer
                else None
            )
            return final_answer or execution_log
        finally:
            await self.cleanup()
