"""
ToolCallAgent - 基于 archive/app/original/toolcall.py 的简洁实现。

保持原始逻辑：无 _handle_no_tool_round、无 _synthesize_final_answer、
无 max_identical_tool_calls 等复杂逻辑，确保步骤执行清晰可靠。
支持 AntiLoopInterceptor 防死循环与低级错误拦截。
"""
import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, PrivateAttr

from app.agent.react import ReActAgent
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import (
    TOOL_CHOICE_TYPE,
    AgentState,
    Function,
    Message,
    ToolCall,
    ToolChoice,
)
from app.tool import Terminate, ToolCollection


TOOL_CALL_REQUIRED = "Tool calls required but none provided"


def _parse_tool_arguments(raw: str, tool_name: str = "") -> Dict[str, Any]:
    """
    解析工具参数字符串为 dict。当 json.loads 失败时，尝试修复常见 JSON 问题。

    根因：LLM 输出包含未转义的引号或特殊字符，或JSON被截断。
    """
    s = (raw or "").strip()
    if not s:
        return {}

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 对 python_execute, finance_extraction_skill, planning 尝试修复
    if tool_name not in ("python_execute", "finance_extraction_skill", "planning"):
        raise json.JSONDecodeError("Invalid JSON", s, 0)

    # 修复0: 尝试补全缺失的闭合括号（JSON被截断的情况）
    open_braces = s.count('{') - s.count('}')
    open_brackets = s.count('[') - s.count(']')
    if open_braces > 0 or open_brackets > 0:
        repaired = s + ']' * open_brackets + '}' * open_braces
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    # 修复1: LLM 误输出 ","+str 应为 "+str（逗号多余且 " 未转义）。替换为 \"+str
    repaired = re.sub(r',(?<!\\)"(?=[ \t]*\+str\()', r'\\"', s)
    # 修复2: =" 后接 +str 的 " 未转义，替换为 \"
    repaired = re.sub(r'=(?<!\\)"(?=[ \t]*\+str\()', r'=\\"', repaired)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # 修复3: python_execute 缺失闭合 " 时，从 "code": " 到 } 前提取内容并反转义
    if tool_name == "python_execute":
        code_match = re.search(r'"code"\s*:\s*"', repaired, re.IGNORECASE)
        if code_match:
            start = code_match.end()
            rbrace = repaired.rfind("}")
            if rbrace > start:
                code_raw = repaired[start:rbrace]
                code = code_raw.replace("\\\\", "\x00").replace('\\"', '"').replace("\\n", "\n")
                code = code.replace("\x00", "\\")
                return {"code": code}

    # 修复4: finance_extraction_skill 解析失败时，尝试提取 variables
    if tool_name == "finance_extraction_skill":
        vars_match = re.search(r'"variables"\s*:\s*\[([^\]]*)\]', s, re.IGNORECASE)
        if vars_match:
            vars_str = vars_match.group(1)
            variables = [v.strip().strip('"\'') for v in vars_str.split(",") if v.strip()]
            if variables:
                return {"variables": variables, "use_context_image": True}

    # 修复5: planning 工具解析失败时，尝试提取关键字段
    if tool_name == "planning":
        result = {}
        command_match = re.search(r'"command"\s*:\s*"([^"]+)"', s, re.IGNORECASE)
        if command_match:
            result["command"] = command_match.group(1)
        plan_id_match = re.search(r'"plan_id"\s*:\s*"([^"]+)"', s, re.IGNORECASE)
        if plan_id_match:
            result["plan_id"] = plan_id_match.group(1)
        title_match = re.search(r'"title"\s*:\s*"([^"]+)"', s, re.IGNORECASE)
        if title_match:
            result["title"] = title_match.group(1)
        steps_match = re.search(r'"steps"\s*:\s*\[(.*)\]', s, re.DOTALL | re.IGNORECASE)
        if steps_match:
            steps_str = steps_match.group(1)
            steps = re.findall(r'"([^"]*(?:\'[^\']*\')?[^"]*)"', steps_str)
            if steps:
                result["steps"] = [step.strip() for step in steps if step.strip()]
        if result.get("command") and result.get("plan_id"):
            return result

    raise json.JSONDecodeError("Invalid JSON", s, 0)


class ToolCallAgent(ReActAgent):
    """Base agent class for handling tool/function calls with enhanced abstraction"""

    name: str = "toolcall"
    description: str = "an agent that can execute tool calls."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    # 保持基础默认最小化，子类（如 FinanceAgent）注入领域工具
    available_tools: ToolCollection = ToolCollection(Terminate())
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    tool_calls: List[ToolCall] = Field(default_factory=list)
    _current_base64_image: Optional[str] = None

    # 由 flow 注入的防死循环拦截器，每个样本开始时重置
    anti_loop_interceptor: Optional[Any] = Field(default=None, exclude=True)

    # zero_tool 早停：连续 N 次无工具调用时强制终止，减少无效轮次
    _consecutive_zero_tool_rounds: int = PrivateAttr(default=0)
    # P0: 连续被 AntiLoop 拦截次数，>=2 时在结果中追加强制 terminate 提示
    _consecutive_antiloop_blocks: int = PrivateAttr(default=0)

    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None

    async def think(self) -> bool:
        """Process current state and decide next actions using tools"""
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        try:
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

        # Fallback: when model returns no tool_calls but content matches parseable format
        if not tool_calls and content:
            parsed = self._parse_content_for_tool_calls(content)
            if parsed:
                self.tool_calls = tool_calls = parsed
                logger.info(
                    f"📋 Parsed {len(parsed)} tool call(s) from content (fallback)"
                )

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

            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if content:
                    self.memory.add_message(Message.assistant_message(content))
                    return True
                return False

            assistant_msg = (
                Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
                if self.tool_calls
                else Message.assistant_message(content)
            )
            self.memory.add_message(assistant_msg)

            if self.tool_calls:
                self._consecutive_zero_tool_rounds = 0

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                self._consecutive_zero_tool_rounds += 1
                if self._consecutive_zero_tool_rounds >= 3:
                    logger.warning(
                        f"🛑 Zero-tool early stop (REQUIRED): {self._consecutive_zero_tool_rounds} "
                        "consecutive rounds without tool calls"
                    )
                    self.state = AgentState.FINISHED
                    return False
                if self._consecutive_zero_tool_rounds >= 2:
                    self.memory.add_message(
                        Message.user_message(
                            "URGENT: You did not call any tool in the previous round(s). "
                            "You MUST invoke a tool via tool_call (python_execute or terminate). Do not output tool code as text."
                        )
                    )
                return True

            # AUTO 模式：无工具调用时，有内容则继续（原始逻辑）
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                self._consecutive_zero_tool_rounds += 1
                if self._consecutive_zero_tool_rounds >= 3:
                    logger.warning(
                        f"🛑 Zero-tool early stop: {self._consecutive_zero_tool_rounds} "
                        "consecutive rounds without tool calls"
                    )
                    self.state = AgentState.FINISHED
                    return False
                if self._consecutive_zero_tool_rounds >= 2:
                    self.memory.add_message(
                        Message.user_message(
                            "URGENT: You did not call any tool in the previous round(s). "
                            "You MUST invoke a tool via tool_call (python_execute or terminate). Do not output tool code as text."
                        )
                    )
                return bool(content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"Error encountered while processing: {str(e)}"
                )
            )
            return False

    async def act(self) -> str:
        """Execute tool calls and handle their results"""
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)
            return self.messages[-1].content or "No content or commands to execute"

        results = []
        for command in self.tool_calls:
            self._current_base64_image = None
            result = await self.execute_tool(command)

            if self.max_observe:
                result = result[: self.max_observe]

            logger.info(
                f"🎯 Tool '{command.function.name}' completed its mission! Result: {result}"
            )

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
            args = _parse_tool_arguments(command.function.arguments or "{}", tool_name=name)

            logger.info(f"🔧 Activating tool: '{name}'...")

            # P0: 连续被 block 2+ 次时，强制执行 terminate 以打断重复循环
            if (
                self._consecutive_antiloop_blocks >= 2
                and name in ("python_execute", "str_replace_editor")
            ):
                if self.available_tools.get_tool("terminate"):
                    logger.warning("🛑 Force-terminate: blocked 2+ times, redirecting to terminate")
                    result = await self.available_tools.execute(
                        name="terminate", tool_input={"status": "success"}
                    )
                    await self._handle_special_tool(name="terminate", result=result)
                    return (
                        f"Observed output of cmd `terminate` executed:\n{str(result)}\n"
                        f"(Forced: repeated block detected; python_execute was not executed.)"
                    )

            if self.anti_loop_interceptor and name in (
                "python_execute",
                "str_replace_editor",
            ):
                result = await self.anti_loop_interceptor.execute_with_reflection(
                    tool_name=name,
                    kwargs=args,
                    actual_execute_func=lambda **kw: self.available_tools.execute(
                        name=name, tool_input=kw
                    ),
                )
                # P0: 连续拦截时强制 terminate 提示
                is_block = isinstance(result, str) and "System Intercept" in str(result)
                if is_block:
                    self._consecutive_antiloop_blocks += 1
                else:
                    self._consecutive_antiloop_blocks = 0  # 非拦截时重置
                if self._consecutive_antiloop_blocks >= 2:
                    result = (
                        str(result)
                        + "\n\nCRITICAL: You have been blocked 2+ times in a row. "
                        "You MUST call terminate(status='success') via tool_call. Do not output tool names as text."
                    )
            else:
                self._consecutive_antiloop_blocks = 0
                result = await self.available_tools.execute(name=name, tool_input=args)

            await self._handle_special_tool(name=name, result=result)

            if hasattr(result, "base64_image") and result.base64_image:
                self._current_base64_image = result.base64_image

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

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """Handle special tool execution and state changes"""
        if not self._is_special_tool(name):
            return
        if self._should_finish_execution(name=name, result=result, **kwargs):
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        return True

    def _is_special_tool(self, name: str) -> bool:
        return name.lower() in [n.lower() for n in self.special_tool_names]

    def _parse_content_for_tool_calls(self, content: str) -> Optional[List[ToolCall]]:
        """
        Override in subclasses to parse tool calls from text when model returns none.
        Returns None if content cannot be parsed; otherwise returns list of ToolCall.
        """
        return None

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

    async def run(
        self,
        request: Optional[str] = None,
        base64_image: Optional[str] = None,
        base64_images: Optional[List[str]] = None,
    ) -> str:
        """Run the agent with cleanup when done. Supports multi-image via base64_images."""
        self._consecutive_zero_tool_rounds = 0
        self._consecutive_antiloop_blocks = 0
        try:
            return await super().run(
                request,
                base64_image=base64_image,
                base64_images=base64_images,
            )
        finally:
            await self.cleanup()
