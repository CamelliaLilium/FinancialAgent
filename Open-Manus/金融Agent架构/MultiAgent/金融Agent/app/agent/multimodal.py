"""
MultimodalAgent - 多模态任务专用智能体

使用 Qwen3-VL-8B-Instruct 等视觉模型处理带图片的输入。
仅在多模态输入和任务时被调用，不影响纯文本流程。

当 vision 模型不返回原生 tool_calls 时，通过 _parse_content_for_tool_calls
解析 "Thought / Action / 代码块" 格式的文本输出，转为 synthetic tool_calls。
"""
import json
import re
from typing import List, Optional

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.llm import LLM
from app.prompt.multimodal import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import Function, ToolCall
from app.skill import FinanceExtractionSkill
from app.tool import Terminate, ToolCollection
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor


class MultimodalAgent(ToolCallAgent):
    """
    多模态任务专用智能体

    使用 vision 配置的 LLM（如 Qwen3-VL-8B-Instruct）处理图片+文本输入。
    能力：图像理解、图表数据提取、视觉证据转文本、多模态推理。
    """

    name: str = "Multimodal"
    description: str = (
        "A specialized multimodal agent for image-text tasks: chart/table extraction, "
        "visual evidence understanding, and vision-language reasoning. Uses vision models. "
        "Prefer finance_extraction_skill for numeric extraction from tables/charts (OCR+VLM Point-Check). "
        "Tool-first execution: MUST call finance_extraction_skill or python_execute to persist data, or terminate when done."
    )

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 15000
    max_steps: int = 20

    # 使用 vision 配置的 LLM（Qwen3-VL-8B-Instruct 等）
    llm: LLM = Field(default_factory=lambda: LLM(config_name="vision"))

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            FinanceExtractionSkill(),
            PythonExecute(),
            StrReplaceEditor(),
            Terminate(),
        )
    )
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    def _parse_content_for_tool_calls(self, content: str) -> Optional[List[ToolCall]]:
        """
        解析 QWEN 格式的文本输出：Thought / Action / 代码块。
        当 vision 模型不返回原生 tool_calls 时，从 content 中提取并转为 synthetic tool_calls。
        """
        content = content or ""
        action_match = re.search(
            r"Action:\s*(finance_extraction_skill|python_execute|terminate)",
            content,
            re.IGNORECASE,
        )
        if not action_match:
            return None
        action = action_match.group(1).lower()

        if action == "finance_extraction_skill":
            # 解析 variables: ["var1", "var2"] 或 Variables: var1, var2
            vars_match = re.search(
                r"[Vv]ariables?\s*[:\[]\s*\[([^\]]*)\]", content, re.IGNORECASE
            )
            if vars_match:
                vars_str = vars_match.group(1)
                variables = [v.strip().strip('"\'') for v in vars_str.split(",") if v.strip()]
            else:
                vars_match = re.search(
                    r"[Vv]ariables?\s*[:\=]\s*([^\n]+)", content, re.IGNORECASE
                )
                if vars_match:
                    variables = [
                        v.strip().strip('"\'')
                        for v in vars_match.group(1).split(",")
                        if v.strip()
                    ]
                else:
                    return None
            if not variables:
                return None
            return [
                ToolCall(
                    id="fallback_multimodal_1",
                    type="function",
                    function=Function(
                        name="finance_extraction_skill",
                        arguments=json.dumps(
                            {"variables": variables, "use_context_image": True}
                        ),
                    ),
                )
            ]
        if action == "python_execute":
            code_match = re.search(r"```python\s*\n(.*?)```", content, re.DOTALL)
            if not code_match:
                return None
            code = code_match.group(1).strip()
            if not code:
                return None
            return [
                ToolCall(
                    id="fallback_multimodal_1",
                    type="function",
                    function=Function(
                        name="python_execute",
                        arguments=json.dumps({"code": code}),
                    ),
                )
            ]
        if action == "terminate":
            status = "failure" if re.search(r"failure|DATA_UNREADABLE", content, re.I) else "success"
            return [
                ToolCall(
                    id="fallback_multimodal_1",
                    type="function",
                    function=Function(
                        name="terminate",
                        arguments=json.dumps({"status": status}),
                    ),
                )
            ]
        return None
