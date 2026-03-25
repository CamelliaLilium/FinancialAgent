from typing import Optional

from pydantic import Field, model_validator

from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.prompt.planning import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection, WorkflowStateTool


class PlanningAgent(ToolCallAgent):
    """
    A focused planner agent that creates and updates execution plans.

    This agent is intentionally lightweight:
    - It owns planning decisions.
    - It does not write workflow state in plan-synthesis phase.
    - It can terminate when planning is complete.
    """

    name: str = "Planning"
    description: str = (
        "A specialized planning agent that decomposes tasks into actionable steps "
        "for flow-controlled execution."
    )

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 12
    # 在多智能体 flow 中，最终答复由 PlanningFlow 统一收口，避免重复输出。
    enable_final_answer: bool = False

    workflow_state_tool: WorkflowStateTool = Field(default_factory=WorkflowStateTool)
    available_tools: Optional[ToolCollection] = None
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    @model_validator(mode="after")
    def init_tools(self) -> "PlanningAgent":
        """
        Ensure planning agent stays synthesis-only in this flow.

        Workflow state updates are committed by PlanningFlow at controlled commit points.
        """
        if self.available_tools is None:
            self.available_tools = ToolCollection(
                Terminate(),
            )
        return self
