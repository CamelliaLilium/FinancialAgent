import asyncio
import base64
import time
from pathlib import Path

from app.agent.finance import FinanceAgent
from app.agent.multimodal import MultimodalAgent
from app.agent.planning import PlanningAgent
from app.config import config
from app.flow.flow_factory import FlowFactory, FlowType
from app.logger import logger
from app.tool import WorkflowStateTool


def _load_images_from_paths(paths: list[str]) -> list[str]:
    """Load images from file paths and return base64 strings."""
    result = []
    for p in paths:
        path = Path(p).expanduser().resolve()
        if path.exists():
            with path.open("rb") as f:
                result.append(base64.b64encode(f.read()).decode("utf-8"))
    return result


def _has_vision_config() -> bool:
    """Check if [llm.vision] is configured for multimodal agent."""
    return "vision" in config.llm


async def run_flow():
    workflow_state_tool = WorkflowStateTool()
    agents = {
        "finance": FinanceAgent(),
    }
    executor_keys = ["finance"]

    if config.run_flow_config.use_planning_agent:
        agents["planning"] = PlanningAgent(workflow_state_tool=workflow_state_tool)

    # 多模态 Agent：仅在配置启用且 vision 模型存在时添加
    if (
        config.run_flow_config.use_multimodal_agent
        and _has_vision_config()
    ):
        agents["multimodal"] = MultimodalAgent()
        executor_keys = ["finance", "multimodal"]

    try:
        prompt = input("Enter your prompt: ")

        if prompt.strip().isspace() or not prompt:
            logger.warning("Empty prompt provided.")
            return

        # 可选：解析图片路径。格式 "!image:path1.png" 或 "!image:path1.png,path2.jpg"
        base64_images = None
        if "!image:" in prompt:
            parts = prompt.split("!image:", 1)
            prompt = parts[0].strip()
            path_part = parts[1].split()[0] if parts[1] else ""
            if path_part:
                image_paths = [p.strip() for p in path_part.split(",")]
                base64_images = _load_images_from_paths(image_paths)
                if base64_images:
                    logger.info(f"Loaded {len(base64_images)} image(s) for multimodal task")

        logger.warning("Processing your request...")

        try:
            start_time = time.time()
            primary_key = "planning" if "planning" in agents else "finance"
            flow = FlowFactory.create_flow(
                flow_type=FlowType.PLANNING,
                agents=agents,
                primary_agent_key=primary_key,
                executors=executor_keys,
                workflow_state_tool=workflow_state_tool,
            )
            result = await asyncio.wait_for(
                flow.execute(prompt, base64_images=base64_images),
                timeout=3600,  # 60 minute timeout for the entire execution
            )

            elapsed_time = time.time() - start_time
            logger.info(f"Request processed in {elapsed_time:.2f} seconds")
            logger.info(result)
        except asyncio.TimeoutError:
            logger.error("Request processing timed out after 1 hour")
            logger.info(
                "Operation terminated due to timeout. Please try a simpler request."
            )

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
    except Exception as e:
        logger.error(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(run_flow())
