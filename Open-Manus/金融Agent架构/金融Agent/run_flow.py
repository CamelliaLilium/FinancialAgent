import asyncio
import time

from app.agent.finance import FinanceAgent
from app.agent.planning import PlanningAgent
from app.config import config
from app.flow.flow_factory import FlowFactory, FlowType
from app.logger import logger
from app.tool import WorkflowStateTool


async def run_flow():
    workflow_state_tool = WorkflowStateTool()
    agents = {
        "finance": FinanceAgent(),
    }
    executor_keys = ["finance"]

    if config.run_flow_config.use_planning_agent:
        agents["planning"] = PlanningAgent(workflow_state_tool=workflow_state_tool)

    try:
        prompt = input("Enter your prompt: ")

        if prompt.strip().isspace() or not prompt:
            logger.warning("Empty prompt provided.")
            return

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
                flow.execute(prompt),
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
