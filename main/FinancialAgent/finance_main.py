import argparse
import asyncio

from app.agent.finance import FinanceAgent
from app.logger import logger


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Finance agent with a prompt")
    parser.add_argument(
        "--prompt", type=str, required=False, help="Input prompt for the finance agent"
    )
    args = parser.parse_args()

    # Create Finance agent
    agent = FinanceAgent()
    try:
        # Use command line prompt if provided, otherwise ask for input
        prompt = args.prompt if args.prompt else input("Enter your prompt: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning("Processing your request...")
        result = await agent.run(prompt)
        logger.info(result)
        logger.info("Request processing completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        # Cleanup if needed
        if hasattr(agent, "cleanup"):
            await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
