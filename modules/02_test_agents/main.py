from agents import Agent, Runner, logger
from dotenv import load_dotenv

load_dotenv()
logger.logger.disabled = True


async def main() -> None:
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
        model="gpt-4.1",
    )

    runner = await Runner.run(agent, input="What is the capital of Indonesia?")
    print(runner.final_output)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
