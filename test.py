import asyncio
import os
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

# Load Gemini API key securely from environment
GEMINI_API_KEY = "AIzaSyDKOKL1CHtUhjZpsfuIbYdu11U4FhuW5jI"

# Ensure key exists
if not GEMINI_API_KEY:
    raise ValueError("Missing Gemini API key. Set GEMINI_API_KEY as an environment variable.")

# Initialize Gemini client with OpenAI-compatible interface
client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Disable tracing for cleaner runs (optional)
set_tracing_disabled(disabled=True)

async def run_agent():
    # Define agent with haiku-only instructions
    agent = Agent(
        name="HaikuBot",
        instructions="You only respond in haikus.",
        model=OpenAIChatCompletionsModel(
            model="gemini-2.0-flash",
            openai_client=client
        ),
    )

    # Input prompt
    prompt = "Tell me about recursion in programming."

    # Run the agent
    result = await Runner.run(agent, prompt)

    # Display result
    print(result.final_output)

if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except Exception as e:
        print(f"Error occurred: {e}")
