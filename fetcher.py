# fetcher.py
# Standalone script that runs the Fetcher agent (uses MCP).

import sys
import asyncio
import os
from pathlib import Path

# Windows + MCP compatibility — must run before crewai imports
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

# Load .env from the project root
load_dotenv()
assert os.getenv('OPENROUTER_API_KEY'), 'OPENROUTER_API_KEY missing in .env'

# Configure LLM via OpenRouter
llm = LLM(
    model='openrouter/openai/gpt-oss-20b:free',
    base_url='https://openrouter.ai/api/v1',
    api_key=os.getenv('OPENROUTER_API_KEY'),
)

# Configure MCP server (run with the same python this script uses)
server_params = StdioServerParameters(
    command=sys.executable,
    args=[str(Path(__file__).parent / 'review_server.py')],
    env={**os.environ},
)

# Run the Fetcher agent
with MCPServerAdapter(server_params) as mcp_tools:
    print(f'MCP tools discovered: {[t.name for t in mcp_tools]}')

    fetcher = Agent(
        role='Review Fetcher',
        goal='Retrieve the exact text of an Amazon review given its ID.',
        backstory=(
            'You are a meticulous data retriever. You never paraphrase. '
            'You always call the get_review tool and return the raw text '
            'it gives you, character for character.'
        ),
        tools=mcp_tools,
        llm=llm,
        verbose=True,
    )

    fetch_task = Task(
        description=(
            "Call the get_review tool with review_id='{review_id}'. "
            'Return ONLY the raw review text.'
        ),
        expected_output='The complete, unmodified review text.',
        agent=fetcher,
    )

    crew = Crew(
        agents=[fetcher],
        tasks=[fetch_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff(inputs={'review_id': '1'})

print('\n=========== FETCHER OUTPUT ===========')
print(result)