# full_crew.py
# Full 3-agent pipeline: Fetcher (MCP) -> Analyst -> Classifier.

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

load_dotenv()
assert os.getenv('OPENROUTER_API_KEY'), 'OPENROUTER_API_KEY missing in .env'

# Same LLM for all three agents
llm = LLM(
    model='openrouter/openai/gpt-oss-120b:free',
    base_url='https://openrouter.ai/api/v1',
    api_key=os.getenv('OPENROUTER_API_KEY'),
)

# MCP server config (only the Fetcher will use it)
server_params = StdioServerParameters(
    command=sys.executable,
    args=[str(Path(__file__).parent / 'review_server.py')],
    env={**os.environ},
)

# Which review to process — change to '1', '2', '3', '4', '5', or '6'
REVIEW_ID = '4'

with MCPServerAdapter(server_params) as mcp_tools:

    # ---------- AGENTS ----------
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

    analyst = Agent(
        role='Senior Review Analyst',
        goal=(
            'Break a review into structured insights: topic, praises, '
            'complaints, emotional tone, and factual claims.'
        ),
        backstory=(
            'Ten years of e-commerce consumer-insights experience. '
            'You read between the lines and separate opinion from fact.'
        ),
        llm=llm,
        verbose=True,
    )

    classifier = Agent(
        role='Sentiment Classifier',
        goal=(
            "Read the analyst's report and commit to a final sentiment "
            'label: POSITIVE, NEGATIVE, or NEUTRAL.'
        ),
        backstory=(
            'A precise classification specialist. You always pick exactly '
            'one of three labels and back it with a confidence score.'
        ),
        llm=llm,
        verbose=True,
    )

    # ---------- TASKS (chained via context=) ----------
    fetch_task = Task(
        description=(
            "Call the get_review tool with review_id='{review_id}'. "
            'Return ONLY the raw review text.'
        ),
        expected_output='The complete, unmodified review text.',
        agent=fetcher,
    )

    analyze_task = Task(
        description=(
            'Analyze the review text from the previous step. Produce a '
            'structured report with these five sections:\n'
            '1. Main topic / product feature\n'
            '2. Specific praises (bullet list)\n'
            '3. Specific complaints (bullet list)\n'
            '4. Emotional tone (e.g. excited, frustrated, indifferent)\n'
            '5. Any factual claims worth verifying'
        ),
        expected_output='A clear, bulleted analysis covering all five sections.',
        agent=analyst,
        context=[fetch_task],            # uses fetcher's output
    )

    classify_task = Task(
        description=(
            "Based on the analyst's report, output the final verdict in "
            'EXACTLY this format:\n'
            'Sentiment: <POSITIVE|NEGATIVE|NEUTRAL>\n'
            'Confidence: <number between 0.0 and 1.0>\n'
            'Justification: <one short sentence>'
        ),
        expected_output='Three lines: Sentiment, Confidence, Justification.',
        agent=classifier,
        context=[analyze_task],          # uses analyst's output
    )

    # ---------- CREW ----------
    crew = Crew(
        agents=[fetcher, analyst, classifier],
        tasks=[fetch_task, analyze_task, classify_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff(inputs={'review_id': REVIEW_ID})

print(f'\n========== FINAL VERDICT for review {REVIEW_ID} ==========')
print(result)