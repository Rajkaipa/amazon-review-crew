# crew_lib.py
# Shared crew-building logic. Used by full_crew.py, evaluate.py, and app.py.

import os
import sys
import asyncio
from pathlib import Path
from typing import Optional, Tuple

# Windows + MCP compatibility — apply before any crewai/mcp import
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv()


def build_llm() -> LLM:
    """Build the LLM used by all agents. Centralized so swapping models is one edit."""
    return LLM(
        model='openrouter/openai/gpt-oss-20b:free',
        base_url='https://openrouter.ai/api/v1',
        api_key=os.getenv('OPENROUTER_API_KEY'),
    )


def analyze_review_text(review_text: str, verbose: bool = False) -> Tuple[str, str]:
    """
    Run the Analyst → Classifier pipeline on a piece of review text.

    Skips the Fetcher (no MCP needed) because we already have the text.
    Returns (analysis_text, classification_text).
    """
    llm = build_llm()

    analyst = Agent(
        role='Senior Review Analyst',
        goal='Break a review into structured insights: topic, praises, complaints, tone, factual claims.',
        backstory='Ten years of e-commerce consumer-insights experience. You read between the lines.',
        llm=llm,
        verbose=verbose,
    )
    classifier = Agent(
        role='Sentiment Classifier',
        goal="Read the analyst's report and commit to POSITIVE, NEGATIVE, or NEUTRAL.",
        backstory='A precise classification specialist. Always picks exactly one label.',
        llm=llm,
        verbose=verbose,
    )

    analyze_task = Task(
        description=(
            'Analyze the following customer review and produce a structured report with:\n'
            '1. Main topic / product feature\n'
            '2. Specific praises (bullet list)\n'
            '3. Specific complaints (bullet list)\n'
            '4. Emotional tone\n'
            '5. Any factual claims worth verifying\n\n'
            'REVIEW:\n{review_text}'
        ),
        expected_output='A clear, bulleted analysis covering all five sections.',
        agent=analyst,
    )
    classify_task = Task(
        description=(
        "Based on the analyst's report above, assign a final sentiment label.\n\n"
            "LABEL DEFINITIONS:\n"
            "- POSITIVE: The reviewer is satisfied. They recommend, praise, or "
            "express clear approval. Use this even if they mention one or two "
            "minor complaints, as long as the overall verdict is favorable.\n"
            "- NEGATIVE: The reviewer is dissatisfied. They warn others, "
            "complain strongly, express frustration, want a refund, or would "
            "not buy again.\n"
            "- NEUTRAL: The reviewer is genuinely on the fence. Use this when:\n"
            "    (a) the review describes a product as 'okay', 'average', "
            "'nothing special', or 'fine for the price'; OR\n"
            "    (b) the reviewer lists pros AND cons in roughly equal weight "
            "without a clear verdict; OR\n"
            "    (c) the reviewer is informational rather than evaluative.\n"
            "  Do NOT use NEUTRAL just because there are some complaints — "
            "if the overall tone is negative, label it NEGATIVE.\n\n"
            "DECISION RULE: If you are torn between NEUTRAL and another label, "
            "ask yourself: 'would the reviewer buy this product again?' "
            "If clearly yes → POSITIVE. If clearly no → NEGATIVE. "
            "If genuinely unclear or 'maybe' → NEUTRAL.\n\n"
            "ADDITIONAL GUIDANCE: If the reviewer plans to KEEP the product, "
            "accepts a moderate flaw without strong complaint, expresses "
            "uncertainty ('I think', 'I am worried', 'seems okay'), or "
            "describes the experience as 'fine for the price' / 'gets the job "
            "done' — lean NEUTRAL even if they list specific issues. "
            "Reserve NEGATIVE for reviews where the reviewer is clearly "
            "frustrated, regrets the purchase, or actively warns others away.\n\n"
            "Output EXACTLY this format and nothing else:\n"
            "Sentiment: <POSITIVE|NEGATIVE|NEUTRAL>\n"
            "Confidence: <number between 0.0 and 1.0>\n"
            "Justification: <one short sentence>"
        ),
        expected_output='Three lines: Sentiment, Confidence, Justification.',
        agent=classifier,
        context=[analyze_task],
    )

    crew = Crew(
        agents=[analyst, classifier],
        tasks=[analyze_task, classify_task],
        process=Process.sequential,
        verbose=verbose,
    )

    result = crew.kickoff(inputs={'review_text': review_text})
    analysis = str(analyze_task.output) if analyze_task.output else ''
    classification = str(result)
    return analysis, classification


def parse_verdict(classification_text: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """
    Extract Sentiment / Confidence / Justification from the Classifier's text.
    Returns (sentiment, confidence, justification) — any of which can be None on parse failure.
    """
    sentiment = None
    confidence = None
    justification = None
    for line in classification_text.splitlines():
        line = line.strip()
        if line.lower().startswith('sentiment:'):
            value = line.split(':', 1)[1].strip().upper()
            for label in ('POSITIVE', 'NEGATIVE', 'NEUTRAL'):
                if label in value:
                    sentiment = label
                    break
        elif line.lower().startswith('confidence:'):
            try:
                confidence = float(line.split(':', 1)[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.lower().startswith('justification:'):
            justification = line.split(':', 1)[1].strip()
    return sentiment, confidence, justification