# amazon-review-crew
Beginner CrewAI + MCP project for review sentiment analysis
# Amazon Review Sentiment Crew

A beginner-friendly multi-agent project using **CrewAI** + **MCP** + **OpenRouter** to classify Amazon reviews as Positive, Negative, or Neutral.

## Architecture

A 3-agent sequential pipeline:

1. **Fetcher** — pulls a review by ID via an MCP server tool.
2. **Analyst** — produces a structured breakdown (topic, praises, complaints, tone, factual claims).
3. **Classifier** — outputs the final sentiment label with a confidence score.

## Project structure

- `data/reviews.json` — synthetic review dataset
- `review_server.py` — MCP server exposing `get_review` and `list_review_ids` tools
- `fetcher.py` — standalone Fetcher script (MCP-using)
- `full_crew.py` — full 3-agent pipeline
- `notebooks/01_analyst_agent.ipynb` — Analyst in isolation
- `notebooks/02_classifier_agent.ipynb` — Classifier in isolation

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env and paste your OpenRouter key
python full_crew.py
```

## Tech

- CrewAI for agent orchestration
- Model Context Protocol (MCP) for the review-fetching tool
- OpenRouter as the LLM gateway (currently `openai/gpt-oss-120b:free`)

## Status

Learning project — not production.

