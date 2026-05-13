# Amazon Review Sentiment Crew

**Status:** 73.3% accuracy on a balanced 60-review test set — see [Evaluation](#evaluation).

A beginner-friendly multi-agent project using **CrewAI** + **MCP** + **OpenRouter** to classify Amazon reviews as Positive, Negative, or Neutral.

## Architecture

A 3-agent sequential pipeline:

1. **Fetcher** — pulls a review by ID via an MCP server tool.
2. **Analyst** — produces a structured breakdown (topic, praises, complaints, tone, factual claims).
3. **Classifier** — outputs the final sentiment label with a confidence score.

## Project structure

- `data/reviews.json` — 200 real Amazon reviews from Hugging Face (`SetFit/amazon_reviews_multi_en`)
- `data/labels.json` — ground-truth ratings and sentiment labels for evaluation
- `prepare_data.py` — one-time script to download and sample the dataset
- `review_server.py` — MCP server exposing `get_review` and `list_review_ids` tools
- `crew_lib.py` — shared crew-building logic (used by `full_crew.py` and `evaluate.py`)
- `fetcher.py` — standalone Fetcher script (MCP-using)
- `full_crew.py` — full 3-agent pipeline
- `evaluate.py` — runs the crew against the dataset and reports accuracy / precision / recall
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
- OpenRouter as the LLM gateway (currently `openai/gpt-oss-20b:free`)

## Evaluation

The system was evaluated against a balanced subset of 60 real Amazon reviews
drawn from `SetFit/amazon_reviews_multi_en` on Hugging Face — 20 each of
NEGATIVE (1-2 star), NEUTRAL (3 star), and POSITIVE (4-5 star).

### Results

| Metric | Baseline | After Prompt Iteration |
|---|---:|---:|
| Overall accuracy | 71.7% | **73.3%** |
| Parse failures | 0 | 0 |
| POSITIVE — precision / recall | 0.93 / 0.70 | 0.89 / **0.80** |
| NEGATIVE — precision / recall | 0.68 / 0.96 | 0.71 / **1.00** |
| NEUTRAL — precision / recall | 0.43 / 0.23 | 0.25 / 0.08 |
| Avg time per review | 59.6s | 31.8s |

### What the iteration changed

The Classifier prompt was extended with explicit label definitions, a
decision rule ("would the reviewer buy this again?"), and guidance to
reserve NEGATIVE for clearly frustrated reviewers. This improved POSITIVE
recall by 10 points and brought NEGATIVE recall to 100%, but pushed
NEUTRAL detection in the wrong direction — a textbook case of prompt
fragility where fixing one class shifts confusion elsewhere.

### Known limitations

- **Label noise:** Amazon's 1-5 star ratings don't perfectly track sentiment
  in the review text. Manual inspection of "wrong" NEUTRAL predictions
  shows many are defensible reads — reviewers giving 3 stars to text that
  reads as clearly negative.
- **NEUTRAL is genuinely hard:** with the current data and model, the
  classifier collapses much of the NEUTRAL class into NEGATIVE. Further
  prompt-tuning on the same free-tier model is hitting diminishing returns.
- **Free-tier latency:** ~32 seconds per review on `gpt-oss-20b:free`,
  with occasional rate-limit pauses. A paid frontier model
  (Gemini Flash / Claude Haiku / GPT-4o-mini) would likely close most of
  the NEUTRAL gap and run ~10× faster.

### Reproducing the evaluation

```bash
python prepare_data.py        # downloads 200 reviews from Hugging Face
python evaluate.py --limit 60 # runs balanced subset, prints metrics
```

The full report (per-review predictions, justifications, timings) is saved
to `evaluation_report.json`.