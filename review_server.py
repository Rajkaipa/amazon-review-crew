# review_server.py
# MCP server that exposes Amazon reviews loaded from data/reviews.json.

import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Create the server with a human-readable name.
mcp = FastMCP("amazon-review-server")

# Load the dataset once at startup. Using pathlib makes this work
# regardless of which directory you launch the server from.
DATA_PATH = Path(__file__).parent / "data" / "reviews.json"
with open(DATA_PATH, "r", encoding="utf-8") as f:
    REVIEWS: dict[str, str] = json.load(f)


@mcp.tool()
def get_review(review_id: str) -> str:
    """Fetch a single Amazon review by its ID. Returns the full review text."""
    review = REVIEWS.get(review_id)
    if review is None:
        return f"ERROR: no review found for ID '{review_id}'."
    return review


@mcp.tool()
def list_review_ids() -> list[str]:
    """List all available review IDs that can be passed to get_review."""
    return list(REVIEWS.keys())


if __name__ == "__main__":
    # 'stdio' = the server talks over standard input/output.
    # Simplest transport, perfect for local development.
    mcp.run(transport="stdio")