# prepare_data.py
# One-time utility: download real Amazon reviews from Hugging Face,
# sample evenly across 1-5 star ratings, save to data/reviews.json
# (for the MCP server) and data/labels.json (for evaluation).

import json
from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)


def star_to_label(rating: int) -> str:
    if rating <= 2:
        return "NEGATIVE"
    elif rating == 3:
        return "NEUTRAL"
    else:
        return "POSITIVE"


# SetFit/amazon_reviews_multi_en: English subset, balanced across 5 stars,
# auto-converted to Parquet so it works without dataset scripts.
print("Loading dataset from Hugging Face...")
ds = load_dataset("SetFit/amazon_reviews_multi_en", split="train", streaming=True)

TARGET_PER_STAR = 40
MIN_LEN = 30
MAX_LEN = 1500
buckets = {1: [], 2: [], 3: [], 4: [], 5: []}

print(f"Sampling up to {TARGET_PER_STAR} reviews per star rating...")
for row in ds:
    # SetFit/amazon_reviews_multi_en uses 'label' with values 0-4 (0 = 1-star, 4 = 5-star).
    # Some other Amazon datasets use 'stars' directly as 1-5. Handle both.
    if "stars" in row:
        star = int(row["stars"])
    elif "label" in row:
        star = int(row["label"]) + 1     # shift 0-4 -> 1-5
    else:
        continue

    if star not in buckets or len(buckets[star]) >= TARGET_PER_STAR:
        continue

    # Text might be under 'text' or 'review_body'
    text = (row.get("text") or row.get("review_body") or "").strip()
    if MIN_LEN <= len(text) <= MAX_LEN:
        buckets[star].append({"text": text, "rating": star})

    if all(len(v) >= TARGET_PER_STAR for v in buckets.values()):
        break

reviews = {}
labels = {}
review_id = 1
for star in [1, 2, 3, 4, 5]:
    for item in buckets[star]:
        rid = str(review_id)
        reviews[rid] = item["text"]
        labels[rid] = {"rating": item["rating"], "label": star_to_label(item["rating"])}
        review_id += 1

with open(OUT_DIR / "reviews.json", "w", encoding="utf-8") as f:
    json.dump(reviews, f, indent=2, ensure_ascii=False)
with open(OUT_DIR / "labels.json", "w", encoding="utf-8") as f:
    json.dump(labels, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(reviews)} reviews to data/reviews.json")
print(f"Saved {len(labels)} labels to data/labels.json")
print("\nDistribution by star rating:")
for star in [1, 2, 3, 4, 5]:
    print(f"  {star} stars: {sum(1 for v in labels.values() if v['rating'] == star)}")
print("\nDistribution by sentiment label:")
for label in ["NEGATIVE", "NEUTRAL", "POSITIVE"]:
    print(f"  {label}: {sum(1 for v in labels.values() if v['label'] == label)}")