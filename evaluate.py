# evaluate.py
# Run the crew against every (or N) reviews in data/reviews.json
# and compare predictions to data/labels.json. Prints accuracy,
# a confusion matrix, per-class metrics, and writes a JSON report.

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from crew_lib import analyze_review_text, parse_verdict

DATA_DIR = Path(__file__).parent / 'data'
REPORT_PATH = Path(__file__).parent / 'evaluation_report.json'

LABELS = ('POSITIVE', 'NEGATIVE', 'NEUTRAL')


def load_data():
    with open(DATA_DIR / 'reviews.json', encoding='utf-8') as f:
        reviews = json.load(f)
    with open(DATA_DIR / 'labels.json', encoding='utf-8') as f:
        labels = json.load(f)
    return reviews, labels


def evaluate(limit: int | None) -> dict:
    reviews, labels = load_data()
    ids = list(reviews.keys())
    if limit:
        step = max(1, len(ids) // limit)
        ids = ids[::step][:limit]

    results = []                # per-review outcomes
    confusion = defaultdict(int)   # (true_label, pred_label) -> count
    correct = 0
    parse_failures = 0
    started = time.time()

    print(f'Running evaluation on {len(ids)} reviews...\n')

    for i, review_id in enumerate(ids, start=1):
        text = reviews[review_id]
        true_label = labels[review_id]['label']
        true_rating = labels[review_id]['rating']

        t0 = time.time()
        try:
            _, classification = analyze_review_text(text, verbose=False)
            pred, confidence, justification = parse_verdict(classification)
        except Exception as e:
            pred, confidence, justification = None, None, f'ERROR: {e}'
        elapsed = time.time() - t0

        if pred is None:
            parse_failures += 1
            confusion[(true_label, 'PARSE_FAIL')] += 1
        else:
            confusion[(true_label, pred)] += 1
            if pred == true_label:
                correct += 1

        is_correct = pred == true_label
        results.append({
            'id': review_id,
            'rating': true_rating,
            'true_label': true_label,
            'predicted': pred,
            'confidence': confidence,
            'justification': justification,
            'correct': is_correct,
            'seconds': round(elapsed, 1),
        })

        marker = '✓' if is_correct else '✗'
        print(f'  [{i:>3}/{len(ids)}] id={review_id:>3}  {true_label:>8} → {str(pred):>8}  {marker}  ({elapsed:.1f}s)')

    total_time = time.time() - started

    # --- Summary stats ---
    accuracy = correct / len(ids) if ids else 0.0

    # Per-class precision / recall
    per_class = {}
    for label in LABELS:
        tp = confusion[(label, label)]
        fp = sum(confusion[(other, label)] for other in LABELS if other != label)
        fn = sum(confusion[(label, other)] for other in LABELS if other != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        per_class[label] = {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'support': sum(confusion[(label, p)] for p in LABELS + ('PARSE_FAIL',)),
        }

    # --- Print report ---
    print('\n' + '=' * 60)
    print('EVALUATION REPORT')
    print('=' * 60)
    print(f'Total reviews: {len(ids)}')
    print(f'Correct: {correct}  |  Parse failures: {parse_failures}')
    print(f'Accuracy: {accuracy:.1%}')
    print(f'Total time: {total_time:.0f}s  |  Avg per review: {total_time / len(ids):.1f}s')
    print('\nConfusion matrix (rows = true, columns = predicted):')
    cols = LABELS + ('PARSE_FAIL',)
    print(f'  {"":>10} {" ".join(f"{c:>11}" for c in cols)}')
    for true in LABELS:
        row = ' '.join(f'{confusion[(true, p)]:>11}' for p in cols)
        print(f'  {true:>10} {row}')
    print('\nPer-class metrics:')
    for label, m in per_class.items():
        print(f'  {label:>8}:  precision={m["precision"]:.2f}  recall={m["recall"]:.2f}  support={m["support"]}')

    # --- Save full report ---
    report = {
        'summary': {
            'total': len(ids),
            'correct': correct,
            'accuracy': round(accuracy, 4),
            'parse_failures': parse_failures,
            'total_seconds': round(total_time, 1),
            'avg_seconds_per_review': round(total_time / len(ids), 2),
        },
        'per_class': per_class,
        'confusion_matrix': {f'{t}_vs_{p}': c for (t, p), c in confusion.items()},
        'results': results,
    }
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'\nFull report saved to {REPORT_PATH.name}')

    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None,
                        help='Evaluate only the first N reviews (default: all).')
    args = parser.parse_args()
    evaluate(args.limit)