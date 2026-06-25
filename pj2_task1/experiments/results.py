"""
Results tracking: CSV read/write, summary tables, comparison charts.
"""

import csv
import os
import json
from typing import Dict, List
import numpy as np


RESULTS_CSV_PATH = os.path.join(os.path.dirname(__file__), 'results.csv')

CSV_COLUMNS = [
    'experiment_id', 'phase', 'model_name', 'activation', 'loss_fn',
    'optimizer_name', 'lr', 'weight_decay', 'momentum', 'label_smoothing',
    'epochs', 'batch_size', 'num_params', 'best_val_acc', 'test_acc',
    'test_loss', 'train_time_seconds', 'seed',
]


def save_result(result: Dict, csv_path: str = None):
    """Append one experiment result to CSV."""
    if csv_path is None:
        csv_path = RESULTS_CSV_PATH

    # Extract rows
    row = {col: result.get(col, '') for col in CSV_COLUMNS}

    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # Save full history as JSON alongside
    json_path = csv_path.replace('.csv', f"_{result['experiment_id']}.json")
    json_data = {k: v for k, v in result.items()
                 if k not in ['model_class']}
    # Convert non-serializable values
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_safe = {k: convert(v) for k, v in json_data.items()}
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_safe, f, indent=2, default=str)


def load_results(csv_path: str = None) -> List[Dict]:
    """Load all results from CSV."""
    if csv_path is None:
        csv_path = RESULTS_CSV_PATH
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_best_result(phase: str = None, csv_path: str = None) -> Dict:
    """Get best result by test accuracy, optionally filtered by phase."""
    results = load_results(csv_path)
    if phase:
        results = [r for r in results if r.get('phase') == phase]
    if not results:
        return {}
    return max(results, key=lambda r: float(r.get('test_acc', 0)))


def generate_summary_table(csv_path: str = None) -> str:
    """Generate a markdown summary table of all results."""
    results = load_results(csv_path)
    if not results:
        return "No results found."

    lines = [
        "| Experiment | Phase | Model | Params | Val Acc | Test Acc | Loss | Time(s) |",
        "|-----------|-------|-------|--------|---------|----------|------|---------|",
    ]
    for r in results:
        lines.append(
            f"| {r['experiment_id']} | {r['phase']} | {r['model_name']} | "
            f"{int(float(r['num_params'])):,} | {float(r['best_val_acc']):.4f} | "
            f"{float(r['test_acc']):.4f} | {float(r['test_loss']):.4f} | "
            f"{float(r['train_time_seconds']):.0f} |"
        )
    return '\n'.join(lines)
