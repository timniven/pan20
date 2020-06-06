"""Saving outputs for software submission."""
import json
import os


def save(preds, output_dir):
    """Save outputs for software submission.

    Args:
      preds: List of dictionaries.
      output_dir: String. Saves preds as jsonl to output_dir/answers.jsonl.
    """
    file_path = os.path.join(output_dir, 'answers.jsonl')
    preds = [json.dumps(p) for p in preds]
    preds = '\n'.join(preds)
    with open(file_path, 'w+') as f:
        f.write(preds)
