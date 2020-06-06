"""For managing file inputs for software submission."""
import json
import os


def parse(input_dir):
    file_path = os.path.join(input_dir, 'pairs.jsonl')
    with open(file_path) as f:
        for line in f.readlines():
            x = json.loads(line)
            yield x
