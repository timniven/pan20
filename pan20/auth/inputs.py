"""For managing file inputs for software submission."""
import json
import os


def parse(input_dir, subset):
    file_path = os.path.join(input_dir, 'pairs.jsonl')
    i = 0
    with open(file_path) as f:
        data = [json.loads(x) for x in f.readlines()]
        if subset:
            return data[0:subset]
        return data
