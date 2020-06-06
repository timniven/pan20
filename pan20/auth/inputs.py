"""For managing file inputs for software submission."""
import json
import os


def parse(input_dir, subset):
    file_path = os.path.join(input_dir, 'pairs.jsonl')
    i = 0
    with open(file_path) as f:
        for line in f.readlines():
            if i >= subset:
                raise StopIteration
            x = json.loads(line)
            i += 1
            yield x
