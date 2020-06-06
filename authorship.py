"""Script to run authorship verification."""
import argparse

from pan20.auth import inputs, models, outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--subset', type=int, default=None)
    args = parser.parse_args()

    data = inputs.parse(args.input_dir, args.subset)
    predictions = models.predict(data)
    outputs.save(predictions, args.output_dir)
