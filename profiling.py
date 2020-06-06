"""Script to run spreaders of fake news profiling."""
import argparse

from pan20.fake import inputs, models, outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()

    data = inputs.parse(args.file_path)
    preds = models.predict(data)
    outputs.save(preds, args.output_dir)
