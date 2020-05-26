"""Script to run spreaders of fake news profiling."""
import argparse

from pan20.fake import features, inputs, models, outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()

    data = inputs.parse(args.file_path)
    feats = features.early_bird(data)

    model = models.early_bird()
    preds = model.predict(feats.values)  # list of 0, 1 as long as len(data)
    preds = [{'author': data.iloc[i].author, 'pred': preds[i]}
             for i in range(len(data))]

    outputs.save(preds, args.output_dir)
