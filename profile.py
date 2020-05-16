"""Script to run spreaders of fake news profiling."""
import argparse

from pan20.fake import feats, inputs, models, outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    args = parser.parse_args()

    data = inputs.parse(args.file_path)
    data = feats.early_bird(data)

    model = models.EarlyBird()
    preds = model.pred(data)  # list of 0, 1 as long as len(data)
    preds = [{'author': data.iloc[i].author, 'pred': preds[i]}
             for i in range(len(data))]

    outputs.save(preds)
