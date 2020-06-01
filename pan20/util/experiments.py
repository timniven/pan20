import json
import os

import pandas as pd
from tqdm.notebook import tqdm

from pan20.util.pytorch import metrics, training
from pan20 import util


class Results:
    """Basic wrapper for experiment results."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.metric = metrics.get(cfg.train.metric)
        self.metrics = []
        self.preds = []

    def df_metrics(self):
        return pd.DataFrame(self.metrics)

    def df_preds(self):
        return pd.DataFrame(self.preds)

    @classmethod
    def load(cls, cfg):
        with open(f'{cfg.results_dir}/results.json', 'r') as f:
            results = json.loads(f.read())
        obj = cls(cfg)
        obj.metrics = results['metrics']
        obj.preds = results['preds']
        return obj

    def report_metrics(self, run_no, seed, train, dev=None, test=None):
        results = zip(('train', 'dev', 'test'), (train, dev, test))
        for split, metric in results:
            if metric:
                self.metrics.append({
                    'run_no': run_no,
                    'seed': seed,
                    'split': split,
                    self.metric.abbr: metric,
                })

    def report_preds(self, run_no, seed, train, dev=None, test=None):
        predictions = zip(('train', 'dev', 'test'), (train, dev, test))
        for split, preds in predictions:
            if preds:
                for pred in preds:
                    self.preds.append({
                        'run_no': run_no,
                        'seed': seed,
                        'split': split,
                        'pred': pred,
                    })

    def save(self):
        results = {
            'cfg': self.cfg.to_dict(),
            'metrics': self.metrics,
            'preds': self.preds,
        }
        with open(f'{self.cfg.results_dir}/results.json', 'w+') as f:
            f.write(json.dumps(results))

    def summarize(self):
        df = self.df_metrics()
        train = df[df.split == 'train']
        dev = df[df.split == 'dev']
        test = df[df.split == 'test']
        summary = f'{self.cfg.experiment_name} results:'
        if len(train) > 0:
            summary += f'\tTrain {self.metric.abbr}:'
            summary += '\t\tMax: %5.4f' % train[self.metric.abbr].max()
            summary += '\t\tMean: %5.4f' % train[self.metric.abbr].mean()
            summary += '\t\tStd: %5.4f' % train[self.metric.abbr].std()
        if len(dev) > 0:
            summary += f'\tDev {self.metric.abbr}:'
            summary += '\t\tMax: %5.4f' % dev[self.metric.abbr].max()
            summary += '\t\tMean: %5.4f' % dev[self.metric.abbr].mean()
            summary += '\t\tStd: %5.4f' % dev[self.metric.abbr].std()
        if len(test) > 0:
            summary += f'\tTest {self.metric.abbr}:'
            summary += '\t\tMax: %5.4f' % test[self.metric.abbr].max()
            summary += '\t\tMean: %5.4f' % test[self.metric.abbr].mean()
            summary += '\t\tStd: %5.4f' % test[self.metric.abbr].std()
        return summary


def new_or_load_results(experiment):
    if os.path.exists(f'{experiment.cfg.results_dir}/results.json'):
        return Results.load(experiment.cfg)
    else:
        return Results(experiment.cfg)


def run(experiment, memory_limit):
    # if there is a memory limit, set it on the config
    if memory_limit:
        experiment.cfg.train.memory_limit = memory_limit

    # load the data
    train_loader, dev_loader, test_loader = experiment.dataloaders_fn(
        collate=experiment.collate,
        train_batch_size=experiment.cfg.train.train_batch_size,
        tune_batch_size=experiment.cfg.train.tune_batch_size)

    # do grid search if required
    if experiment.grid_space:
        # TODO: the search
        pass

    results = new_or_load_results(experiment)

    for run_no in range(1, experiment.cfg.n_runs + 1):
        seed = util.new_random_seed()
        util.set_random_seed(seed)

        # init model and train
        model = experiment.model_cls(**experiment.cfg.model)
        model = training.TrainableModel(model, experiment.cfg)
        model.train(train_loader, dev_loader)

        # obtain evaluation results and predictions
        train_metric, train_preds = model.evaluate(train_loader)
        dev_metric, dev_preds = model.evaluate(dev_loader)
        test_metric, test_preds = model.evaluate(test_loader)

        # save results
        results.report_metrics(train_metric, dev_metric, test_metric)
        results.report_preds(train_preds, dev_preds, test_preds)

    # report results
    tqdm.write(results.summarize())
