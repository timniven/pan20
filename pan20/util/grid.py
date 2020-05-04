"""Utilities for experiments."""
import copy
import gc
import itertools
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import glovar
from arct import util
from arct.util import training


# TODO: GridSpace should be an object which can handle my various cases.
# TODO: RandomSearch is a similar case: abstractions should generalize.


def accs_path(experiment_name):
    return os.path.join(glovar.RESULTS_DIR, experiment_name, 'accs.csv')


def attns_path(experiment_name, subset, run_no):
    attns_folder = os.path.join(glovar.RESULTS_DIR, experiment_name, 'attns')
    if not os.path.exists(attns_folder):
        os.mkdir(attns_folder)
    return os.path.join(attns_folder, f'attns_{subset}_run_{run_no}.csv')


def grid_path(experiment_name):
    return os.path.join(glovar.RESULTS_DIR, experiment_name, 'grid.csv')


def mix_weights_path(experiment_name):
    return os.path.join(glovar.RESULTS_DIR, experiment_name, 'mix_weights.csv')


def preds_path(experiment_name):
    return os.path.join(glovar.RESULTS_DIR, experiment_name, 'preds.csv')


def params_path(experiment_name):
    return os.path.join(
        glovar.RESULTS_DIR, experiment_name, 'best_params.json')


class GridSearch:

    def __init__(self, model, cfg, train_loader, dev_loader, search_space):
        self.experiment_name = experiment_name
        self.model_constructor = model_constructor
        self.data_loaders = data_loaders
        self.config = config
        self.search_space = search_space
        self.search_keys = search_space.keys()
        self.grid_path = grid_path(experiment_name)
        self.params_path = params_path(experiment_name)
        self.data, self.columns = self.get_or_load_data()

    def __call__(self):
        tqdm.write('Conducting grid search for %s...' % self.experiment_name)

        for combination in self.combinations:
            if not self.evaluated(combination):
                self.evaluate(combination)
            else:
                print('Already evaluated this combination:')
                for key, value in combination.items():
                    print('\t%s:\t%s' % (key, value))

        best_acc, combinations = self.winning_combinations()
        if best_acc == 0.5:  # i.e. random performance on dev
            tqdm.write('All dev accs are random - taking best train acc.')
            # take the run with the best training acc
            best_acc, combinations = self.winning_train_acc_combinations()
            while len(combinations) > 1:
                tqdm.write('Found %s combinations with best train acc of %s.'
                           % (len(combinations), best_acc))
                tqdm.write('Performing tie break...')
                for _ in range(5):
                    seed = random.choice(range(10000))
                    for combination in combinations:
                        self.evaluate(combination, seed, tie_break=True)
                best_acc, combinations = self.winning_combinations()
        else:
            while len(combinations) > 1:
                tqdm.write('Found %s combinations with best acc of %s.'
                           % (len(combinations), best_acc))
                tqdm.write('Performing tie break...')
                for _ in range(5):
                    seed = random.choice(range(10000))
                    for combination in combinations:
                        self.evaluate(combination, seed, tie_break=True)
                best_acc, combinations = self.winning_combinations()

        best_params = combinations[0]
        tqdm.write('Grid search complete. Best acc: %s. Params:' % best_acc)
        util.aligned_print(
            keys=list(best_params.keys()),
            values=list(best_params.values()),
            indent=1)

        tqdm.write('Saving grid best params...')
        with open(self.params_path, 'w') as f:
            best_config = self.config.copy()
            for key, value in best_params.items():
                setattr(best_config, key, value)
            f.write(json.dumps(best_config.__dict__))

        return best_acc, best_params

    @property
    def combinations(self):
        keys = self.search_space.keys()
        values = list(self.search_space.values())
        i = 0
        for _values in itertools.product(*values):
            combination = dict(zip(keys, _values))
            combination['id'] = i
            i += 1
            yield combination

    def evaluate(self, combination, seed=42, tie_break=False):
        tqdm.write('Evaluating param combination%s:'
                   % ' (tie break)' if tie_break else '')
        args = copy.deepcopy(self.config)
        for key, value in combination.items():
            setattr(args, key, value)
            self.data[key].append(value)
        args.seed = seed
        args.print()
        model = self.model_constructor(args)
        accs, _, __ = training.train(args, model, self.data_loaders)
        self.data['seed'].append(args.seed)
        self.data['train_acc'].append(accs['train'])
        self.data['dev_acc'].append(accs['dev'])
        self.data['test_acc'].append(accs['test'])
        df = pd.DataFrame(data=self.data, columns=self.columns)
        df.to_csv(grid_path(self.experiment_name), index=False)

    def evaluated(self, combination):
        if not os.path.exists(self.grid_path):
            return False
        df = pd.read_csv(self.grid_path)
        for key, value in combination.items():
            if isinstance(value, float):
                df = df[np.isclose(df[key], value)]
            else:
                df = df[df[key] == value]
        return len(df) > 0

    def get_or_load_data(self):
        # init the dict and columns
        data = {'id': []}
        columns = ['id']
        for key in self.search_keys:
            data[key] = []
            columns.append(key)
        data['seed'] = []
        data['train_acc'] = []
        data['dev_acc'] = []
        data['test_acc'] = []
        columns += ['seed', 'train_acc', 'dev_acc', 'test_acc']

        # load any old data
        if os.path.exists(self.grid_path):
            df = pd.read_csv(self.grid_path)
            data['id'] = list(df.id.values)
            for key in self.search_keys:
                data[key] = list(df[key].values)
            data['train_acc'] = list(df.train_acc.values)
            data['dev_acc'] = list(df.dev_acc.values)
            data['test_acc'] = list(df.test_acc.values)
            data['seed'] = list(df.seed.values)

        return data, columns

    @staticmethod
    def get_query(combination):
        query = ''
        for key, value in combination.items():
            if isinstance(value, str):
                value = "'%s'" % value
            else:
                value = str(value)
            query += ' & %s == %s' % (key, value)
        query = query[3:]
        return query

    @staticmethod
    def parse_dict(_dict):
        # wish I didn't need this hack for pandas
        # github issues reckons it should be solved in 24.0?
        keys = _dict.keys()
        values = []
        for value in _dict.values():
            if isinstance(value, np.bool_):
                value = bool(value)
            if isinstance(value, np.float64):
                value = float(value)
            if isinstance(value, np.int64):
                value = int(value)
            values.append(value)
        return dict(zip(keys, values))

    def winning_combinations(self):
        df = pd.read_csv(self.grid_path)
        best_acc = df.dev_acc.max()
        rows = df[df.dev_acc == best_acc]
        wanted_columns = list(self.search_keys) + ['id']
        column_selector = [c in wanted_columns for c in df.columns]
        if len(rows) > 1:  # have a tie break
            ids = rows.id.unique()
            ids_avgs = []
            for _id in ids:
                id_rows = df[df.id == _id]
                avg = id_rows.dev_acc.mean()
                ids_avgs.append((_id, avg))
            best_avg_acc = max(x[1] for x in ids_avgs)
            best_ids = [x[0] for x in ids_avgs if x[1] == best_avg_acc]
            combinations = []
            for _id in best_ids:
                rows = df[df.id == _id].loc[:, column_selector]
                combinations.append(rows.iloc[0].to_dict())
            best_acc = max(best_acc, best_avg_acc)
        else:
            rows = rows.loc[:, column_selector]
            combinations = [r[1].to_dict() for r in rows.iterrows()]
        combinations = [self.parse_dict(d) for d in combinations]
        return best_acc, combinations

    def winning_train_acc_combinations(self):
        df = pd.read_csv(self.grid_path)
        best_acc = df.train_acc.max()
        rows = df[df.train_acc == best_acc]
        wanted_columns = list(self.search_keys) + ['id']
        column_selector = [c in wanted_columns for c in df.columns]
        if len(rows) > 1:  # have a tie break
            ids = rows.id.unique()
            ids_avgs = []
            for _id in ids:
                id_rows = df[df.id == _id]
                avg = id_rows.train_acc.mean()
                ids_avgs.append((_id, avg))
            best_avg_acc = max(x[1] for x in ids_avgs)
            best_ids = [x[0] for x in ids_avgs if x[1] == best_avg_acc]
            combinations = []
            for _id in best_ids:
                rows = df[df.id == _id].loc[:, column_selector]
                combinations.append(rows.iloc[0].to_dict())
            best_acc = max(best_acc, best_avg_acc)
        else:
            rows = rows.loc[:, column_selector]
            combinations = [r[1].to_dict() for r in rows.iterrows()]
        combinations = [self.parse_dict(d) for d in combinations]
        return best_acc, combinations


def to_dict(df):
    data = {}
    for key, value in df.to_dict().items():
        data[key] = list(value.values())
    return data


def run(config, model_constructor, data_loaders_constructor, grid_space,
        n_experiments, train_fn=training.train, do_grid=True):
    # create the experiment folder if it doesn't already exist
    experiment_path = os.path.join(glovar.RESULTS_DIR, config.experiment_name)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    if do_grid:
        # create the data loaders
        data_loaders = data_loaders_constructor(config)

        # conduct grid if best params not already found
        if not os.path.exists(params_path(config.experiment_name)):
            grid_search = GridSearch(
                experiment_name=config.experiment_name,
                model_constructor=model_constructor,
                data_loaders=data_loaders,
                config=config,
                search_space=grid_space)
            best_acc, best_params = grid_search()
        else:
            tqdm.write('Loading best grid params...')
            with open(params_path(config.experiment_name), 'r') as f:
                best_params = json.loads(f.read())

        # merge best params, but keep memory limit
        for attr, value in best_params.items():
            if attr == 'memory_limit':
                continue
            setattr(config, attr, value)

    # run the experiments
    tqdm.write('Running experiments...')

    # load or init new accs and preds data
    _accs_path = accs_path(config.experiment_name)
    if os.path.exists(_accs_path):
        accs = to_dict(pd.read_csv(_accs_path))
    else:
        accs = {
            'run_no': [],
            'seed': [],
            'train': [],
            'dev': [],
            'test': []}
    _preds_path = preds_path(config.experiment_name)
    if os.path.exists(_preds_path):
        preds = to_dict(pd.read_csv(_preds_path))
    else:
        preds = {
            'run_no': [],
            'dataset': [],
            'id': [],
            'prob0': [],
            'prob1': [],
            'pred': [],
            'correct': []}

    # conduct the experiments
    while len(accs['run_no']) < n_experiments:
        run_no = len(accs['run_no']) + 1

        # set seed
        random.seed(run_no)
        config.seed = random.choice(range(10000))
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.device_count() > 0:
            torch.cuda.manual_seed_all(config.seed)

        # create the data loaders - should have been after set seed in
        # original experiments, fixed here
        data_loaders = data_loaders_constructor(config)

        # print info
        tqdm.write('Experiment %s' % run_no)
        config.print()

        model = model_constructor(config)
        _accs, _preds, attentions = train_fn(config, model, data_loaders)

        # update accs
        accs['run_no'].append(run_no)
        accs['seed'].append(config.seed)
        accs['train'].append(_accs['train'])
        accs['dev'].append(_accs['dev'])
        accs['test'].append(_accs['test'])

        # update preds
        for dataset in ['train', 'dev', 'test']:
            for _pred in _preds[dataset]:
                preds['run_no'].append(run_no)
                preds['dataset'].append(dataset)
                preds['id'].append(_pred['id'])
                preds['prob0'].append(_pred['prob0'])
                preds['prob1'].append(_pred['prob1'])
                preds['pred'].append(_pred['pred'])
                preds['correct'].append(_pred['correct'])

        _accs = pd.DataFrame(data=accs, columns=accs.keys())
        _accs.to_csv(_accs_path, index=False)
        _preds = pd.DataFrame(data=preds, columns=preds.keys())
        _preds.to_csv(_preds_path, index=False)

        # update attentions
        # NOTE: don't waste space doing it for MaxLayer models
        if 'layer' not in config.experiment_name:
            for subset in ['train', 'dev', 'test']:
                attns = pd.DataFrame(attentions[subset])
                attns['run_no'] = [run_no] * len(attns)
                attns.to_csv(attns_path(config.experiment_name, subset, run_no),
                             index=False)

        # save mixing weights for MaxLayer models
        if 'layer' in config.experiment_name and config.max_layer > 0:
            # can pull is straight off the model, thankfully
            weights = model.mixing_weights.detach()
            weights = model.softmax(weights)
            weights = list(weights.cpu().numpy()[0])
            row = {
                'experiment_name': config.experiment_name,
                'run_no': str(run_no),
                'max_layers': config.max_layer,
            }
            for layer, weight in enumerate(weights):
                row[f'layer{layer}'] = weight
            weights = pd.DataFrame([row])

            weights_path = mix_weights_path(config.experiment_name)
            if os.path.exists(weights_path):
                df = pd.read_csv(weights_path)
                weights = pd.concat([df, weights], axis=0)
            weights.to_csv(weights_path, index=False)

        # garbage collection - have been getting memory leaks
        del model, data_loaders
        gc.collect()

    # report results
    util.report(accs)
