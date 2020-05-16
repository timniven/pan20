"""Common utilities for PyTorch."""
import inspect
import json

import numpy as np


class Config:
    """Base config class."""

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if isinstance(val, dict):
                setattr(self, key, Config(**val))
            else:
                setattr(self, key, val)

    def __repr__(self):
        cfg = self.to_dict()
        lens = []  # attribute name lengths
        parent_attrs = {k: v for k, v in cfg.items()
                        if not isinstance(v, Config)}
        child_attrs = {k: v for k, v in cfg.items()
                       if isinstance(v, Config)}
        lens += [len(k) for k in parent_attrs]
        lens += [len(k) for k in child_attrs]
        for children in child_attrs.values():
            lens += [len(k) for k in children]
        max_len = np.max(lens)
        cfg = 'Config:\n'

        def pad(attr):
            while len(attr) < max_len:
                attr += ' '
            return attr

        for attr, val in sorted(parent_attrs.items()):
            attr = pad(attr)
            cfg += f'{attr}:\t\t{val}\n'
        for child in sorted(child_attrs.items()):
            cfg += f'{child}\n'
            for attr, val in sorted(child.items()):
                attr = pad(attr)
                cfg += f'{attr}:\t\t{val}'

        return cfg

    def copy(self):
        """Return a copy of this config.

        Useful, e.g., during grid search.

        Returns:
          Config.
        """
        return Config(**self.__dict__)

    @classmethod
    def load(cls, file_path):
        """Load a saved config.

        Args:
          file_path: String, path to the saved config. Must be json data.

        Returns:
          Config.
        """
        with open(file_path) as f:
            params = json.loads(f.read())
        return cls(**params)

    def save(self, file_path):
        """Save config as JSON.

        Args:
          file_path: String. Will be saved as JSON data, so a .json extension
            makes sense.
        """
        with open(file_path, 'w+') as f:
            f.write(json.dumps(self.to_dict()))

    def properties(self):
        def is_property(v):
            return isinstance(v, property)
        return inspect.getmembers(self, is_property)

    def to_dict(self):
        cfg = {}
        for attr, val in self.__dict__.items():
            # NOTE: at most one level of depth in these config trees.
            if isinstance(val, Config):
                cfg[attr] = {}
                for child_attr, child_val in val.__dict__.items():
                    cfg[attr][child_attr] = child_val
                for child_attr, child_val in val.properties():
                    cfg[attr][child_attr] = child_val
            else:
                cfg[attr] = val
        return cfg


class ExperimentConfig(Config):
    """Config for an experiment."""

    def __init__(self, experiment_name, ckpt_dir, results_dir, model, train,
                 anneal, optim, stop, n_runs=20, **kwargs):
        """Create a new Config class.

        Args:
          experiment_name: String.
          ckpt_dir: String, where to save checkpoints and experiment data.
          results_dir: String, where to save experiment results.
          n_runs: Integer, number of training runs. Defaults to 20.
          model: Config object for model specific settings.
          train: Config object for general training settings.
          anneal: Config object for annealing config settings.
          optim: Config object for optimizer config settings.
          stop: Config object for early stopping settings.
          kwargs: for any other desired config settings.
        """
        super().__init__(**kwargs)
        self.experiment_name = experiment_name
        self.ckpt_dir = ckpt_dir
        self.results_dir = results_dir
        self.n_runs = n_runs
        self.model = model
        self.train = train
        self.anneal = anneal
        self.optim = optim
        self.stop = stop
