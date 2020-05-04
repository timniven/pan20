"""Optimization config and helpers."""
from torch import optim
import transformers

from . import config


def get(cfg, model_parameters):
    if cfg.optimizer == 'adam':
        return optim.Adam(
            params=model_parameters,
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay)
    if cfg.optimizer == 'adamw':
        return transformers.AdamW(
            params=model_parameters,
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            correct_bias=cfg.correct_bias)
    else:
        raise ValueError(f'Unexpected optimizer: {cfg.optimizer}')


class OptimizerConfig(config.Config):
    """Base class for optimization config."""

    def __init__(self, optimizer, lr, weight_decay):
        super().__init__()
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay


class AdamConfig(OptimizerConfig):
    """Config for Adam optimization."""

    def __init__(self, lr, weight_decay=0., beta1=0.9, beta2=0.999, eps=1e-08):
        super().__init__(
            optimizer='adam',
            lr=lr,
            weight_decay=weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps


class AdamWConfig(OptimizerConfig):
    """Config for AdamW."""

    def __init__(self, lr, weight_decay=0., beta1=0.9, beta2=0.999,
                 eps=1e-08, correct_bias=True):
        super().__init__(
            optimizer='adamw',
            lr=lr,
            weight_decay=weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.correct_bias = correct_bias
