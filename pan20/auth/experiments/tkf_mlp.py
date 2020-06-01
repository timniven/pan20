"""Top-k n-gram freqs MLP."""
from pan20.auth.models import mlp
from pan20.auth import pytorch
from pan20.util.pytorch import anneal, config, opt, stopping, training


# experiment_name (follows file name)
experiment_name = __name__.split('.')[-1]


# config
cfg = config.ExperimentConfig(
    experiment_name=experiment_name,
    ckpt_dir=f'ckpts/{experiment_name}',
    results_dir=f'results/{experiment_name}',
    n_runs=1,
    model=mlp.Config(
        k1=1024,
        k2=0,
        k3=0,
        # NOTE: in_features defined as k1 + k2 + k3
        hidden_size=512,
        n_layers=1,
        dropout=0.,
        weight_decay=0.
    ),
    train=training.TrainConfig(
        n_epochs=20,
        seed=42,
        train_batch_size=32,
        tune_batch_size=64,
        metric='acc',
        weight_decay=0.
    ),
    anneal=anneal.ReduceLROnPlateauConfig(
        factor=0.5,
        patience=3,
    ),
    optim=opt.AdamWConfig(
        lr=1e-1,
    ),
    stop=stopping.NoDevImprovementConfig(
        patience=3,
        k=3,
        metric='acc'
    )
)


# model
model_cls = mlp.MLP


# dataset
collate = mlp.CollateTopKNGrams(
    k1=cfg.model.k1,
    k2=cfg.model.k2,
    k3=cfg.model.k3)
dataloaders_fn = pytorch.dataloaders_small


# grid space (None to skip grid)
grid_space = None
