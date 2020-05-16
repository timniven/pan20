import functools

from pan20.auth.trans import distrob
from pan20.util.pytorch import anneal, config, opt, stopping, training
from pan20.auth import pytorch


# experiment_name (follows file name)
experiment_name = __name__.split('.')[-1]


# config
cfg = config.ExperimentConfig(
    experiment_name=experiment_name,
    ckpt_dir=f'ckpts/{experiment_name}',
    results_dir=f'results/{experiment_name}',
    lambda_fd=0.5,
    lambda_grad=1.,
    model=(

    ),
    train=training.TrainConfig(
        n_epochs=20,
        seed=42,
        train_batch_size=32,
        tune_batch_size=64,
        p_drop=0.1,
        dev_metric='acc',
        weight_decay=0.
    ),
    anneal=anneal.ReduceLROnPlateauConfig(
        factor=0.5,
        patience=3,
    ),
    optim=opt.AdamWConfig(
        lr=6e-5,
    ),
    stop=stopping.NoDevImprovementConfig(
        patience=3,
        k=3,
        metric='acc'
    )
)


# model
model_cls = distrob.DistilRoBERTaComparison1


# dataset
collate = distrob.CollateFirstK()
dataloaders_fn = pytorch.dataloaders_small


# grid space (None to skip grid)
grid_space = None
