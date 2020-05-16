"""PyTorch training utilities."""
import os

from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm

from . import config
from . import anneal, metrics, opt, stopping


class TrainConfig(config.Config):
    """Config class for common training settings."""

    def __init__(self, n_epochs, seed, train_batch_size, p_drop, metric,
                 tune_batch_size, run_no=1, memory_limit=None, no_cuda=False,
                 **kwargs):
        super().__init__(**kwargs)
        # NOTE: this is a target, but account for memory limit, so use property
        self._train_batch_size = train_batch_size
        self.p_drop = p_drop
        self.no_cuda = no_cuda
        self.seed = seed
        self.run_no = run_no
        self.n_epochs = n_epochs
        self._tune_batch_size = tune_batch_size
        self.metric = metric
        self._memory_limit = memory_limit

    @property
    def grad_accum_steps(self):
        # this is a function of batch_size and memory limits for specific models
        # the memory limits are also computer (i.e. GPU) dependent.
        # the limits are controlled in this config by memory_limit.
        n_steps = max(int(self._train_batch_size / self.memory_limit), 1)
        if n_steps == 0:
            raise ValueError(
                'Erroring here: gradient_accumulation_steps should be '
                'greater than zero.\n'
                f'\ttrain_batch_size: {self._train_batch_size}\t'
                f'\tmemory_limit" {self.memory_limit}')
        return n_steps

    @property
    def memory_limit(self):
        if self._memory_limit:
            return self._memory_limit
        else:
            return max(self._train_batch_size, self._tune_batch_size)

    @property
    def train_batch_size(self):
        return int(self._train_batch_size / self.grad_accum_steps)

    @property
    def tune_batch_size(self):
        return int(self._tune_batch_size / self.grad_accum_steps)


class Batch:
    """Base class for a batch."""

    def __init__(self, **kwargs):
        for attr, val in kwargs.items():
            setattr(self, attr, val)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __iter__(self):
        for key in self.__dict__.keys():
            yield key

    def __len__(self):
        return len(self.__dict__)

    def __contains__(self, item):
        return item in self.__dict__

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def get(self, item):
        return self.__dict__[item]

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return False

    def to(self, device):
        """Calls to(device) on all torch.Tensors on this batch object."""
        for attr, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))


class ClassificationBatch(Batch):
    """Base class for a batch for a classification task."""

    def __init__(self, labels, **kwargs):
        self.labels = labels
        super().__init__(**kwargs)


class TrainableModel:
    """Training wrapper for a nn.Module."""

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.metric = metrics.get(cfg.train.metric)
        self.train_state = TrainState()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.saver = Saver(ckpt_dir=self.cfg.ckpt_dir)
        self.stop = stopping.get(cfg.stop)

    def train(self, train_loader, dev_loader):
        # make ckpts dir if not existing
        if not os.path.exists(self.cfg.ckpt_dir):
            os.mkdir(self.cfg.ckpt_dir)

        # use cuda if available
        self.model.to(self.device)

        # initialize the optimizer
        # NOTE: models have to implement optim_params() method
        optimizer = opt.get(self.cfg.optim, self.model.optim_params())

        # lr schedule
        annealing = anneal.get(self.cfg.anneal, optimizer)

        # variables for training loop
        self.train_state.reset()
        epoch_pbar = TqdmWrapper(self.cfg.train.n_epochs, 'epoch')
        iter_pbar = TqdmWrapper(len(train_loader), 'iter')
        eval_pbar = TqdmWrapper(len(dev_loader), 'tune')

        # training loop
        self.model.train()
        # start an epoch
        for epoch in range(1, self.cfg.train.n_epochs + 1):
            self.train_state.epoch += 1
            iter_pbar.restart()

            # start an iter
            for batch in train_loader:
                self.train_state.step += 1
                batch.to(self.device)
                loss, logits = self.model(**batch)
                preds = logits.max(dim=1).indices.detach().cpu().numpy()
                labels = batch.labels.detach().cpu().numpy()
                tr_metric = self.metric(labels, preds)
                if self.cfg.train.grad_accum_steps > 1:
                    loss = loss / self.cfg.train.grad_accum_steps
                loss.backward()
                self.train_state.n_tr_x += batch.labels.size(0)
                self.train_state.cum_tr_metric += tr_metric
                self.train_state.cum_tr_loss += loss.detach().cpu().numpy()

                if (self.train_state.step + 1) \
                        % self.cfg.train.grad_accum_steps == 0:
                    optimizer.step()
                    self.model.zero_grad()
                    metric = self.train_state.cum_tr_metric \
                             / self.train_state.step
                    loss = self.train_state.cum_tr_loss \
                           / self.train_state.n_tr_x
                    self.train_state.train_metrics.append({
                        'step': self.train_state.step,
                        self.metric.abbr: self.train_state.cum_tr_metric
                                          / self.train_state.step,
                    })
                    self.train_state.train_losses.append({
                        'step': self.train_state.step,
                        'loss': self.train_state.cum_tr_loss
                                / self.train_state.n_tr_x,
                    })
                    iter_pbar.set_description(
                        '(J: %4.3f, M: %3.2f)' % (loss, metric))

                # thus ends an iteration
                iter_pbar.update()

            # tuning
            dev_metric, _, = self.evaluate(dev_loader, eval_pbar)
            self.train_state.dev_metrics.append({
                'step': self.train_state.step,
                self.metric.abbr: dev_metric
            })
            is_best = self.metric.is_best(
                score=dev_metric, scores=self.train_state.dev_metrics)

            # learning rate annealing
            if self.cfg.anneal.epoch:
                annealing.step(dev_metric)

            # plot diagnostics
            self.plot_diagnostics()

            # save params
            self.saver.save(self.model, self.cfg.experiment_name, is_best)

            # thus ends an epoch
            epoch_pbar.update()
            epoch_pbar.set_description(
                '(best: %5.2f)'
                % self.cfg.metric.best(self.train_state.dev_metrics))

            # early stopping
            stop, message = self.stop(self.train_state)
            if stop:
                tqdm.write(message)
                break

        # report end of training and load best model
        tqdm.write('Training completed.')
        self.load_best()

    def load_best(self):
        # TODO: will raise error if no ckpt, should have readable error message
        self.saver.load(
            model=self.model,
            name=self.cfg.experiment_name,
            is_best=True,
            load_optimizer=False)

    def evaluate(self, data_loader, pbar=None):
        self.model.eval()
        cum_metric = 0.
        n_steps, n_x = 0, 0
        predictions = []
        if not pbar:
            pbar = TqdmWrapper(len(data_loader), 'eval')
        else:
            pbar.restart()

        for i, batch in enumerate(data_loader):
            i += 1
            batch.to(self.device)

            with torch.no_grad():
                _, logits = self.model(**batch)

            n_x += batch.labels.size(0)
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
            preds = logits.max(dim=1).indices.detach().cpu().numpy()
            labels = batch.labels.detach().cpu().numpy()
            tmp_metric = self.metric(labels, preds)
            cum_metric += tmp_metric
            n_steps += 1
            correct = preds == labels

            # TODO: make more general
            # for i in range(len(batch)):
            #     predictions.append({
            #         'prob0': probs[i][0],
            #         'prob1': probs[i][1],
            #         'pred': preds[i],
            #         'correct': correct[i]})

            pbar.update()
            pbar.set_description('(%5.2f)' % (cum_metric / i))

        metric = cum_metric / len(data_loader)

        return metric, predictions

    def plot_diagnostics(self):
        # train losses
        x1 = [x['step'] for x in self.train_state.train_losses]
        y1 = [x['loss'] for x in self.train_state.train_losses]
        plt.plot(x1, y1)
        plt.title('Train Loss')
        plt.ylabel('loss')
        plt.xlabel('global step')
        loss_path = os.path.join(self.cfg.ckpt_dir, 'loss.png')
        plt.savefig(loss_path)
        plt.clf()

        # train and dev metrics
        x1 = [x['step'] for x in self.train_state.train_metrics]
        y1 = [x[self.metric.abbr] for x in self.train_state.train_metrics]
        plt.plot(x1, y1)
        x1 = [x['step'] for x in self.train_state.dev_metrics]
        y1 = [x[self.metric.abbr] for x in self.train_state.dev_metrics]
        plt.plot(x1, y1)
        metrics_path = os.path.join(self.cfg.ckpt_dir, 'metrics.png')
        plt.savefig(metrics_path)
        plt.clf()


class Saver:

    def __init__(self, ckpt_dir):
        self.ckpt_dir = ckpt_dir

    def ckpt_path(self, name, module, is_best):
        return os.path.join(
            self.ckpt_dir,
            '%s_%s_%s' % (name, module, 'best' if is_best else 'latest'))

    def load(self, model, name, is_best=False, load_to_cpu=False,
             load_optimizer=True, replace_model=None, ignore_missing=False):
        model_path = self.ckpt_path(name, 'model', is_best)
        model_state_dict = self.get_state_dict(model_path, load_to_cpu)
        model_state_dict = self.replace_model_state(
            model_state_dict, replace_model)
        if ignore_missing:
            model_state_dict = self.drop_missing(model, model_state_dict)
        model.load_state_dict(model_state_dict)
        if load_optimizer:
            optim_path = self.ckpt_path(name, 'optim', is_best)
            optim_state_dict = self.get_state_dict(optim_path, load_to_cpu)
            model.optimizer.load_state_dict(optim_state_dict)

    @staticmethod
    def drop_missing(model, saved_state_dict):
        return {k: v for k, v in saved_state_dict.items()
                if k in model.state_dict().keys()}

    @staticmethod
    def replace_model_state(state_dict, replace):
        if replace is not None:
            for name, tensor in replace.items():
                state_dict[name] = tensor
        return state_dict

    @staticmethod
    def filter_optim_state_dict(state_dict, exclude):
        if exclude is not None:
            raise NotImplementedError  # TODO
        else:
            return state_dict

    @staticmethod
    def get_state_dict(path, load_to_cpu):
        if not torch.cuda.is_available() or load_to_cpu:
            return torch.load(path, map_location=lambda storage, loc: storage)
        else:
            return torch.load(path)

    def save(self, model, name, is_best, save_optim=False):
        model_path = self.ckpt_path(name, 'model', False)
        torch.save(model.state_dict(), model_path)
        if is_best:
            model_path = self.ckpt_path(name, 'model', True)
            torch.save(model.state_dict(), model_path)
        if save_optim:
            optim_path = self.ckpt_path(name, 'optim', is_best)
            torch.save(model.optimizer.state_dict(), optim_path)


class TqdmWrapper:
    """Wraps tqdm progress bars with a restart method and base description."""

    def __init__(self, total, desc):
        self.pbar = tqdm(total=total, desc=desc)
        self.base_desc = desc

    def reset(self, total=None):
        self.pbar.reset(total)
        self.restart()

    def restart(self):
        self.pbar.n = 0
        self.pbar.last_print_n = 0
        self.pbar.refresh()
        self.pbar.set_description(self.base_desc)

    def set_description(self, desc):
        desc = f'{self.base_desc}: {desc}'
        self.pbar.set_description(desc)

    def update(self):
        self.pbar.update()


class TrainState:
    """Wraps info about state of training."""

    def __init__(self):
        self.epoch = 0
        self.step = 0
        self.n_tr_x = 0
        self.cum_tr_loss = 0.
        self.cum_tr_metric = 0.
        self.train_losses = []
        self.train_metrics = []
        self.dev_metrics = []

    def reset(self):
        self.epoch = 0
        self.step = 0
        self.n_tr_x = 0
        self.cum_tr_loss = 0.
        self.cum_tr_metric = 0.
        self.train_losses = []
        self.train_metrics = []
        self.dev_metrics = []

    # TODO: load and save
