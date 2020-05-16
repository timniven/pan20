import argparse
import importlib

from pan20.util import experiments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--memory_limit', type=int, default=None)
    args = parser.parse_args()

    experiment = importlib.import_module(
        f'{args.task}.experiments.{args.experiment_name}')

    experiments.run(
        experiment=experiment,
        memory_limit=args.memory_limit)
