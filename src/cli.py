import argparse
from task import get_task


parser = argparse.ArgumentParser(prog="Fedas Codes Predictor")

subparsers = parser.add_subparsers()

train_parser = subparsers.add_parser('train', help='')
train_parser.add_argument("train_data_path")
train_parser.add_argument("--validation_split", "-v", type=float)
train_parser.add_argument("--model-name", "-n")

infer_parser = subparsers.add_parser('infer', help='')
infer_parser.add_argument("model_checkpoint_path")
infer_parser.add_argument("--infer_data_path", "-i")


def to_kwargs(args):
    return vars(args)


def cli():
    args = parser.parse_args()
    kwargs = to_kwargs(args)
    task = get_task(**kwargs)
    task.run()
