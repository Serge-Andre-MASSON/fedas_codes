from os.path import dirname
from pathlib import Path
from yaml import load, Loader
import torch
from torch import LongTensor
from torch.utils.data import DataLoader, Dataset

CONF_PATH = Path(f"{dirname(__file__)}") / "default_dl.yaml"


class TrainDS(Dataset):
    def __init__(self, X, y) -> None:
        device = self.get_device()
        self.X = torch.tensor(X).long().to(device)
        y = torch.tensor(y).long().to(device)

        self.y_in = y[:, :-1]
        self.y_out = y[:, 1:]

    def get_device(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def __getitem__(self, index):
        return self.X[index], self.y_in[index], self.y_out[index]

    def __len__(self):
        return self.X.size(0)

    def to_loader(self, batch_size, shuffle):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


class ValDS(Dataset):
    def __init__(self, X) -> None:
        device = self.get_device()  # TODO:  May be moved somewhere else
        self.X = torch.tensor(X).long().to(device)

    def get_device(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return self.X.size(0)

    def to_loader(self, batch_size, shuffle):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


def get_dataloader(X: LongTensor, y: LongTensor = None, conf_path=CONF_PATH):
    with open(conf_path) as f:
        conf = load(f, Loader)

    if y is None:
        conf = conf["inference"]
        return ValDS(X).to_loader(**conf)

    conf = conf["train"]
    return TrainDS(X, y).to_loader(**conf)
