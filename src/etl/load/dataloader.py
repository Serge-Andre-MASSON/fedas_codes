import torch
from torch.utils.data import DataLoader, Dataset


class DS(Dataset):
    def __init__(self, X, y) -> None:
        device = self.get_device()  # TODO:  May be moved somewhere else
        X = torch.tensor(X).long().to(device)
        y = torch.tensor(y).long().to(device)

        self.X = X
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


# TODO: Add a conf file for batch_size and shuffle and other stuff
def get_dataloader(X, y, batch_size=2048, shuffle=True):
    return DS(X, y).to_loader(batch_size, shuffle)
