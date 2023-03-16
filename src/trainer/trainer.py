from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.nn.functional import cross_entropy

from conf.conf import get_conf
from model.model import Model

from .metrics import batch_loss, batch_accurracies
from .scheduler import get_scheduler
from .summary import TrainingSummary
from collections import deque


class Trainer:
    def __init__(self, model: Model, summary: TrainingSummary) -> None:
        self.model = model
        self.conf = get_conf("trainer")

        self.optimizer = Adam(
            model.parameters(),
            lr=self.conf["lr"]
        )

        self.summary = summary

    def fit(self, train_dl, test_dl=None, epochs: int = None):

        if epochs is None:
            epochs = self.conf["epochs"]

        self.scheduler = get_scheduler(
            self.optimizer,
            n_steps=epochs*len(train_dl)
        )

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1} / {epochs}")
            self.model.train()

            train_loss = 0
            for batch in tqdm(train_dl):
                loss = batch_loss(self.model, cross_entropy, batch)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.scheduler.step()

                train_loss += loss.detach().cpu().numpy()

            train_loss /= len(train_dl)
            self.summary.update_train_loss(train_loss)

            if test_dl is not None:
                self.model.eval()

                val_loss = 0
                val_acc = 0
                val_sub_acc = 0

                for batch in test_dl:
                    with torch.no_grad():
                        loss = batch_loss(
                            self.model,
                            cross_entropy,
                            batch
                        )

                    val_loss += loss.cpu().numpy()

                    batch_acc, batch_sub_acc = batch_accurracies(
                        self.model, batch)
                    val_acc += batch_acc.cpu().numpy()
                    val_sub_acc += batch_sub_acc.cpu().numpy()

                val_loss /= len(test_dl)
                self.summary.update_val_loss(val_loss)

                val_acc /= len(test_dl)
                val_sub_acc /= len(test_dl)

                self.summary.update_accuracies(val_acc, val_sub_acc)

            print("learning rate :", self.scheduler.get_last_lr())
