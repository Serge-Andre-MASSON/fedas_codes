from tqdm import tqdm
import torch

from model.model import Model
from torch.nn.functional import cross_entropy
from torch.optim import Adam


def batch_loss(model, loss_function, batch):
    x, y_in, y_out = batch
    y = model(x, y_in)

    y = y.view(-1, y.size(-1))
    y_out = y_out.view(-1)

    y_reshaped = y.view(-1, y.size(-1))
    loss = loss_function(y_reshaped, y_out)
    return loss


class Trainer:
    def __init__(self, model: Model) -> None:
        self.model = model
        self.optimizer = Adam(model.parameters())

    def fit(self, train_dl, test_dl=None, epochs=10):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1} / {epochs}")
            losses = []
            self.model.train()
            for batch in tqdm(train_dl):
                loss = batch_loss(self.model, cross_entropy, batch)
                if loss.isnan():
                    print("Alert Nan")
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses.append(loss.detach().cpu().numpy())
            train_loss = sum(losses) / len(losses)
            print(f"train loss  : {train_loss}")

            if test_dl is not None:
                self.model.eval()
                losses = []
                matches = None
                for batch in test_dl:
                    losses.append(batch_loss(self.model, cross_entropy,
                                             batch).detach().cpu().numpy())
                    X, _, y_out = batch
                    expect = y_out[:, :-1]
                    assert expect.size(1) == 4

                    _, pred = self.model.predict(X)
                    match = ((pred-expect) == 0)

                    if matches is None:
                        matches = match
                    else:
                        matches = torch.cat([matches, match])

                matches = matches.float()
                val_loss = sum(losses) / len(losses)

                val_accuracy = (matches.sum(axis=1) == 4).float().mean()
                val_sub_accuracy = matches.mean()

                print(f"validation loss  : {val_loss}")
                print(f"validation accurracy  : {val_accuracy}")
                print(f"validation sub accurracy  : {val_sub_accuracy}")
