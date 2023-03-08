from tqdm import tqdm

from model.model import Model
from torch.nn.functional import cross_entropy
from torch.optim import Adam


# TODO: Read more about pytorch-lightning. Maybe useful.


def batch_loss(model, loss_function, batch, max_output_length=5):
    x, y_in, y_out = batch
    y = model(x, y_in)

    l = max_output_length

    y = y[:, -l, :].contiguous().view(-1, y.size(-1))
    y_out = y_out[:, -l].contiguous().view(-1)

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
                for batch in test_dl:
                    losses.append(batch_loss(self.model, cross_entropy,
                                             batch).detach().cpu().numpy())
                val_loss = sum(losses) / len(losses)

                print(f"validation loss  : {val_loss}")
            self.model.train()
