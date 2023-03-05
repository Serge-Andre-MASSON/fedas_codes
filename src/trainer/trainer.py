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

    def fit(self, train_dl, epochs=10):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1} / {epochs}")
            for batch in tqdm(train_dl):
                loss = batch_loss(self.model, cross_entropy, batch)
                if loss.isnan():
                    print("Alert Nan")
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            print(f"loss  : {loss}")
