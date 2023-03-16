def batch_loss(model, loss_function, batch):
    x, y_in, y_out = batch
    y = model(x, y_in)

    y = y.view(-1, y.size(-1))
    y_out = y_out.view(-1)

    y_reshaped = y.view(-1, y.size(-1))
    loss = loss_function(y_reshaped, y_out)
    return loss


def batch_accurracies(model, batch):
    X, _, y_out = batch
    expect = y_out[:, :-1]

    _, pred = model.predict(X)
    match = ((pred-expect) == 0).float()

    val_accuracy = (match.sum(axis=1) == 4).float().mean()
    val_sub_accuracy = match.mean()
    return val_accuracy, val_sub_accuracy
