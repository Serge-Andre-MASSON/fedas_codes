from dataclasses import dataclass, field


@dataclass
class TrainingSummary:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    val_sub_accuracy: list[float] = field(default_factory=list)

    def update_train_loss(self, loss):
        self.train_loss.append(loss)
        print(f"train loss  : {loss}")

    def update_val_loss(self, loss):
        self.val_loss.append(loss)
        print(f"validation loss  : {loss}")

    def update_accuracies(self, acc, sub_acc):
        self.val_accuracy.append(acc)
        self.val_sub_accuracy.append(sub_acc)

        print(f"validation accurracy  : {acc}")
        print(f"validation sub accurracy  : {sub_acc}")
