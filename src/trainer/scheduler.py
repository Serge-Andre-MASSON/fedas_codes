from torch.optim import Optimizer
from conf.conf import get_conf


class Scheduler:
    def __init__(self, optimizer: Optimizer, n_steps: int, max_lr_factor):
        self.optimizer_params = optimizer.param_groups[0]
        self.n_steps = n_steps

        self.lr = self.optimizer_params["lr"]
        self.last_lr: None

        self.start_slop = self.lr*(max_lr_factor - 1) / (self.n_steps * 2 / 3)
        self.end_slop = self.lr*(max_lr_factor - 1) / (self.n_steps / 3)

        self.count_step = 0

    def step(self):
        self.count_step += 1
        self.last_lr = self.lr

        if self.count_step <= self.n_steps * 2 / 3:
            self.lr += self.start_slop
        else:
            self.lr -= self.end_slop

        # Sometime the lr decline a little bit too much
        self.optimizer_params['lr'] = max(self.lr, 0)

    def get_last_lr(self):
        return self.last_lr


def get_scheduler(optimizer, n_steps: int) -> Scheduler:
    conf = get_conf("scheduler")
    scheduler = Scheduler(
        optimizer,
        n_steps=n_steps,
        max_lr_factor=conf["max_lr_factor"]
    )
    return scheduler
