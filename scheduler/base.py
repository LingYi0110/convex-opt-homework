from abc import ABC, abstractmethod


class Scheduler(ABC):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.last_epoch = 0
        self.last_step = 0

    @abstractmethod
    def step(self, X, y):
        raise NotImplementedError()

    def get_lr(self):
        return self.optimizer.lr

    def set_epoch(self, epoch):
        self.last_epoch = epoch

    def set_step(self, step):
        self.last_step = step