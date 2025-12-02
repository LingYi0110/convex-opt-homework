from abc import ABC, abstractmethod


class Scheduler(ABC):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.last_epoch = 0

    @abstractmethod
    def step(self):
        raise NotImplementedError()

    def get_lr(self):
        return self.optimizer.lr