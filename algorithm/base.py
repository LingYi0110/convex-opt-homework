from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr # 其实步长alpha就是学习率，反正我就这么写了

    @abstractmethod
    def step(self, X, y):
        raise NotImplementedError()