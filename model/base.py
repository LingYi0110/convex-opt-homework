from abc import ABC, abstractmethod
from backend import xp


class Weight:
    def __init__(self, data):
        self.data = xp.asarray(data)
        self.grad = xp.zeros_like(self.data)


class BaseModel(ABC):
    def __init__(self, weight):
        super().__init__()
        self.weight = Weight(weight)

    @abstractmethod
    def forward(self, X):
        raise NotImplementedError()

    @abstractmethod
    def loss(self, X, y):
        raise NotImplementedError()

    @abstractmethod
    def grad(self, X, y):
        raise NotImplementedError()
