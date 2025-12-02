from abc import ABC, abstractmethod
from backend import xp


class Parameter:
    def __init__(self, data):
        self.data = xp.asarray(data)
        self.grad = xp.zeros_like(self.data)


class BaseModel(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, X):
        raise NotImplementedError()

    @abstractmethod
    def loss(self, X, y):
        raise NotImplementedError()

    @abstractmethod
    def grad(self, X, y):
        raise NotImplementedError()

    def parameters(self):
        # 只要是Parameter就yield出来
        for _, i in self.__dict__.items():
            if isinstance(i, Parameter):
                yield i