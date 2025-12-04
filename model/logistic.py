from backend import xp
from model.base import BaseModel, Parameter
from utils import *


class Logistic(BaseModel):
    def __init__(self, input_dim, lam, norm='l1', subgrad = 'off'):
        super().__init__()
        self.input_dim = input_dim
        self.lam = lam
        self.norm = norm
        self.subgrad = subgrad

        self.weight = Parameter(xp.random.randn(input_dim) * 0.01)

    def forward(self, X):
        # 越大，倾向于+1类
        # 越小，倾向于-1类
        return X @ self.weight.data

    def loss(self, X, y):
        f = xp.log1p(xp.exp(-y * self.forward(X))).mean()

        if self.norm == 'l1':
            g = self.lam * l1_norm(self.weight.data)
        elif self.norm == 'l2':
            g = self.lam * l2_norm(self.weight.data)
        else:
            raise NotImplementedError(f'Not Supported Norm :{self.norm}')

        return f + g

    def grad(self, X, y):
        m = X.shape[0]

        tmp = (-y / (1 + xp.exp(y * self.forward(X)))) / m
        grad = X.T @ tmp

        if self.norm == 'l1':
            grad += self.lam * l1_subgrad(self.weight.data, self.subgrad)
        else:
            grad += self.lam * self.weight.data / l2_norm(self.weight.data)

        self.weight.grad = grad