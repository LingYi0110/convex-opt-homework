import utils
from model.base import BaseModel, Weight
from utils import *
from backend import xp


class LASSO(BaseModel):
    def __init__(self, input_dim: int, lam: float, sub_gradient: str = 'off', weight=None):
        if weight is None:
            weight = xp.random.randn(input_dim) * 0.01
        super().__init__(weight)

        self.lam = lam
        self.subgrad = sub_gradient

    def forward(self, X):
        return X @ self.weight.data

    def loss(self, X, y):
        # 最小化这一部分，怎么看都像是一个MSE的损失函数加上一个L1正则
        # 不过也能理解，其实就是既要能使得预测值和真实值接近，又要使得权重尽力变小
        # 最后的结果应该就是选择重要的特征
        residual = self.forward(X) - y
        f = 0.5 * l2_norm(residual) ** 2
        g = self.lam * l1_norm(self.weight.data)
        return f + g

    def grad(self, X, y):
        # 没有自动求导 :(
        residual = self.forward(X) - y
        self.weight.grad = X.T @ residual + self.lam * l1_subgrad(self.weight.data, self.subgrad)

    def prox(self, v, lam):
        return utils.prox_l1(v, lam)
