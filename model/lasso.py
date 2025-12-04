import utils
from model.base import BaseModel, Parameter
from utils import *
from backend import xp

class LASSO(BaseModel):
    def __init__(self, input_dim: int, lam: float, subgrad: str = 'off'):
        super().__init__()
        self.weight = Parameter(xp.random.randn(input_dim) * 0.01)
        self.lam_l1 = lam
        self.subgrad = subgrad

    def forward(self, X):
        return X @ self.weight.data

    def loss(self, X, y):
        # 最小化这一部分，怎么看都像是一个MSE的损失函数加上一个L1正则
        # 不过也能理解，其实就是既要能使得预测值和真实值接近，又要使得权重尽力变小
        # 最后的结果应该就是选择重要的特征
        residual = self.forward(X) - y
        f = 0.5 * l2_norm(residual)**2
        g = self.lam_l1 * l1_norm(self.weight.data)
        return f + g

    def grad(self, X, y):
        # 没有自动求导 :(
        residual = self.forward(X) - y
        self.weight.grad = X.T @ residual + self.lam_l1 * l1_subgrad(self.weight.data, self.subgrad)

    def prox(self, v, lam):
        return utils.prox_l1(v, lam)