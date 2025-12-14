from scheduler.base import Scheduler
from backend import xp
import math


class StepLR(Scheduler):
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self, X, y):
        # 每step_size次就会衰减一次
        if self.last_epoch % self.step_size == 0 and self.last_epoch != 0:
            self.optimizer.lr *= self.gamma


class CosineAnnealingLR(Scheduler):
    def __init__(self, optimizer, T_max, eta_min=0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
        self.eta_t = self.optimizer.lr  # 取初始lr作为起始点

    def step(self, X, y):
        # 参考了 https://github.com/pytorch/pytorch/blob/v2.9.1/torch/optim/lr_scheduler.py
        eta_next = self.eta_min + 0.5 * (self.eta_t - self.eta_min) * (
                    1 + math.cos(math.pi * self.last_epoch / self.T_max))
        self.optimizer.lr = eta_next


class LineSearch(Scheduler):
    def __init__(self, optimizer, c1=1e-4, c2=0.9):
        super().__init__(optimizer)
        self.alpha = optimizer.lr
        self.c1 = c1
        self.c2 = c2

        self.model = self.optimizer.model

    def _phi(self, alpha, weight, pk, X, y):
        old = xp.copy(weight)
        weight[...] = old + alpha * pk
        loss = self.model.loss(X, y)
        weight[...] = old
        return loss

    def _der_phi(self, alpha, weight, pk, X, y):
        old = xp.copy(weight)
        weight[...] = old + alpha * pk
        g = self.model.grad(X, y)
        weight[...] = old
        return g.T @ pk

    def _zoom(self, a_min, a_max, X, y):
        weight = self.optimizer.model.weight.data
        v = self.model.loss(X, y)
        v_grad = self.model.grad(X, y)
        while True:
            a_star = (a_min + a_max) / 2
            left = self._phi(a_star, weight, -v_grad, X, y)
            right = v + self.c1 * a_star * v_grad.T @ -v_grad
            if left >= right:
                a_max = a_star
                continue

            left = self._der_phi(a_star, weight, -v_grad, X, y)
            right = self.c2 * v_grad.T @ -v_grad
            if abs(left) <= -right:
                break

            if left * (a_max - a_min) >= 0:
                a_max = a_min
                continue

            a_min = a_star
        return a_star


    def step(self, X, y):
        # 使用回退法找到了满足Armijo条件的步长后，接下来要找到能满足Wolfe条件的步长
        # 但是还用回退法的话，不一定能找到满足的步长，因为有可能会退多了，跳过了最优点
        # 这个问题非常致命，因为在之前的试验中，会有很多时候回退法是找不到满足条件的步长的
        # 这一块我不会，所以下面的代码参考了
        # https://github.com/scipy/scipy/blob/main/scipy/optimize/_linesearch.py
        weight = self.optimizer.model.weight.data
        grad = self.optimizer.model.weight.grad

        alpha0 = 0
        alpha1 = self.alpha

        v = self.model.loss(X, y)
        v_grad = self.model.grad(X, y)

        while True:
            # 判断Armijo条件
            left = self._phi(alpha1, weight, -v_grad, X, y)
            right = v + self.c1 * alpha1 * v_grad.T @ -v_grad
            if left >= right:
                self.alpha = self._zoom(alpha0, alpha1, X, y)
                self.optimizer.lr = alpha1
                break
            # 判断曲率条件，这里似乎是强Wolfe条件
            left = self._der_phi(alpha1, weight, -v_grad, X, y)
            right = self.c2 * v_grad.T @ -v_grad
            if abs(left) <= -right:
                self.alpha = alpha1
                self.optimizer.lr = alpha1
                break
            # 判断斜率是否变正
            if left >= 0:
                self.alpha = self._zoom(alpha1, alpha0, X, y)
                self.optimizer.lr = alpha1
                break
            # 如果上面三个条件都不满足，说明alpha太小
            alpha0 = alpha1
            alpha1 *= 2

class BarzilaiBorwein(Scheduler):
    def __init__(self, optimizer, lr_type='BB1', c1=1e-4, decay=0.5, memory_size=5, lr_min=1e-10, lr_max=1e4, max_iter=2000):
        super().__init__(optimizer)
        self.lr_type = lr_type
        self.c1 = c1
        self.decay = decay
        self.memory_size = memory_size
        self.alpha = optimizer.lr
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.max_iter = max_iter

        self.model = self.optimizer.model
        self._max_val = float('-inf')

    def step(self, X, y):
        weight = self.optimizer.model.weight.data
        grad = self.optimizer.model.weight.grad

        if self.model.loss(X, y) > self._max_val and self.last_step <= self.memory_size:
            self._max_val = self.model.loss(X, y)

        # 非单调Armijo条件
        sk = xp.copy(weight)
        yk = xp.copy(grad)

        self.model.grad(X, y)
        v1 = self._max_val - self.c1 * self.alpha * grad.T @ grad

        for i in range(self.max_iter):
            self.model.grad(X, y)
            weight -= self.alpha * grad
            v2 = self.model.loss(X, y)
            if v2 >= v1:
                weight[...] = sk
                self.alpha *= self.decay  # 回退
                self.model.grad_zero()
                continue
            else:
                sk = weight - sk
                yk = grad - yk
                weight += self.alpha * grad
                self.model.grad_zero()
                break

        # 计算下一步的BB步长
        if self.lr_type == 'BB1':
            self.alpha = (sk.T @ yk) / (yk.T @ yk)
        elif self.lr_type == 'BB2':
            self.alpha = (sk.T @ sk) / (sk.T @ yk)

        # 对步长进行限制
        self.alpha = max(min(self.alpha, self.lr_max), self.lr_min)

        self.optimizer.lr = self.alpha
