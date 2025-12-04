from scheduler.base import Scheduler


class StepLR(Scheduler):
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self):
        #每step_size次就会衰减一次
        if self.last_epoch % self.step_size == 0 and self.last_epoch != 0:
            self.optimizer.lr *= self.gamma

        # 调用次数自增
        self.last_epoch += 1