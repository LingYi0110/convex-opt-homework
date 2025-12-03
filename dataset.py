import os
from typing import *

from sklearn.datasets import load_svmlight_file
from backend import xp
import numpy as np

# Dataset抽象基类
class Dataset:
    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError()


class LibSVMDataset(Dataset):
    def __init__(self, path: str, dtype = np.float64):
        if not os.path.exists(path):
            raise FileNotFoundError(f'File {path} not found')

        X, y = load_svmlight_file(path)

        # 不整什么fp16加速的烂活了，还要梯度放缩啥的，麻烦死了
        self.X = X.astype(dtype, copy=False)
        self.y = y.astype(dtype, copy=False)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> Any:
        row = self.X.getrow(index)
        X_xp = xp.asarray(row.toarray().ravel()) # 被log1p.E2006这个数据集坑了，全用稠密矩阵内存直接爆炸
        y_xp = xp.asarray(self.y[index])
        return X_xp, y_xp

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: Optional[int] = None):
        self.dataset = dataset

        if batch_size is None:
            self.batch_size = len(dataset)
        else:
            self.batch_size = batch_size

    def __len__(self) -> int:
        # 一个epoch中的batch数量
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple]:
        num = len(self.dataset)

        for start in range(0, num, self.batch_size):
            end = min(start + self.batch_size, num) # 防止数据不够一个batch

            X_list = []
            y_list = []
            # 切片每一个batch的元素
            for i in range(start, end):
                X, y = self.dataset[i]
                X_list.append(X)
                y_list.append(y)

            X_batch = xp.stack(X_list)
            y_batch = xp.stack(y_list)
            yield X_batch, y_batch
