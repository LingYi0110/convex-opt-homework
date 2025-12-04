import os
from typing import *

from sklearn.datasets import load_svmlight_file
from backend import xp, get_backend
import numpy as np

# Dataset抽象基类
class Dataset:
    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, index: int | slice) -> Any:
        raise NotImplementedError()


class LibSVMDataset(Dataset):
    def __init__(self, path: str, dtype = np.float64):
        if not os.path.exists(path):
            raise FileNotFoundError(f'File {path} not found')

        X_sparse, y = load_svmlight_file(path)

        # 稀疏矩阵运算支持
        if get_backend() == 'cupy':
            import cupyx.scipy.sparse as csp
            self.X = csp.csr_matrix(X_sparse).astype(dtype)
        else:
            self.X = X_sparse.astype(dtype)

        # 对于分类问题，有的数据集标签不是-1, +1, 这里统一改成-1, +1
        unique = np.unique(y)
        if len(unique) == 2:
            y = np.where(y == unique.max(), 1.0, -1.0)

        self.y = xp.asarray(y.astype(dtype, copy=False))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int | slice) -> Any:
        # 切片操作支持
        if isinstance(index, slice):
            return self.X[index], self.y[index]

        return self.X[index], self.y[index]

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

            yield self.dataset[start:end]
