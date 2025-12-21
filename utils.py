from backend import xp, get_backend
import numpy as np
from scipy.special import expit as np_expit


def sigmoid(x):
    if isinstance(x, np.ndarray):
        return np_expit(x)

    try:
        import cupy as cp
        from cupyx.scipy.special import expit as cp_expit
        if isinstance(x, cp.ndarray):
            return cp_expit(x)
        else:
            x_np = np.asarray(x)
            return np_expit(x_np)
    except ImportError:
        raise RuntimeError("Cupy is not installed, cannot compute sigmoid for non-numpy arrays.")


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x

    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return x.get()
        if isinstance(x, cp.generic):
            return cp.asnumpy(x)
    except ImportError:
        pass

    return np.asarray(x)

def l1_norm(x):
    x = xp.asarray(x)
    return xp.sum(xp.abs(x))

def l2_norm(x):
    x = xp.asarray(x)
    return xp.sqrt(xp.sum(xp.square(x)))

# L1范数的近段算子
def prox_l1(v, lam):
    return xp.sign(v) * xp.maximum(xp.abs(v) - lam, 0)    # 推导看报告吧，这里不写了

# L2范数的近段算子
def prox_l2(v, lam):
    norm = l2_norm(v)
    if norm == 0:
        return xp.zeros_like(v)
    return v * xp.maximum(0, 1 - lam / norm)

# L1范数的次梯度
def l1_subgrad(weight, mode):
        if mode == 'off':
            return xp.zeros_like(weight)
        elif mode == 'zero':
            return xp.sign(weight)
        elif mode == 'random':
            g = xp.sign(weight)
            zero = (weight == 0)
            g[zero] = xp.random.uniform(-1.0, 1.0, size=zero.sum().item())
            return g
        else:
            raise NotImplementedError(f'Not Supported SubGradient Mode:{mode}')