import numpy as np


class Backend:
    def __init__(self):
        self._backend = np
        self._type = 'numpy'

    def set_backend(self, backend: str):
        if backend == 'cupy':
            try:
                import cupy as cp
                self._backend = cp
                self._type = 'cupy'
            except ImportError:
                self._backend = np
                self._type = 'numpy'
        elif backend == 'numpy':
            self._backend = np
            self._type = 'numpy'
        else:
            # 代码健壮性这一块还是要的
            raise NotImplementedError(f'Backend {backend} not implemented')

    def get_backend(self):
        return self._type

    def __getattr__(self, name):
        return getattr(self._backend, name) # 把对应的调用送到指定后端


xp = Backend()

def set_backend(backend: str):
    xp.set_backend(backend)

def get_backend():
    return xp.get_backend()