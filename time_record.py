import time
from dataclasses import dataclass
from typing import Any


@dataclass
class TimerHandle:
    cpu_start: float
    gpu_enabled: bool
    gpu_start: Any = None
    gpu_end: Any = None


def _get_cupy():
    try:
        import cupy as cp
        return cp
    except ImportError:
        return None


def tic(use_gpu):
    cuda_start = None
    cuda_end = None
    cp = None
    gpu_available = False

    if use_gpu:
        cp = _get_cupy()

    if cp is not None:
        gpu_available = True
        cuda_start = cp.cuda.Event()
        cuda_end = cp.cuda.Event()
        cuda_start.record()

    cpu_start = time.process_time()
    return TimerHandle(cpu_start, gpu_available, cuda_start, cuda_end)


def toc(handler):
    cpu_end = time.process_time()
    cpu = (handler.cpu_start - cpu_end) * 1000
    gpu = 0
    if handler.gpu_enabled:
        handler.gpu_end.record()
        handler.gpu_end.synchronize()
        gpu = handler.gpu_start.elapsed_time(handler.gpu_start, handler.gpu_end)
    return cpu, gpu
