import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TimerHandle:
    cpu_start: float
    gpu_enabled: bool
    cp: Any = None
    stream: Any = None
    gpu_start: Any = None
    gpu_end: Any = None


def _get_cupy():
    try:
        import cupy as cp
        return cp
    except ImportError:
        return None


def tic(use_gpu: bool = True):
    cpu_start = time.process_time()

    cp = _get_cupy() if use_gpu else None
    if cp is None:
        return TimerHandle(cpu_start=cpu_start, gpu_enabled=False)

    stream = cp.cuda.get_current_stream()


    stream.synchronize()

    ev_start = cp.cuda.Event()
    ev_end = cp.cuda.Event()
    ev_start.record(stream)

    return TimerHandle(
        cpu_start=cpu_start,
        gpu_enabled=True,
        cp=cp,
        stream=stream,
        gpu_start=ev_start,
        gpu_end=ev_end,
    )


def toc(handler):
    cpu_end = time.process_time()
    cpu_ms = (cpu_end - handler.cpu_start) * 1000

    gpu_ms = 0
    if handler.gpu_enabled:
        handler.gpu_end.record(handler.stream)
        handler.gpu_end.synchronize()

        gpu_ms = handler.cp.cuda.get_elapsed_time(handler.gpu_start, handler.gpu_end)

    return cpu_ms, gpu_ms
