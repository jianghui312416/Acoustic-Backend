"""声场工作进程之间共享的 NetCDF/HDF5 读取锁。"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from functools import wraps
from typing import Any, Iterator


_PROCESS_LOCK: Any | None = None
_THREAD_LOCK = threading.RLock()
_THREAD_STATE = threading.local()


def install_process_lock(lock: Any) -> None:
    """由声场进程池 initializer 注入父进程创建的共享锁。"""

    global _PROCESS_LOCK
    _PROCESS_LOCK = lock


@contextmanager
def locked() -> Iterator[None]:
    """串行化当前进程线程，并在配置后串行化多个工作进程。

    锁允许嵌套：外层入口可以保护完整的 NetCDF 业务阶段，内部的具体
    Dataset 读取仍可独立使用本上下文，不会重复获取非重入的进程锁。
    """

    lock = _PROCESS_LOCK
    _THREAD_LOCK.acquire()
    depth = int(getattr(_THREAD_STATE, "depth", 0))
    acquired_process_lock = False
    try:
        if depth == 0 and lock is not None:
            lock.acquire()
            acquired_process_lock = True
        _THREAD_STATE.depth = depth + 1
        yield
    finally:
        _THREAD_STATE.depth = depth
        if acquired_process_lock:
            lock.release()
        _THREAD_LOCK.release()


def serialized(function):
    """把一个同步 NetCDF 入口包装为可嵌套的串行操作。"""

    @wraps(function)
    def wrapper(*args, **kwargs):
        with locked():
            return function(*args, **kwargs)

    return wrapper
