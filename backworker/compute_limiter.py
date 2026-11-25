"""按用户限制并发的声场计算，超出并发时进入等待队列。"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from . import auth_config


class ComputeQueueError(RuntimeError):
    """等待空闲计算槽位超时。"""


class ComputeLimiter:
    def __init__(self) -> None:
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._active: dict[str, int] = {}
        self._queued: dict[str, int] = {}
        self._lock = threading.Lock()

    def _limit(self) -> int:
        return auth_config.max_concurrent_jobs()

    def max_concurrent(self) -> int:
        return self._limit()

    def _max_wait(self) -> float:
        return auth_config.job_queue_wait_seconds()

    def _semaphore(self, key: str) -> asyncio.Semaphore:
        with self._lock:
            semaphore = self._semaphores.get(key)
            if semaphore is None:
                semaphore = asyncio.Semaphore(self._limit())
                self._semaphores[key] = semaphore
            return semaphore

    @asynccontextmanager
    async def slot(self, user_key: str) -> AsyncIterator[None]:
        semaphore = self._semaphore(user_key)
        with self._lock:
            self._queued[user_key] = self._queued.get(user_key, 0) + 1
        acquired = False
        try:
            try:
                await asyncio.wait_for(semaphore.acquire(), timeout=self._max_wait())
                acquired = True
            except asyncio.TimeoutError as exc:
                raise ComputeQueueError(
                    "当前账号的计算任务排队超时，请等待已有任务完成后再试。"
                ) from exc
            finally:
                with self._lock:
                    self._queued[user_key] = max(0, self._queued.get(user_key, 1) - 1)
            with self._lock:
                self._active[user_key] = self._active.get(user_key, 0) + 1
            yield
        finally:
            if acquired:
                with self._lock:
                    self._active[user_key] = max(0, self._active.get(user_key, 1) - 1)
                semaphore.release()

    def stats(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {
                key: {
                    "active": self._active.get(key, 0),
                    "queued": self._queued.get(key, 0),
                }
                for key in set(self._active) | set(self._queued)
            }


compute_limiter = ComputeLimiter()
