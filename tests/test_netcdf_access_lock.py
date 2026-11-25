from __future__ import annotations

import unittest
import threading
import time

from backworker import auth_config, netcdf_access_lock
from backworker.api_endpoints import dispatcher
from backworker.bellhop_model import runner


class _RecordingLock:
    def __init__(self) -> None:
        self.events: list[str] = []

    def acquire(self) -> None:
        self.events.append("acquire")

    def release(self) -> None:
        self.events.append("release")


class NetcdfAccessLockTests(unittest.TestCase):
    def tearDown(self) -> None:
        netcdf_access_lock.install_process_lock(None)

    def test_locked_uses_the_injected_process_lock(self) -> None:
        lock = _RecordingLock()
        netcdf_access_lock.install_process_lock(lock)

        with netcdf_access_lock.locked():
            lock.events.append("body")

        self.assertEqual(lock.events, ["acquire", "body", "release"])

    def test_locked_releases_after_an_error(self) -> None:
        lock = _RecordingLock()
        netcdf_access_lock.install_process_lock(lock)

        with self.assertRaisesRegex(RuntimeError, "boom"):
            with netcdf_access_lock.locked():
                raise RuntimeError("boom")

        self.assertEqual(lock.events, ["acquire", "release"])

    def test_nested_locked_only_acquires_process_lock_once(self) -> None:
        lock = _RecordingLock()
        netcdf_access_lock.install_process_lock(lock)

        with netcdf_access_lock.locked():
            with netcdf_access_lock.locked():
                lock.events.append("body")

        self.assertEqual(lock.events, ["acquire", "body", "release"])

    def test_locked_serializes_threads_in_the_main_process(self) -> None:
        active = 0
        maximum_active = 0
        guard = threading.Lock()
        start = threading.Barrier(3)

        def worker() -> None:
            nonlocal active, maximum_active
            start.wait()
            with netcdf_access_lock.locked():
                with guard:
                    active += 1
                    maximum_active = max(maximum_active, active)
                time.sleep(0.03)
                with guard:
                    active -= 1

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join(timeout=2.0)

        self.assertEqual(maximum_active, 1)


class AcousticConcurrencyTests(unittest.TestCase):
    def test_global_route_limit_is_four_processes_times_four_bearings(self) -> None:
        self.assertEqual(dispatcher.ACOUSTIC_PROCESS_COUNT, 4)
        self.assertEqual(dispatcher.ACOUSTIC_BEARING_THREADS_PER_PROCESS, 4)
        self.assertEqual(dispatcher.ACOUSTIC_GLOBAL_BEARING_LIMIT, 16)
        self.assertEqual(auth_config.max_concurrent_jobs(), 4)

    def test_one_field_uses_four_bearing_threads(self) -> None:
        plan = runner._hierarchical_parallel_plan(
            source_count=1,
            bearing_count=8,
            settings={
                "outer_parallel_enabled": True,
                "max_2d_workers": 12,
                "max_source_processes": 4,
                "bearing_threads_per_source": 4,
            },
        )
        self.assertEqual(plan, (1, 4, 4))


if __name__ == "__main__":
    unittest.main()
