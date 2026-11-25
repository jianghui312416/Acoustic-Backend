from __future__ import annotations

import unittest
import threading
import time

import numpy as np

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


class HycomProfileExtrapolationTests(unittest.TestCase):
    @staticmethod
    def _cube() -> dict[str, np.ndarray | float]:
        return {
            "depth": np.asarray([0.0, 50.0, 100.0, 125.0, 150.0]),
            "lat": np.asarray([10.0]),
            "lon": np.asarray([120.0]),
            "centre_lon": 120.0,
            "values": np.asarray([20.0, 15.0, 10.0, 8.0, np.nan]).reshape(5, 1, 1),
        }

    def test_terminal_gradient_is_used_below_deepest_valid_layer(self) -> None:
        profiles, inverse = runner._profile_cell_cache(
            self._cube(),
            np.asarray([100.0, 125.0, 130.0, 150.0]),
            np.asarray([120.0]),
            np.asarray([10.0]),
        )

        np.testing.assert_array_equal(inverse, [0])
        # 100--125 m 的末端梯度为 -2/25；130 m 应继续该梯度，而不是保持 8。
        np.testing.assert_allclose(profiles[:, 0], [10.0, 8.0, 7.6, 6.0])

    def test_terminal_extrapolation_is_limited_to_one_native_interval(self) -> None:
        profiles, _ = runner._profile_cell_cache(
            self._cube(),
            np.asarray([150.0, 200.0, 1000.0]),
            np.asarray([120.0]),
            np.asarray([10.0]),
        )

        np.testing.assert_allclose(profiles[:, 0], [6.0, 6.0, 6.0])


if __name__ == "__main__":
    unittest.main()
