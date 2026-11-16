"""Use the local GEBCO grid to remove AIS positions that fall on land."""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np


GEBCO_PATH_ENV = "AIS_GEBCO_PATH"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_GEBCO_PATHS = (
    PROJECT_ROOT / "HYdata" / "GEBCO_2026_sub_ice.nc",
    PROJECT_ROOT / "HYdata" / "GEBCO_2026_sub_ice" / "GEBCO_2026_sub_ice.nc",
)


def resolve_gebco_path() -> Path:
    """Resolve the same local GEBCO 2026 file used by the terrain view."""
    configured = os.environ.get(GEBCO_PATH_ENV, "").strip()
    candidates = (Path(configured).expanduser(),) if configured else _DEFAULT_GEBCO_PATHS
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    locations = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"找不到 AIS 海陆筛选所需的 GEBCO 文件：{locations}")


class GebcoLandMask:
    """Vectorised GEBCO land classifier with a compact on-disk index.

    The 7 GB GEBCO dataset is contiguous and uncompressed.  It is sampled once
    into a packed 0.05-degree index; subsequent global AIS snapshots use the
    small in-memory mask rather than reading a world-sized elevation rectangle.
    """

    COARSE_STEP_DEG = 0.05
    CACHE_VERSION = 1

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        with h5py.File(self.path, "r") as dataset:
            lon = dataset["lon"]
            lat = dataset["lat"]
            elevation = dataset["elevation"]
            if lon.size < 2 or lat.size < 2 or elevation.ndim != 2:
                raise ValueError("GEBCO 文件缺少有效的 lon、lat 或 elevation 网格。")
            if elevation.chunks is not None or elevation.compression is not None:
                raise ValueError("AIS 海陆筛选要求连续、未压缩的 GEBCO elevation 网格。")
            offset = elevation.id.get_offset()
            if offset is None:
                raise ValueError("无法定位 GEBCO elevation 数据。")
            self.lon0 = float(lon[0])
            self.lat0 = float(lat[0])
            self.lon_step = float(lon[1] - lon[0])
            self.lat_step = float(lat[1] - lat[0])
            self.shape = tuple(int(value) for value in elevation.shape)
            self.dtype = elevation.dtype
            self.fill_value = elevation.fillvalue
        self._elevation = np.memmap(
            self.path,
            mode="r",
            dtype=self.dtype,
            offset=int(offset),
            shape=self.shape,
            order="C",
        )
        self._coarse_land = self._load_or_build_coarse_land_mask()

    @property
    def cache_path(self) -> Path:
        return self.path.with_name(f"{self.path.stem}.ais_land_0p05.npz")

    def _load_or_build_coarse_land_mask(self) -> np.ndarray:
        """Load a compact global prefilter, rebuilding it when GEBCO changes."""
        source_stat = self.path.stat()
        try:
            with np.load(self.cache_path, allow_pickle=False) as cached:
                if (
                    int(cached["version"]) == self.CACHE_VERSION
                    and int(cached["source_size"]) == source_stat.st_size
                    and int(cached["source_mtime_ns"]) == source_stat.st_mtime_ns
                ):
                    rows = int(cached["rows"])
                    columns = int(cached["columns"])
                    return np.unpackbits(cached["packed"], axis=1, count=columns).astype(
                        bool, copy=False
                    ).reshape(rows, columns)
        except (FileNotFoundError, KeyError, OSError, ValueError):
            pass

        rows = int(round(180.0 / self.COARSE_STEP_DEG))
        columns = int(round(360.0 / self.COARSE_STEP_DEG))
        longitude = -180.0 + (np.arange(columns) + 0.5) * self.COARSE_STEP_DEG
        latitude = -90.0 + (np.arange(rows) + 0.5) * self.COARSE_STEP_DEG
        source_x = np.clip(
            np.rint((longitude - self.lon0) / self.lon_step).astype(np.int64),
            0,
            self.shape[1] - 1,
        )
        source_y = np.clip(
            np.rint((latitude - self.lat0) / self.lat_step).astype(np.int64),
            0,
            self.shape[0] - 1,
        )
        land = np.empty((rows, columns), dtype=bool)
        for output_y, input_y in enumerate(source_y):
            # Copy a whole row so the source file is read sequentially rather
            # than issuing thousands of page reads for strided point access.
            elevation_row = np.array(self._elevation[input_y, :], copy=True)
            sampled = elevation_row[source_x]
            land[output_y, :] = sampled >= 0
            if self.fill_value is not None:
                land[output_y, sampled == self.fill_value] = False

        packed = np.packbits(land, axis=1)
        temporary = self.cache_path.with_name(
            f"{self.cache_path.name}.{os.getpid()}.tmp.npz"
        )
        try:
            np.savez_compressed(
                temporary,
                version=self.CACHE_VERSION,
                source_size=source_stat.st_size,
                source_mtime_ns=source_stat.st_mtime_ns,
                rows=rows,
                columns=columns,
                packed=packed,
            )
            os.replace(temporary, self.cache_path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return land

    def sample(self, longitude: Sequence[float], latitude: Sequence[float]) -> np.ndarray:
        """Return non-negative land and negative ocean classification values."""
        lon = np.asarray(longitude, dtype=np.float64)
        lat = np.asarray(latitude, dtype=np.float64)
        if lon.shape != lat.shape:
            raise ValueError("AIS 经度和纬度数组长度不一致。")
        result = np.full(lon.shape, np.nan, dtype=np.float64)
        valid = np.isfinite(lon) & np.isfinite(lat)
        valid &= (lon >= -180.0) & (lon <= 180.0)
        valid &= (lat >= -90.0) & (lat <= 90.0)
        if not np.any(valid):
            return result
        coarse_x = np.floor(
            (lon[valid] + 180.0) / self.COARSE_STEP_DEG
        ).astype(np.int64)
        coarse_y = np.floor(
            (lat[valid] + 90.0) / self.COARSE_STEP_DEG
        ).astype(np.int64)
        coarse_x = np.clip(coarse_x, 0, self._coarse_land.shape[1] - 1)
        coarse_y = np.clip(coarse_y, 0, self._coarse_land.shape[0] - 1)
        # The current cell decides the result so coastal land positions are not
        # counted.  The compact lookup keeps every refresh O(N) without random
        # reads from the 7 GB source grid.
        sampled = np.where(self._coarse_land[coarse_y, coarse_x], 0.0, -1.0)
        if self.fill_value is not None:
            sampled[sampled == float(self.fill_value)] = np.nan
        result[valid] = sampled
        return result


_LAND_MASK: GebcoLandMask | None = None
_LAND_MASK_LOCK = threading.Lock()


def get_land_mask() -> GebcoLandMask:
    global _LAND_MASK
    if _LAND_MASK is None:
        with _LAND_MASK_LOCK:
            if _LAND_MASK is None:
                _LAND_MASK = GebcoLandMask(resolve_gebco_path())
    return _LAND_MASK


def filter_land_vessels(
    vessels: Sequence[Mapping[str, Any]],
    *,
    land_mask: GebcoLandMask | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove positions whose GEBCO elevation is at or above sea level."""
    items = [dict(vessel) for vessel in vessels]
    try:
        classifier = land_mask or get_land_mask()
        longitude = [float(vessel.get("longitude", math.nan)) for vessel in items]
        latitude = [float(vessel.get("latitude", math.nan)) for vessel in items]
        elevation = classifier.sample(longitude, latitude)
        is_land = np.isfinite(elevation) & (elevation >= 0.0)
        kept = [vessel for vessel, on_land in zip(items, is_land) if not on_land]
        return kept, {
            "land_filter_applied": True,
            "land_filter_source": "GEBCO 2026",
            "land_filter_threshold_m": 0.0,
            "land_filter_resolution_deg": getattr(
                classifier, "COARSE_STEP_DEG", None
            ),
            "observed_vessel_count": len(items),
            "land_vessel_count": int(np.count_nonzero(is_land)),
            "unclassified_vessel_count": int(np.count_nonzero(~np.isfinite(elevation))),
        }
    except Exception as exc:
        # Keep AIS usable if the optional local terrain file is unavailable, but
        # expose the failure explicitly instead of pretending filtering succeeded.
        return items, {
            "land_filter_applied": False,
            "land_filter_source": "GEBCO 2026",
            "land_filter_error": str(exc),
            "observed_vessel_count": len(items),
            "land_vessel_count": 0,
            "unclassified_vessel_count": len(items),
        }
