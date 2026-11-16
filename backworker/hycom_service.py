"""HYCOM 官方 THREDDS/NCSS 温盐数据服务。

前端只提交一个有效时间。这里按时间自动选择当前分析/预报、ESPC-D-V02
年度历史归档或 GOFS 3.1 旧历史归档，并只下载仿真区域所需的温盐子集。
"""

from __future__ import annotations

import math
import re
import tempfile
import threading
import time
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import requests
from netCDF4 import Dataset, num2date
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


UTC = timezone.utc
HYCOM_DATA_SOURCE = "HYCOM"
_METADATA_LOCK = threading.Lock()
_METADATA_CACHE: dict[str, Any] = {"expires": 0.0, "times": []}
_ICE_CACHE_LOCK = threading.Lock()


class HycomServiceError(RuntimeError):
    """HYCOM 服务、时间或返回数据不可用。"""


def _service_config(config: Mapping[str, Any]) -> dict[str, Any]:
    services = config.get("DATA_SERVICES")
    if not isinstance(services, Mapping) or not isinstance(services.get("hycom"), Mapping):
        raise HycomServiceError("data_config.json 缺少 DATA_SERVICES.hycom 配置。")
    return dict(services["hycom"])


def _service_urls(options: Mapping[str, Any], plural_key: str, singular_key: str) -> list[str]:
    """读取按优先级排列的官方地址，并兼容旧的单地址配置。"""
    raw = options.get(plural_key)
    values = raw if isinstance(raw, (list, tuple)) else [options.get(singular_key)]
    urls: list[str] = []
    for value in values:
        url = str(value or "").strip()
        if url and url not in urls:
            urls.append(url)
    return urls


def _parse_datetime(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise HycomServiceError("HYCOM 请求缺少时间。")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text.replace(" ", "T"))
    except ValueError as exc:
        raise HycomServiceError(f"无效的 HYCOM 时间：{value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).replace(microsecond=0)


def _iso_hour(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _display_time(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _new_session(options: Mapping[str, Any]) -> requests.Session:
    retries = max(0, int(options.get("retries", 2)))
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers["User-Agent"] = "Acoustic-Field-GeoVis-HYCOM/1.0"
    return session


def _request_text(session: requests.Session, url: str, timeout: float) -> str:
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HycomServiceError(f"HYCOM 时间索引请求失败：{exc}") from exc
    return response.text


def _time_units(das: str, variable: str) -> str:
    block = re.search(rf"(?ms)^\s*{re.escape(variable)}\s*\{{(.*?)^\s*\}}", das)
    if block is None:
        raise HycomServiceError(f"HYCOM 元数据缺少时间变量 {variable}。")
    match = re.search(r'String\s+units\s+"([^"]+)"', block.group(1))
    if match is None:
        raise HycomServiceError(f"HYCOM 时间变量 {variable} 缺少 units。")
    return match.group(1)


def _ascii_values(payload: str, variable: str) -> np.ndarray:
    marker = re.search(rf"(?m)^\s*{re.escape(variable)}\[\d+\]\s*$", payload)
    if marker is None:
        raise HycomServiceError(f"HYCOM 时间响应缺少 {variable} 数组。")
    numbers = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        payload[marker.end():],
    )
    if not numbers:
        raise HycomServiceError(f"HYCOM 时间响应中的 {variable} 为空。")
    return np.asarray([float(item) for item in numbers], dtype=np.float64)


def _python_datetime(value: Any) -> datetime:
    return datetime(
        int(value.year), int(value.month), int(value.day),
        int(value.hour), int(value.minute), int(value.second), tzinfo=UTC,
    )


def _fetch_latest_times(config: Mapping[str, Any]) -> list[datetime]:
    options = _service_config(config)
    metadata_url = str(options.get("latest_opendap_url") or "").rstrip("/")
    if not metadata_url:
        raise HycomServiceError("DATA_SERVICES.hycom.latest_opendap_url 不能为空。")
    timeout = max(5.0, float(options.get("metadata_timeout_seconds", 30.0)))
    session = _new_session(options)
    try:
        das = _request_text(session, metadata_url + ".das", timeout)
        time_sets: list[set[datetime]] = []
        for variable in ("time", "time3"):
            units = _time_units(das, variable)
            payload = _request_text(session, metadata_url + f".ascii?{variable}", timeout)
            values = _ascii_values(payload, variable)
            decoded = num2date(values, units=units, only_use_cftime_datetimes=True)
            time_sets.append({_python_datetime(item) for item in np.asarray(decoded).reshape(-1)})
        common = set.intersection(*time_sets)
        if not common:
            raise HycomServiceError("HYCOM 当前温度与盐度没有共同有效时次。")
        return sorted(common)
    finally:
        session.close()


def latest_times(config: Mapping[str, Any], *, force: bool = False) -> list[datetime]:
    options = _service_config(config)
    ttl = max(30.0, float(options.get("metadata_cache_seconds", 600.0)))
    now_mono = time.monotonic()
    with _METADATA_LOCK:
        cached = list(_METADATA_CACHE.get("times") or [])
        if not force and cached and now_mono < float(_METADATA_CACHE.get("expires") or 0.0):
            return cached
        try:
            fetched = _fetch_latest_times(config)
        except HycomServiceError:
            if cached:
                return cached
            # 官方时间索引短暂不可用时仍允许前端工作；实际取数会由 NCSS
            # 对时次做最终校验。
            step = max(1, int(options.get("time_step_hours", 3)))
            now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
            now -= timedelta(hours=now.hour % step)
            past = max(step, int(options.get("latest_fallback_history_hours", 72)))
            future = max(step, int(options.get("forecast_hours", 192)))
            fetched = [
                now + timedelta(hours=offset)
                for offset in range(-past, future + step, step)
            ]
        _METADATA_CACHE["times"] = list(fetched)
        _METADATA_CACHE["expires"] = now_mono + ttl
        return list(fetched)


def default_time(config: Mapping[str, Any]) -> str:
    now = datetime.now(UTC)
    values = latest_times(config)
    available = [value for value in values if value <= now]
    selected = available[-1] if available else values[0]
    return _display_time(selected)


def dataset_list() -> list[dict[str, Any]]:
    return [{
        "id": HYCOM_DATA_SOURCE,
        "name": "HYCOM 历史与预报",
        "categoryId": "ocean",
        "categoryName": "海洋环境",
        "pid": HYCOM_DATA_SOURCE,
        "datatype": ["DT_ENV_TEMP", "DT_ENV_SALT"],
        "datarange": ["history", "forecast"],
        "normalizeLayers": [],
    }]


def time_list(
    config: Mapping[str, Any], query_month: int = 0, *, now: datetime | None = None
) -> dict[str, Any]:
    options = _service_config(config)
    step = max(1, int(options.get("time_step_hours", 3)))
    start = _parse_datetime(options.get("history_start", "2018-12-04T00:00:00Z"))
    current = (now or datetime.now(UTC)).astimezone(UTC)
    live = latest_times(config)
    live_start = live[0]
    history_end = live_start - timedelta(hours=step)
    values: list[datetime] = []
    cursor = start
    while cursor <= history_end:
        if not query_month or cursor.month == query_month:
            values.append(cursor)
        cursor += timedelta(hours=step)
    values.extend(value for value in live if not query_month or value.month == query_month)
    values = sorted(set(values))
    analysed = [value for value in live if value <= current]
    selected = analysed[-1] if analysed else live[0]
    forecasts = [value for value in live if value > current]
    return {
        "dataSource": HYCOM_DATA_SOURCE,
        "isEnv": True,
        "displayList": [_display_time(value) for value in values],
        "standardTimeList": [_display_time(value) for value in values],
        "defaultTime": _display_time(selected),
        "forecastStartTime": _display_time(forecasts[0]) if forecasts else None,
        "historyStartTime": _display_time(values[0]) if values else None,
        "latestTime": _display_time(values[-1]) if values else None,
        "timeStepHours": step,
        "provider": "HYCOM THREDDS/NCSS",
    }


def _product_for_time(
    config: Mapping[str, Any], requested: datetime, live: Sequence[datetime]
) -> str:
    options = _service_config(config)
    start = _parse_datetime(options.get("history_start", "2018-12-04T00:00:00Z"))
    active = _parse_datetime(options.get("active_archive_start", "2024-08-10T00:00:00Z"))
    tolerance = timedelta(hours=max(1, int(options.get("time_step_hours", 3))))
    if requested < start:
        raise HycomServiceError(f"HYCOM 历史数据最早支持 {_display_time(start)} UTC。")
    if live and live[0] - tolerance <= requested <= live[-1] + tolerance:
        return "latest"
    if live and requested > live[-1] + tolerance:
        raise HycomServiceError(
            f"所选时间超过 HYCOM 当前预报末端 {_display_time(live[-1])} UTC。"
        )
    return "active_archive" if requested >= active else "legacy_archive"


class HycomNcssReader:
    """按所选时次从 HYCOM NCSS 下载局部温盐与海冰场。"""

    temperature_tag = "water_temp"
    salinity_tag = "salinity"
    hs_tag = ""
    tp_tag = ""
    direction_tag = ""
    topography_tag = ""
    ice_tag = "sih"
    bottom_geoacoustics_tag = ""

    def __init__(self, config: Mapping[str, Any], *, time_value: str = "") -> None:
        self._config = config
        self._options = _service_config(config)
        self._requested = _parse_datetime(time_value or default_time(config))
        self._live_times = latest_times(config)
        self._product = _product_for_time(config, self._requested, self._live_times)
        self._session = _new_session(self._options)
        self._timeout = max(10.0, float(self._options.get("request_timeout_seconds", 90.0)))
        self._temporary = tempfile.TemporaryDirectory(
            prefix="hycom_ncss_", ignore_cleanup_errors=True
        )
        self._files: dict[tuple[Any, ...], Path] = {}
        self._lock = threading.Lock()

    @property
    def product(self) -> str:
        return self._product

    def _bounds(self, query_lon: np.ndarray, query_lat: np.ndarray) -> tuple[float, ...]:
        lon = np.asarray(query_lon, dtype=np.float64).reshape(-1)
        lat = np.asarray(query_lat, dtype=np.float64).reshape(-1)
        if not lon.size or not lat.size or not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
            raise HycomServiceError("HYCOM 区域经纬度必须是非空有限数组。")
        centre = float(np.mean(lon))
        unwrapped = centre + ((lon - centre + 180.0) % 360.0) - 180.0
        margin = max(0.01, float(self._options.get("request_margin_degrees", 0.10)))
        west = float(np.min(unwrapped) - margin)
        east = float(np.max(unwrapped) + margin)
        south = max(-80.0, float(np.min(lat) - margin))
        north = min(90.0, float(np.max(lat) + margin))
        maximum_span = max(0.5, float(self._options.get("max_request_span_degrees", 12.0)))
        if east - west > maximum_span or north - south > maximum_span:
            raise HycomServiceError(
                f"HYCOM 单次仿真区域不能超过 {maximum_span:g}°；请减小计算半径。"
            )
        while west < 0.0:
            west += 360.0
            east += 360.0
        while west >= 360.0:
            west -= 360.0
            east -= 360.0
        return west, east, south, north, centre

    def _request_spec(self, tag: str) -> tuple[list[str], tuple[str, ...]]:
        if tag == self.ice_tag:
            if self._product == "latest":
                urls = _service_urls(
                    self._options, "ice_latest_ncss_urls", "ice_latest_ncss_url"
                )
            elif self._product == "active_archive":
                urls = _service_urls(
                    self._options, "active_ice_ncss_urls", "active_ice_ncss_url"
                )
            else:
                templates = self._options.get("legacy_ice_ncss_templates")
                if not isinstance(templates, (list, tuple)):
                    templates = [self._options.get("legacy_ice_ncss_template")]
                urls = [
                    str(template).format(year=self._requested.year)
                    for template in templates
                    if str(template or "").strip()
                ]
            return urls, ("sic", "sih")
        if self._product == "latest":
            urls = _service_urls(self._options, "latest_ncss_urls", "latest_ncss_url")
            return urls, ("salinity", "water_temp")
        if self._product == "legacy_archive":
            urls = _service_urls(self._options, "legacy_ncss_urls", "legacy_ncss_url")
            return urls, ("salinity", "water_temp")
        key = "active_temperature_ncss_template" if tag == self.temperature_tag else "active_salinity_ncss_template"
        template = str(self._options.get(key) or "")
        return [template.format(year=self._requested.year)] if template else [], (tag,)

    def _download(self, tag: str, bounds: tuple[float, ...]) -> Path:
        west, east, south, north, _centre = bounds
        urls, variables = self._request_spec(tag)
        if not urls:
            raise HycomServiceError(f"HYCOM {self._product} NCSS 地址未配置。")
        key = (
            tuple(urls), variables, _iso_hour(self._requested),
            round(west, 5), round(east, 5), round(south, 5), round(north, 5),
        )
        with self._lock:
            cached = self._files.get(key)
            if cached is not None and cached.is_file():
                return cached
            params: list[tuple[str, str]] = [("var", name) for name in variables]
            params.extend([
                ("north", f"{north:.6f}"),
                ("south", f"{south:.6f}"),
                ("west", f"{west:.6f}"),
                ("east", f"{east:.6f}"),
                ("horizStride", "1"),
                ("time", _iso_hour(self._requested)),
                ("accept", "netCDF4"),
                ("addLatLon", "true"),
            ])
            content = b""
            failures: list[str] = []
            last_error: Exception | None = None
            failover_rounds = max(1, int(self._options.get("endpoint_failover_rounds", 3)))
            for attempt in range(failover_rounds):
                for url in urls:
                    try:
                        response = self._session.get(url, params=params, timeout=self._timeout)
                        response.raise_for_status()
                        candidate = response.content
                        if not (candidate.startswith(b"CDF") or candidate.startswith(b"\x89HDF")):
                            failures.append(
                                f"{url}: 未返回 NetCDF（{candidate[:180].decode('utf-8', 'replace')}）"
                            )
                            continue
                        content = candidate
                        break
                    except requests.RequestException as exc:
                        last_error = exc
                        detail = ""
                        error_response = getattr(exc, "response", None)
                        if error_response is not None:
                            detail = str(error_response.text or "")[:300]
                        failures.append(f"第{attempt + 1}轮 {url}: {exc} {detail}".strip())
                if content:
                    break
                if attempt + 1 < failover_rounds:
                    self._session.close()
                    self._session = _new_session(self._options)
                    time.sleep(0.8 * (attempt + 1))
            if not content:
                message = "；".join(failures)
                raise HycomServiceError(
                    f"HYCOM {self._product} 子集请求失败，已进行 {failover_rounds} 轮主备重连：{message}"
                ) from last_error
            path = Path(self._temporary.name) / f"subset_{len(self._files):03d}.nc"
            path.write_bytes(content)
            self._files[key] = path
            return path

    def read_profile_cube(
        self, tag: str, query_lon: np.ndarray, query_lat: np.ndarray
    ) -> dict[str, Any]:
        if tag not in {self.temperature_tag, self.salinity_tag, self.ice_tag}:
            raise HycomServiceError(f"HYCOM 读取器不支持变量 {tag!r}。")
        bounds = self._bounds(query_lon, query_lat)
        path = self._download(tag, bounds)
        with Dataset(path, "r") as dataset:
            if tag not in dataset.variables:
                raise HycomServiceError(f"HYCOM 返回文件缺少变量 {tag}。")
            variable = dataset.variables[tag]
            dimensions = list(variable.dimensions)
            key: list[Any] = [0 if name.startswith("time") else slice(None) for name in dimensions]
            raw = variable[tuple(key)]
            kept = [name for name in dimensions if not name.startswith("time")]
            values = np.asarray(np.ma.filled(raw, np.nan), dtype=np.float64)
            if tag == self.ice_tag:
                if not {"lat", "lon"}.issubset(kept):
                    raise HycomServiceError(f"HYCOM {tag} 维度无法识别：{dimensions}")
                values = np.transpose(values, [kept.index("lat"), kept.index("lon")])[None, :, :]
                if "sic" in dataset.variables:
                    concentration_var = dataset.variables["sic"]
                    concentration_dims = list(concentration_var.dimensions)
                    concentration_key = [
                        0 if name.startswith("time") else slice(None)
                        for name in concentration_dims
                    ]
                    concentration_kept = [
                        name for name in concentration_dims if not name.startswith("time")
                    ]
                    concentration = np.asarray(
                        np.ma.filled(concentration_var[tuple(concentration_key)], np.nan),
                        dtype=np.float64,
                    )
                    concentration = np.transpose(
                        concentration,
                        [concentration_kept.index("lat"), concentration_kept.index("lon")],
                    )
                    values[0, ~np.isfinite(values[0]) & (concentration <= 0.0)] = 0.0
                depth = np.asarray([0.0], dtype=np.float64)
            else:
                if not {"depth", "lat", "lon"}.issubset(kept):
                    raise HycomServiceError(f"HYCOM {tag} 维度无法识别：{dimensions}")
                values = np.transpose(
                    values,
                    [kept.index("depth"), kept.index("lat"), kept.index("lon")],
                )
                depth = np.asarray(dataset.variables["depth"][:], dtype=np.float64)
            lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
            lon = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
            time_name = next((name for name in dimensions if name.startswith("time")), None)
            if time_name and time_name in dataset.variables:
                time_var = dataset.variables[time_name]
                returned = _python_datetime(num2date(time_var[0], units=time_var.units))
                max_delta = max(1.6, float(self._options.get("time_tolerance_hours", 1.6)))
                if abs((returned - self._requested).total_seconds()) > max_delta * 3600.0:
                    raise HycomServiceError(
                        f"HYCOM 返回时次 {_display_time(returned)} 与请求 {_display_time(self._requested)} 不匹配。"
                    )

        centre = bounds[4]
        lon_unwrapped = centre + ((lon - centre + 180.0) % 360.0) - 180.0
        lon_order = np.argsort(lon_unwrapped)
        lat_order = np.argsort(lat)
        values = values[:, lat_order, :][:, :, lon_order]
        query_lon_array = np.asarray(query_lon, dtype=np.float64)
        query_unwrapped = centre + ((query_lon_array - centre + 180.0) % 360.0) - 180.0
        return {
            "depth": depth,
            "lat": lat[lat_order],
            "lon": lon_unwrapped[lon_order],
            "values": values,
            "query_lon": query_unwrapped,
            "centre_lon": centre,
            "provider": "HYCOM THREDDS/NCSS",
            "product": self._product,
            "requested_time": _display_time(self._requested),
        }

    def close(self) -> None:
        self._session.close()
        self._files.clear()
        self._temporary.cleanup()


def cache_ice_subset(
    config: Mapping[str, Any],
    *,
    time_value: str,
    query_lon: np.ndarray,
    query_lat: np.ndarray,
    cache_root: str | Path,
) -> str:
    """下载并缓存声学计算需要的 HYCOM sic/sih 区域文件。"""
    reader = HycomNcssReader(config, time_value=time_value)
    try:
        bounds = reader._bounds(query_lon, query_lat)
        source = reader._download(reader.ice_tag, bounds)
        identity = "|".join([
            reader.product,
            _iso_hour(reader._requested),
            *(f"{value:.6f}" for value in bounds[:4]),
        ])
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        root = Path(cache_root)
        root.mkdir(parents=True, exist_ok=True)
        destination = root / f"hycom_ice_{reader._requested:%Y%m%dT%H%M%SZ}_{digest}.nc"
        with _ICE_CACHE_LOCK:
            if not destination.is_file():
                temporary = destination.with_suffix(".tmp")
                temporary.write_bytes(source.read_bytes())
                temporary.replace(destination)
        return str(destination)
    finally:
        reader.close()


