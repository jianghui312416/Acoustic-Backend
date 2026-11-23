"""AISStream 指定区域实时船舶快照服务。"""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import monotonic
from typing import Any

import websockets

from .ais_land_filter import filter_land_vessels


AISSTREAM_URL = "wss://stream.aisstream.io/v0/stream"
API_KEY_ENV = "AISSTREAM_API_KEY"
API_KEY_FILE = Path(__file__).resolve().parents[2] / "config" / "aisstream_api_key.txt"
MESSAGE_TYPES = (
    "PositionReport",
    "StandardClassBPositionReport",
    "ExtendedClassBPositionReport",
    "LongRangeAisBroadcastMessage",
    "ShipStaticData",
    "StaticDataReport",
)

_POSITION_TYPES = frozenset(MESSAGE_TYPES[:4])
_STATIC_CACHE: dict[int, dict[str, Any]] = {}
_STATIC_CACHE_LOCK = threading.Lock()
_MAX_STATIC_CACHE_SIZE = 50_000
ROLLING_HISTORY_SECONDS = 5 * 60
INITIAL_WARMUP_SECONDS = 5
GLOBAL_BOUNDING_BOX = [[90.0, -180.0], [-90.0, 180.0]]


class AISStreamError(RuntimeError):
    """AISStream 配置、连接或上游响应错误。"""


def resolve_api_key() -> str:
    """优先从环境变量读取密钥，其次读取本机忽略提交的密钥文件。"""
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key and API_KEY_FILE.is_file():
        key = API_KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise AISStreamError(
            f"未配置 AISStream API key；请设置环境变量 {API_KEY_ENV}，"
            f"或写入 {API_KEY_FILE}。"
        )
    return key


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _clean_time(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    if text.endswith(" +0000 UTC"):
        text = text[: -len(" +0000 UTC")].replace(" ", "T", 1) + "Z"
    return text


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mmsi(metadata: Mapping[str, Any], message: Mapping[str, Any]) -> int | None:
    value = metadata.get("MMSI", message.get("UserID"))
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if 100_000_000 <= number <= 999_999_999 else None


def _dimensions(value: Any) -> tuple[float | None, float | None]:
    if not isinstance(value, Mapping):
        return None, None
    a = _finite_float(value.get("A"))
    b = _finite_float(value.get("B"))
    c = _finite_float(value.get("C"))
    d = _finite_float(value.get("D"))
    length = a + b if a is not None and b is not None and a + b > 0 else None
    width = c + d if c is not None and d is not None and c + d > 0 else None
    return length, width


def ship_type_name(code: int | None) -> str | None:
    """把 AIS 船型代码归并为稳定、可读的中文类别。"""
    if code is None or code <= 0:
        return None
    if 20 <= code <= 29:
        return "翼地效应船"
    if code == 30:
        return "渔船"
    names = {
        31: "拖船",
        32: "拖船",
        33: "疏浚或水下作业船",
        34: "潜水作业船",
        35: "军舰",
        36: "帆船",
        37: "游艇",
        40: "高速船",
        50: "引航船",
        51: "搜救船",
        52: "拖船",
        53: "港口补给船",
        54: "防污染船",
        55: "执法船",
        58: "医疗运输船",
        59: "非战斗舰船",
    }
    if code in names:
        return names[code]
    if 40 <= code <= 49:
        return "高速船"
    if 60 <= code <= 69:
        return "客船"
    if 70 <= code <= 79:
        return "货船"
    if 80 <= code <= 89:
        return "油轮"
    if 90 <= code <= 99:
        return "其他船舶"
    return "未定义船型"


def _static_fields(message_type: str, data: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    dimension: Any = None
    if message_type == "ShipStaticData":
        fields["vessel_name"] = str(data.get("Name") or "").strip() or None
        fields["ship_type"] = data.get("Type")
        dimension = data.get("Dimension")
        draught = _finite_float(data.get("MaximumStaticDraught"))
        fields["draught_m"] = draught if draught is not None and draught > 0 else None
    elif message_type == "ExtendedClassBPositionReport":
        fields["vessel_name"] = str(data.get("Name") or "").strip() or None
        fields["ship_type"] = data.get("Type")
        dimension = data.get("Dimension")
    elif message_type == "StaticDataReport":
        report_a = data.get("ReportA")
        report_b = data.get("ReportB")
        if isinstance(report_a, Mapping) and report_a.get("Valid"):
            fields["vessel_name"] = str(report_a.get("Name") or "").strip() or None
        if isinstance(report_b, Mapping) and report_b.get("Valid"):
            fields["ship_type"] = report_b.get("ShipType")
            dimension = report_b.get("Dimension")

    if dimension is not None:
        fields["length_m"], fields["width_m"] = _dimensions(dimension)
    if "ship_type" in fields:
        try:
            fields["ship_type"] = int(fields["ship_type"])
        except (TypeError, ValueError):
            fields["ship_type"] = None
        fields["ship_type_name"] = ship_type_name(fields["ship_type"])
    return {name: value for name, value in fields.items() if value is not None}


def _remember_static(mmsi: int, fields: Mapping[str, Any]) -> None:
    if not fields:
        return
    with _STATIC_CACHE_LOCK:
        entry = _STATIC_CACHE.setdefault(mmsi, {})
        entry.update(fields)
        entry["_updated_monotonic"] = monotonic()
        if len(_STATIC_CACHE) > _MAX_STATIC_CACHE_SIZE:
            oldest = min(
                _STATIC_CACHE,
                key=lambda key: _STATIC_CACHE[key].get("_updated_monotonic", 0.0),
            )
            _STATIC_CACHE.pop(oldest, None)


def _cached_static(mmsi: int) -> dict[str, Any]:
    with _STATIC_CACHE_LOCK:
        return {
            key: value
            for key, value in _STATIC_CACHE.get(mmsi, {}).items()
            if not key.startswith("_")
        }


class _AisCollector:
    def __init__(self, max_vessels: int | None = None):
        self.max_vessels = max_vessels
        self.positions: dict[int, dict[str, Any]] = {}

    def process(
        self,
        event: Mapping[str, Any],
        received_at: str | None = None,
        received_monotonic: float | None = None,
    ) -> None:
        message_type = str(event.get("MessageType") or "")
        envelope = event.get("Message")
        metadata = event.get("MetaData")
        if not isinstance(envelope, Mapping):
            return
        data = envelope.get(message_type)
        if not isinstance(data, Mapping):
            return
        if not isinstance(metadata, Mapping):
            metadata = {}
        vessel_mmsi = _mmsi(metadata, data)
        if vessel_mmsi is None or data.get("Valid") is False:
            return

        static = _static_fields(message_type, data)
        metadata_name = str(metadata.get("ShipName") or "").strip()
        if metadata_name:
            static.setdefault("vessel_name", metadata_name)
        _remember_static(vessel_mmsi, static)

        if message_type not in _POSITION_TYPES:
            return
        if (
            self.max_vessels is not None
            and vessel_mmsi not in self.positions
            and len(self.positions) >= self.max_vessels
        ):
            return

        longitude = _finite_float(data.get("Longitude", metadata.get("longitude", metadata.get("Longitude"))))
        latitude = _finite_float(data.get("Latitude", metadata.get("latitude", metadata.get("Latitude"))))
        if longitude is None or latitude is None or not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
            return
        sog = _finite_float(data.get("Sog"))
        cog = _finite_float(data.get("Cog"))
        heading = _finite_float(data.get("TrueHeading"))
        fallback_time = received_at or _utc_now()
        data_time = _clean_time(metadata.get("time_utc"), fallback_time)
        received_tick = received_monotonic if received_monotonic is not None else monotonic()
        previous = self.positions.get(vessel_mmsi, {})
        reports = list(previous.get("_reports", ()))
        reports.append((received_tick, data_time))
        track = list(previous.get("_track", ()))
        track_point = {
            "longitude": longitude,
            "latitude": latitude,
            "sog_kn": sog if sog is not None and 0 <= sog < 102.3 else None,
            "cog_deg": cog if cog is not None and 0 <= cog < 360 else None,
            "heading_deg": heading if heading is not None and 0 <= heading < 360 else None,
            "data_time_utc": data_time,
            "_received_monotonic": received_tick,
        }
        # AISStream 偶尔会重发完全相同的位置报文。单船航迹只保留一次，
        # 避免前端折线出现大量零长度线段。
        if not track or any(
            track[-1].get(name) != track_point.get(name)
            for name in ("longitude", "latitude", "data_time_utc")
        ):
            track.append(track_point)
        position = {
            "longitude": longitude,
            "latitude": latitude,
            "sog_kn": sog if sog is not None and 0 <= sog < 102.3 else None,
            "cog_deg": cog if cog is not None and 0 <= cog < 360 else None,
            "heading_deg": heading if heading is not None and 0 <= heading < 360 else None,
            "mmsi": vessel_mmsi,
            "data_time_utc": data_time,
            "_received_monotonic": received_tick,
            "_reports": reports,
            "_track": track,
        }
        position.update(_cached_static(vessel_mmsi))
        for field in (
            "vessel_name",
            "ship_type",
            "ship_type_name",
            "length_m",
            "width_m",
            "draught_m",
        ):
            position.setdefault(field, None)
        self.positions[vessel_mmsi] = position

    def prune(self, cutoff_monotonic: float) -> None:
        """丢弃滚动窗口之外的船舶与位置报文统计。"""
        stale = []
        for mmsi, position in self.positions.items():
            reports = [
                report
                for report in position.get("_reports", ())
                if report[0] >= cutoff_monotonic
            ]
            if not reports or position.get("_received_monotonic", 0.0) < cutoff_monotonic:
                stale.append(mmsi)
            else:
                position["_reports"] = reports
                position["_track"] = [
                    point
                    for point in position.get("_track", ())
                    if point.get("_received_monotonic", 0.0) >= cutoff_monotonic
                ]
        for mmsi in stale:
            self.positions.pop(mmsi, None)

    def result(
        self,
        *,
        since_monotonic: float | None = None,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> list[dict[str, Any]]:
        vessels = []
        for mmsi, position in self.positions.items():
            if (
                since_monotonic is not None
                and position.get("_received_monotonic", 0.0) < since_monotonic
            ):
                continue
            if bounds is not None:
                west, south, east, north = bounds
                if not (
                    west <= position["longitude"] <= east
                    and south <= position["latitude"] <= north
                ):
                    continue
            merged = dict(position)
            reports = [
                report
                for report in merged.pop("_reports", ())
                if since_monotonic is None or report[0] >= since_monotonic
            ]
            merged.pop("_received_monotonic", None)
            merged.pop("_track", None)
            if reports:
                merged["first_seen_utc"] = reports[0][1]
                merged["last_seen_utc"] = reports[-1][1]
                merged["position_report_count"] = len(reports)
            merged.update(_cached_static(mmsi))
            vessels.append(merged)
        return sorted(vessels, key=lambda item: item["mmsi"])


async def _collect(
    bounding_box: list[list[float]],
    duration_seconds: float,
    max_vessels: int | None,
    api_key: str,
) -> list[dict[str, Any]]:
    collector = _AisCollector(max_vessels)
    subscription = {
        "APIKey": api_key,
        "BoundingBoxes": [bounding_box],
        "FilterMessageTypes": list(MESSAGE_TYPES),
    }
    confirmed = False
    try:
        async with websockets.connect(
            AISSTREAM_URL,
            compression="deflate",
            open_timeout=45,
            close_timeout=2,
            ping_interval=20,
            ping_timeout=20,
            max_size=2**20,
        ) as socket:
            await socket.send(json.dumps(subscription, separators=(",", ":")))
            deadline = monotonic() + duration_seconds
            while True:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(socket.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                event = json.loads(raw)
                if not isinstance(event, Mapping):
                    continue
                if event.get("MessageType") == "SubscriptionConfirmation":
                    confirmed = True
                    continue
                if event.get("error") or event.get("Error"):
                    raise AISStreamError(str(event.get("error") or event.get("Error")))
                collector.process(event)
    except AISStreamError:
        raise
    except (OSError, ValueError, websockets.WebSocketException) as exc:
        raise AISStreamError(f"AISStream 实时连接失败：{exc}") from exc
    if not confirmed:
        raise AISStreamError("AISStream 未返回订阅确认。")
    return collector.result()


class _RollingAisFeed:
    """单进程持续订阅全球 AIS，并保留最近五分钟的位置报文。"""

    def __init__(self, retention_seconds: float = ROLLING_HISTORY_SECONDS):
        self.retention_seconds = float(retention_seconds)
        self.collector = _AisCollector()
        self.lock = threading.RLock()
        self.ready = threading.Event()
        self.thread: threading.Thread | None = None
        self.api_key = ""
        self.coverage_started_monotonic: float | None = None
        self.connected = False
        self.last_error: str | None = None

    def ensure_started(self, api_key: str) -> None:
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return
            self.api_key = api_key
            self.ready.clear()
            self.connected = False
            self.last_error = None
            self.thread = threading.Thread(
                target=self._thread_main,
                name="aisstream-rolling-feed",
                daemon=True,
            )
            self.thread.start()

    def _thread_main(self) -> None:
        asyncio.run(self._run_forever())

    async def _run_forever(self) -> None:
        backoff_seconds = 1.0
        while True:
            try:
                subscription = {
                    "APIKey": self.api_key,
                    "BoundingBoxes": [GLOBAL_BOUNDING_BOX],
                    "FilterMessageTypes": list(MESSAGE_TYPES),
                }
                async with websockets.connect(
                    AISSTREAM_URL,
                    compression="deflate",
                    open_timeout=45,
                    close_timeout=2,
                    ping_interval=20,
                    ping_timeout=20,
                    max_size=2**20,
                ) as socket:
                    await socket.send(json.dumps(subscription, separators=(",", ":")))
                    confirmed = False
                    last_prune = monotonic()
                    async for raw in socket:
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8")
                        event = json.loads(raw)
                        if not isinstance(event, Mapping):
                            continue
                        if event.get("MessageType") == "SubscriptionConfirmation":
                            confirmed = True
                            with self.lock:
                                self.connected = True
                                self.last_error = None
                                # 每次重连都重新累计连续覆盖时长；连接中断期间的缺口
                                # 不能被误报为完整的五分钟历史窗口。
                                self.coverage_started_monotonic = monotonic()
                            self.ready.set()
                            backoff_seconds = 1.0
                            continue
                        if event.get("error") or event.get("Error"):
                            raise AISStreamError(str(event.get("error") or event.get("Error")))
                        if not confirmed:
                            continue
                        received_tick = monotonic()
                        received_at = _utc_now()
                        with self.lock:
                            self.collector.process(
                                event,
                                received_at=received_at,
                                received_monotonic=received_tick,
                            )
                            if received_tick - last_prune >= 10:
                                self.collector.prune(
                                    received_tick - self.retention_seconds
                                )
                                last_prune = received_tick
            except Exception as exc:  # 后台订阅必须自动恢复，错误由快照元数据暴露。
                with self.lock:
                    self.connected = False
                    self.last_error = str(exc)
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 30.0)

    def wait_for_initial_window(self) -> None:
        if not self.ready.wait(timeout=10):
            with self.lock:
                detail = self.last_error or "AISStream 订阅确认超时"
            raise AISStreamError(detail)
        with self.lock:
            started = self.coverage_started_monotonic
        if started is None:
            return
        remaining = INITIAL_WARMUP_SECONDS - (monotonic() - started)
        if remaining > 0:
            threading.Event().wait(remaining)

    def snapshot(
        self,
        *,
        bottom_left_lon: float,
        bottom_left_lat: float,
        top_right_lon: float,
        top_right_lat: float,
        window_seconds: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        now_tick = monotonic()
        cutoff = now_tick - window_seconds
        with self.lock:
            self.collector.prune(now_tick - self.retention_seconds)
            vessels = self.collector.result(
                since_monotonic=cutoff,
                bounds=(
                    bottom_left_lon,
                    bottom_left_lat,
                    top_right_lon,
                    top_right_lat,
                ),
            )
            started = self.coverage_started_monotonic
            connected = self.connected
            last_error = self.last_error
        coverage_seconds = (
            min(window_seconds, max(0.0, now_tick - started))
            if started is not None
            else 0.0
        )
        return vessels, {
            "history_coverage_seconds": round(coverage_seconds, 3),
            "history_complete": coverage_seconds >= window_seconds - 0.5,
            "stream_connected": connected,
            "stream_error": last_error,
        }

    def vessel_track(
        self,
        *,
        mmsi: int,
        window_seconds: float,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
        """返回一艘船的最新状态和滚动窗口内的真实位置序列。"""
        now_tick = monotonic()
        cutoff = now_tick - window_seconds
        with self.lock:
            self.collector.prune(now_tick - self.retention_seconds)
            raw = self.collector.positions.get(mmsi)
            vessel = None
            track: list[dict[str, Any]] = []
            if raw is not None:
                vessel = dict(raw)
                vessel.pop("_received_monotonic", None)
                vessel.pop("_reports", None)
                vessel.pop("_track", None)
                vessel.update(_cached_static(mmsi))
                for point in raw.get("_track", ()):
                    if point.get("_received_monotonic", 0.0) < cutoff:
                        continue
                    clean = dict(point)
                    clean.pop("_received_monotonic", None)
                    track.append(clean)
            started = self.coverage_started_monotonic
            connected = self.connected
            last_error = self.last_error
        coverage_seconds = (
            min(window_seconds, max(0.0, now_tick - started))
            if started is not None
            else 0.0
        )
        return vessel, track, {
            "history_coverage_seconds": round(coverage_seconds, 3),
            "history_complete": coverage_seconds >= window_seconds - 0.5,
            "stream_connected": connected,
            "stream_error": last_error,
        }


_ROLLING_FEED = _RollingAisFeed()


def fetch_vessels(
    *,
    bottom_left_lon: float,
    bottom_left_lat: float,
    top_right_lon: float,
    top_right_lat: float,
    duration_seconds: float = ROLLING_HISTORY_SECONDS,
    max_vessels: int | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """返回当前时刻向前滚动窗口内出现过的全部唯一船舶。"""
    del max_vessels  # 兼容旧调用；滚动快照不再设置船舶数量上限。
    window_seconds = float(duration_seconds)
    feed = _ROLLING_FEED
    feed.ensure_started(api_key or resolve_api_key())
    feed.wait_for_initial_window()
    snapshot_time = datetime.now(timezone.utc)
    vessels, feed_meta = feed.snapshot(
        bottom_left_lon=bottom_left_lon,
        bottom_left_lat=bottom_left_lat,
        top_right_lon=top_right_lon,
        top_right_lat=top_right_lat,
        window_seconds=window_seconds,
    )
    vessels, land_filter_meta = filter_land_vessels(vessels)
    return {
        "source": "AISStream",
        "area": {
            "bottom_left": [bottom_left_lon, bottom_left_lat],
            "top_right": [top_right_lon, top_right_lat],
        },
        "snapshot_time_utc": snapshot_time.isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
        "history_window_started_at_utc": (
            snapshot_time - timedelta(seconds=window_seconds)
        ).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "history_window_ended_at_utc": snapshot_time.isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z"),
        "history_window_seconds": window_seconds,
        **feed_meta,
        **land_filter_meta,
        "vessel_count": len(vessels),
        "position_report_count": sum(
            int(vessel.get("position_report_count", 0)) for vessel in vessels
        ),
        "vessels": vessels,
        "static_data_note": (
            "船型、船长、船宽和吃水来自低频静态 AIS 报文；"
            "本次窗口及服务缓存均未收到时返回 null。"
        ),
    }


def fetch_vessel_track(
    *,
    mmsi: int,
    duration_seconds: float = ROLLING_HISTORY_SECONDS,
    api_key: str | None = None,
) -> dict[str, Any]:
    """返回指定 MMSI 的最新状态及最近一段时间的真实 AIS 航迹。"""
    vessel_mmsi = int(mmsi)
    if not 100_000_000 <= vessel_mmsi <= 999_999_999:
        raise ValueError("mmsi 必须是 9 位有效 MMSI")
    window_seconds = min(float(ROLLING_HISTORY_SECONDS), max(2.0, float(duration_seconds)))
    feed = _ROLLING_FEED
    feed.ensure_started(api_key or resolve_api_key())
    feed.wait_for_initial_window()
    snapshot_time = datetime.now(timezone.utc)
    vessel, track, feed_meta = feed.vessel_track(
        mmsi=vessel_mmsi,
        window_seconds=window_seconds,
    )
    return {
        "source": "AISStream",
        "mmsi": vessel_mmsi,
        "snapshot_time_utc": snapshot_time.isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
        "history_window_seconds": window_seconds,
        **feed_meta,
        "found": vessel is not None,
        "vessel": vessel,
        "track_point_count": len(track),
        "track": track,
    }
