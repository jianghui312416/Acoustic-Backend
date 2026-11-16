"""二维声线动态帧构建。

把 ``Analyse2DRTRD`` 的静态声线返回（``trackData.data``）展开为随时间变化的
可见声线片段，供前端实现「流星 / 水流」式声线动画。输入和输出都沿用门户
二维声线接口的坐标约定：

- ``x``：水平距离，单位米；
- ``z``：深度，单位米，向下为正。

每一帧只保留当前时刻的「尾 → 头」片段。``trail_ratio`` / ``trail_length_m``
都为 0 时尾迹固定在声源，声线从声源逐步伸长到终点。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

PORTAL_INVALID_DATA_VALUE = 9999.0

DEFAULT_FRAME_COUNT = 60
DEFAULT_TRAIL_RATIO = 0.25
DEFAULT_OUTPUT_DECIMALS = 6
DEFAULT_SOUND_SPEED_MPS = 1500.0


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round_number(value: float, decimals: int) -> float:
    return round(float(value), decimals)


def _axis_bracket(axis: list[float], target: float) -> tuple[int, int, float]:
    if len(axis) <= 1 or target <= axis[0]:
        return 0, 0, 0.0
    if target >= axis[-1]:
        last = len(axis) - 1
        return last, last, 0.0
    lo = 0
    hi = len(axis) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if axis[mid] <= target:
            lo = mid
        else:
            hi = mid
    span = axis[hi] - axis[lo]
    return lo, hi, (target - axis[lo]) / span if span > 0.0 else 0.0


def _normalise_profile(depths: Any, speeds: Any) -> tuple[list[float], list[float]]:
    pairs = []
    depth_values = [] if depths is None else depths
    speed_values = [] if speeds is None else speeds
    for depth_raw, speed_raw in zip(depth_values, speed_values):
        depth = _finite_number(depth_raw)
        speed = _finite_number(speed_raw)
        if depth is not None and speed is not None and speed > 0.0:
            pairs.append((depth, speed))
    pairs.sort(key=lambda item: item[0])
    return [item[0] for item in pairs], [item[1] for item in pairs]


def _normalise_sound_speed(sound_speed: Mapping[str, Any] | None) -> dict[str, Any]:
    source = sound_speed or {}
    depths, profile = _normalise_profile(
        source.get("ssp_depth_m"), source.get("ssp_sound_speed_mps")
    )
    model: dict[str, Any] = {
        "depths": depths,
        "profile": profile,
        "fallback": DEFAULT_SOUND_SPEED_MPS,
        "source": "source_ssp" if depths else "constant_1500_mps_fallback",
    }

    range_source = source.get("sound_speed_range_m")
    ranges = [_finite_number(value) for value in ([] if range_source is None else range_source)]
    raw_grid = source.get("section_sound_speed_mps")
    if depths and raw_grid is not None and ranges and all(value is not None for value in ranges):
        range_values = [float(value) for value in ranges]
        rows = [list(row) for row in raw_grid]
        if len(rows) == len(depths) and all(len(row) == len(range_values) for row in rows):
            grid = rows
        elif len(rows) == len(range_values) and all(len(row) == len(depths) for row in rows):
            grid = [list(column) for column in zip(*rows)]
        else:
            grid = []
        if grid:
            order = sorted(range(len(range_values)), key=range_values.__getitem__)
            model["ranges"] = [range_values[index] for index in order]
            model["grid"] = [[row[index] for index in order] for row in grid]
            model["source"] = "range_dependent_section_ssp"
    return model


def _sound_speed_at(model: Mapping[str, Any], range_m: float, depth_m: float) -> float:
    depths = model["depths"]
    profile = model["profile"]
    if not depths:
        return float(model["fallback"])
    depth_lo, depth_hi, depth_fraction = _axis_bracket(depths, depth_m)
    ranges = model.get("ranges")
    grid = model.get("grid")
    if ranges and grid:
        range_lo, range_hi, range_fraction = _axis_bracket(ranges, range_m)
        weighted = []
        for depth_index, depth_weight in (
            (depth_lo, 1.0 - depth_fraction), (depth_hi, depth_fraction)
        ):
            for range_index, range_weight in (
                (range_lo, 1.0 - range_fraction), (range_hi, range_fraction)
            ):
                value = _finite_number(grid[depth_index][range_index])
                weight = depth_weight * range_weight
                if value is not None and value > 0.0 and weight > 0.0:
                    weighted.append((weight, value))
        if weighted:
            total = sum(weight for weight, _ in weighted)
            return sum(weight * value for weight, value in weighted) / total
    first = profile[depth_lo]
    second = profile[depth_hi]
    return first + (second - first) * depth_fraction


def _build_track(
    points: list[tuple[float, float]], sound_speed_model: Mapping[str, Any]
) -> dict[str, Any] | None:
    if len(points) < 2:
        return None
    cumulative = [0.0]
    cumulative_time_s = [0.0]
    for index in range(1, len(points)):
        dx = points[index][0] - points[index - 1][0]
        dz = points[index][1] - points[index - 1][1]
        segment_m = math.hypot(dx, dz)
        cumulative.append(cumulative[-1] + segment_m)
        midpoint_x = 0.5 * (points[index][0] + points[index - 1][0])
        midpoint_z = 0.5 * (points[index][1] + points[index - 1][1])
        speed_mps = _sound_speed_at(sound_speed_model, midpoint_x, midpoint_z)
        cumulative_time_s.append(cumulative_time_s[-1] + segment_m / speed_mps)
    length_m = cumulative[-1]
    if not (length_m > 0.0):
        return None
    travel_time_s = cumulative_time_s[-1]
    return {
        "points": points,
        "cumulative": cumulative,
        "cumulative_time_s": cumulative_time_s,
        "length_m": length_m,
        "travel_time_s": travel_time_s,
        "mean_sound_speed_mps": length_m / travel_time_s,
    }


def _build_3d_track(
    points: list[tuple[float, float, float]],
    sound_speed_model: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build one local East/North/depth ray with physical travel time."""
    if len(points) < 2:
        return None
    cumulative = [0.0]
    cumulative_time_s = [0.0]
    for index in range(1, len(points)):
        start = points[index - 1]
        end = points[index]
        segment_m = math.sqrt(sum((end[axis] - start[axis]) ** 2 for axis in range(3)))
        cumulative.append(cumulative[-1] + segment_m)
        midpoint_x = 0.5 * (start[0] + end[0])
        midpoint_y = 0.5 * (start[1] + end[1])
        midpoint_z = 0.5 * (start[2] + end[2])
        speed_mps = _sound_speed_at(
            sound_speed_model,
            math.hypot(midpoint_x, midpoint_y),
            midpoint_z,
        )
        cumulative_time_s.append(cumulative_time_s[-1] + segment_m / speed_mps)
    length_m = cumulative[-1]
    if not (length_m > 0.0):
        return None
    travel_time_s = cumulative_time_s[-1]
    return {
        "points": points,
        "cumulative": cumulative,
        "cumulative_time_s": cumulative_time_s,
        "length_m": length_m,
        "travel_time_s": travel_time_s,
        "mean_sound_speed_mps": length_m / travel_time_s,
    }


def _distance_at_time(track: Mapping[str, Any], elapsed_s: float) -> float:
    times = track["cumulative_time_s"]
    distances = track["cumulative"]
    if elapsed_s <= 0.0:
        return 0.0
    if elapsed_s >= times[-1]:
        return track["length_m"]
    lo = 0
    hi = len(times) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if times[mid] <= elapsed_s:
            lo = mid
        else:
            hi = mid
    span = times[hi] - times[lo]
    fraction = (elapsed_s - times[lo]) / span if span > 0.0 else 0.0
    return distances[lo] + (distances[hi] - distances[lo]) * fraction


def _point_at_distance(track: Mapping[str, Any], distance_m: float) -> list[float]:
    points = track["points"]
    cumulative = track["cumulative"]
    length_m = track["length_m"]
    if distance_m <= 0.0:
        return list(points[0])
    if distance_m >= length_m:
        return list(points[-1])

    lo = 0
    hi = len(cumulative) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cumulative[mid] <= distance_m:
            lo = mid
        else:
            hi = mid

    start = points[lo]
    end = points[hi]
    span = cumulative[hi] - cumulative[lo]
    ratio = (distance_m - cumulative[lo]) / span if span > 0.0 else 0.0
    return [
        start[axis] + (end[axis] - start[axis]) * ratio
        for axis in range(len(start))
    ]


def _sample_track_segment(
    track: Mapping[str, Any],
    tail_distance_m: float,
    head_distance_m: float,
    decimals: int,
) -> dict[str, Any]:
    head_distance = min(max(head_distance_m, 0.0), track["length_m"])
    tail_distance = min(max(tail_distance_m, 0.0), head_distance)
    if not (head_distance > 0.0):
        return {
            "x": [],
            "z": [],
            "head": None,
            "tail": None,
            "headDistanceM": 0.0,
            "tailDistanceM": 0.0,
        }

    xs: list[float] = []
    zs: list[float] = []
    tail_point = _point_at_distance(track, tail_distance)
    xs.append(_round_number(tail_point[0], decimals))
    zs.append(_round_number(tail_point[1], decimals))

    cumulative = track["cumulative"]
    for index, point in enumerate(track["points"]):
        distance = cumulative[index]
        if distance <= tail_distance + 1e-9 or distance >= head_distance - 1e-9:
            continue
        x = _round_number(point[0], decimals)
        z = _round_number(point[1], decimals)
        if abs(x - xs[-1]) <= 1e-9 and abs(z - zs[-1]) <= 1e-9:
            continue
        xs.append(x)
        zs.append(z)

    head_point = _point_at_distance(track, head_distance)
    head_x = _round_number(head_point[0], decimals)
    head_z = _round_number(head_point[1], decimals)
    if abs(head_x - xs[-1]) > 1e-9 or abs(head_z - zs[-1]) > 1e-9:
        xs.append(head_x)
        zs.append(head_z)

    return {
        "x": xs,
        "z": zs,
        "head": [head_x, head_z],
        "tail": [_round_number(tail_point[0], decimals), _round_number(tail_point[1], decimals)],
        "headDistanceM": _round_number(head_distance, decimals),
        "tailDistanceM": _round_number(tail_distance, decimals),
    }


def build_dynamic_ray_frames(
    ray_response: Mapping[str, Any],
    *,
    frame_count: int = DEFAULT_FRAME_COUNT,
    trail_ratio: float = DEFAULT_TRAIL_RATIO,
    trail_length_m: float = 0.0,
    decimals: int = DEFAULT_OUTPUT_DECIMALS,
    sound_speed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """在静态二维声线返回上追加 ``frames`` 动态帧。

    返回值保留原返回的 ``sampling`` / ``trackData``（冰区还保留 ``iceThickness``），
    因此调用方既可以用 ``frames`` 播放动画，也可以继续使用原来的静态字段。
    """

    frame_count = max(2, min(2000, int(frame_count)))
    trail_ratio = min(1.0, max(0.0, float(trail_ratio)))
    trail_length_m = max(0.0, float(trail_length_m)) if math.isfinite(float(trail_length_m)) else 0.0
    decimals = max(0, min(12, int(decimals)))

    result = dict(ray_response)
    tracks: list[dict[str, Any]] = []
    sound_speed_model = _normalise_sound_speed(sound_speed)
    raw_tracks = (ray_response.get("trackData") or {}).get("data") or []
    for raw_track in raw_tracks:
        if not isinstance(raw_track, Mapping):
            continue
        xs = raw_track.get("x") or []
        zs = raw_track.get("z") or []
        points: list[tuple[float, float]] = []
        for x_raw, z_raw in zip(xs, zs):
            x = _finite_number(x_raw)
            z = _finite_number(z_raw)
            # 门户用 9999 表示缺测；深度 9999 m 在本接口中不是合法声线点。
            if x is None or z is None or z == PORTAL_INVALID_DATA_VALUE:
                continue
            points.append((x, z))
        track = _build_track(points, sound_speed_model)
        if track is not None:
            tracks.append(track)

    duration_s = max((track["travel_time_s"] for track in tracks), default=0.0)

    max_dist_m = max(
        (point[0] for track in tracks for point in track["points"]),
        default=0.0,
    )
    max_depth_m = max(
        (point[1] for track in tracks for point in track["points"]),
        default=0.0,
    )
    source_depth_m = tracks[0]["points"][0][1] if tracks else None

    frames: list[dict[str, Any]] = []
    for frame_index in range(frame_count):
        progress = frame_index / (frame_count - 1)
        elapsed_s = progress * duration_s
        frame_rays: list[dict[str, Any]] = []
        for ray_id, track in enumerate(tracks):
            head_distance_m = _distance_at_time(track, elapsed_s)
            tail_length = trail_length_m if trail_length_m > 0.0 else track["length_m"] * trail_ratio
            tail_distance_m = max(0.0, head_distance_m - tail_length) if tail_length > 0.0 else 0.0
            segment = _sample_track_segment(track, tail_distance_m, head_distance_m, decimals)
            frame_rays.append({
                "id": ray_id,
                **segment,
                "progress": min(1.0, head_distance_m / track["length_m"]),
            })
        frames.append({
            "index": frame_index,
            "timeMs": _round_number(elapsed_s * 1000.0, decimals),
            "progress": progress,
            "rays": frame_rays,
        })

    result.update({
        "frameCount": frame_count,
        "timeModel": "path_integral_ds_over_c",
        "soundSpeedSource": sound_speed_model["source"],
        "trailRatio": trail_ratio,
        "trailLengthM": _round_number(trail_length_m, decimals),
        "rayCount": len(tracks),
        "maxDistM": _round_number(max_dist_m, decimals),
        "maxDepthM": _round_number(max_depth_m, decimals),
        "sourceDepthM": None if source_depth_m is None else _round_number(source_depth_m, decimals),
        "rays": [
            {
                "id": ray_id,
                "lengthM": _round_number(track["length_m"], decimals),
                "travelTimeMs": _round_number(track["travel_time_s"] * 1000.0, decimals),
                "meanSoundSpeedMps": _round_number(track["mean_sound_speed_mps"], decimals),
                "pointCount": len(track["points"]),
            }
            for ray_id, track in enumerate(tracks)
        ],
        "frames": frames,
    })
    return result


def _sample_3d_track_segment(
    track: Mapping[str, Any],
    tail_distance_m: float,
    head_distance_m: float,
    decimals: int,
) -> dict[str, Any]:
    head_distance = min(max(head_distance_m, 0.0), track["length_m"])
    tail_distance = min(max(tail_distance_m, 0.0), head_distance)
    if not (head_distance > 0.0):
        return {
            "x": [], "y": [], "z": [], "head": None, "tail": None,
            "headDistanceM": 0.0, "tailDistanceM": 0.0,
        }

    points = []
    tail_point = _point_at_distance(track, tail_distance)
    points.append(tail_point)
    for index, point in enumerate(track["points"]):
        distance = track["cumulative"][index]
        if tail_distance + 1.0e-9 < distance < head_distance - 1.0e-9:
            points.append(list(point))
    head_point = _point_at_distance(track, head_distance)
    if any(abs(head_point[axis] - points[-1][axis]) > 1.0e-9 for axis in range(3)):
        points.append(head_point)
    rounded = [
        [_round_number(point[axis], decimals) for axis in range(3)]
        for point in points
    ]
    return {
        "x": [point[0] for point in rounded],
        "y": [point[1] for point in rounded],
        "z": [point[2] for point in rounded],
        "head": rounded[-1],
        "tail": rounded[0],
        "headDistanceM": _round_number(head_distance, decimals),
        "tailDistanceM": _round_number(tail_distance, decimals),
    }


def build_dynamic_3d_ray_frames(
    ray_response: Mapping[str, Any],
    *,
    frame_count: int = DEFAULT_FRAME_COUNT,
    trail_ratio: float = 0.0,
    trail_length_m: float = 0.0,
    decimals: int = DEFAULT_OUTPUT_DECIMALS,
    sound_speed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append synchronized dynamic frames to a multi-bearing 3-D ray response.

    Static tracks and frame segments use local Cartesian metres: x points east,
    y points north and z is positive-down depth.
    """
    frame_count = max(2, min(2000, int(frame_count)))
    trail_ratio = min(1.0, max(0.0, float(trail_ratio)))
    trail_length_m = (
        max(0.0, float(trail_length_m))
        if math.isfinite(float(trail_length_m)) else 0.0
    )
    decimals = max(0, min(12, int(decimals)))
    sound_speed_model = _normalise_sound_speed(sound_speed)

    tracks: list[dict[str, Any]] = []
    for raw_track in (ray_response.get("trackData") or {}).get("data") or []:
        if not isinstance(raw_track, Mapping):
            continue
        points: list[tuple[float, float, float]] = []
        for x_raw, y_raw, z_raw in zip(
            raw_track.get("x") or [], raw_track.get("y") or [], raw_track.get("z") or []
        ):
            x = _finite_number(x_raw)
            y = _finite_number(y_raw)
            z = _finite_number(z_raw)
            if x is None or y is None or z is None or z == PORTAL_INVALID_DATA_VALUE:
                continue
            points.append((x, y, z))
        track = _build_3d_track(points, sound_speed_model)
        if track is not None:
            track["id"] = int(raw_track.get("id", len(tracks)))
            track["azimuth"] = float(raw_track.get("azimuth", 0.0)) % 360.0
            tracks.append(track)

    duration_s = max((track["travel_time_s"] for track in tracks), default=0.0)
    frames: list[dict[str, Any]] = []
    for frame_index in range(frame_count):
        progress = frame_index / (frame_count - 1)
        elapsed_s = progress * duration_s
        frame_rays = []
        for track in tracks:
            head_distance_m = _distance_at_time(track, elapsed_s)
            tail_length = (
                trail_length_m if trail_length_m > 0.0
                else track["length_m"] * trail_ratio
            )
            tail_distance_m = (
                max(0.0, head_distance_m - tail_length) if tail_length > 0.0 else 0.0
            )
            segment = _sample_3d_track_segment(
                track, tail_distance_m, head_distance_m, decimals
            )
            frame_rays.append({
                "id": track["id"],
                "azimuth": _round_number(track["azimuth"], decimals),
                **segment,
                "progress": min(1.0, head_distance_m / track["length_m"]),
            })
        frames.append({
            "index": frame_index,
            "timeMs": _round_number(elapsed_s * 1000.0, decimals),
            "progress": progress,
            "rays": frame_rays,
        })

    result = dict(ray_response)
    result.update({
        "frameCount": frame_count,
        "timeModel": "path_integral_ds_over_c",
        "soundSpeedSource": sound_speed_model["source"],
        "trailRatio": trail_ratio,
        "trailLengthM": _round_number(trail_length_m, decimals),
        "rayCount": len(tracks),
        "rays": [
            {
                "id": track["id"],
                "azimuth": _round_number(track["azimuth"], decimals),
                "lengthM": _round_number(track["length_m"], decimals),
                "travelTimeMs": _round_number(track["travel_time_s"] * 1000.0, decimals),
                "meanSoundSpeedMps": _round_number(track["mean_sound_speed_mps"], decimals),
                "pointCount": len(track["points"]),
            }
            for track in tracks
        ],
        "frames": frames,
    })
    return result
