# -*- coding: utf-8 -*-
"""真正的 Bellhop3D 三维声线计算。

本模块一次向 ``bellhop3d.exe`` 提交仿真三维声速场、三维海底和
一个“仰角 × 方位角”声线扇面。它不会调用多个 Bellhop 2D 任务，
也不会把二维切面旋转后拼接成三维结果。
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .. import acoustic_environment
from ..bellhop_model import results as bellhop_results
from ..bellhop_model import runner as bellhop_runner
from ..ocean_environment_source import DEFAULT_SOURCE as environment_source


ROOT_NAME = "native_3d_ray"


def _write_vector(handle, values: Sequence[float], *, slash: bool = False) -> None:
    array = np.asarray(values, dtype=np.float64).ravel()
    for start in range(0, max(1, array.size), 8):
        chunk = array[start : start + 8]
        handle.write(" ".join(f"{value:.8f}" for value in chunk))
        handle.write(" /\n" if slash and start + 8 >= array.size else "\n")


def _validate_context(context: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = context["grid"]
    environment = context["environment"]
    bathymetry = context["bathymetry"]
    x_km = np.asarray(grid["x_rel_axis_km"], dtype=np.float64).ravel()
    y_km = np.asarray(grid["y_rel_axis_km"], dtype=np.float64).ravel()
    depth_m = np.asarray(environment["depth_m"], dtype=np.float64).ravel()
    speed = np.asarray(environment["sound_speed_mps"], dtype=np.float64)
    bottom_m = np.asarray(bathymetry["depth_m"], dtype=np.float64)
    expected_3d = (depth_m.size, y_km.size, x_km.size)
    expected_2d = (y_km.size, x_km.size)
    if speed.shape != expected_3d:
        raise ValueError(f"三维声速场尺寸错误：{speed.shape}，期望 {expected_3d}。")
    if bottom_m.shape != expected_2d:
        raise ValueError(f"三维海底尺寸错误：{bottom_m.shape}，期望 {expected_2d}。")
    if any(axis.size < 2 or np.any(~np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0)
           for axis in (x_km, y_km, depth_m)):
        raise ValueError("Bellhop3D 的 x/y/深度坐标轴必须是有限且严格递增的数组。")
    if np.any(~np.isfinite(speed)) or np.any((speed < 1300.0) | (speed > 1800.0)):
        raise ValueError("三维声速场含有无效数值。")
    if np.any(~np.isfinite(bottom_m)) or np.any(bottom_m <= 0.0):
        raise ValueError("三维海底场含有无效水深。")
    return x_km, y_km, depth_m, speed, bottom_m


def _write_ssp(path: Path, x_km: np.ndarray, y_km: np.ndarray,
               depth_m: np.ndarray, speed: np.ndarray) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"{x_km.size}\n")
        _write_vector(handle, x_km)
        handle.write(f"{y_km.size}\n")
        _write_vector(handle, y_km)
        handle.write(f"{depth_m.size}\n")
        _write_vector(handle, depth_m)
        # Acoustics Toolbox H 插值按 z -> y -> x 的顺序读取。
        for layer in speed:
            for row in layer:
                _write_vector(handle, row)


def _write_bathymetry(path: Path, x_km: np.ndarray, y_km: np.ndarray,
                      bottom_m: np.ndarray) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("'R'\n")
        handle.write(f"{x_km.size}\n")
        _write_vector(handle, x_km)
        handle.write(f"{y_km.size}\n")
        _write_vector(handle, y_km)
        for row in bottom_m:
            _write_vector(handle, row)


def _solver_betas_from_bearings(bearings_deg: np.ndarray) -> np.ndarray:
    # 门户方位角从地理北向开始顺时针增加；Bellhop3D beta 从 +x（东）
    # 开始逆时针增加。本地 x/y 分别是东/北，因此需要转换。
    converted = np.mod(90.0 - np.asarray(bearings_deg, dtype=np.float64), 360.0)
    # Bellhop3D 的角度读取器要求扇面单调；跨越 0° 的地理方位角
    # 转换后会换序，所以在写入求解器前排序。
    return np.sort(converted)


def _bearings_from_solver_betas(solver_betas: np.ndarray) -> np.ndarray:
    return np.mod(90.0 - np.asarray(solver_betas, dtype=np.float64), 360.0)


def _write_environment(path: Path, params: Mapping[str, Any], settings: Mapping[str, Any],
                       context: Mapping[str, Any], bearings_deg: np.ndarray,
                       line_count: int, depth_m: np.ndarray, speed: np.ndarray) -> None:
    model_depth = float(depth_m[-1])
    source_depth = float(params["source_depth_m"])
    radius_km = float(params["radius_km"])
    receiver_bottom = max(0.1, model_depth - float(settings["depth_guard_m"]))
    centre_y = speed.shape[1] // 2
    centre_x = speed.shape[2] // 2
    profile = speed[:, centre_y, centre_x]
    bottom = settings["bottom_fixed"]
    solver_betas = _solver_betas_from_bearings(bearings_deg)
    box_km = radius_km + float(settings["trace_extra_km"])
    lines = [
        "'Native Bellhop3D ray trace'",
        f"{float(params['freq_hz']):.8f}",
        "1",
        "'HVW   '",
        f"{depth_m.size} 0.0 {model_depth:.6f}",
    ]
    lines.extend(f"{z:.6f} {c:.8f} /" for z, c in zip(depth_m, profile))
    lines.extend([
        f"'A*' {float(settings['bottom_roughness_m']):.6f}",
        f"{model_depth:.6f} {float(bottom['compressional_speed_mps']):.8f} "
        f"{float(bottom['shear_speed_mps']):.8f} {float(bottom['density_g_cm3']):.8f} "
        f"{float(bottom['compressional_attenuation']):.8f} "
        f"{float(bottom['shear_attenuation']):.8f} /",
        "1", "0.0 /",                 # NSx, Sx (km)
        "1", "0.0 /",                 # NSy, Sy (km)
        "1", f"{source_depth:.6f} /", # NSz, Sz (m)
        "2", f"0.0 {receiver_bottom:.6f} /",
        "2", f"0.0 {radius_km:.8f} /",
        str(len(solver_betas)),
        " ".join(f"{value:.8f}" for value in solver_betas) + " /",
        "'RG RR3'",
        # 第二个 0 是旧版 Acoustics Toolbox 的 ISINGLE。新版只读第一个数，
        # 同行剩余字段会被安全忽略，因此该写法同时兼容新旧版。
        f"{line_count} 0",
        f"{float(params['alpha_min']):.8f} {float(params['alpha_max']):.8f} /",
        f"{len(solver_betas)} 0",
        " ".join(f"{value:.8f}" for value in solver_betas) + " /",
        f"{float(settings['step_m']):.6f} {box_km:.8f} {box_km:.8f} "
        f"{model_depth + float(settings['zbox_margin_m']):.6f}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _run_solver(executable: Path, work_dir: Path, timeout_sec: float) -> None:
    command = [str(executable), ROOT_NAME]
    (work_dir / "bellhop3d_command.txt").write_text(
        subprocess.list2cmdline(command), encoding="utf-8"
    )
    try:
        completed = subprocess.run(
            command,
            cwd=str(work_dir),
            timeout=float(timeout_sec),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Bellhop3D 运行超过 {timeout_sec:.0f} 秒。") from exc
    except OSError as exc:
        raise RuntimeError(f"Bellhop3D 可执行文件无法启动：{executable}；{exc}") from exc
    (work_dir / "bellhop3d_stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (work_dir / "bellhop3d_stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    problem = bellhop_runner._prt_error(work_dir / f"{ROOT_NAME}.prt")
    if completed.returncode != 0 or problem:
        detail = problem or completed.stderr.strip() or completed.stdout.strip() or "未生成有效结果"
        raise RuntimeError(
            f"Bellhop3D 计算失败（返回码 {completed.returncode}）：{detail[-1500:]}"
        )


def _clip_points(points: np.ndarray, radius_km: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    good = np.all(np.isfinite(values), axis=1)
    values = values[good]
    if values.size == 0:
        return values.reshape(0, 3)
    radius_m = float(radius_km) * 1000.0
    radial = np.hypot(values[:, 0], values[:, 1])
    outside = np.flatnonzero(radial > radius_m)
    if outside.size:
        stop = int(outside[0])
        if stop > 0:
            p0, p1 = values[stop - 1], values[stop]
            r0, r1 = radial[stop - 1], radial[stop]
            fraction = 0.0 if r1 <= r0 else (radius_m - r0) / (r1 - r0)
            crossing = p0 + np.clip(fraction, 0.0, 1.0) * (p1 - p0)
            values = np.vstack((values[:stop], crossing))
        else:
            values = values[:1]
    return values


def build_response(ray_path: Path, source_lon: float, source_lat: float,
                   bearings_deg: Sequence[float], line_count: int,
                   radius_km: float) -> dict[str, Any]:
    header, rays = bellhop_results.parse_ray_file(ray_path)
    if header.get("simulation_dimension") != "3D" or "xyz" not in str(header.get("coord_type", "")):
        raise ValueError("Bellhop3D 返回的不是 XYZ 三维声线文件。")
    expected = len(bearings_deg) * int(line_count)
    if len(rays) != expected:
        raise ValueError(
            f"Bellhop3D 声线数量不完整：期望 {expected} 条，实际 {len(rays)} 条。"
        )
    response_items = []
    for index, ray in enumerate(rays):
        bearing = float(bearings_deg[index // int(line_count)])
        points = _clip_points(np.asarray(ray["points"], dtype=np.float64), radius_km)
        if points.shape[0] == 0:
            raise ValueError(f"Bellhop3D 第 {index + 1} 条声线没有有效坐标点。")
        longitude, latitude = bellhop_results.local_offset_to_lonlat(
            points[:, 0] / 1000.0,
            points[:, 1] / 1000.0,
            float(source_lon),
            float(source_lat),
        )
        depth = np.abs(points[:, 2])
        response_items.append({
            "bearingDeg": bearing,
            "points": {
                "size": int(points.shape[0]),
                "longitude": np.round(longitude, 6).tolist(),
                "latitude": np.round(latitude, 6).tolist(),
                "depthM": np.round(depth, 6).tolist(),
            },
        })
    response_items.sort(key=lambda item: float(item["bearingDeg"]))
    return {"size": len(response_items), "bearings": response_items}


def run(job_dir: str | Path, params: dict[str, Any], config: Mapping[str, Any],
        sector: int, line_count: int) -> dict[str, Any]:
    started = time.time()
    output = Path(job_dir)
    native_dir = output / "native_bellhop3d"
    native_dir.mkdir(parents=True, exist_ok=True)
    merged = bellhop_runner._task_defaults(config)
    merged.update(params)
    params.clear()
    params.update(merged)
    bellhop_runner.validate_params(params)
    settings = bellhop_runner._effective_settings(params, config)
    executable = Path(str(config.get("BELLHOP_CXX3d") or "")).expanduser()
    if not executable.is_file():
        raise FileNotFoundError(f"Bellhop3D 可执行文件不存在：{executable or '(空路径)'}")

    points = bellhop_runner._normalise_source_points(params, 1)
    prepared = acoustic_environment.acquire(params, config, settings, points, environment_source)
    try:
        context = prepared.build_common_source_context(0)
    finally:
        prepared.close()

    x_km, y_km, depth_m, speed, bottom_m = _validate_context(context)
    bearings = np.linspace(0.0, 360.0, int(sector), endpoint=False, dtype=np.float64)
    _write_ssp(native_dir / f"{ROOT_NAME}.ssp", x_km, y_km, depth_m, speed)
    _write_bathymetry(native_dir / f"{ROOT_NAME}.bty", x_km, y_km, bottom_m)
    _write_environment(
        native_dir / f"{ROOT_NAME}.env",
        params,
        settings,
        context,
        bearings,
        int(line_count),
        depth_m,
        speed,
    )
    manifest = {
        "engine": "Bellhop3D native 3D ray tracing",
        "calculation_method": "native_bellhop3d_hexahedral_ssp",
        "executable": str(executable),
        "sector": int(sector),
        "line_count_per_sector": int(line_count),
        "expected_ray_count": int(sector) * int(line_count),
        "bearings_deg": bearings.tolist(),
        "solver_beta_deg": _solver_betas_from_bearings(bearings).tolist(),
        "ssp_shape_z_y_x": list(speed.shape),
        "bathymetry_shape_y_x": list(bottom_m.shape),
    }
    (native_dir / "bellhop3d_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _run_solver(executable, native_dir, float(settings["timeout_sec"]))
    solver_betas = _solver_betas_from_bearings(bearings)
    response = build_response(
        native_dir / f"{ROOT_NAME}.ray",
        float(context["source_lon"]),
        float(context["source_lat"]),
        _bearings_from_solver_betas(solver_betas),
        int(line_count),
        float(params["radius_km"]),
    )
    return {
        "status": "ready",
        "elapsed": time.time() - started,
        "response": response,
        "manifest": manifest,
    }
