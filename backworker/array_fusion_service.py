"""拖曳线阵多域联动的实时 WebSocket 计算服务。

本模块不创建 FastAPI 应用，也不修改现有任务系统。``backend_worker`` 只需把
WebSocket 对象交给 :func:`serve_array_fusion_websocket`，即可获得独立会话级的
阵元信号仿真、时频谱、常规波束形成和检测结果。客户端可先发送
``{"type":"bellhop_bind","bellhop_job_id":"..."}``，把已完成任务中的实际
海底地形、声速剖面、二维 TL 和声线几何绑定到会话；后续 frame 会自动使用同一
声场计算传播损失与多途参数，并按 VirTEX 一阶平台运动模型对每条特征声线分别
计算路径伸缩、多普勒和相位。``bellhop_series_bind`` 可绑定多个时刻的多方位
Bellhop 任务，并在距离、深度、方位和时间上插值。``field_req`` 可重新取得声场；
``static_channel_req`` 返回固定声源和固定阵列的每阵元 TL、复数冲激响应和声线。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import shutil
from functools import lru_cache
from pathlib import Path
from dataclasses import asdict, dataclass, field as dataclass_field, replace
from typing import Any, Callable, Mapping

import numpy as np
from fastapi import WebSocketDisconnect
from scipy.signal import resample_poly


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEASURED_NOISE_FILE = PROJECT_ROOT / "HYdata" / "blue_noise.mat"
BEAM_POINT_COUNT = 121
SPECTRUM_POINT_COUNT = 72
FRAME_SAMPLE_COUNT = 512
FIELD_MAX_RANGE_POINTS = 180
FIELD_MAX_DEPTH_POINTS = 100
MAX_MULTIPATH_COUNT = 8
MAX_NATIVE_ARRIVAL_COUNT = 64
STATIC_CHANNEL_MAX_ELEMENTS = 512
MAX_SERIES_SNAPSHOTS = 64
MAX_RAY_POINTS_PER_DIRECTION = 800
_JOB_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{4,80}")


def _parse_bellhop_ascii_arrivals(path: Path) -> dict[str, Any]:
    """读取 Bellhop 二维 ASCII arrivals，保留原生复幅相和角度。"""

    tokens = path.read_text(encoding="ascii", errors="replace").split()
    position = 0

    def take_float() -> float:
        nonlocal position
        if position >= len(tokens):
            raise ValueError(f"Bellhop arrivals 文件意外结束：{path}")
        token = tokens[position].replace("D", "E").replace("d", "e")
        position += 1
        try:
            return float(token)
        except ValueError as exc:
            raise ValueError(f"Bellhop arrivals 数值无效：{token!r}") from exc

    def take_int() -> int:
        return int(round(take_float()))

    if not tokens:
        raise ValueError(f"Bellhop arrivals 文件为空：{path}")
    format_token = tokens[0].strip("'\"").upper()
    if format_token == "2D":
        position = 1
    elif format_token == "3D":
        raise ValueError("阵列联动当前仅支持 Bellhop 二维 ASCII arrivals")
    frequency_hz = take_float()
    source_count, receiver_count, range_count = take_int(), take_int(), take_int()
    if min(source_count, receiver_count, range_count) <= 0:
        raise ValueError(f"Bellhop arrivals 网格维度无效：{path}")
    source_depths_m = [take_float() for _ in range(source_count)]
    receiver_depths_m = [take_float() for _ in range(receiver_count)]
    raw_ranges = [take_float() for _ in range(range_count)]
    max_arrivals = take_int()
    records: list[dict[str, Any]] = []
    for source_depth_m in source_depths_m:
        for receiver_depth_m in receiver_depths_m:
            for raw_range in raw_ranges:
                arrival_count = take_int()
                if not 0 <= arrival_count <= max_arrivals:
                    raise ValueError(
                        f"Bellhop arrivals 数量 {arrival_count} 超出 [0, {max_arrivals}]：{path}"
                    )
                for arrival_index in range(arrival_count):
                    records.append({
                        "arrival_index": arrival_index + 1,
                        "source_depth_m": source_depth_m,
                        "receiver_depth_m": receiver_depth_m,
                        "raw_range": raw_range,
                        "amplitude": take_float(),
                        "phase_deg": take_float(),
                        "delay_s": take_float(),
                        "delay_imag_s": take_float(),
                        "source_angle_deg": take_float(),
                        "arrival_angle_deg": take_float(),
                        "top_bounces": take_int(),
                        "bottom_bounces": take_int(),
                    })
    return {
        "path": str(path),
        "frequency_hz": frequency_hz,
        "source_depths_m": source_depths_m,
        "receiver_depths_m": receiver_depths_m,
        "raw_ranges": raw_ranges,
        "max_arrivals": max_arrivals,
        "records": records,
    }


def _write_point_arrival_environment(
    template_path: Path,
    target_path: Path,
    receiver_depth_m: float,
    ranges_km: list[float],
    beam_count: int,
) -> None:
    """由本项目生成的 TL ENV 派生同环境的接收点 arrivals ENV。"""

    lines = template_path.read_text(encoding="ascii", errors="strict").splitlines()
    if len(lines) < 15:
        raise ValueError(f"Bellhop ENV 结构不完整：{template_path}")
    # environment.write_environment_file 的末十行固定为：源深、接收深、距离、
    # 运行类型、波束数、角范围和计算盒。只替换接收网格与运行类型，水体、
    # 海面、海底和声源参数均保持与 TL 任务一致。
    tail = lines[-10:]
    try:
        source_count = int(tail[0].strip())
        float(tail[1].split()[0])
        int(tail[2].strip())
        int(tail[4].strip())
        int(tail[7].strip())
    except (ValueError, IndexError) as exc:
        raise ValueError(f"无法识别 Bellhop ENV 接收网格：{template_path}") from exc
    if source_count != 1:
        raise ValueError("静态阵列 arrivals 当前要求 Bellhop ENV 只有一个声源深度")
    unique_ranges = sorted({round(float(value), 9) for value in ranges_km})
    if not unique_ranges or unique_ranges[0] <= 0.0:
        raise ValueError("arrivals 接收距离必须为正数")
    range_lines = []
    for start in range(0, len(unique_ranges), 8):
        chunk = unique_ranges[start : start + 8]
        suffix = " /" if start + 8 >= len(unique_ranges) else ""
        range_lines.append(" ".join(f"{value:.9f}" for value in chunk) + suffix)
    replacement = [
        "1",
        tail[1],
        "1",
        f"{float(receiver_depth_m):.6f} /",
        str(len(unique_ranges)),
        *range_lines,
        "'AG'",
        str(int(beam_count)),
        tail[8],
        tail[9],
    ]
    target_path.write_text(
        "\n".join([*lines[:-10], *replacement]) + "\n",
        encoding="ascii",
    )


def _friendly_arrival_solver_error(value: Any) -> str:
    text = str(value)
    lowered = text.lower()
    if "winerror 623" in lowered or "dll 重定位" in text:
        return (
            "原生 arrivals 求解器与当前 Windows 不兼容（DLL 重定位错误）；"
            "请在 bellhop_config.json 中替换为兼容的 BellhopCXX，当前结果已回退为声线几何重建。"
        )
    if "no kernel image is available" in lowered:
        return (
            "当前 bellhopcuda 不包含本机 GPU 架构对应内核；请改用兼容的 "
            "BellhopCXX，当前结果已回退为声线几何重建。"
        )
    return f"原生 arrivals 求解器不可用：{text}"


def _generate_static_native_arrivals(
    jobs_dir: Path,
    field: "BellhopFieldSnapshot",
    receiver_depth_m: float,
    ranges_km: list[float],
    beam_count: int,
) -> dict[str, Any]:
    """在已有 Bellhop 剖面环境上为静态阵列生成并缓存原生 arrivals。"""

    job_dir = (jobs_dir.resolve() / field.job_id).resolve()
    if jobs_dir.resolve() not in job_dir.parents:
        raise ValueError("Bellhop arrivals 任务目录越界")
    templates = list(job_dir.glob("sources/source_*/bearings_2d/*/Point2D_TL.env"))
    if not templates:
        templates = list(job_dir.rglob("Point2D_TL.env"))
    if not templates:
        raise FileNotFoundError("Bellhop 任务未保留 Point2D_TL.env，无法生成原生 arrivals")

    def template_bearing(path: Path) -> float:
        first_line = path.read_text(encoding="ascii", errors="replace").splitlines()[0]
        match = re.search(r"bearing\s+([-+0-9.]+)\s+deg", first_line, re.IGNORECASE)
        return float(match.group(1)) if match else field.section_theta_deg

    template = min(
        templates,
        key=lambda path: _bearing_difference(template_bearing(path), field.section_theta_deg),
    )
    from .bellhop_model.config import load_config
    from .bellhop_model.runner import _run_solver

    bellhop_config = load_config()
    executable = Path(str(bellhop_config["BELLHOP_CXX"]))
    signature_data = {
        "job_id": field.job_id,
        "template": str(template.relative_to(job_dir)),
        "receiver_depth_m": round(float(receiver_depth_m), 6),
        "ranges_km": [round(float(value), 9) for value in ranges_km],
        "beam_count": int(beam_count),
        "solver": str(executable),
        "solver_mtime_ns": executable.stat().st_mtime_ns if executable.is_file() else None,
    }
    signature = hashlib.sha256(
        json.dumps(signature_data, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    work_dir = job_dir / "communication_arrivals" / f"case_{signature}"
    work_dir.mkdir(parents=True, exist_ok=True)
    root_name = "StaticArrivals"
    arrival_path = work_dir / f"{root_name}.arr"
    failure_path = work_dir / "generation_error.txt"
    if failure_path.is_file() and not arrival_path.is_file():
        raise RuntimeError(_friendly_arrival_solver_error(
            failure_path.read_text(encoding="utf-8", errors="replace")
        ))
    if not arrival_path.is_file():
        _write_point_arrival_environment(
            template,
            work_dir / f"{root_name}.env",
            receiver_depth_m,
            ranges_km,
            beam_count,
        )
        template_root = template.with_suffix("")
        for extension in (".ssp", ".bty", ".ati", ".trc"):
            source_path = template_root.with_suffix(extension)
            if source_path.is_file():
                shutil.copy2(source_path, work_dir / f"{root_name}{extension}")
        try:
            _run_solver(
                str(executable),
                root_name,
                work_dir,
                float(bellhop_config["MAX_WAIT_TIME"]),
                True,
            )
        except (OSError, RuntimeError) as exc:
            message = _friendly_arrival_solver_error(exc)
            failure_path.write_text(message, encoding="utf-8")
            raise RuntimeError(message) from exc
        if not arrival_path.is_file():
            message = "Bellhop arrivals 计算结束但未生成 .arr 文件"
            failure_path.write_text(message, encoding="utf-8")
            raise RuntimeError(message)
    return _parse_bellhop_ascii_arrivals(arrival_path)


def _finite_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} 必须是布尔值")


@lru_cache(maxsize=4)
def _load_measured_noise(path: str) -> dict[str, Any]:
    """读取水声实测噪声混合系数文件，兼容 MAT v7.3 与 v5。"""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"找不到实测噪声文件：{file_path}")

    values: dict[str, Any]
    try:
        import h5py

        with h5py.File(file_path, "r") as handle:
            values = {name: np.asarray(handle[name][()]) for name in handle.keys()}
    except (ImportError, OSError):
        from scipy.io import loadmat

        raw = loadmat(file_path)
        values = {name: np.asarray(value) for name, value in raw.items() if not name.startswith("__")}

    required = ("Fs", "R", "alpha", "beta", "fc", "version")
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"实测噪声文件缺少字段：{', '.join(missing)}")

    beta = np.asarray(values["beta"], dtype=float)
    if beta.ndim == 3 and beta.shape[0] != beta.shape[1]:
        beta = np.transpose(beta, (2, 1, 0))
    if beta.ndim != 3 or beta.shape[0] != beta.shape[1]:
        raise ValueError("实测噪声 beta 必须是 [M, M, K] 混合系数张量")

    return {
        "path": str(file_path),
        "Fs": float(np.asarray(values["Fs"]).ravel()[0]),
        "R": float(np.asarray(values["R"]).ravel()[0]),
        "alpha": float(np.asarray(values["alpha"]).ravel()[0]),
        "beta": beta,
        "fc": float(np.asarray(values["fc"]).ravel()[0]),
        "version": float(np.asarray(values["version"]).ravel()[0]),
    }


def _expand_measured_beta(beta: np.ndarray, n_elem: int) -> tuple[np.ndarray, list[int], str]:
    """把实测 12 阵元混合系数映射到请求的阵元数。"""
    measured_count = int(beta.shape[0])
    if n_elem <= measured_count:
        return beta[:n_elem], list(range(n_elem)), "measured_subset"

    positions = np.linspace(0.0, measured_count - 1.0, n_elem)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, measured_count - 1)
    fraction = positions - lower
    expanded = (
        (1.0 - fraction)[:, None, None] * beta[lower]
        + fraction[:, None, None] * beta[upper]
    )
    norms = np.sqrt(np.sum(expanded ** 2, axis=(1, 2), keepdims=True))
    expanded /= np.maximum(norms, 1.0e-300)
    return expanded, list(range(n_elem)), "spatial_interpolated"


def _generate_measured_noise(
    sample_count: int,
    sample_rate_hz: float,
    n_elem: int,
    rng: np.random.Generator,
    path: str | Path = DEFAULT_MEASURED_NOISE_FILE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """按实测混合系数生成 [阵元, 样点] 的时空相关噪声。"""
    model = _load_measured_noise(str(Path(path).resolve()))
    measured_count = int(model["beta"].shape[0])
    beta, array_index, spatial_mode = _expand_measured_beta(model["beta"], n_elem)
    alpha = float(model["alpha"])
    native_fs = float(model["Fs"])
    mix_taps = int(beta.shape[2])

    native_count = int(math.ceil(sample_count / sample_rate_hz * native_fs))
    if alpha == 2.0:
        innovations = rng.standard_normal((native_count + mix_taps, measured_count))
    else:
        from scipy.stats import levy_stable

        innovations = levy_stable.rvs(
            alpha, 0, scale=1.0 / math.sqrt(2.0),
            size=(native_count + mix_taps, measured_count), random_state=rng,
        )

    noise = np.zeros((native_count, n_elem), dtype=float)
    for tap in range(mix_taps):
        noise += innovations[tap : tap + native_count, :] @ beta[:, :, tap].T

    ratio = sample_rate_hz / native_fs
    if abs(ratio - 1.0) > 1.0e-12:
        from fractions import Fraction

        fraction = Fraction(ratio).limit_denominator(100_000)
        noise = resample_poly(noise, fraction.numerator, fraction.denominator, axis=0)

    if noise.shape[0] < sample_count:
        noise = np.pad(noise, ((0, sample_count - noise.shape[0]), (0, 0)))
    noise = noise[:sample_count, :]

    metadata = {
        "source": model["path"],
        "spatial_mode": spatial_mode,
        "native_sample_rate_hz": native_fs,
        "analysis_bandwidth_hz": float(model["R"]),
        "center_frequency_hz": float(model["fc"]),
        "stability_index": alpha,
        "measured_channel_count": measured_count,
        "array_channel_index": array_index,
        "sample_rate_hz": float(sample_rate_hz),
        "mix_tap_count": mix_taps,
        "band_matched": float(sample_rate_hz) >= 2.0 * (
            float(model["fc"]) + 0.5 * float(model["R"])
        ),
    }
    return noise.T, metadata


def _generate_environment_time_frequency_noise(
    sample_count: int,
    sample_rate_hz: float,
    n_elem: int,
    rng: np.random.Generator,
    source: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """按环境噪声模块的时频 PSD 合成 [阵元, 样点] 非平稳有色噪声。"""
    times = np.asarray(source.get("time_s"), dtype=float)
    frequencies = np.asarray(source.get("frequency_hz"), dtype=float)
    psd_db = np.asarray(source.get("psd_db"), dtype=float)
    if times.ndim != 1 or times.size < 1:
        raise ValueError("environment_noise.time_s 至少需要一个时刻")
    if frequencies.ndim != 1 or frequencies.size < 2:
        raise ValueError("environment_noise.frequency_hz 至少需要两个频点")
    if psd_db.shape != (times.size, frequencies.size):
        raise ValueError("environment_noise.psd_db 维度必须为 [时刻, 频率]")
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(frequencies)) and np.all(np.isfinite(psd_db))):
        raise ValueError("environment_noise 时频数据包含无效数值")

    time_order = np.argsort(times)
    frequency_order = np.argsort(frequencies)
    times = times[time_order]
    frequencies = frequencies[frequency_order]
    psd_db = psd_db[time_order][:, frequency_order]
    if np.any(np.diff(frequencies) <= 0) or np.any(frequencies <= 0):
        raise ValueError("environment_noise.frequency_hz 必须严格递增且大于 0")
    times = times - times[0]

    # PSD 的绝对标度用于计算环境带内总噪声级；合成后仍统一归一化，再按该
    # 声级缩放到接收机量纲，因此既保留谱形/时变性，也保持原有链路幅度口径。
    band_power = np.trapezoid(np.power(10.0, psd_db / 10.0), frequencies, axis=1)
    ambient_noise_db = float(10.0 * np.log10(max(float(np.median(band_power)), 1.0e-30)))
    ambient_noise_db = float(np.clip(ambient_noise_db, 20.0, 180.0))

    fft_size = min(512, max(64, 2 ** int(math.floor(math.log2(max(64, sample_count))))))
    hop = max(1, fft_size // 2)
    synthesis_frequencies = np.fft.rfftfreq(fft_size, 1.0 / sample_rate_hz)
    window = np.sqrt(np.hanning(fft_size))
    output = np.zeros((n_elem, sample_count + fft_size), dtype=float)
    weight = np.zeros(sample_count + fft_size, dtype=float)
    log_source_frequency = np.log(np.maximum(frequencies, 1.0e-9))
    log_target_frequency = np.log(np.maximum(synthesis_frequencies, frequencies[0]))
    floor_db = float(np.min(psd_db) - 30.0)

    for start in range(0, sample_count, hop):
        relative_time = start / sample_rate_hz
        if times.size > 1 and times[-1] > 0:
            relative_time %= times[-1]
            upper = int(np.searchsorted(times, relative_time, side="right"))
            upper = min(max(1, upper), times.size - 1)
            lower = upper - 1
            ratio = (relative_time - times[lower]) / max(times[upper] - times[lower], 1.0e-12)
            frame_psd = psd_db[lower] * (1.0 - ratio) + psd_db[upper] * ratio
        else:
            frame_psd = psd_db[0]
        target_psd = np.interp(
            log_target_frequency,
            log_source_frequency,
            frame_psd,
            left=floor_db,
            right=floor_db,
        )
        target_psd[synthesis_frequencies < frequencies[0]] = floor_db
        target_psd[synthesis_frequencies > frequencies[-1]] = floor_db
        power = np.power(10.0, target_psd / 10.0)
        amplitude = np.sqrt(power * sample_rate_hz * fft_size / 4.0)
        common = rng.standard_normal(amplitude.size) + 1j * rng.standard_normal(amplitude.size)
        for element_index in range(n_elem):
            independent = rng.standard_normal(amplitude.size) + 1j * rng.standard_normal(amplitude.size)
            spectrum = amplitude * (0.35 * common + math.sqrt(1.0 - 0.35 ** 2) * independent)
            spectrum[0] = complex(float(np.real(spectrum[0])), 0.0)
            if fft_size % 2 == 0:
                spectrum[-1] = complex(float(np.real(spectrum[-1])), 0.0)
            frame = np.fft.irfft(spectrum, n=fft_size) * window
            output[element_index, start : start + fft_size] += frame
        weight[start : start + fft_size] += window ** 2

    output = output[:, :sample_count] / np.sqrt(np.maximum(weight[:sample_count], 1.0e-12))[None, :]
    return output, {
        "model": "environment_time_frequency",
        "source": str(source.get("source") or "环境噪声时频结果"),
        "query_time": str(source.get("query_time") or ""),
        "variant": str(source.get("variant") or "normal"),
        "ambient_noise_db": ambient_noise_db,
        "time_bin_count": int(times.size),
        "frequency_bin_count": int(frequencies.size),
        "frequency_range_hz": [float(frequencies[0]), float(frequencies[-1])],
        "sample_rate_hz": float(sample_rate_hz),
        "spatial_mode": "partially_correlated_spectral_synthesis",
    }


def _unit_rms_per_channel(noise: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(noise ** 2, axis=1, keepdims=True))
    return noise / np.maximum(rms, 1.0e-300)


def _finite_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数值") from exc
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} 必须位于 {minimum:g} 到 {maximum:g} 之间")
    return number


def _finite_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    number = _finite_float(value, name, minimum, maximum)
    integer = int(number)
    if integer != number:
        raise ValueError(f"{name} 必须是整数")
    return integer


@dataclass(frozen=True)
class ArrayFusionConfig:
    """单个浏览器连接使用的线阵参数。"""

    n_elem: int = 128
    d: float = 0.75
    f0: float = 1000.0
    c: float = 1500.0
    depth: float = 50.0
    az_scan: float = 90.0
    sensitivity: float = 0.0
    threshold: float = 6.0
    sample_rate: float = 8000.0

    @property
    def array_length(self) -> float:
        return (self.n_elem - 1) * self.d

    def updated(self, payload: Mapping[str, Any]) -> "ArrayFusionConfig":
        values = {
            "n_elem": _finite_int(payload.get("n_elem", self.n_elem), "n_elem", 4, 1024),
            "d": _finite_float(payload.get("d", self.d), "d", 0.01, 20.0),
            "f0": _finite_float(payload.get("f0", self.f0), "f0", 20.0, 12000.0),
            "c": _finite_float(payload.get("c", self.c), "c", 1300.0, 1800.0),
            "depth": _finite_float(payload.get("depth", self.depth), "depth", 0.0, 12000.0),
            "az_scan": _finite_float(payload.get("az_scan", self.az_scan), "az_scan", 5.0, 90.0),
            "sensitivity": _finite_float(
                payload.get("sensitivity", self.sensitivity), "sensitivity", -60.0, 60.0
            ),
            "threshold": _finite_float(payload.get("threshold", self.threshold), "threshold", -40.0, 60.0),
            "sample_rate": _finite_float(
                payload.get("sample_rate", self.sample_rate), "sample_rate", 1000.0, 96000.0
            ),
        }
        # 防止中心频率越过奈奎斯特频率；保留用户显式设置的更高采样率。
        values["sample_rate"] = max(values["sample_rate"], values["f0"] * 4.0)
        return replace(self, **values)

    def acknowledgement(self) -> dict[str, Any]:
        result = asdict(self)
        result.update({"type": "config_ack", "array_length": round(self.array_length, 3)})
        return result


def _nc_values(variable: Any) -> np.ndarray:
    """把 NetCDF 掩码数组统一转换为含 NaN 的 float64 数组。"""

    values = variable[:]
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=np.float64)


def _json_values(values: Any, decimals: int = 4) -> Any:
    """生成严格 JSON 可编码的数值列表，缺测值使用 null。"""

    array = np.asarray(values, dtype=np.float64)
    rounded = np.round(array, decimals)
    if rounded.ndim == 0:
        return float(rounded) if np.isfinite(rounded) else None
    if rounded.ndim == 1:
        return [float(value) if np.isfinite(value) else None for value in rounded]
    return [_json_values(row, decimals) for row in rounded]


def _sample_1d(x: np.ndarray, values: np.ndarray, target: float) -> float:
    good = np.isfinite(x) & np.isfinite(values)
    if not np.any(good):
        return math.nan
    xx = np.asarray(x[good], dtype=np.float64)
    yy = np.asarray(values[good], dtype=np.float64)
    order = np.argsort(xx)
    xx, yy = xx[order], yy[order]
    xx, unique = np.unique(xx, return_index=True)
    yy = yy[unique]
    return float(np.interp(float(target), xx, yy, left=yy[0], right=yy[-1]))


def _wrap_bearing(degrees: float) -> float:
    """把方位角归一化到 [0, 360)。"""

    return float(degrees) % 360.0


def _bearing_difference(first_deg: float, second_deg: float) -> float:
    """返回两个方位角的最小绝对差。"""

    return abs((_wrap_bearing(first_deg) - _wrap_bearing(second_deg) + 180.0) % 360.0 - 180.0)


def _distance_and_bearing(
    source_lon: float,
    source_lat: float,
    target_lon: float,
    target_lat: float,
) -> tuple[float, float]:
    """用大圆近似计算两经纬度点的距离和初始方位。"""

    radius_m = 6_371_008.8
    lat1 = math.radians(source_lat)
    lat2 = math.radians(target_lat)
    delta_lat = lat2 - lat1
    delta_lon = math.radians(target_lon - source_lon)
    sin_lat = math.sin(delta_lat / 2.0)
    sin_lon = math.sin(delta_lon / 2.0)
    haversine = sin_lat * sin_lat + math.cos(lat1) * math.cos(lat2) * sin_lon * sin_lon
    central_angle = 2.0 * math.asin(min(1.0, math.sqrt(max(0.0, haversine))))
    y = math.sin(delta_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    return radius_m * central_angle, _wrap_bearing(math.degrees(math.atan2(y, x)))


def _destination(lon: float, lat: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    """从起点沿大圆方向平移一段距离。"""

    radius_m = 6_371_008.8
    angular_distance = distance_m / radius_m
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )
    return (math.degrees(lon2) + 540.0) % 360.0 - 180.0, math.degrees(lat2)


def _knots_to_mps(speed_knots: float) -> float:
    return float(speed_knots) * 0.5144444444444445


def _velocity_components(speed_knots: float, heading_deg: float) -> tuple[float, float]:
    """返回东、北方向速度分量，单位 m/s。"""

    speed_mps = _knots_to_mps(speed_knots)
    heading = math.radians(_wrap_bearing(heading_deg))
    return speed_mps * math.sin(heading), speed_mps * math.cos(heading)


def _path_motion(
    arrival: Mapping[str, Any],
    *,
    source_radial_speed_mps: float,
    receiver_radial_speed_mps: float,
    source_depth_rate_mps: float,
    receiver_depth_rate_mps: float,
    sound_speed_mps: float,
    source_frequency_hz: float,
) -> dict[str, float]:
    """按 VirTEX Platform Motion 的一阶模型计算单条声线的运动参数。

    径向轴从声源指向接收端，深度轴向下为正。声源和接收端速度分别
    投影到声线的发射、到达切向；两者之差就是该路径长度的变化率。
    """

    source_angle_deg = float(arrival.get("source_angle_deg", arrival.get("arrival_angle_deg", 0.0)))
    arrival_angle_deg = float(arrival.get("arrival_angle_deg", source_angle_deg))
    source_angle = math.radians(source_angle_deg)
    arrival_angle = math.radians(arrival_angle_deg)
    source_projection = (
        source_radial_speed_mps * math.cos(source_angle)
        + source_depth_rate_mps * math.sin(source_angle)
    )
    receiver_projection = (
        receiver_radial_speed_mps * math.cos(arrival_angle)
        + receiver_depth_rate_mps * math.sin(arrival_angle)
    )
    path_length_rate = receiver_projection - source_projection
    stretch_factor = 1.0 - path_length_rate / sound_speed_mps
    doppler_hz = source_frequency_hz * (stretch_factor - 1.0)
    return {
        "source_angle_deg": source_angle_deg,
        "arrival_angle_deg": arrival_angle_deg,
        "source_velocity_projection_mps": source_projection,
        "receiver_velocity_projection_mps": receiver_projection,
        "path_length_rate_mps": path_length_rate,
        "stretch_factor": stretch_factor,
        "doppler_hz": doppler_hz,
        "frequency_hz": source_frequency_hz + doppler_hz,
    }


def _analytic_channel_sample(
    range_km: float,
    source_depth_m: float,
    receiver_depth_m: float,
    bottom_depth_m: float,
    sound_speed_mps: float,
) -> dict[str, Any]:
    """在没有 Bellhop 场时，为单个阵元生成三路径解析回退。"""

    horizontal_range_m = float(range_km) * 1000.0
    direct_angle_deg = math.degrees(math.atan2(
        receiver_depth_m - source_depth_m, max(horizontal_range_m, 1.0)
    ))
    surface_angle_deg = math.degrees(math.atan2(
        source_depth_m + receiver_depth_m, max(horizontal_range_m, 1.0)
    ))
    bottom_angle_deg = math.degrees(math.atan2(
        2.0 * bottom_depth_m - source_depth_m - receiver_depth_m,
        max(horizontal_range_m, 1.0),
    ))
    path_models = [
        (
            -1,
            math.hypot(horizontal_range_m, source_depth_m - receiver_depth_m),
            0.0,
            direct_angle_deg,
            direct_angle_deg,
        ),
        (
            -2,
            math.hypot(horizontal_range_m, source_depth_m + receiver_depth_m),
            -6.0,
            -surface_angle_deg,
            surface_angle_deg,
        ),
        (
            -3,
            math.hypot(
                horizontal_range_m,
                2.0 * bottom_depth_m - source_depth_m - receiver_depth_m,
            ),
            -10.0,
            bottom_angle_deg,
            -bottom_angle_deg,
        ),
    ]
    first_delay_s = min(distance_m / sound_speed_mps for _, distance_m, _, _, _ in path_models)
    arrivals = [{
        "ray_index": ray_index,
        "delay_s": round(distance_m / sound_speed_mps, 9),
        "excess_delay_ms": round(
            (distance_m / sound_speed_mps - first_delay_s) * 1000.0, 6
        ),
        "relative_gain_db": relative_gain_db,
        "source_angle_deg": round(source_angle_deg, 6),
        "arrival_angle_deg": round(arrival_angle_deg, 6),
        "path_length_m": round(distance_m, 6),
    } for ray_index, distance_m, relative_gain_db, source_angle_deg, arrival_angle_deg in path_models]
    return {
        "transmission_loss_db": 20.0 * math.log10(max(horizontal_range_m, 1.0)),
        "bottom_depth_m": bottom_depth_m,
        "arrivals": arrivals,
    }


def _relative_broadside_angle(arrival_bearing_deg: float, array_axis_heading_deg: float) -> float:
    """把绝对来波方位折叠为 ULA 的 [-90, 90] 宽侧角。"""

    relative = (arrival_bearing_deg - (array_axis_heading_deg + 90.0) + 180.0) % 360.0 - 180.0
    if relative > 90.0:
        relative = 180.0 - relative
    elif relative < -90.0:
        relative = -180.0 - relative
    return float(relative)


def _compact_series(values: np.ndarray, count: int = 128, decimals: int = 6) -> list[float]:
    """对长序列等间隔取样，限制实时协议尺寸。"""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size > count:
        indexes = np.linspace(0, array.size - 1, count).round().astype(int)
        array = array[indexes]
    return np.round(array, decimals).tolist()


@dataclass
class BellhopFieldSnapshot:
    """从一个已完成 Bellhop 任务加载的真实环境与二维声场快照。"""

    job_id: str
    source_lon: float
    source_lat: float
    source_depth_m: float
    frequency_hz: float
    section_theta_deg: float
    range_km: np.ndarray
    depth_m: np.ndarray
    bottom_depth_m: np.ndarray
    tl_db: np.ndarray
    ssp_depth_m: np.ndarray
    ssp_sound_speed_mps: np.ndarray
    boundary_range_m: np.ndarray
    boundary_bottom_m: np.ndarray
    rays: list[tuple[np.ndarray, np.ndarray]]
    ray_job_id: str | None = None
    arrival_job_id: str | None = None
    native_arrival_sets: list[dict[str, Any]] = dataclass_field(default_factory=list)

    @property
    def mean_sound_speed_mps(self) -> float:
        valid = self.ssp_sound_speed_mps[np.isfinite(self.ssp_sound_speed_mps)]
        return float(np.median(valid)) if valid.size else 1500.0

    @classmethod
    def load(
        cls,
        jobs_dir: str | Path,
        job_id: str,
        ray_job_id: str | None = None,
        arrival_job_id: str | None = None,
    ) -> "BellhopFieldSnapshot":
        """安全地读取 TL、声线，以及可选独立任务中的原生 arrivals。"""

        from netCDF4 import Dataset

        job_id = str(job_id or "").strip()
        if not _JOB_ID_PATTERN.fullmatch(job_id):
            raise ValueError("Bellhop 任务号格式无效")
        root = Path(jobs_dir).resolve()
        job_dir = (root / job_id).resolve()
        if root not in job_dir.parents:
            raise ValueError("Bellhop 任务目录越界")
        ray_job_id = str(ray_job_id or job_id).strip()
        if not _JOB_ID_PATTERN.fullmatch(ray_job_id):
            raise ValueError("Bellhop 声线任务号格式无效")
        ray_job_dir = (root / ray_job_id).resolve()
        if root not in ray_job_dir.parents:
            raise ValueError("Bellhop 声线任务目录越界")
        normalized_arrival_job_id = str(arrival_job_id or "").strip() or None
        if normalized_arrival_job_id and not _JOB_ID_PATTERN.fullmatch(normalized_arrival_job_id):
            raise ValueError("Bellhop 到达结构任务号格式无效")
        arrival_job_dir = (
            (root / normalized_arrival_job_id).resolve()
            if normalized_arrival_job_id else None
        )
        if arrival_job_dir is not None and root not in arrival_job_dir.parents:
            raise ValueError("Bellhop 到达结构任务目录越界")
        tl_path = job_dir / "tl2d_section.nc"
        if not tl_path.is_file():
            raise FileNotFoundError(f"任务 {job_id} 没有 tl2d_section.nc")

        # Windows 下 HDF5 对中文路径支持不稳定；以内存只读方式避免路径编码损坏。
        with Dataset("array_fusion_tl.nc", "r", memory=tl_path.read_bytes()) as dataset:
            required = ("range_km", "depth_m", "bottom_depth_m", "tl_db")
            missing = [name for name in required if name not in dataset.variables]
            if missing:
                raise ValueError(f"Bellhop 声场缺少变量：{', '.join(missing)}")
            range_km = _nc_values(dataset.variables["range_km"]).ravel()
            depth_m = _nc_values(dataset.variables["depth_m"]).ravel()
            bottom_depth_m = _nc_values(dataset.variables["bottom_depth_m"]).ravel()
            tl_db = _nc_values(dataset.variables["tl_db"])
            if tl_db.shape == (len(depth_m), len(range_km)):
                tl_db = tl_db.T
            if not len(range_km) or not len(depth_m) or not len(bottom_depth_m):
                raise ValueError("Bellhop 声场坐标轴为空")
            if tl_db.shape != (len(range_km), len(depth_m)):
                raise ValueError("Bellhop TL 数组与距离/深度坐标不匹配")
            ssp_depth_m = (
                _nc_values(dataset.variables["ssp_depth_m"]).ravel()
                if "ssp_depth_m" in dataset.variables else np.asarray([], dtype=np.float64)
            )
            ssp_sound_speed_mps = (
                _nc_values(dataset.variables["ssp_sound_speed_mps"]).ravel()
                if "ssp_sound_speed_mps" in dataset.variables else np.asarray([], dtype=np.float64)
            )
            boundary_range_m = (
                _nc_values(dataset.variables["boundary_range_m"]).ravel()
                if "boundary_range_m" in dataset.variables else range_km * 1000.0
            )
            boundary_bottom_m = (
                _nc_values(dataset.variables["boundary_bottom_m"]).ravel()
                if "boundary_bottom_m" in dataset.variables else bottom_depth_m
            )
            source_lon = float(getattr(dataset, "source_longitude", math.nan))
            source_lat = float(getattr(dataset, "source_latitude", math.nan))
            source_depth_m = float(getattr(dataset, "source_depth_m", 0.0))
            frequency_hz = float(getattr(dataset, "frequency_hz", math.nan))
            if "section_theta_deg" in dataset.variables:
                section_theta_deg = float(np.asarray(dataset.variables["section_theta_deg"][...]))
            else:
                section_theta_deg = float(getattr(dataset, "section_theta_deg", 0.0))

        rays: list[tuple[np.ndarray, np.ndarray]] = []
        ray_path = ray_job_dir / "ray2d_section.nc"
        if ray_path.is_file():
            with Dataset("array_fusion_ray.nc", "r", memory=ray_path.read_bytes()) as dataset:
                ray_required = ("ray_point_count", "ray_start_index", "range_km", "depth_m")
                if all(name in dataset.variables for name in ray_required):
                    counts = _nc_values(dataset.variables["ray_point_count"]).astype(np.int64).ravel()
                    starts = _nc_values(dataset.variables["ray_start_index"]).astype(np.int64).ravel()
                    ray_range = _nc_values(dataset.variables["range_km"]).ravel()
                    ray_depth = _nc_values(dataset.variables["depth_m"]).ravel()
                    for count, start in zip(counts, starts):
                        end = int(start + count)
                        rr = ray_range[int(start):end]
                        zz = ray_depth[int(start):end]
                        good = np.isfinite(rr) & np.isfinite(zz)
                        if np.count_nonzero(good) >= 2:
                            rays.append((rr[good], zz[good]))

        arrival_roots: list[Path] = []
        for candidate in (arrival_job_dir, job_dir, ray_job_dir):
            if candidate is not None and candidate.is_dir() and candidate not in arrival_roots:
                arrival_roots.append(candidate)
        native_arrival_sets: list[dict[str, Any]] = []
        seen_arrival_paths: set[Path] = set()
        for arrival_root in arrival_roots:
            for arrival_path in sorted(arrival_root.rglob("*.arr"))[:64]:
                resolved_path = arrival_path.resolve()
                if resolved_path in seen_arrival_paths:
                    continue
                seen_arrival_paths.add(resolved_path)
                try:
                    native_arrival_sets.append(_parse_bellhop_ascii_arrivals(arrival_path))
                except (OSError, ValueError):
                    # 兼容没有 arrivals 或历史二进制 arrivals 的既有任务。
                    continue

        return cls(
            job_id=job_id,
            ray_job_id=ray_job_id,
            source_lon=source_lon,
            source_lat=source_lat,
            source_depth_m=source_depth_m,
            frequency_hz=frequency_hz,
            section_theta_deg=section_theta_deg,
            range_km=range_km,
            depth_m=depth_m,
            bottom_depth_m=bottom_depth_m,
            tl_db=tl_db,
            ssp_depth_m=ssp_depth_m,
            ssp_sound_speed_mps=ssp_sound_speed_mps,
            boundary_range_m=boundary_range_m,
            boundary_bottom_m=boundary_bottom_m,
            rays=rays,
            arrival_job_id=normalized_arrival_job_id,
            native_arrival_sets=native_arrival_sets,
        )

    def sample(self, range_km: float, depth_m: float) -> tuple[float, float]:
        """在接收位置对 TL 和海深做二维/一维插值。"""

        range_min = float(np.nanmin(self.range_km))
        range_max = float(np.nanmax(self.range_km))
        depth_min = float(np.nanmin(self.depth_m))
        depth_max = float(np.nanmax(self.depth_m))
        requested_range_km = float(range_km)
        requested_depth_m = float(depth_m)
        # NetCDF 坐标通常以 float32 保存，例如请求 287.78 m 会还原成
        # 287.7799988 m。边界点应吸附到轴端点，不能因此误判为声场范围外。
        range_tolerance_km = max(
            1.0e-6,
            float(np.finfo(np.float32).eps) * 8.0
            * max(1.0, abs(range_min), abs(range_max), abs(requested_range_km)),
        )
        depth_tolerance_m = max(
            1.0e-3,
            float(np.finfo(np.float32).eps) * 8.0
            * max(1.0, abs(depth_min), abs(depth_max), abs(requested_depth_m)),
        )
        if (
            requested_range_km < range_min - range_tolerance_km
            or requested_range_km > range_max + range_tolerance_km
            or requested_depth_m < depth_min - depth_tolerance_m
            or requested_depth_m > depth_max + depth_tolerance_m
        ):
            return math.nan, math.nan
        sample_range_km = float(np.clip(requested_range_km, range_min, range_max))
        sample_depth_m = float(np.clip(requested_depth_m, depth_min, depth_max))
        values_by_range = np.asarray([
            _sample_1d(self.depth_m, row, sample_depth_m) for row in self.tl_db
        ])
        tl_db = _sample_1d(self.range_km, values_by_range, sample_range_km)
        bottom_depth_m = _sample_1d(self.range_km, self.bottom_depth_m, sample_range_km)
        if math.isfinite(bottom_depth_m) and sample_depth_m > bottom_depth_m:
            tl_db = math.nan
        return tl_db, bottom_depth_m

    def native_multipath(
        self,
        range_km: float,
        receiver_depth_m: float,
    ) -> list[dict[str, Any]]:
        """从原生 arrivals 中选择最接近当前接收点的一组特征声线。"""

        if not self.native_arrival_sets:
            return []
        target_range_m = float(range_km) * 1000.0
        best: tuple[float, dict[str, Any], float, float, float, float] | None = None
        for arrival_set in self.native_arrival_sets:
            if not arrival_set["records"]:
                continue
            frequency_error = (
                abs(float(arrival_set["frequency_hz"]) - self.frequency_hz)
                if math.isfinite(self.frequency_hz) else 0.0
            )
            if (
                math.isfinite(self.frequency_hz)
                and frequency_error > max(1.0, abs(self.frequency_hz) * 0.01)
            ):
                continue
            source_depth = min(
                arrival_set["source_depths_m"],
                key=lambda value: abs(float(value) - self.source_depth_m),
            )
            selected_depth = min(
                arrival_set["receiver_depths_m"],
                key=lambda value: abs(float(value) - receiver_depth_m),
            )
            raw_ranges = np.asarray(arrival_set["raw_ranges"], dtype=np.float64)
            metre_error = np.abs(raw_ranges - target_range_m)
            kilometre_error = np.abs(raw_ranges * 1000.0 - target_range_m)
            range_scale_to_m = (
                1.0
                if float(np.min(metre_error)) <= float(np.min(kilometre_error))
                else 1000.0
            )
            ranges_m = raw_ranges * range_scale_to_m
            selected_range_m = float(
                ranges_m[int(np.argmin(np.abs(ranges_m - target_range_m)))]
            )
            score = (
                abs(float(source_depth) - self.source_depth_m) * 1000.0
                + abs(float(selected_depth) - receiver_depth_m) * 100.0
                + abs(selected_range_m - target_range_m)
                + frequency_error * 10.0
            )
            if best is None or score < best[0]:
                best = (
                    score,
                    arrival_set,
                    float(source_depth),
                    float(selected_depth),
                    selected_range_m,
                    range_scale_to_m,
                )
        if best is None:
            return []
        _, arrival_set, source_depth, selected_depth, selected_range_m, range_scale_to_m = best
        selected_raw_range = selected_range_m / range_scale_to_m
        selected = [
            record
            for record in arrival_set["records"]
            if record["source_depth_m"] == source_depth
            and record["receiver_depth_m"] == selected_depth
            and record["raw_range"] == selected_raw_range
            and math.isfinite(float(record["amplitude"]))
            and float(record["amplitude"]) > 0.0
            and math.isfinite(float(record["delay_s"]))
        ]
        if len(selected) > MAX_NATIVE_ARRIVAL_COUNT:
            selected = sorted(
                selected,
                key=lambda record: float(record["amplitude"]),
                reverse=True,
            )[:MAX_NATIVE_ARRIVAL_COUNT]
        selected.sort(
            key=lambda record: (float(record["delay_s"]), int(record["arrival_index"]))
        )
        if not selected:
            return []
        maximum_amplitude = max(float(record["amplitude"]) for record in selected)
        first_delay_s = min(float(record["delay_s"]) for record in selected)
        result: list[dict[str, Any]] = []
        for record in selected:
            amplitude = float(record["amplitude"])
            result.append({
                "ray_index": int(record["arrival_index"]),
                "arrival_index": int(record["arrival_index"]),
                "delay_s": round(float(record["delay_s"]), 9),
                "delay_imag_s": float(record["delay_imag_s"]),
                "excess_delay_ms": round(
                    (float(record["delay_s"]) - first_delay_s) * 1000.0, 6
                ),
                "amplitude": amplitude,
                "relative_gain_db": round(
                    20.0 * math.log10(max(amplitude / maximum_amplitude, 1.0e-300)),
                    6,
                ),
                "phase_deg": float(record["phase_deg"]),
                "source_angle_deg": float(record["source_angle_deg"]),
                "arrival_angle_deg": float(record["arrival_angle_deg"]),
                "top_bounces": int(record["top_bounces"]),
                "bottom_bounces": int(record["bottom_bounces"]),
                "selected_source_depth_m": source_depth,
                "selected_receiver_depth_m": selected_depth,
                "selected_range_m": selected_range_m,
                "requested_range_m": target_range_m,
                "range_scale_to_m": range_scale_to_m,
                "parameter_source": "bellhop_native_arrivals",
                "arrival_source_file": arrival_set["path"],
            })
        return result

    def multipath(
        self,
        range_km: float,
        receiver_depth_m: float,
        *,
        prefer_native: bool = True,
    ) -> list[dict[str, Any]]:
        """由 Bellhop 声线在接收距离处的交点估算多途时延与到达角。"""

        native = self.native_multipath(range_km, receiver_depth_m) if prefer_native else []
        if native:
            return native

        candidates: list[dict[str, float]] = []
        target = float(range_km)
        sound_speed = self.mean_sound_speed_mps
        for ray_index, (rr, zz) in enumerate(self.rays):
            dr_m = np.diff(rr) * 1000.0
            dz_m = np.diff(zz)
            segment_m = np.hypot(dr_m, dz_m)
            cumulative_m = np.concatenate(([0.0], np.cumsum(segment_m)))
            valid_source_segments = np.flatnonzero(segment_m > 1.0e-9)
            source_angle_deg = (
                math.degrees(math.atan2(
                    float(dz_m[valid_source_segments[0]]),
                    float(dr_m[valid_source_segments[0]]),
                ))
                if valid_source_segments.size else 0.0
            )
            crossings = np.flatnonzero((rr[:-1] - target) * (rr[1:] - target) <= 0.0)
            best: dict[str, float] | None = None
            for index in crossings:
                delta = rr[index + 1] - rr[index]
                if abs(delta) < 1.0e-12:
                    continue
                fraction = float(np.clip((target - rr[index]) / delta, 0.0, 1.0))
                arrival_depth = float(zz[index] + fraction * (zz[index + 1] - zz[index]))
                path_length_m = float(cumulative_m[index] + fraction * segment_m[index])
                angle_deg = math.degrees(math.atan2(float(dz_m[index]), float(dr_m[index])))
                item = {
                    "ray_index": float(ray_index),
                    "arrival_depth_m": arrival_depth,
                    "depth_error_m": abs(arrival_depth - receiver_depth_m),
                    "path_length_m": path_length_m,
                    "source_angle_deg": source_angle_deg,
                    "arrival_angle_deg": angle_deg,
                    "delay_s": path_length_m / sound_speed,
                }
                if best is None or item["depth_error_m"] < best["depth_error_m"]:
                    best = item
            if best is not None:
                candidates.append(best)

        candidates.sort(key=lambda item: (item["depth_error_m"], item["path_length_m"]))
        if not candidates:
            direct_m = math.hypot(target * 1000.0, receiver_depth_m - self.source_depth_m)
            candidates = [{
                "ray_index": -1.0,
                "arrival_depth_m": receiver_depth_m,
                "depth_error_m": 0.0,
                "path_length_m": direct_m,
                "source_angle_deg": math.degrees(
                    math.atan2(receiver_depth_m - self.source_depth_m, max(target * 1000.0, 1.0))
                ),
                "arrival_angle_deg": math.degrees(
                    math.atan2(receiver_depth_m - self.source_depth_m, max(target * 1000.0, 1.0))
                ),
                "delay_s": direct_m / sound_speed,
            }]

        selected = candidates[:MAX_MULTIPATH_COUNT]
        finite_steps = np.diff(self.depth_m)
        finite_steps = finite_steps[np.isfinite(finite_steps) & (finite_steps > 0.0)]
        receiver_bottom = _sample_1d(self.range_km, self.bottom_depth_m, target)
        depth_scale = max(
            10.0,
            float(np.median(finite_steps)) * 3.0 if finite_steps.size else 10.0,
            receiver_bottom * 0.15 if math.isfinite(receiver_bottom) else 10.0,
        )
        weights = np.asarray([
            math.exp(-item["depth_error_m"] / depth_scale)
            / math.sqrt(max(item["path_length_m"], 1.0))
            for item in selected
        ])
        weights /= max(float(np.max(weights)), 1.0e-12)
        first_delay = min(item["delay_s"] for item in selected)
        result = []
        for item, weight in zip(selected, weights):
            result.append({
                "ray_index": int(item["ray_index"]),
                "delay_s": round(float(item["delay_s"]), 6),
                "excess_delay_ms": round((float(item["delay_s"]) - first_delay) * 1000.0, 3),
                "relative_gain_db": round(20.0 * math.log10(max(float(weight), 1.0e-6)), 3),
                "source_angle_deg": round(float(item["source_angle_deg"]), 3),
                "arrival_angle_deg": round(float(item["arrival_angle_deg"]), 3),
                "arrival_depth_m": round(float(item["arrival_depth_m"]), 3),
                "depth_error_m": round(float(item["depth_error_m"]), 3),
                "path_length_m": round(float(item["path_length_m"]), 3),
                "parameter_source": "ray_geometry_reconstruction",
            })
        return result

    def channel_sample(
        self,
        range_km: float,
        receiver_depth_m: float,
        *,
        prefer_native_arrivals: bool = True,
    ) -> dict[str, Any]:
        """统一的位置查询内核，供静态信道和运动信道共同使用。"""

        tl_db, bottom_depth_m = self.sample(range_km, receiver_depth_m)
        if not math.isfinite(tl_db):
            raise ValueError("接收位置位于 Bellhop 声场范围外、海底以下或该位置无有效 TL")
        arrivals = self.multipath(
            range_km,
            receiver_depth_m,
            prefer_native=prefer_native_arrivals,
        )
        if not arrivals:
            raise ValueError("接收位置没有可用多径")
        return {
            "range_km": float(range_km),
            "receiver_depth_m": float(receiver_depth_m),
            "transmission_loss_db": float(tl_db),
            "bottom_depth_m": float(bottom_depth_m),
            "arrivals": arrivals,
        }

    def profile_response(self) -> dict[str, Any]:
        source_bottom = self.bottom_depth_m[0] if self.bottom_depth_m.size else math.nan
        water = np.isfinite(self.ssp_depth_m) & np.isfinite(self.ssp_sound_speed_mps)
        if math.isfinite(source_bottom) and source_bottom > 0.0:
            water &= self.ssp_depth_m <= source_bottom
        return {
            "type": "profile_resp",
            "lon": self.source_lon,
            "lat": self.source_lat,
            "depth": _json_values(self.ssp_depth_m[water], 3),
            "c": _json_values(self.ssp_sound_speed_mps[water], 4),
            "bottom_depth_m": _json_values(source_bottom, 3),
            "data_source": "bellhop_job",
            "bellhop_job_id": self.job_id,
            "ray_job_id": self.ray_job_id or self.job_id,
            "arrival_job_id": self.arrival_job_id,
            "actual_environment": True,
        }

    def field_response(self) -> dict[str, Any]:
        range_step = max(1, int(math.ceil(len(self.range_km) / FIELD_MAX_RANGE_POINTS)))
        depth_step = max(1, int(math.ceil(len(self.depth_m) / FIELD_MAX_DEPTH_POINTS)))
        range_index = np.arange(0, len(self.range_km), range_step, dtype=np.int64)
        depth_index = np.arange(0, len(self.depth_m), depth_step, dtype=np.int64)
        if range_index[-1] != len(self.range_km) - 1:
            range_index = np.append(range_index, len(self.range_km) - 1)
        if depth_index[-1] != len(self.depth_m) - 1:
            depth_index = np.append(depth_index, len(self.depth_m) - 1)
        source_bottom = self.bottom_depth_m[0] if self.bottom_depth_m.size else math.nan
        water = np.isfinite(self.ssp_depth_m) & np.isfinite(self.ssp_sound_speed_mps)
        if math.isfinite(source_bottom) and source_bottom > 0.0:
            water &= self.ssp_depth_m <= source_bottom
        ray_rows = []
        for ray_index, (ray_range, ray_depth) in enumerate(self.rays[:24]):
            ray_step = max(1, int(math.ceil(len(ray_range) / 240)))
            point_index = np.arange(0, len(ray_range), ray_step, dtype=np.int64)
            if point_index[-1] != len(ray_range) - 1:
                point_index = np.append(point_index, len(ray_range) - 1)
            ray_rows.append({
                "ray_index": ray_index,
                "range_km": _json_values(ray_range[point_index], 5),
                "depth_m": _json_values(ray_depth[point_index], 3),
            })
        return {
            "type": "field_resp",
            "engine": "BellhopCXX",
            "bellhop_job_id": self.job_id,
            "ray_job_id": self.ray_job_id or self.job_id,
            "arrival_job_id": self.arrival_job_id,
            "actual_environment": True,
            "source": {
                "lon": _json_values(self.source_lon, 6),
                "lat": _json_values(self.source_lat, 6),
                "depth_m": round(self.source_depth_m, 3),
                "frequency_hz": _json_values(self.frequency_hz, 3),
                "section_theta_deg": round(self.section_theta_deg, 3),
            },
            "range_km": _json_values(self.range_km[range_index], 5),
            "depth_m": _json_values(self.depth_m[depth_index], 3),
            # 绘图端通常以 [depth][range] 消费剖面矩阵。
            "tl_db": _json_values(self.tl_db[np.ix_(range_index, depth_index)].T, 3),
            "bathymetry": {
                "range_m": _json_values(self.boundary_range_m, 3),
                "bottom_depth_m": _json_values(self.boundary_bottom_m, 3),
            },
            "ssp": {
                "depth_m": _json_values(self.ssp_depth_m[water], 3),
                "sound_speed_mps": _json_values(self.ssp_sound_speed_mps[water], 4),
            },
            "ray_count": len(self.rays),
            "rays": ray_rows,
            "multipath_available": bool(self.native_arrival_sets or self.rays),
            "multipath_method": (
                "bellhop_native_arrivals"
                if self.native_arrival_sets else "ray_section_crossing_estimate"
            ),
            "native_arrival_file_count": len(self.native_arrival_sets),
        }


def _cyclic_bracket(values_deg: list[float], target_deg: float) -> tuple[int, int, float]:
    """返回环形方位轴上包围目标的两个索引及线性权重。"""

    if not values_deg:
        raise ValueError("Bellhop 快照没有可用方位")
    if len(values_deg) == 1:
        return 0, 0, 0.0
    values = np.asarray([_wrap_bearing(value) for value in values_deg], dtype=np.float64)
    order = np.argsort(values)
    sorted_values = values[order]
    target = _wrap_bearing(target_deg)
    if target < sorted_values[0]:
        target += 360.0
    extended = np.append(sorted_values, sorted_values[0] + 360.0)
    upper = int(np.searchsorted(extended, target, side="right"))
    upper = min(max(upper, 1), len(extended) - 1)
    lower = upper - 1
    span = float(extended[upper] - extended[lower])
    fraction = 0.0 if span <= 1.0e-12 else float((target - extended[lower]) / span)
    return int(order[lower % len(order)]), int(order[upper % len(order)]), fraction


def _linear_bracket(values: list[float], target: float, name: str) -> tuple[int, int, float]:
    """在严格有序的普通坐标轴上确定插值区间，不允许越界外推。"""

    if not values:
        raise ValueError(f"{name} 轴为空")
    if len(values) == 1:
        return 0, 0, 0.0
    if target < values[0] or target > values[-1]:
        raise ValueError(f"{name} {target:.6g} 位于快照范围 [{values[0]:.6g}, {values[-1]:.6g}] 外")
    if target == values[-1]:
        return len(values) - 1, len(values) - 1, 0.0
    upper = int(np.searchsorted(np.asarray(values), target, side="right"))
    lower = max(0, upper - 1)
    span = values[upper] - values[lower]
    return lower, upper, 0.0 if span <= 1.0e-12 else float((target - values[lower]) / span)


def _weighted_angle_deg(rows: list[tuple[float, float]]) -> float:
    """对射线俯仰角做圆周加权，避免跨越 ±180° 时跳变。"""

    sine = sum(weight * math.sin(math.radians(value)) for weight, value in rows)
    cosine = sum(weight * math.cos(math.radians(value)) for weight, value in rows)
    return math.degrees(math.atan2(sine, cosine))


def _interpolate_channel_samples(
    corners: list[tuple[float, dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    """对时间/方位角点的 TL 与多径参数做物理量域插值。"""

    active = [(weight, sample, meta) for weight, sample, meta in corners if weight > 1.0e-12]
    if not active:
        raise ValueError("Bellhop 快照插值权重为空")
    total_weight = sum(weight for weight, _, _ in active)
    active = [(weight / total_weight, sample, meta) for weight, sample, meta in active]
    received_power = sum(
        weight * 10.0 ** (-float(sample["transmission_loss_db"]) / 10.0)
        for weight, sample, _ in active
    )
    tl_db = -10.0 * math.log10(max(received_power, 1.0e-300))
    bottom_depth_m = sum(weight * float(sample["bottom_depth_m"]) for weight, sample, _ in active)

    arrivals_by_ray: dict[int, list[tuple[float, Mapping[str, Any]]]] = {}
    for weight, sample, _ in active:
        for arrival in sample["arrivals"]:
            arrivals_by_ray.setdefault(int(arrival["ray_index"]), []).append((weight, arrival))

    arrivals = []
    for ray_index, support in arrivals_by_ray.items():
        support_weight = sum(weight for weight, _ in support)
        if support_weight <= 1.0e-12:
            continue
        rows = [(weight / support_weight, arrival) for weight, arrival in support]
        amplitude = sum(
            weight * 10.0 ** (float(arrival["relative_gain_db"]) / 20.0)
            for weight, arrival in rows
        )
        arrivals.append({
            "ray_index": ray_index,
            "delay_s": sum(weight * float(arrival["delay_s"]) for weight, arrival in rows),
            "relative_gain_linear": amplitude,
            "source_angle_deg": _weighted_angle_deg([
                (weight, float(arrival.get("source_angle_deg", 0.0))) for weight, arrival in rows
            ]),
            "arrival_angle_deg": _weighted_angle_deg([
                (weight, float(arrival.get("arrival_angle_deg", 0.0))) for weight, arrival in rows
            ]),
            "arrival_depth_m": sum(
                weight * float(arrival.get("arrival_depth_m", 0.0)) for weight, arrival in rows
            ),
            "depth_error_m": sum(
                weight * float(arrival.get("depth_error_m", 0.0)) for weight, arrival in rows
            ),
            "path_length_m": sum(
                weight * float(arrival.get("path_length_m", 0.0)) for weight, arrival in rows
            ),
            "interpolation_support": round(support_weight, 9),
        })
    arrivals.sort(key=lambda item: (-item["relative_gain_linear"], item["delay_s"]))
    arrivals = arrivals[:MAX_MULTIPATH_COUNT]
    if not arrivals:
        raise ValueError("Bellhop 快照插值后没有可用多径")
    maximum_amplitude = max(float(item["relative_gain_linear"]) for item in arrivals)
    first_delay = min(float(item["delay_s"]) for item in arrivals)
    for item in arrivals:
        amplitude = float(item.pop("relative_gain_linear"))
        item["delay_s"] = round(float(item["delay_s"]), 9)
        item["excess_delay_ms"] = round((float(item["delay_s"]) - first_delay) * 1000.0, 6)
        item["relative_gain_db"] = round(20.0 * math.log10(max(amplitude / maximum_amplitude, 1.0e-6)), 6)
        for name in ("source_angle_deg", "arrival_angle_deg", "arrival_depth_m", "depth_error_m", "path_length_m"):
            item[name] = round(float(item[name]), 6)

    return {
        "transmission_loss_db": tl_db,
        "bottom_depth_m": bottom_depth_m,
        "arrivals": arrivals,
        "interpolation": {
            "method": "range_depth_then_cyclic_bearing_and_time; TL interpolated in linear power",
            "corners": [
                {**meta, "weight": round(weight, 9)} for weight, _, meta in active
            ],
        },
    }


def _load_job_directional_fields(jobs_dir: Path, job_id: str) -> list[BellhopFieldSnapshot]:
    """读取一个任务中的全部方位；旧任务自动退回所选二维剖面。"""

    base = BellhopFieldSnapshot.load(jobs_dir, job_id)
    job_dir = (jobs_dir / job_id).resolve()
    tl3d_path = job_dir / "tl3d_depth_theta_range.nc"
    if not tl3d_path.is_file():
        return [base]

    from netCDF4 import Dataset

    with Dataset("array_fusion_tl3d.nc", "r", memory=tl3d_path.read_bytes()) as dataset:
        required = ("theta_deg", "range_km", "depth_m", "bottom_depth_m", "tl_db")
        if any(name not in dataset.variables for name in required):
            return [base]
        bearings = _nc_values(dataset.variables["theta_deg"]).ravel()
        ranges = _nc_values(dataset.variables["range_km"]).ravel()
        depths = _nc_values(dataset.variables["depth_m"]).ravel()
        bottom = _nc_values(dataset.variables["bottom_depth_m"])
        tl = _nc_values(dataset.variables["tl_db"])
        if bottom.ndim == 3:
            bottom = bottom[0]
        if tl.ndim == 4:
            tl = tl[0]
        if bottom.ndim == 1 and len(bearings) == 1:
            bottom = bottom[None, :]
        if tl.ndim == 2 and len(bearings) == 1:
            tl = tl[:, None, :]
        if bottom.shape != (len(bearings), len(ranges)):
            return [base]
        if tl.shape != (len(depths), len(bearings), len(ranges)):
            return [base]

    ray_groups: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in bearings]
    ray3d_path = job_dir / "ray3d_wgs84.nc"
    if ray3d_path.is_file():
        with Dataset("array_fusion_ray3d.nc", "r", memory=ray3d_path.read_bytes()) as dataset:
            needed = ("ray_point_count", "ray_start_index", "local_x_km", "local_y_km", "depth_m")
            if all(name in dataset.variables for name in needed):
                counts = _nc_values(dataset.variables["ray_point_count"]).astype(np.int64).ravel()
                starts = _nc_values(dataset.variables["ray_start_index"]).astype(np.int64).ravel()
                xx = _nc_values(dataset.variables["local_x_km"]).ravel()
                yy = _nc_values(dataset.variables["local_y_km"]).ravel()
                zz = _nc_values(dataset.variables["depth_m"]).ravel()
                for count, start in zip(counts, starts):
                    end = int(start + count)
                    ray_x, ray_y, ray_z = xx[int(start):end], yy[int(start):end], zz[int(start):end]
                    good = np.isfinite(ray_x) & np.isfinite(ray_y) & np.isfinite(ray_z)
                    if np.count_nonzero(good) < 2:
                        continue
                    ray_x, ray_y, ray_z = ray_x[good], ray_y[good], ray_z[good]
                    radius = np.hypot(ray_x, ray_y)
                    farthest = int(np.argmax(radius))
                    bearing = _wrap_bearing(math.degrees(math.atan2(ray_x[farthest], ray_y[farthest])))
                    bearing_index = min(
                        range(len(bearings)), key=lambda index: _bearing_difference(bearing, float(bearings[index]))
                    )
                    if len(radius) > MAX_RAY_POINTS_PER_DIRECTION:
                        indexes = np.linspace(0, len(radius) - 1, MAX_RAY_POINTS_PER_DIRECTION).round().astype(int)
                        radius, ray_z = radius[indexes], ray_z[indexes]
                    ray_groups[bearing_index].append((radius, ray_z))

    fields = []
    for index, bearing in enumerate(bearings):
        fields.append(BellhopFieldSnapshot(
            job_id=job_id,
            source_lon=base.source_lon,
            source_lat=base.source_lat,
            source_depth_m=base.source_depth_m,
            frequency_hz=base.frequency_hz,
            section_theta_deg=float(bearing),
            range_km=np.asarray(ranges, dtype=np.float64),
            depth_m=np.asarray(depths, dtype=np.float64),
            bottom_depth_m=np.asarray(bottom[index], dtype=np.float64),
            tl_db=np.asarray(tl[:, index, :].T, dtype=np.float64),
            ssp_depth_m=base.ssp_depth_m,
            ssp_sound_speed_mps=base.ssp_sound_speed_mps,
            boundary_range_m=np.asarray(ranges, dtype=np.float64) * 1000.0,
            boundary_bottom_m=np.asarray(bottom[index], dtype=np.float64),
            rays=ray_groups[index],
        ))
    return fields or [base]


@dataclass
class BellhopDirectionalFrame:
    time_s: float
    fields: list[BellhopFieldSnapshot]

    @property
    def bearings_deg(self) -> list[float]:
        return [float(field.section_theta_deg) for field in self.fields]

    def field_weights(self, bearing_deg: float) -> list[tuple[float, BellhopFieldSnapshot]]:
        lower, upper, fraction = _cyclic_bracket(self.bearings_deg, bearing_deg)
        weighted = [(1.0 - fraction, self.fields[lower])]
        if upper != lower and fraction > 1.0e-12:
            weighted.append((fraction, self.fields[upper]))
        return weighted

    def corner_samples(
        self, bearing_deg: float, range_km: float, depth_m: float,
    ) -> list[tuple[float, dict[str, Any], dict[str, Any]]]:
        corners = []
        for weight, field in self.field_weights(bearing_deg):
            sample = field.channel_sample(range_km, depth_m)
            corners.append((weight, sample, {
                "time_s": round(self.time_s, 6),
                "bearing_deg": round(_wrap_bearing(field.section_theta_deg), 6),
                "job_id": field.job_id,
            }))
        return corners

    def channel_sample(self, bearing_deg: float, range_km: float, depth_m: float) -> dict[str, Any]:
        corners = self.corner_samples(bearing_deg, range_km, depth_m)
        result = _interpolate_channel_samples(corners)
        result.update({"range_km": range_km, "receiver_depth_m": depth_m})
        return result


@dataclass
class BellhopFieldTimeline:
    frames: list[BellhopDirectionalFrame]

    @property
    def times_s(self) -> list[float]:
        return [float(frame.time_s) for frame in self.frames]

    @classmethod
    def load(cls, jobs_dir: str | Path, snapshots: Any) -> "BellhopFieldTimeline":
        if not isinstance(snapshots, list) or not 1 <= len(snapshots) <= MAX_SERIES_SNAPSHOTS:
            raise ValueError(f"snapshots 必须是包含 1 到 {MAX_SERIES_SNAPSHOTS} 项的数组")
        root = Path(jobs_dir).resolve()
        frames = []
        seen_times: set[float] = set()
        for index, item in enumerate(snapshots):
            if not isinstance(item, Mapping):
                raise ValueError(f"snapshots[{index}] 必须是对象")
            time_s = _finite_float(item.get("t", item.get("time_s")), f"snapshots[{index}].t", 0.0, 86400.0)
            if time_s in seen_times:
                raise ValueError("Bellhop 时间快照不可重复")
            seen_times.add(time_s)
            job_id = str(item.get("bellhop_job_id") or item.get("job_id") or "").strip()
            if not job_id:
                raise ValueError(f"snapshots[{index}] 缺少 bellhop_job_id")
            fields = _load_job_directional_fields(root, job_id)
            if len(fields) == 1 and "bearing_deg" in item:
                fields[0].section_theta_deg = _wrap_bearing(_finite_float(
                    item["bearing_deg"], f"snapshots[{index}].bearing_deg", -360.0, 360.0
                ))
            frames.append(BellhopDirectionalFrame(time_s=time_s, fields=fields))
        frames.sort(key=lambda frame: frame.time_s)
        return cls(frames=frames)

    @property
    def representative(self) -> BellhopFieldSnapshot:
        return self.frames[0].fields[0]

    def channel_sample(self, time_s: float, bearing_deg: float, range_km: float, depth_m: float) -> dict[str, Any]:
        lower, upper, fraction = _linear_bracket(self.times_s, time_s, "时间")
        frame_corners = [(lower, 1.0 - fraction)]
        if upper != lower and fraction > 1.0e-12:
            frame_corners.append((upper, fraction))
        corners = []
        for index, time_weight in frame_corners:
            for bearing_weight, sample, meta in self.frames[index].corner_samples(
                bearing_deg, range_km, depth_m
            ):
                corners.append((time_weight * bearing_weight, sample, meta))
        result = _interpolate_channel_samples(corners)
        result.update({"range_km": range_km, "receiver_depth_m": depth_m})
        return result

    def transmission_loss(self, time_s: float, bearing_deg: float, range_km: float, depth_m: float) -> float:
        """只插值 TL，供逐阵元查询使用，避免重复构造全部多径。"""

        lower, upper, fraction = _linear_bracket(self.times_s, time_s, "时间")
        frame_weights = [(1.0 - fraction, self.frames[lower])]
        if upper != lower and fraction > 1.0e-12:
            frame_weights.append((fraction, self.frames[upper]))
        power = 0.0
        total_weight = 0.0
        for time_weight, frame in frame_weights:
            for bearing_weight, field in frame.field_weights(bearing_deg):
                tl_db, _ = field.sample(range_km, depth_m)
                if math.isfinite(tl_db):
                    weight = time_weight * bearing_weight
                    power += weight * 10.0 ** (-tl_db / 10.0)
                    total_weight += weight
        if total_weight <= 1.0e-12:
            raise ValueError("阵元位置位于 Bellhop 快照声场范围外")
        return -10.0 * math.log10(max(power / total_weight, 1.0e-300))

    def source_position(self, time_s: float) -> dict[str, float]:
        """按时间插值 Bellhop 快照的声源位置，用于核对外部运动轨迹。"""

        lower, upper, fraction = _linear_bracket(self.times_s, time_s, "时间")
        first = self.frames[lower].fields[0]
        second = self.frames[upper].fields[0]
        lon_delta = (second.source_lon - first.source_lon + 180.0) % 360.0 - 180.0
        return {
            "lon": (first.source_lon + fraction * lon_delta + 540.0) % 360.0 - 180.0,
            "lat": first.source_lat + fraction * (second.source_lat - first.source_lat),
            "depth_m": first.source_depth_m + fraction * (second.source_depth_m - first.source_depth_m),
        }

    def acknowledgement(self) -> dict[str, Any]:
        return {
            "type": "field_series_resp",
            "mode": "bellhop_time_bearing_interpolation",
            "time_bounds_s": [self.times_s[0], self.times_s[-1]],
            "snapshot_count": len(self.frames),
            "snapshots": [{
                "t": frame.time_s,
                "job_ids": sorted({field.job_id for field in frame.fields}),
                "bearings_deg": sorted(round(_wrap_bearing(value), 6) for value in frame.bearings_deg),
                "source": {
                    "lon": round(frame.fields[0].source_lon, 8),
                    "lat": round(frame.fields[0].source_lat, 8),
                    "depth_m": round(frame.fields[0].source_depth_m, 3),
                },
            } for frame in self.frames],
            "interpolation": {
                "range_depth": "per Bellhop section",
                "bearing": "cyclic linear",
                "time": "linear without extrapolation",
                "transmission_loss": "linear received power",
                "multipath": "ray-index matched amplitude/delay/angle interpolation",
            },
        }


class ArrayFusionSession:
    """一个 WebSocket 连接对应一个实例，参数和随机过程互不串扰。"""

    def __init__(self, config: ArrayFusionConfig | None = None, jobs_dir: str | Path | None = None):
        self.config = config or ArrayFusionConfig()
        self.jobs_dir = Path(jobs_dir).resolve() if jobs_dir is not None else None
        self.field: BellhopFieldSnapshot | None = None
        self.field_timeline: BellhopFieldTimeline | None = None
        self.motion_config: dict[str, Any] = {
            "source": {
                "lon": 114.24, "lat": 14.01, "depth_m": 180.0,
                "speed_knots": 15.0, "heading_deg": 225.0, "depth_rate_mps": 0.0,
            },
            "platform": {
                "lon": 113.82, "lat": 13.46, "depth_m": 0.0,
                "speed_knots": 17.0, "heading_deg": 45.0, "depth_rate_mps": 0.0,
            },
            "array": {
                "tow_length_m": 300.0, "heading_mode": "follow", "heading_deg": 225.0,
                "depth_rate_mps": 0.0,
            },
            "duration_s": 120.0,
            "frame_step_s": 0.2,
            "selected_element": 0,
        }

    def configure(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.config = self.config.updated(payload)
        return self.config.acknowledgement()

    def configure_motion(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """保存运动学配置；后续帧也可覆盖当前位置用于航路插值模式。"""

        current = self.motion_config

        def state(name: str) -> dict[str, float]:
            provided = payload.get(name)
            source = provided if isinstance(provided, Mapping) else {}
            fallback = current[name]
            return {
                "lon": _finite_float(source.get("lon", fallback["lon"]), f"{name}.lon", -180.0, 180.0),
                "lat": _finite_float(source.get("lat", fallback["lat"]), f"{name}.lat", -90.0, 90.0),
                "depth_m": _finite_float(source.get("depth_m", fallback["depth_m"]), f"{name}.depth_m", 0.0, 12000.0),
                "speed_knots": _finite_float(source.get("speed_knots", fallback["speed_knots"]), f"{name}.speed_knots", 0.0, 100.0),
                "heading_deg": _wrap_bearing(_finite_float(source.get("heading_deg", fallback["heading_deg"]), f"{name}.heading_deg", -360.0, 360.0)),
                "depth_rate_mps": _finite_float(
                    source.get("depth_rate_mps", fallback.get("depth_rate_mps", 0.0)),
                    f"{name}.depth_rate_mps", -20.0, 20.0,
                ),
            }

        array_payload = payload.get("array") if isinstance(payload.get("array"), Mapping) else {}
        array_current = current["array"]
        heading_mode = str(array_payload.get("heading_mode", array_current["heading_mode"])).lower()
        if heading_mode not in {"follow", "fixed"}:
            raise ValueError("array.heading_mode 必须为 follow 或 fixed")
        self.motion_config = {
            "source": state("source"),
            "platform": state("platform"),
            "array": {
                "tow_length_m": _finite_float(array_payload.get("tow_length_m", array_current["tow_length_m"]), "array.tow_length_m", 0.0, 20_000.0),
                "heading_mode": heading_mode,
                "heading_deg": _wrap_bearing(_finite_float(array_payload.get("heading_deg", array_current["heading_deg"]), "array.heading_deg", -360.0, 360.0)),
                "depth_rate_mps": _finite_float(
                    array_payload.get("depth_rate_mps", array_current.get("depth_rate_mps", 0.0)),
                    "array.depth_rate_mps", -20.0, 20.0,
                ),
            },
            "duration_s": _finite_float(payload.get("duration_s", current["duration_s"]), "duration_s", 0.1, 86400.0),
            "frame_step_s": _finite_float(payload.get("frame_step_s", current["frame_step_s"]), "frame_step_s", 0.02, 10.0),
            "selected_element": _finite_int(payload.get("selected_element", current["selected_element"]), "selected_element", 0, max(0, self.config.n_elem - 1)),
        }
        return {"type": "motion_config_ack", **self.motion_config}

    def process_motion_frame(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """由速度、航向和拖缆长度生成运动位置，并计算同一时刻的六域结果。"""

        config = self.motion_config
        time_s = _finite_float(payload.get("t", 0.0), "t", 0.0, 86400.0)

        def moving_state(name: str) -> dict[str, float]:
            base = config[name]
            supplied = payload.get(f"{name}_state")
            if isinstance(supplied, Mapping) and "lon" in supplied and "lat" in supplied:
                lon = _finite_float(supplied["lon"], f"{name}_state.lon", -180.0, 180.0)
                lat = _finite_float(supplied["lat"], f"{name}_state.lat", -90.0, 90.0)
                heading = _wrap_bearing(_finite_float(supplied.get("heading_deg", base["heading_deg"]), f"{name}_state.heading_deg", -360.0, 360.0))
                speed = _finite_float(supplied.get("speed_knots", base["speed_knots"]), f"{name}_state.speed_knots", 0.0, 100.0)
                depth = _finite_float(supplied.get("depth_m", base["depth_m"]), f"{name}_state.depth_m", 0.0, 12000.0)
                depth_rate = _finite_float(
                    supplied.get("depth_rate_mps", base.get("depth_rate_mps", 0.0)),
                    f"{name}_state.depth_rate_mps", -20.0, 20.0,
                )
            else:
                heading = float(base["heading_deg"])
                speed = float(base["speed_knots"])
                depth_rate = float(base.get("depth_rate_mps", 0.0))
                depth = _finite_float(
                    float(base["depth_m"]) + depth_rate * time_s,
                    f"{name}.depth_m(t)", 0.0, 12000.0,
                )
                lon, lat = _destination(float(base["lon"]), float(base["lat"]), heading, _knots_to_mps(speed) * time_s)
            east_mps, north_mps = _velocity_components(speed, heading)
            return {
                "lon": lon, "lat": lat, "depth_m": depth, "speed_knots": speed,
                "speed_mps": _knots_to_mps(speed), "heading_deg": heading,
                "depth_rate_mps": depth_rate,
                "east_mps": east_mps, "north_mps": north_mps,
            }

        source = moving_state("source")
        platform = moving_state("platform")
        array_cfg = config["array"]
        array_heading = (
            _wrap_bearing(platform["heading_deg"] + 180.0)
            if array_cfg["heading_mode"] == "follow"
            else float(array_cfg["heading_deg"])
        )
        array_lon, array_lat = _destination(
            platform["lon"], platform["lat"],
            _wrap_bearing(platform["heading_deg"] + 180.0),
            float(array_cfg["tow_length_m"]),
        )
        range_m, source_to_array_bearing = _distance_and_bearing(
            source["lon"], source["lat"], array_lon, array_lat
        )
        if range_m < 10.0:
            raise ValueError("声源与拖曳阵列距离必须大于 10 m")
        arrival_bearing = _wrap_bearing(source_to_array_bearing + 180.0)
        az_true = _relative_broadside_angle(arrival_bearing, array_heading)
        array_depth_rate_mps = float(array_cfg.get("depth_rate_mps", 0.0))
        receiver_depth_m = _finite_float(
            payload.get("receiver_depth_m", self.config.depth + array_depth_rate_mps * time_s),
            "receiver_depth_m", 0.0, 12000.0,
        )
        elevation = math.degrees(math.atan2(source["depth_m"] - receiver_depth_m, max(range_m, 1.0)))

        los_east = math.sin(math.radians(source_to_array_bearing))
        los_north = math.cos(math.radians(source_to_array_bearing))
        radial_speed_mps = (
            (platform["east_mps"] - source["east_mps"]) * los_east
            + (platform["north_mps"] - source["north_mps"]) * los_north
        )
        source_radial_speed_mps = source["east_mps"] * los_east + source["north_mps"] * los_north
        receiver_radial_speed_mps = platform["east_mps"] * los_east + platform["north_mps"] * los_north
        frame_payload = {
            **payload,
            "t": time_s,
            "az": az_true,
            "el": elevation,
            "range_km": range_m / 1000.0,
            "receiver_depth_m": receiver_depth_m,
            "source_depth_m": source["depth_m"],
            "array_heading": array_heading,
            "radial_speed_mps": radial_speed_mps,
            "source_radial_speed_mps": source_radial_speed_mps,
            "receiver_radial_speed_mps": receiver_radial_speed_mps,
            "source_east_mps": source["east_mps"],
            "source_north_mps": source["north_mps"],
            "receiver_east_mps": platform["east_mps"],
            "receiver_north_mps": platform["north_mps"],
            "source_depth_rate_mps": source["depth_rate_mps"],
            "receiver_depth_rate_mps": array_depth_rate_mps,
            "bearing_deg": source_to_array_bearing,
            "source_lon": source["lon"],
            "source_lat": source["lat"],
            "selected_element": payload.get("selected_element", config["selected_element"]),
        }
        result = self.process_frame(frame_payload)
        result["type"] = "motion_frame"
        result["kinematics"] = {
            "source": {key: round(value, 8) if isinstance(value, float) else value for key, value in source.items() if key not in {"east_mps", "north_mps"}},
            "platform": {key: round(value, 8) if isinstance(value, float) else value for key, value in platform.items() if key not in {"east_mps", "north_mps"}},
            "array": {
                "lon": round(array_lon, 8), "lat": round(array_lat, 8),
                "depth_m": round(receiver_depth_m, 3), "heading_deg": round(array_heading, 6),
                "depth_rate_mps": round(array_depth_rate_mps, 6),
                "tow_length_m": round(float(array_cfg["tow_length_m"]), 3),
                "heading_mode": array_cfg["heading_mode"],
            },
            "relative": {
                "range_km": round(range_m / 1000.0, 6),
                "bearing_deg": round(source_to_array_bearing, 6),
                "arrival_bearing_deg": round(arrival_bearing, 6),
                "broadside_angle_deg": round(az_true, 6),
                "range_rate_mps": round(radial_speed_mps, 6),
            },
        }
        return result

    def bind_bellhop(
        self,
        job_id: str,
        ray_job_id: str | None = None,
        arrival_job_id: str | None = None,
    ) -> dict[str, Any]:
        if self.jobs_dir is None:
            raise ValueError("服务端未配置 Bellhop 任务目录")
        had_timeline = self.field_timeline is not None
        self.field_timeline = None
        normalized_ray_job_id = str(ray_job_id or job_id).strip()
        normalized_arrival_job_id = str(arrival_job_id or "").strip() or None
        if (
            had_timeline
            or self.field is None
            or self.field.job_id != str(job_id).strip()
            or str(self.field.ray_job_id or self.field.job_id) != normalized_ray_job_id
            or self.field.arrival_job_id != normalized_arrival_job_id
        ):
            self.field = BellhopFieldSnapshot.load(
                self.jobs_dir,
                job_id,
                normalized_ray_job_id,
                normalized_arrival_job_id,
            )
            frequency = (
                self.field.frequency_hz
                if math.isfinite(self.field.frequency_hz) and self.field.frequency_hz > 0.0
                else self.config.f0
            )
            self.config = replace(
                self.config,
                c=self.field.mean_sound_speed_mps,
                f0=frequency,
                sample_rate=max(self.config.sample_rate, frequency * 4.0),
            )
        return self.field.field_response()

    def bind_bellhop_series(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """绑定多个 Bellhop 时间快照；每个任务可自动展开其多方位三维输出。"""

        if self.jobs_dir is None:
            raise ValueError("服务端未配置 Bellhop 任务目录")
        timeline = BellhopFieldTimeline.load(self.jobs_dir, payload.get("snapshots"))
        representative = timeline.representative
        frequencies = [
            field.frequency_hz for frame in timeline.frames for field in frame.fields
            if math.isfinite(field.frequency_hz) and field.frequency_hz > 0.0
        ]
        if frequencies and max(frequencies) - min(frequencies) > max(1.0, float(np.median(frequencies)) * 0.01):
            raise ValueError("Bellhop 时间快照的频率不一致")
        self.field_timeline = timeline
        self.field = representative
        frequency = float(np.median(frequencies)) if frequencies else self.config.f0
        sound_speeds = [
            field.mean_sound_speed_mps for frame in timeline.frames for field in frame.fields
        ]
        self.config = replace(
            self.config,
            c=float(np.median(sound_speeds)),
            f0=frequency,
            sample_rate=max(self.config.sample_rate, frequency * 4.0),
        )
        return timeline.acknowledgement()

    def sound_speed_profile(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        requested_job = str(payload.get("bellhop_job_id") or "").strip()
        if requested_job:
            self.bind_bellhop(requested_job)
        if self.field is not None:
            return self.field.profile_response()

        lon = _finite_float(payload.get("lon", 113.8), "lon", -180.0, 180.0)
        lat = _finite_float(payload.get("lat", 13.5), "lat", -90.0, 90.0)
        depth = np.linspace(0.0, 5000.0, 41)
        seasonal = 2.2 * math.sin(math.radians(lat)) + 0.6 * math.cos(math.radians(lon))
        sound_speed = 1523.0 + seasonal - 0.017 * np.minimum(depth, 900.0)
        sound_speed += 0.00125 * np.maximum(depth - 900.0, 0.0)
        return {
            "type": "profile_resp",
            "lon": lon,
            "lat": lat,
            "depth": np.round(depth, 2).tolist(),
            "c": np.round(sound_speed, 3).tolist(),
            "data_source": "analytic_fallback",
            "actual_environment": False,
        }

    def static_channel(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """计算固定声源到固定线阵各阵元的复数稀疏冲激响应。

        Bellhop 任务提供选定方位剖面的 TL 和声线。每个阵元按其实际
        经纬度重新计算距离，并在同一剖面上采样 TL 与多径。复数抽头的总非相干
        功率严格归一到该阵元的 Bellhop TL，便于后续时域、频域与波束域共用。
        """

        if self.field is None:
            raise ValueError("static_channel 需要先绑定 bellhop_job_id")
        field = self.field
        config = self.config
        n_elem = _finite_int(
            payload.get("n_elem", config.n_elem), "n_elem", 1, STATIC_CHANNEL_MAX_ELEMENTS
        )
        spacing_m = _finite_float(
            payload.get("element_spacing_m", payload.get("d", config.d)),
            "element_spacing_m",
            0.001,
            100.0,
        )
        receiver_depth_m = _finite_float(
            payload.get("receiver_depth_m", payload.get("array_depth_m", config.depth)),
            "receiver_depth_m",
            0.0,
            12000.0,
        )
        array_heading_deg = _finite_float(
            payload.get("array_heading_deg", payload.get("array_heading", field.section_theta_deg + 90.0)),
            "array_heading_deg",
            -360.0,
            360.0,
        )
        frequency_hz = _finite_float(
            payload.get("frequency_hz", payload.get("f0", config.f0)),
            "frequency_hz",
            1.0,
            100_000.0,
        )
        impulse_sample_rate_hz = _finite_float(
            payload.get("impulse_sample_rate_hz", payload.get("sample_rate", config.sample_rate)),
            "impulse_sample_rate_hz",
            10.0,
            1_000_000.0,
        )
        gate_period_s = _finite_float(
            payload.get("gate_period_s", 0.016),
            "gate_period_s",
            0.001,
            0.2,
        )
        transmission_mode = str(
            payload.get("transmission_mode", "single_pulse")
        ).strip().lower()
        transmission_mode = {
            "single": "single_pulse",
            "single_shot": "single_pulse",
            "one_shot": "single_pulse",
            "pulse_train": "periodic",
        }.get(transmission_mode, transmission_mode)
        if transmission_mode not in {"single_pulse", "periodic"}:
            raise ValueError("transmission_mode 只能是 single_pulse 或 periodic")
        waveform_type = str(payload.get("waveform_type", "cw")).strip().lower()
        if waveform_type not in {"cw", "lfm"}:
            raise ValueError("waveform_type 只能是 cw 或 lfm")
        lfm_bandwidth_hz = _finite_float(
            payload.get("lfm_bandwidth_hz", max(10.0, frequency_hz * 0.4)),
            "lfm_bandwidth_hz",
            0.1,
            100_000.0,
        )
        if "pulse_duration_s" in payload:
            gate_on_duration_s = _finite_float(
                payload.get("pulse_duration_s"), "pulse_duration_s", 0.0001, gate_period_s
            )
            gate_duty_cycle = gate_on_duration_s / gate_period_s
            blanking_ratio = 1.0 - gate_duty_cycle
        elif "blanking_ratio" in payload:
            blanking_ratio = _finite_float(
                payload.get("blanking_ratio"), "blanking_ratio", 0.0, 0.95
            )
            gate_duty_cycle = 1.0 - blanking_ratio
            gate_on_duration_s = gate_duty_cycle * gate_period_s
        else:
            gate_duty_cycle = _finite_float(
                payload.get("gate_duty_cycle", 0.6),
                "gate_duty_cycle",
                0.05,
                1.0,
            )
            blanking_ratio = 1.0 - gate_duty_cycle
            gate_on_duration_s = gate_duty_cycle * gate_period_s
        max_section_error_deg = _finite_float(
            payload.get("max_section_error_deg", 5.0),
            "max_section_error_deg",
            0.01,
            180.0,
        )
        noise_enabled = _finite_bool(payload.get("noise_enabled", True), "noise_enabled")
        environment_noise = payload.get("environment_noise")
        if environment_noise is not None and not isinstance(environment_noise, Mapping):
            raise ValueError("environment_noise 必须是对象")
        noise_file = Path(payload.get("noise_model_file", DEFAULT_MEASURED_NOISE_FILE))
        noise_seed = _finite_int(
            payload.get("noise_seed", 20261015), "noise_seed", 0, 2_147_483_647
        )
        source_level_db = _finite_float(
            payload.get("source_level_db", 190.0), "source_level_db", 80.0, 260.0
        )
        ambient_noise_db = _finite_float(
            payload.get("ambient_noise_db", 78.0), "ambient_noise_db", 20.0, 180.0
        )

        if "source_lon" in payload or "source_lat" in payload:
            requested_source_lon = _finite_float(
                payload.get("source_lon"), "source_lon", -180.0, 180.0
            )
            requested_source_lat = _finite_float(
                payload.get("source_lat"), "source_lat", -90.0, 90.0
            )
            if not (math.isfinite(field.source_lon) and math.isfinite(field.source_lat)):
                raise ValueError("Bellhop 任务没有可用的声源经纬度")
            source_error_m, _ = _distance_and_bearing(
                field.source_lon, field.source_lat, requested_source_lon, requested_source_lat
            )
            if source_error_m > 25.0:
                raise ValueError(
                    f"请求声源与 Bellhop 任务声源相差 {source_error_m:.1f} m，"
                    "请先用当前声源位置重新计算 Bellhop 任务"
                )

        has_receiver_coordinates = "receiver_lon" in payload or "receiver_lat" in payload
        if has_receiver_coordinates:
            if not (math.isfinite(field.source_lon) and math.isfinite(field.source_lat)):
                raise ValueError("Bellhop 任务没有可用的声源经纬度")
            receiver_lon = _finite_float(payload.get("receiver_lon"), "receiver_lon", -180.0, 180.0)
            receiver_lat = _finite_float(payload.get("receiver_lat"), "receiver_lat", -90.0, 90.0)
            center_distance_m, center_bearing_deg = _distance_and_bearing(
                field.source_lon, field.source_lat, receiver_lon, receiver_lat
            )
            if center_distance_m <= 0.0:
                raise ValueError("阵列中心不能与声源重合")
            section_error_deg = _bearing_difference(center_bearing_deg, field.section_theta_deg)
            if section_error_deg > max_section_error_deg:
                raise ValueError(
                    f"阵列中心方位 {center_bearing_deg:.3f}° 与 Bellhop 剖面 "
                    f"{field.section_theta_deg:.3f}° 相差 {section_error_deg:.3f}°，超过限制 "
                    f"{max_section_error_deg:.3f}°"
                )
        else:
            center_distance_m = 1000.0 * _finite_float(
                payload.get("range_km", 50.0), "range_km", 0.001, 1000.0
            )
            center_bearing_deg = _wrap_bearing(field.section_theta_deg)
            if math.isfinite(field.source_lon) and math.isfinite(field.source_lat):
                receiver_lon, receiver_lat = _destination(
                    field.source_lon,
                    field.source_lat,
                    center_bearing_deg,
                    center_distance_m,
                )
            else:
                receiver_lon = receiver_lat = math.nan
            section_error_deg = 0.0

        field_frequency_matched = bool(
            math.isfinite(field.frequency_hz)
            and abs(frequency_hz - field.frequency_hz) <= max(1.0, field.frequency_hz * 0.01)
        )
        element_offsets_m = (
            np.arange(n_elem, dtype=np.float64) - (n_elem - 1) / 2.0
        ) * spacing_m
        element_positions: list[dict[str, float]] = []
        for element_index, offset_m in enumerate(element_offsets_m):
            if math.isfinite(receiver_lon) and math.isfinite(receiver_lat):
                element_lon, element_lat = _destination(
                    receiver_lon, receiver_lat, array_heading_deg, float(offset_m)
                )
                element_distance_m, element_bearing_deg = _distance_and_bearing(
                    field.source_lon, field.source_lat, element_lon, element_lat
                )
            else:
                projection_m = float(offset_m) * math.cos(
                    math.radians(array_heading_deg - center_bearing_deg)
                )
                element_distance_m = center_distance_m + projection_m
                element_bearing_deg = center_bearing_deg
                element_lon = element_lat = math.nan
            element_positions.append({
                "element_index": float(element_index),
                "offset_m": float(offset_m),
                "lon": float(element_lon),
                "lat": float(element_lat),
                "distance_m": float(element_distance_m),
                "bearing_deg": float(element_bearing_deg),
            })

        native_arrivals_enabled = _finite_bool(
            payload.get("native_arrivals_enabled", True),
            "native_arrivals_enabled",
        )
        arrival_beam_count = _finite_int(
            payload.get("arrival_beam_count", 0),
            "arrival_beam_count",
            0,
            100_000,
        )
        native_arrival_generation_error: str | None = None
        if (
            native_arrivals_enabled
            and field_frequency_matched
            and field.arrival_job_id is None
            and self.jobs_dir is not None
        ):
            try:
                generated = _generate_static_native_arrivals(
                    Path(self.jobs_dir),
                    field,
                    receiver_depth_m,
                    [item["distance_m"] / 1000.0 for item in element_positions],
                    arrival_beam_count,
                )
                generated_path = str(generated["path"])
                field.native_arrival_sets = [
                    item for item in field.native_arrival_sets
                    if str(item.get("path")) != generated_path
                ] + [generated]
            except (OSError, RuntimeError, ValueError) as exc:
                native_arrival_generation_error = str(exc)

        elements: list[dict[str, Any]] = []
        all_delays: list[float] = []
        tl_values: list[float] = []
        for position in element_positions:
            element_index = int(position["element_index"])
            offset_m = position["offset_m"]
            element_lon = position["lon"]
            element_lat = position["lat"]
            element_distance_m = position["distance_m"]
            element_bearing_deg = position["bearing_deg"]
            range_km = element_distance_m / 1000.0
            try:
                channel_sample = field.channel_sample(
                    range_km,
                    receiver_depth_m,
                    prefer_native_arrivals=(
                        native_arrivals_enabled and field_frequency_matched
                    ),
                )
            except ValueError as exc:
                raise ValueError(f"阵元 {element_index}: {exc}") from exc
            tl_db = channel_sample["transmission_loss_db"]
            bottom_depth_m = channel_sample["bottom_depth_m"]
            arrivals = channel_sample["arrivals"]
            uses_native_arrivals = all(
                path.get("parameter_source") == "bellhop_native_arrivals"
                for path in arrivals
            )
            if uses_native_arrivals:
                amplitudes = np.asarray(
                    [float(path["amplitude"]) for path in arrivals],
                    dtype=np.float64,
                )
                normalization = "bellhop_native_complex_amplitude"
            else:
                relative_amplitudes = np.asarray(
                    [10.0 ** (float(path["relative_gain_db"]) / 20.0) for path in arrivals],
                    dtype=np.float64,
                )
                target_power = 10.0 ** (-float(tl_db) / 10.0)
                scale = math.sqrt(
                    target_power
                    / max(float(np.sum(relative_amplitudes ** 2)), 1.0e-300)
                )
                amplitudes = relative_amplitudes * scale
                normalization = "geometry_weights_scaled_to_sampled_tl"
            sparse_taps = []
            coherent_sum = 0.0j
            first_delay_s = min(float(path["delay_s"]) for path in arrivals)
            for path, amplitude_value in zip(arrivals, amplitudes):
                delay_s = float(path["delay_s"])
                excess_delay_s = delay_s - first_delay_s
                amplitude = float(amplitude_value)
                phase_rad = (
                    math.radians(float(path["phase_deg"]))
                    if uses_native_arrivals
                    else -2.0 * math.pi * frequency_hz * delay_s
                )
                gain = amplitude * complex(math.cos(phase_rad), math.sin(phase_rad))
                coherent_sum += gain
                all_delays.append(delay_s)
                sparse_taps.append({
                    "tap_index": int(round(excess_delay_s * impulse_sample_rate_hz)),
                    "delay_s": round(delay_s, 9),
                    "excess_delay_s": round(excess_delay_s, 9),
                    "gain_real": float(round(gain.real, 12)),
                    "gain_imag": float(round(gain.imag, 12)),
                    "magnitude": float(round(abs(gain), 12)),
                    "phase_deg": round(math.degrees(math.atan2(gain.imag, gain.real)), 6),
                    "source_angle_deg": path.get("source_angle_deg"),
                    "arrival_angle_deg": path.get("arrival_angle_deg"),
                    "path_length_m": path.get("path_length_m"),
                    "ray_index": path.get("ray_index"),
                    "arrival_index": path.get("arrival_index"),
                    "top_bounces": path.get("top_bounces"),
                    "bottom_bounces": path.get("bottom_bounces"),
                    "parameter_source": path.get(
                        "parameter_source", "ray_geometry_reconstruction"
                    ),
                })
            arrival_incoherent_tl_db = -10.0 * math.log10(
                max(float(np.sum(amplitudes ** 2)), 1.0e-300)
            )
            coherent_tl_db = -20.0 * math.log10(max(abs(coherent_sum), 1.0e-300))
            tl_values.append(float(tl_db))
            elements.append({
                "element_index": element_index,
                "offset_m": round(float(offset_m), 6),
                "lon": _json_values(element_lon, 8),
                "lat": _json_values(element_lat, 8),
                "depth_m": round(receiver_depth_m, 3),
                "range_km": round(range_km, 6),
                "bearing_deg": round(element_bearing_deg, 6),
                "bottom_depth_m": round(float(bottom_depth_m), 3),
                "transmission_loss_db": round(float(tl_db), 6),
                "arrival_incoherent_transmission_loss_db": round(
                    arrival_incoherent_tl_db, 6
                ),
                "arrival_vs_field_tl_delta_db": round(
                    arrival_incoherent_tl_db - float(tl_db), 6
                ),
                "coherent_transmission_loss_db": round(coherent_tl_db, 6),
                "path_count": len(sparse_taps),
                "arrival_parameter_source": (
                    "bellhop_native_arrivals"
                    if uses_native_arrivals else "ray_geometry_reconstruction"
                ),
                "impulse_response": {
                    "representation": "complex_sparse_baseband",
                    "normalization": normalization,
                    "sample_rate_hz": round(impulse_sample_rate_hz, 3),
                    "reference_delay_s": round(first_delay_s, 9),
                    "taps": sparse_taps,
                },
            })

        global_first_arrival_s = min(all_delays)
        global_last_arrival_s = max(all_delays)
        for element in elements:
            element["impulse_response"]["global_reference_delay_s"] = round(
                global_first_arrival_s, 9
            )
            for tap in element["impulse_response"]["taps"]:
                global_excess_delay_s = float(tap["delay_s"]) - global_first_arrival_s
                tap["global_excess_delay_s"] = round(global_excess_delay_s, 9)
                tap["global_tap_index"] = int(round(global_excess_delay_s * impulse_sample_rate_hz))

        # 默认按一次有限时长发射来展示传播事件。只有显式选择 periodic 时才重复发射。
        # 每条 Bellhop 路径都用自己的绝对时延反查发射时间，因此单脉冲到达前、
        # 两组互不重叠的回波之间以及末达结束后，确定性接收信号均为真实零值。
        signal_sample_rate_hz = max(impulse_sample_rate_hz, frequency_hz * 4.0)
        # 时域/频域动态图只保留首达附近的全采样率窗口。不能为了把整个
        # 多途扩展塞进固定点数而降采样，否则双频谱线会在显示数据中混叠。
        # 完整首/末达和所有多途仍由 CIR 及绝对时间概览表示。
        requested_signal_duration_s = max(
            FRAME_SAMPLE_COUNT / signal_sample_rate_hz,
            4.0 * gate_period_s,
        )
        signal_sample_count = min(
            2048,
            max(FRAME_SAMPLE_COUNT, int(math.ceil(requested_signal_duration_s * signal_sample_rate_hz))),
        )
        signal_time = np.arange(signal_sample_count, dtype=np.float64) / signal_sample_rate_hz
        signal_duration_s = signal_sample_count / signal_sample_rate_hz
        receive_pretrigger_s = min(0.25 * signal_duration_s, max(gate_period_s, 0.002))
        playback_start_s = max(0.0, global_first_arrival_s - gate_period_s)
        if transmission_mode == "periodic":
            playback_end_s = global_last_arrival_s + 2.0 * gate_period_s
        else:
            playback_end_s = global_last_arrival_s + gate_on_duration_s + 2.0 * gate_period_s
        default_window_start_s = max(0.0, global_first_arrival_s - receive_pretrigger_s)
        if "receive_window_start_s" in payload:
            receive_window_start_s = _finite_float(
                payload.get("receive_window_start_s"),
                "receive_window_start_s",
                playback_start_s,
                playback_end_s,
            )
        else:
            receive_window_start_s = default_window_start_s
        receive_time_s = receive_window_start_s + signal_time
        primary_frequency_hz = min(frequency_hz, signal_sample_rate_hz * 0.42)
        lfm_start_frequency_hz = max(1.0, primary_frequency_hz - lfm_bandwidth_hz / 2.0)
        lfm_end_frequency_hz = min(
            signal_sample_rate_hz * 0.46,
            primary_frequency_hz + lfm_bandwidth_hz / 2.0,
        )
        if lfm_end_frequency_hz <= lfm_start_frequency_hz:
            lfm_start_frequency_hz = max(1.0, primary_frequency_hz * 0.8)
            lfm_end_frequency_hz = min(signal_sample_rate_hz * 0.46, primary_frequency_hz * 1.2)
        actual_lfm_bandwidth_hz = lfm_end_frequency_hz - lfm_start_frequency_hz

        def probe_at(times_s: np.ndarray) -> np.ndarray:
            times = np.asarray(times_s, dtype=np.float64)
            nonnegative = np.maximum(times, 0.0)
            if transmission_mode == "periodic":
                pulse_time = np.mod(nonnegative, gate_period_s)
                active = (times >= 0.0) & (pulse_time < gate_on_duration_s)
            else:
                pulse_time = nonnegative
                active = (times >= 0.0) & (times < gate_on_duration_s)
            if waveform_type == "lfm":
                slope_hz_s = actual_lfm_bandwidth_hz / max(gate_on_duration_s, 1.0e-12)
                phase = 2.0 * np.pi * (
                    lfm_start_frequency_hz * pulse_time + 0.5 * slope_hz_s * pulse_time ** 2
                )
            else:
                # CW 在关断期间仍保持连续相位。
                phase = 2.0 * np.pi * primary_frequency_hz * times
            return active.astype(np.float64) * math.sqrt(2.0) * np.cos(phase)

        transmit_samples = probe_at(signal_time)
        samples = np.zeros((n_elem, signal_sample_count), dtype=np.float64)
        path_active = np.zeros((n_elem, signal_sample_count), dtype=np.int16)
        for element_index, element in enumerate(elements):
            for tap in element["impulse_response"]["taps"]:
                delay_s = float(tap["delay_s"])
                magnitude = float(tap["magnitude"])
                emission_time_s = receive_time_s - delay_s
                if transmission_mode == "periodic":
                    tap_active = (
                        (emission_time_s >= 0.0)
                        & (np.mod(np.maximum(emission_time_s, 0.0), gate_period_s) < gate_on_duration_s)
                    )
                else:
                    tap_active = (
                        (emission_time_s >= 0.0)
                        & (emission_time_s < gate_on_duration_s)
                    )
                    tap["arrival_start_s"] = round(delay_s, 9)
                    tap["arrival_end_s"] = round(delay_s + gate_on_duration_s, 9)
                path_active[element_index] += tap_active.astype(np.int16)
                samples[element_index] += magnitude * probe_at(emission_time_s)

        clean_samples = samples.copy()
        noise_samples = np.zeros_like(samples)
        noise_metadata = {
            "enabled": noise_enabled,
            "model": "measured_mixing_coefficients",
            "source_level_db": source_level_db,
            "ambient_noise_db": ambient_noise_db,
            "sensitivity_db": float(config.sensitivity),
        }
        if noise_enabled:
            rng = np.random.default_rng(noise_seed)
            if environment_noise:
                noise_samples, model_metadata = _generate_environment_time_frequency_noise(
                    signal_sample_count,
                    signal_sample_rate_hz,
                    n_elem,
                    rng,
                    environment_noise,
                )
                ambient_noise_db = float(model_metadata["ambient_noise_db"])
                noise_metadata["model"] = "environment_time_frequency"
                # Environment PSD is absolute. Only convert absolute pressure to
                # the normalized signal frame (source level); do not re-normalize it.
                noise_samples = noise_samples * 10.0 ** (
                    -(source_level_db + config.sensitivity) / 20.0
                )
                target_noise_rms = 10.0 ** (
                    (ambient_noise_db - source_level_db - config.sensitivity) / 20.0
                )
            else:
                noise_samples, model_metadata = _generate_measured_noise(
                    signal_sample_count,
                    signal_sample_rate_hz,
                    n_elem,
                    rng,
                    noise_file,
                )
                target_noise_rms = 10.0 ** (
                    -(source_level_db - ambient_noise_db + config.sensitivity) / 20.0
                )
                noise_samples = _unit_rms_per_channel(noise_samples) * target_noise_rms
            samples = clean_samples + noise_samples
            noise_metadata.update(model_metadata)
            noise_metadata["target_rms_per_element"] = target_noise_rms
            signal_rms = np.sqrt(np.mean(clean_samples ** 2, axis=1))
            noise_rms = np.sqrt(np.mean(noise_samples ** 2, axis=1))
            noise_metadata["signal_rms_per_element"] = np.round(signal_rms, 12).tolist()
            noise_metadata["noise_rms_per_element"] = np.round(noise_rms, 12).tolist()
            noise_metadata["snr_db_per_element"] = np.round(
                20.0 * np.log10(np.maximum(signal_rms, 1.0e-300) / np.maximum(noise_rms, 1.0e-300)),
                6,
            ).tolist()

        waveform_indexes = np.arange(signal_sample_count, dtype=int)
        waveform_time_ms = 1000.0 * signal_time[waveform_indexes]
        waveform_receive_time_s = receive_time_s[waveform_indexes]
        window = np.hanning(signal_sample_count)
        element_fft = np.abs(np.fft.rfft(samples * window[None, :], axis=1))
        frequencies = np.fft.rfftfreq(signal_sample_count, 1.0 / signal_sample_rate_hz)
        signal_frequency_min = max(1.0, min(100.0, primary_frequency_hz * 0.25))
        signal_frequency_max = (
            lfm_end_frequency_hz if waveform_type == "lfm" else primary_frequency_hz
        ) * 1.35
        noise_frequency_range = noise_metadata.get("frequency_range_hz")
        if (
            noise_enabled
            and noise_metadata.get("model") == "environment_time_frequency"
            and isinstance(noise_frequency_range, list)
            and len(noise_frequency_range) == 2
        ):
            # 环境噪声时频结果已经定义了有效频带。观察阵元频谱/时频图以该频带
            # 为主；若通信信号自身越过该上限，才扩展到足以显示信号的范围。
            frequency_min = max(1.0, min(signal_frequency_min, float(noise_frequency_range[0])))
            frequency_max = min(
                signal_sample_rate_hz * 0.48,
                max(float(noise_frequency_range[1]), signal_frequency_max),
            )
        else:
            frequency_min = signal_frequency_min
            frequency_max = min(signal_sample_rate_hz * 0.48, max(3000.0, signal_frequency_max))
        useful_indexes = np.flatnonzero(
            (frequencies >= frequency_min) & (frequencies <= frequency_max)
        )
        if useful_indexes.size > SPECTRUM_POINT_COUNT:
            useful_indexes = useful_indexes[
                np.linspace(0, useful_indexes.size - 1, SPECTRUM_POINT_COUNT)
                .round()
                .astype(int)
            ]
        element_spectrum_db = 20.0 * np.log10(
            np.maximum(element_fft[:, useful_indexes], 1.0e-300)
        )
        element_spectrum_db -= np.max(element_spectrum_db, axis=1, keepdims=True)
        element_spectrum_db = np.clip(element_spectrum_db - 30.0, -82.0, -30.0)
        for element_index, element in enumerate(elements):
            element_signal_active = path_active[element_index] > 0
            element_signal_present = bool(np.any(element_signal_active))
            element_display_spectrum_db = element_spectrum_db[element_index]
            if not element_signal_present and not noise_enabled:
                element_display_spectrum_db = np.full_like(element_display_spectrum_db, -82.0)
            element["signal"] = {
                "waveform": np.round(samples[element_index, waveform_indexes], 12).tolist(),
                "clean_waveform": np.round(clean_samples[element_index, waveform_indexes], 12).tolist(),
                "noise_waveform": np.round(noise_samples[element_index, waveform_indexes], 12).tolist(),
                "receive_time_s": np.round(waveform_receive_time_s, 9).tolist(),
                "window_start_s": round(receive_window_start_s, 9),
                "window_end_s": round(float(receive_time_s[-1]), 9),
                "spectrum_db": np.round(element_display_spectrum_db, 3).tolist(),
                "signal_present": element_signal_present,
                "display_state": (
                    "signal_plus_noise" if element_signal_present and noise_enabled
                    else "signal" if element_signal_present
                    else "noise_only" if noise_enabled
                    else "blank"
                ),
                "active_path_count": path_active[element_index].astype(int).tolist(),
            }

        demodulator = np.exp(-2j * np.pi * primary_frequency_hz * signal_time)
        snapshot = (2.0 / signal_sample_count) * (samples @ demodulator)
        scan_angles = np.linspace(-config.az_scan, config.az_scan, BEAM_POINT_COUNT)
        steering = np.exp(
            -2j
            * np.pi
            * primary_frequency_hz
            / config.c
            * np.sin(np.radians(scan_angles))[:, None]
            * element_offsets_m[None, :]
        )
        beam_amplitude = np.abs(steering.conj() @ snapshot) / n_elem
        beam_db = 20.0 * np.log10(np.maximum(beam_amplitude, 1.0e-300))
        beam_db -= float(np.max(beam_db))
        beam_db = np.clip(beam_db, -42.0, 0.0)
        window_signal_present = bool(np.any(path_active > 0))
        if not window_signal_present and not noise_enabled:
            beam_db = np.full_like(beam_db, -42.0)
            peak_angle_deg = None
        else:
            peak_index = int(np.argmax(beam_db))
            peak_angle_deg = float(scan_angles[peak_index]) if window_signal_present else None

        # 空间谱必须与播放时间同步。这里提供覆盖完整首达—末达过程的紧凑时间轴；
        # 单脉冲间隙返回 null，而不是沿用上一帧的峰值，避免界面显示“冻结的目标”。
        active_intervals_s: list[tuple[float, float]] = []
        if transmission_mode == "single_pulse":
            raw_intervals = sorted(
                (
                    float(tap["delay_s"]),
                    float(tap["delay_s"]) + gate_on_duration_s,
                )
                for element in elements
                for tap in element["impulse_response"]["taps"]
            )
            for start_s, end_s in raw_intervals:
                if active_intervals_s and start_s <= active_intervals_s[-1][1]:
                    active_intervals_s[-1] = (
                        active_intervals_s[-1][0],
                        max(active_intervals_s[-1][1], end_s),
                    )
                else:
                    active_intervals_s.append((start_s, end_s))

        if active_intervals_s:
            # 每个合并后的回波区间至少取起点、中点和结束前一点，区间之间再取一个
            # 空白帧。这样即使首末达相隔数秒，也不会漏掉毫秒级的短回波。
            beam_time_values = [playback_start_s]
            previous_end_s = playback_start_s
            edge_epsilon_s = 0.25 / signal_sample_rate_hz
            for start_s, end_s in active_intervals_s:
                if start_s > previous_end_s:
                    beam_time_values.append(0.5 * (previous_end_s + start_s))
                beam_time_values.extend((
                    start_s,
                    0.5 * (start_s + end_s),
                    max(start_s, end_s - edge_epsilon_s),
                    end_s,
                ))
                previous_end_s = end_s
            if playback_end_s > previous_end_s:
                beam_time_values.append(0.5 * (previous_end_s + playback_end_s))
            beam_time_values.append(playback_end_s)
            beam_times_s = np.asarray(sorted(set(beam_time_values)), dtype=np.float64)
        else:
            beam_frame_count = min(256, max(64, int(math.ceil(
                (playback_end_s - playback_start_s) / max(gate_on_duration_s / 8.0, 1.0e-4)
            ))))
            beam_times_s = np.linspace(playback_start_s, playback_end_s, beam_frame_count)
        beam_timeline_spectra: list[list[float] | None] = []
        beam_timeline_peaks: list[float | None] = []
        beam_timeline_active_elements: list[int] = []
        beam_timeline_active_paths: list[int] = []
        for beam_time_s in beam_times_s:
            analytic_snapshot = np.zeros(n_elem, dtype=np.complex128)
            active_elements = np.zeros(n_elem, dtype=bool)
            active_paths = 0
            for element_index, element in enumerate(elements):
                for tap in element["impulse_response"]["taps"]:
                    emission_time_s = beam_time_s - float(tap["delay_s"])
                    if transmission_mode == "periodic":
                        pulse_time_s = emission_time_s % gate_period_s if emission_time_s >= 0.0 else 0.0
                        active = emission_time_s >= 0.0 and pulse_time_s < gate_on_duration_s
                    else:
                        pulse_time_s = max(emission_time_s, 0.0)
                        active = 0.0 <= emission_time_s < gate_on_duration_s
                    if not active:
                        continue
                    if waveform_type == "lfm":
                        slope_hz_s = actual_lfm_bandwidth_hz / max(gate_on_duration_s, 1.0e-12)
                        phase_rad = 2.0 * math.pi * (
                            lfm_start_frequency_hz * pulse_time_s
                            + 0.5 * slope_hz_s * pulse_time_s ** 2
                        )
                    else:
                        phase_rad = 2.0 * math.pi * primary_frequency_hz * emission_time_s
                    analytic_snapshot[element_index] += float(tap["magnitude"]) * complex(
                        math.cos(phase_rad), math.sin(phase_rad)
                    )
                    active_elements[element_index] = True
                    active_paths += 1
            active_element_count = int(np.count_nonzero(active_elements))
            beam_timeline_active_elements.append(active_element_count)
            beam_timeline_active_paths.append(active_paths)
            if active_element_count == 0:
                beam_timeline_spectra.append(None)
                beam_timeline_peaks.append(None)
                continue
            frame_amplitude = np.abs(steering.conj() @ analytic_snapshot) / n_elem
            frame_db = 20.0 * np.log10(np.maximum(frame_amplitude, 1.0e-300))
            frame_db -= float(np.max(frame_db))
            frame_db = np.clip(frame_db, -42.0, 0.0)
            frame_peak_index = int(np.argmax(frame_db))
            beam_timeline_spectra.append(np.round(frame_db, 3).tolist())
            beam_timeline_peaks.append(round(float(scan_angles[frame_peak_index]), 3))

        sample_fft = np.fft.rfft(samples, axis=1)
        all_frequencies = np.fft.rfftfreq(signal_sample_count, 1.0 / signal_sample_rate_hz)
        beamforming_angle_deg = peak_angle_deg if peak_angle_deg is not None else 0.0
        peak_distance_m = element_offsets_m * math.sin(math.radians(beamforming_angle_deg))
        correction = np.exp(
            2j
            * np.pi
            * all_frequencies[None, :]
            * peak_distance_m[:, None]
            / config.c
        )
        beam_output = np.fft.irfft(
            np.mean(sample_fft * correction, axis=0), n=signal_sample_count
        )

        field_response = field.field_response()
        arrival_parameter_sources = sorted({
            str(element["arrival_parameter_source"]) for element in elements
        })
        impulse_normalizations = sorted({
            str(element["impulse_response"]["normalization"]) for element in elements
        })
        return {
            "type": "static_channel_resp",
            "module": "static_channel",
            "motion": {
                "source_speed_m_s": 0.0,
                "receiver_speed_m_s": 0.0,
                "doppler_hz": 0.0,
                "time_invariant": True,
            },
            "bellhop_job_id": field.job_id,
            "ray_job_id": field.ray_job_id or field.job_id,
            "arrival_job_id": field.arrival_job_id,
            "engine": "BellhopCXX",
            "frequency_hz": round(frequency_hz, 6),
            "field_frequency_hz": _json_values(field.frequency_hz, 6),
            "frequency_matched": field_frequency_matched,
            "source": {
                "lon": _json_values(field.source_lon, 8),
                "lat": _json_values(field.source_lat, 8),
                "depth_m": round(field.source_depth_m, 3),
            },
            "array": {
                "center_lon": _json_values(receiver_lon, 8),
                "center_lat": _json_values(receiver_lat, 8),
                "depth_m": round(receiver_depth_m, 3),
                "heading_deg": round(_wrap_bearing(array_heading_deg), 6),
                "element_count": n_elem,
                "element_spacing_m": round(spacing_m, 6),
                "length_m": round(float((n_elem - 1) * spacing_m), 6),
                "center_range_km": round(center_distance_m / 1000.0, 6),
                "center_bearing_deg": round(center_bearing_deg, 6),
                "section_bearing_deg": round(_wrap_bearing(field.section_theta_deg), 6),
                "section_error_deg": round(section_error_deg, 6),
            },
            "transmission_loss": {
                "definition": "Bellhop TL sampled independently at each fixed element",
                "reference": "source-centered Bellhop section sampled at the fixed array elements",
                "min_db": round(min(tl_values), 6),
                "max_db": round(max(tl_values), 6),
                "mean_db": round(float(np.mean(tl_values)), 6),
                "per_element_db": [round(value, 6) for value in tl_values],
            },
            "impulse_response": {
                "representation": "per_element_complex_sparse_baseband",
                "arrival_parameter_sources": arrival_parameter_sources,
                "normalization": (
                    impulse_normalizations[0]
                    if len(impulse_normalizations) == 1 else "mixed"
                ),
                "native_arrivals_used": (
                    arrival_parameter_sources == ["bellhop_native_arrivals"]
                ),
                "native_arrivals_enabled": native_arrivals_enabled,
                "arrival_beam_count": arrival_beam_count,
                "fallback_reason": (
                    None
                    if arrival_parameter_sources == ["bellhop_native_arrivals"]
                    else (
                        "native_arrivals_disabled"
                        if not native_arrivals_enabled
                        else "requested_frequency_differs_from_arrival_frequency"
                        if field.native_arrival_sets and not field_frequency_matched
                        else native_arrival_generation_error
                        or "native_arrivals_unavailable"
                    )
                ),
                "sample_rate_hz": round(impulse_sample_rate_hz, 3),
                "global_first_arrival_s": round(global_first_arrival_s, 9),
                "global_last_arrival_s": round(global_last_arrival_s, 9),
            },
            "signal": {
                "definition": "periodically gated CW/LFM probe through each fixed-element complex CIR",
                "waveform_type": waveform_type,
                "sample_rate_hz": round(signal_sample_rate_hz, 3),
                "sound_speed_m_s": round(field.mean_sound_speed_mps, 3),
                "time_ms": np.round(waveform_time_ms, 6).tolist(),
                "transmit_waveform": np.round(transmit_samples[waveform_indexes], 6).tolist(),
                "receive_time_s": np.round(waveform_receive_time_s, 9).tolist(),
                "receive_window_start_s": round(receive_window_start_s, 9),
                "receive_window_end_s": round(float(receive_time_s[-1]), 9),
                "frequencies_hz": np.round(frequencies[useful_indexes], 3).tolist(),
                "analysis_frequency_range_hz": [
                    round(float(frequencies[useful_indexes[0]]), 3),
                    round(float(frequencies[useful_indexes[-1]]), 3),
                ],
                "probe_frequencies_hz": [
                    round(
                        lfm_start_frequency_hz if waveform_type == "lfm" else primary_frequency_hz,
                        6,
                    ),
                    round(
                        lfm_end_frequency_hz if waveform_type == "lfm" else primary_frequency_hz,
                        6,
                    ),
                ],
                "lfm_bandwidth_hz": round(actual_lfm_bandwidth_hz, 6) if waveform_type == "lfm" else 0.0,
                "gating": {
                    "mode": (
                        ("pulse_reset_lfm" if waveform_type == "lfm" else "continuous_phase_periodic")
                        if transmission_mode == "periodic"
                        else ("single_lfm_pulse" if waveform_type == "lfm" else "single_cw_pulse")
                    ),
                    "transmission_mode": transmission_mode,
                    "repeats": transmission_mode == "periodic",
                    "period_s": round(gate_period_s, 9),
                    "duty_cycle": round(gate_duty_cycle, 6),
                    "blanking_ratio": round(blanking_ratio, 6),
                    "on_duration_s": round(gate_on_duration_s, 9),
                    "off_duration_s": round(gate_period_s - gate_on_duration_s, 9),
                    "phase_resets_on_gate": waveform_type == "lfm",
                },
                "timeline": {
                    "transmit_start_s": 0.0,
                    "global_first_arrival_s": round(global_first_arrival_s, 9),
                    "global_last_arrival_s": round(global_last_arrival_s, 9),
                    "playback_start_s": round(playback_start_s, 9),
                    "playback_end_s": round(playback_end_s, 9),
                    "pretrigger_s": round(gate_period_s, 9),
                    "post_tail_s": round(2.0 * gate_period_s, 9),
                    "active_intervals_s": [
                        [round(start_s, 9), round(end_s, 9)]
                        for start_s, end_s in active_intervals_s
                    ],
                    "display_scale": "broken_axis",
                    "propagation_gap_compressed": True,
                },
            },
            "noise": noise_metadata,
            "beam": {
                "definition": "time-resolved fixed-array narrowband scan and wideband delay-and-sum output",
                "scan_angles_deg": np.round(scan_angles, 3).tolist(),
                "spatial_spectrum_db": np.round(beam_db, 3).tolist(),
                "peak_angle_deg": round(peak_angle_deg, 3) if peak_angle_deg is not None else None,
                "signal_present": window_signal_present,
                "display_state": (
                    "signal_plus_noise" if window_signal_present and noise_enabled
                    else "signal" if window_signal_present
                    else "noise_only" if noise_enabled
                    else "blank"
                ),
                "output_time_ms": np.round(waveform_time_ms, 6).tolist(),
                "output_time_s": np.round(waveform_receive_time_s, 9).tolist(),
                "output_waveform": np.round(beam_output[waveform_indexes], 12).tolist(),
                "timeline": {
                    "time_s": np.round(beam_times_s, 9).tolist(),
                    "signal_present": [count > 0 for count in beam_timeline_active_elements],
                    "active_element_count": beam_timeline_active_elements,
                    "active_path_count": beam_timeline_active_paths,
                    "peak_angle_deg": beam_timeline_peaks,
                    "spatial_spectrum_db": beam_timeline_spectra,
                    "inactive_display": "noise_only" if noise_enabled else "blank",
                },
            },
            "elements": elements,
            "ray_geometry": {
                "ray_count": field_response["ray_count"],
                "multipath_method": field_response["multipath_method"],
                "rays": field_response["rays"],
            },
            "sound_field": {
                "range_km": field_response["range_km"],
                "depth_m": field_response["depth_m"],
                "tl_db": field_response["tl_db"],
                "bathymetry": field_response["bathymetry"],
                "ssp": field_response["ssp"],
                "mask_definition": "depth below interpolated bathymetry is terrain, not water-column TL",
            },
        }

    def process_frame(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        config = self.config
        frame = _finite_int(payload.get("frame", 0), "frame", 0, 2_147_483_647)
        time_s = _finite_float(payload.get("t", 0.0), "t", 0.0, 86400.0)
        az_true = _finite_float(payload.get("az", payload.get("az_true", 0.0)), "az", -90.0, 90.0)
        elevation = _finite_float(payload.get("el", 0.0), "el", -90.0, 90.0)
        range_km = _finite_float(payload.get("range_km", 50.0), "range_km", 0.01, 1000.0)
        receiver_depth_m = _finite_float(
            payload.get("receiver_depth_m", config.depth), "receiver_depth_m", 0.0, 12000.0
        )
        source_depth_m = _finite_float(
            payload.get("source_depth_m", receiver_depth_m), "source_depth_m", 0.0, 12000.0
        )
        array_heading = _finite_float(payload.get("array_heading", 180.0), "array_heading", -360.0, 360.0)
        array_curvature = _finite_float(payload.get("array_curvature", 0.0), "array_curvature", -0.2, 0.2)
        radial_speed_mps = _finite_float(
            payload.get("radial_speed_mps", 0.0), "radial_speed_mps", -200.0, 200.0
        )
        source_radial_speed_mps = _finite_float(
            payload.get("source_radial_speed_mps", 0.0),
            "source_radial_speed_mps", -200.0, 200.0,
        )
        receiver_radial_speed_mps = _finite_float(
            payload.get("receiver_radial_speed_mps", source_radial_speed_mps + radial_speed_mps),
            "receiver_radial_speed_mps", -200.0, 200.0,
        )
        if "radial_speed_mps" not in payload and (
            "source_radial_speed_mps" in payload or "receiver_radial_speed_mps" in payload
        ):
            radial_speed_mps = receiver_radial_speed_mps - source_radial_speed_mps
        has_horizontal_velocity_components = all(name in payload for name in (
            "source_east_mps", "source_north_mps",
            "receiver_east_mps", "receiver_north_mps",
        ))
        source_east_mps = source_north_mps = 0.0
        receiver_east_mps = receiver_north_mps = 0.0
        if has_horizontal_velocity_components:
            source_east_mps = _finite_float(
                payload["source_east_mps"], "source_east_mps", -200.0, 200.0
            )
            source_north_mps = _finite_float(
                payload["source_north_mps"], "source_north_mps", -200.0, 200.0
            )
            receiver_east_mps = _finite_float(
                payload["receiver_east_mps"], "receiver_east_mps", -200.0, 200.0
            )
            receiver_north_mps = _finite_float(
                payload["receiver_north_mps"], "receiver_north_mps", -200.0, 200.0
            )
        source_depth_rate_mps = _finite_float(
            payload.get("source_depth_rate_mps", 0.0), "source_depth_rate_mps", -20.0, 20.0
        )
        receiver_depth_rate_mps = _finite_float(
            payload.get("receiver_depth_rate_mps", 0.0), "receiver_depth_rate_mps", -20.0, 20.0
        )
        default_bearing = self.field.section_theta_deg if self.field is not None else 0.0
        bearing_deg = _wrap_bearing(_finite_float(
            payload.get("bearing_deg", default_bearing), "bearing_deg", -360.0, 360.0
        ))
        selected_element = _finite_int(
            payload.get("selected_element", 0), "selected_element", 0, config.n_elem - 1
        )

        event_time = time_s % 5.0
        event_envelope = max(
            math.exp(-0.5 * ((event_time - 1.25) / 0.28) ** 2),
            math.exp(-0.5 * ((event_time - 3.35) / 0.36) ** 2),
        )
        event = event_envelope > 0.40
        propagation: dict[str, Any]
        field_available = self.field is not None
        if self.field_timeline is not None:
            channel_sample = self.field_timeline.channel_sample(
                time_s, bearing_deg, range_km, receiver_depth_m
            )
        elif self.field is not None:
            channel_sample = self.field.channel_sample(range_km, receiver_depth_m)
        else:
            channel_sample = None
        if channel_sample is not None:
            tl_db = channel_sample["transmission_loss_db"]
            bottom_depth_m = channel_sample["bottom_depth_m"]
            arrivals = channel_sample["arrivals"]
            source_level_db = _finite_float(
                payload.get("source_level_db", 190.0), "source_level_db", 80.0, 260.0
            )
            ambient_noise_db = _finite_float(
                payload.get("ambient_noise_db", 78.0), "ambient_noise_db", 20.0, 180.0
            )
            snr = source_level_db - tl_db - ambient_noise_db
            snr += config.sensitivity + 5.0 * event_envelope + 0.8 * math.sin(time_s * 2.4)
            propagation = {
                "mode": (
                    "bellhop_time_bearing_interpolated"
                    if self.field_timeline is not None else "bellhop"
                ),
                "bellhop_job_id": self.field.job_id if self.field is not None else None,
                "tl_db": round(float(tl_db), 3),
                "field_frequency_hz": _json_values(self.field.frequency_hz, 3) if self.field is not None else None,
                "signal_frequency_hz": round(config.f0, 3),
                "frequency_matched": bool(
                    self.field is not None
                    and math.isfinite(self.field.frequency_hz)
                    and abs(config.f0 - self.field.frequency_hz) <= max(1.0, self.field.frequency_hz * 0.01)
                ),
                "bearing_deg": round(bearing_deg, 6),
                "bottom_depth_m": round(float(bottom_depth_m), 3),
                "receiver_depth_m": round(receiver_depth_m, 3),
                "multipath_count": len(arrivals),
                "arrivals": arrivals,
            }
            if self.field_timeline is not None:
                propagation["snapshot_interpolation"] = channel_sample["interpolation"]
                snapshot_source = self.field_timeline.source_position(time_s)
                propagation["snapshot_source"] = {
                    "lon": round(snapshot_source["lon"], 8),
                    "lat": round(snapshot_source["lat"], 8),
                    "depth_m": round(snapshot_source["depth_m"], 3),
                }
                if "source_lon" in payload and "source_lat" in payload:
                    source_error_m, _ = _distance_and_bearing(
                        snapshot_source["lon"], snapshot_source["lat"],
                        float(payload["source_lon"]), float(payload["source_lat"]),
                    )
                    depth_error_m = abs(snapshot_source["depth_m"] - source_depth_m)
                    propagation["snapshot_source_error_m"] = round(
                        math.hypot(source_error_m, depth_error_m), 3
                    )
        else:
            range_penalty = max(0.0, range_km - 35.0) * 0.045
            snr = 4.8 + 11.5 * event_envelope + 1.8 * math.sin(time_s * 2.4)
            snr += config.sensitivity - range_penalty
            bottom_depth_m = _finite_float(
                payload.get("bottom_depth_m", max(1000.0, receiver_depth_m + 500.0)),
                "bottom_depth_m", max(receiver_depth_m, source_depth_m) + 1.0, 12000.0,
            )
            channel_sample = _analytic_channel_sample(
                range_km,
                source_depth_m,
                receiver_depth_m,
                bottom_depth_m,
                config.c,
            )
            tl_db = float(channel_sample["transmission_loss_db"])
            arrivals = channel_sample["arrivals"]
            propagation = {
                "mode": "analytic_fallback",
                "tl_db": round(tl_db, 3),
                "bottom_depth_m": round(bottom_depth_m, 3),
                "source_depth_m": round(source_depth_m, 3),
                "receiver_depth_m": round(receiver_depth_m, 3),
                "multipath_count": len(arrivals),
                "arrivals": arrivals,
            }
        detected = snr >= config.threshold

        sample_rate = max(config.sample_rate, config.f0 * 4.0)
        count = FRAME_SAMPLE_COUNT
        sample_time = np.arange(count, dtype=np.float64) / sample_rate
        # range_rate > 0 表示距离增大，因此接收频率降低。
        doppler_hz = -config.f0 * radial_speed_mps / config.c
        tone_frequency = config.f0 + doppler_hz
        if not field_available and event:
            # 解析回退仍可叠加短时事件，但窄带载频只由物理多普勒决定。
            pass
        tone_frequency = min(tone_frequency, sample_rate * 0.44)
        tone_frequency = max(tone_frequency, 1.0)
        def enrich_path(
            arrival: Mapping[str, Any],
            element_source_radial_mps: float,
            element_receiver_radial_mps: float,
        ) -> dict[str, Any]:
            motion = _path_motion(
                arrival,
                source_radial_speed_mps=element_source_radial_mps,
                receiver_radial_speed_mps=element_receiver_radial_mps,
                source_depth_rate_mps=source_depth_rate_mps,
                receiver_depth_rate_mps=receiver_depth_rate_mps,
                sound_speed_mps=config.c,
                source_frequency_hz=config.f0,
            )
            motion["frequency_hz"] = min(max(motion["frequency_hz"], 1.0), sample_rate * 0.44)
            return {
                **arrival,
                "source_angle_deg": round(motion["source_angle_deg"], 6),
                "arrival_angle_deg": round(motion["arrival_angle_deg"], 6),
                "source_velocity_projection_mps": round(motion["source_velocity_projection_mps"], 6),
                "receiver_velocity_projection_mps": round(motion["receiver_velocity_projection_mps"], 6),
                "path_length_rate_mps": round(motion["path_length_rate_mps"], 6),
                "stretch_factor": round(motion["stretch_factor"], 12),
                "doppler_hz": round(motion["doppler_hz"], 6),
                "frequency_hz": round(motion["frequency_hz"], 6),
            }

        arrivals = [
            enrich_path(arrival, source_radial_speed_mps, receiver_radial_speed_mps)
            for arrival in arrivals
        ]
        center_path_motion_rows = [
            {
                key: enriched[key] for key in (
                    "ray_index", "source_angle_deg", "arrival_angle_deg",
                    "source_velocity_projection_mps", "receiver_velocity_projection_mps",
                    "path_length_rate_mps", "stretch_factor", "doppler_hz", "frequency_hz",
                )
            }
            for enriched in arrivals
        ]
        propagation["arrivals"] = arrivals
        center_path_frequency_span_hz = [
            round(min(float(path["frequency_hz"]) for path in arrivals), 6),
            round(max(float(path["frequency_hz"]) for path in arrivals), 6),
        ]
        propagation["received_frequency_span_hz"] = center_path_frequency_span_hz
        if field_available:
            propagation["signal_frequency_hz"] = round(float(tone_frequency), 3)
            propagation["frequency_matched"] = bool(
                math.isfinite(self.field.frequency_hz)
                and abs(tone_frequency - self.field.frequency_hz)
                <= max(1.0, self.field.frequency_hz * 0.01)
            )

        positions = (np.arange(config.n_elem, dtype=np.float64) - (config.n_elem - 1) / 2.0) * config.d
        normalized_position = positions / max(config.array_length / 2.0, config.d)
        cross_track = array_curvature * config.array_length * np.cos(normalized_position * np.pi / 2.0)
        bearing_radians = math.radians(az_true)
        along_range_m = positions * math.sin(bearing_radians) + cross_track * math.cos(bearing_radians)
        cross_range_m = positions * math.cos(bearing_radians) - cross_track * math.sin(bearing_radians)
        center_range_m = range_km * 1000.0
        element_forward_m = center_range_m + along_range_m
        element_ranges_m = np.hypot(element_forward_m, cross_range_m)
        element_ranges_km = np.maximum(0.001, element_ranges_m / 1000.0)
        element_bearings_deg = np.asarray([
            _wrap_bearing(
                bearing_deg + math.degrees(math.atan2(float(cross), float(forward)))
            )
            for cross, forward in zip(cross_range_m, element_forward_m)
        ], dtype=np.float64)

        reference_power = 10.0 ** (-float(tl_db) / 10.0)
        reference_amplitude = math.sqrt(max(reference_power, 1.0e-300))
        carrier = np.zeros((config.n_elem, count), dtype=np.float64)
        event_carrier = np.zeros((config.n_elem, count), dtype=np.float64)
        element_channels: list[dict[str, Any]] = []
        element_tl_db: list[float] = []
        all_path_frequencies: list[float] = []
        all_path_rates: list[float] = []
        all_element_delays: list[float] = []

        source_lon_for_elements = payload.get("source_lon")
        source_lat_for_elements = payload.get("source_lat")
        if source_lon_for_elements is None and self.field is not None:
            source_lon_for_elements = self.field.source_lon
            source_lat_for_elements = self.field.source_lat

        for element_index, (
            offset_m,
            bend_m,
            element_range_km,
            element_bearing_deg,
        ) in enumerate(zip(positions, cross_track, element_ranges_km, element_bearings_deg)):
            try:
                if self.field_timeline is not None:
                    element_sample = self.field_timeline.channel_sample(
                        time_s,
                        float(element_bearing_deg),
                        float(element_range_km),
                        receiver_depth_m,
                    )
                elif self.field is not None:
                    element_sample = self.field.channel_sample(
                        float(element_range_km), receiver_depth_m
                    )
                else:
                    element_sample = _analytic_channel_sample(
                        float(element_range_km),
                        source_depth_m,
                        receiver_depth_m,
                        float(bottom_depth_m),
                        config.c,
                    )
            except ValueError as exc:
                raise ValueError(f"阵元 {element_index}: {exc}") from exc

            if has_horizontal_velocity_components:
                element_los = math.radians(float(element_bearing_deg))
                element_source_radial_mps = (
                    source_east_mps * math.sin(element_los)
                    + source_north_mps * math.cos(element_los)
                )
                element_receiver_radial_mps = (
                    receiver_east_mps * math.sin(element_los)
                    + receiver_north_mps * math.cos(element_los)
                )
            else:
                element_source_radial_mps = source_radial_speed_mps
                element_receiver_radial_mps = receiver_radial_speed_mps

            element_arrivals = [
                enrich_path(
                    path,
                    element_source_radial_mps,
                    element_receiver_radial_mps,
                )
                for path in element_sample["arrivals"]
            ]
            if not element_arrivals:
                raise ValueError(f"阵元 {element_index}: Bellhop 未提供有效到达结构")
            element_tl = float(element_sample["transmission_loss_db"])
            element_tl_db.append(element_tl)
            relative_amplitudes = np.asarray([
                10.0 ** (float(path["relative_gain_db"]) / 20.0)
                for path in element_arrivals
            ], dtype=np.float64)
            element_power = 10.0 ** (-element_tl / 10.0)
            cir_scale = math.sqrt(
                element_power
                / max(float(np.sum(relative_amplitudes ** 2)), 1.0e-300)
            )
            waveform_scale = math.sqrt(2.0) / reference_amplitude
            first_element_delay_s = min(float(path["delay_s"]) for path in element_arrivals)
            element_taps = []
            for path, relative_amplitude in zip(element_arrivals, relative_amplitudes):
                delay_s = float(path["delay_s"])
                path_frequency_hz = float(path["frequency_hz"])
                path_doppler_hz = float(path["doppler_hz"])
                absolute_amplitude = float(relative_amplitude) * cir_scale
                carrier[element_index] += waveform_scale * absolute_amplitude * np.cos(
                    2.0 * np.pi * path_frequency_hz
                    * (time_s + sample_time - delay_s)
                )
                if event:
                    chirp_frequency = min(path_frequency_hz * 1.45, sample_rate * 0.45)
                    event_carrier[element_index] += waveform_scale * absolute_amplitude * np.cos(
                        2.0 * np.pi * chirp_frequency
                        * (time_s + sample_time - delay_s)
                    )
                phase_rad = (
                    -2.0 * math.pi * path_frequency_hz * delay_s
                    + 2.0 * math.pi * path_doppler_hz * time_s
                )
                gain = absolute_amplitude * complex(
                    math.cos(phase_rad), math.sin(phase_rad)
                )
                all_path_frequencies.append(path_frequency_hz)
                all_path_rates.append(float(path["path_length_rate_mps"]))
                all_element_delays.append(delay_s)
                element_taps.append({
                    "tap_index": int(round((delay_s - first_element_delay_s) * sample_rate)),
                    "delay_s": round(delay_s, 9),
                    "excess_delay_ms": round(
                        (delay_s - first_element_delay_s) * 1000.0, 6
                    ),
                    "gain_real": round(gain.real, 12),
                    "gain_imag": round(gain.imag, 12),
                    "magnitude": round(abs(gain), 12),
                    "phase_deg": round(math.degrees(math.atan2(gain.imag, gain.real)), 6),
                    "ray_index": path.get("ray_index"),
                    "source_angle_deg": path["source_angle_deg"],
                    "arrival_angle_deg": path["arrival_angle_deg"],
                    "path_length_m": path.get("path_length_m"),
                    "source_velocity_projection_mps": path["source_velocity_projection_mps"],
                    "receiver_velocity_projection_mps": path["receiver_velocity_projection_mps"],
                    "path_length_rate_mps": path["path_length_rate_mps"],
                    "stretch_factor": path["stretch_factor"],
                    "doppler_hz": path["doppler_hz"],
                    "frequency_hz": path["frequency_hz"],
                })

            element_lon = element_lat = math.nan
            if source_lon_for_elements is not None and source_lat_for_elements is not None:
                source_lon_value = float(source_lon_for_elements)
                source_lat_value = float(source_lat_for_elements)
                if math.isfinite(source_lon_value) and math.isfinite(source_lat_value):
                    element_lon, element_lat = _destination(
                        source_lon_value,
                        source_lat_value,
                        float(element_bearing_deg),
                        float(element_range_km) * 1000.0,
                    )
            element_channel = {
                "element_index": element_index,
                "offset_m": round(float(offset_m), 6),
                "cross_track_m": round(float(bend_m), 6),
                "lon": _json_values(element_lon, 8),
                "lat": _json_values(element_lat, 8),
                "depth_m": round(receiver_depth_m, 3),
                "range_km": round(float(element_range_km), 9),
                "bearing_deg": round(float(element_bearing_deg), 9),
                "bottom_depth_m": round(float(element_sample["bottom_depth_m"]), 3),
                "transmission_loss_db": round(element_tl, 6),
                "path_count": len(element_arrivals),
                "endpoint_velocity": {
                    "source_radial_mps": round(element_source_radial_mps, 6),
                    "receiver_radial_mps": round(element_receiver_radial_mps, 6),
                    "source_depth_rate_mps": round(source_depth_rate_mps, 6),
                    "receiver_depth_rate_mps": round(receiver_depth_rate_mps, 6),
                },
                "arrivals": element_arrivals,
                "impulse_response": {
                    "representation": "complex_sparse_baseband",
                    "sample_rate_hz": round(float(sample_rate), 3),
                    "reference_delay_s": round(first_element_delay_s, 9),
                    "taps": element_taps,
                },
            }
            if "interpolation" in element_sample:
                element_channel["snapshot_interpolation"] = element_sample["interpolation"]
            element_channels.append(element_channel)

        global_first_delay_s = min(all_element_delays)
        global_last_delay_s = max(all_element_delays)
        for element_channel in element_channels:
            element_channel["impulse_response"]["global_reference_delay_s"] = round(
                global_first_delay_s, 9
            )
            for tap in element_channel["impulse_response"]["taps"]:
                global_excess_delay_s = float(tap["delay_s"]) - global_first_delay_s
                tap["global_excess_delay_s"] = round(global_excess_delay_s, 9)
                tap["global_tap_index"] = int(round(global_excess_delay_s * sample_rate))

        # 以确定性种子生成实测时空相关噪声，便于复现实验和自动化测试。
        seed = (frame * 2654435761 + int(time_s * 1000.0) + config.n_elem * 17) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        noise, noise_metadata = _generate_measured_noise(
            count, sample_rate, config.n_elem, rng
        )
        noise = _unit_rms_per_channel(noise)
        signal_rms = 10.0 ** (snr / 20.0)
        samples = noise + signal_rms * carrier

        if event:
            envelope = np.exp(-0.5 * ((sample_time - sample_time[-1] * 0.55) / 0.012) ** 2)
            samples += event_envelope * signal_rms * envelope[None, :] * event_carrier

        propagation["array_sampling"] = {
            "model": "per_element_bellhop_arrivals" if field_available else "per_element_analytic_arrivals",
            "element_count": config.n_elem,
            "bearing_per_element": self.field_timeline is not None,
            "delay_phase_source": "absolute per-element path delay",
        }

        demodulator = np.exp(-2j * np.pi * tone_frequency * sample_time)
        snapshot = (2.0 / count) * (samples @ demodulator)
        scan_angles = np.linspace(-config.az_scan, config.az_scan, BEAM_POINT_COUNT)
        steering = np.exp(
            -2j
            * np.pi
            * tone_frequency
            / config.c
            * (
                np.sin(np.radians(scan_angles))[:, None] * positions[None, :]
                + np.cos(np.radians(scan_angles))[:, None] * cross_track[None, :]
            )
        )
        beam_amplitude = np.abs(steering.conj() @ snapshot) / config.n_elem
        beam_db = 20.0 * np.log10(np.maximum(beam_amplitude, 1.0e-12))
        beam_db -= float(np.max(beam_db))
        beam_db = np.clip(beam_db, -42.0, 0.0)
        peak_index = int(np.argmax(beam_db))
        az_peak = float(scan_angles[peak_index])

        # 对峰值方向实施宽带延时求和，得到独立的波束输出时域信号。
        sample_fft = np.fft.rfft(samples, axis=1)
        all_frequencies = np.fft.rfftfreq(count, 1.0 / sample_rate)
        peak_angle_rad = math.radians(az_peak)
        peak_distance = positions * math.sin(peak_angle_rad) + cross_track * math.cos(peak_angle_rad)
        correction = np.exp(
            2j * np.pi * all_frequencies[None, :] * peak_distance[:, None] / config.c
        )
        beam_output = np.fft.irfft(np.mean(sample_fft * correction, axis=0), n=count)

        window = np.hanning(count)
        fft = np.fft.rfft(samples[: min(config.n_elem, 32)] * window[None, :], axis=1)
        spectrum = np.sqrt(np.mean(np.abs(fft) ** 2, axis=0))
        frequencies = np.fft.rfftfreq(count, 1.0 / sample_rate)
        useful = (frequencies >= 100.0) & (frequencies <= min(3000.0, sample_rate * 0.48))
        useful_indexes = np.flatnonzero(useful)
        if useful_indexes.size > SPECTRUM_POINT_COUNT:
            useful_indexes = useful_indexes[
                np.linspace(0, useful_indexes.size - 1, SPECTRUM_POINT_COUNT).round().astype(int)
            ]
        spectrum_db = 20.0 * np.log10(np.maximum(spectrum[useful_indexes], 1.0e-12))
        spectrum_db -= float(np.max(spectrum_db))
        spectrum_db = np.clip(spectrum_db - 30.0, -82.0, -30.0)

        selected_waveform = samples[selected_element]
        selected_spectrum = np.abs(np.fft.rfft(selected_waveform * window))
        selected_spectrum_db = 20.0 * np.log10(
            np.maximum(selected_spectrum[useful_indexes], 1.0e-12)
        )
        selected_spectrum_db -= float(np.max(selected_spectrum_db))
        selected_spectrum_db = np.clip(selected_spectrum_db - 30.0, -82.0, -30.0)
        waveform_indexes = np.linspace(0, count - 1, min(128, count)).round().astype(int)
        waveform_time_ms = 1000.0 * sample_time[waveform_indexes]

        selected_channel = element_channels[selected_element]
        selected_arrivals = selected_channel["arrivals"]
        selected_path_motion_rows = [{
            key: path[key] for key in (
                "ray_index", "source_angle_deg", "arrival_angle_deg",
                "source_velocity_projection_mps", "receiver_velocity_projection_mps",
                "path_length_rate_mps", "stretch_factor", "doppler_hz", "frequency_hz",
            )
        } for path in selected_arrivals]
        path_frequency_span_hz = [
            round(min(all_path_frequencies), 6),
            round(max(all_path_frequencies), 6),
        ]

        signal_domains = {
            "selected_element": selected_element,
            "time_ms": np.round(waveform_time_ms, 6).tolist(),
            "waveform": np.round(selected_waveform[waveform_indexes], 6).tolist(),
            "frequencies_hz": np.round(frequencies[useful_indexes], 3).tolist(),
            "spectrum_db": np.round(selected_spectrum_db, 3).tolist(),
        }
        channel = {
            "time_varying": any(abs(value) > 1.0e-9 for value in all_path_rates),
            "motion_model": "virtex_first_order_per_eigenray",
            "motion_scope": "per_element_per_eigenray",
            "arrival_model": (
                "per_element_bellhop" if field_available else "per_element_analytic_fallback"
            ),
            "radial_speed_mps": round(radial_speed_mps, 6),
            "doppler_hz": round(doppler_hz, 6),
            "path_frequency_span_hz": path_frequency_span_hz,
            "endpoint_velocity": {
                "source_radial_mps": round(source_radial_speed_mps, 6),
                "receiver_radial_mps": round(receiver_radial_speed_mps, 6),
                "source_depth_rate_mps": round(source_depth_rate_mps, 6),
                "receiver_depth_rate_mps": round(receiver_depth_rate_mps, 6),
            },
            "center_path_doppler": center_path_motion_rows,
            "path_doppler": selected_path_motion_rows,
            "selected_element": selected_element,
            "per_element_tl_db": np.round(element_tl_db, 6).tolist(),
            "global_first_arrival_s": round(global_first_delay_s, 9),
            "global_last_arrival_s": round(global_last_delay_s, 9),
            "impulse_response": selected_channel["impulse_response"],
            "elements": element_channels,
        }
        beam = {
            "definition": "received array signal after wideband delay-and-sum beamforming",
            "scan_angles_deg": np.round(scan_angles, 3).tolist(),
            "spatial_spectrum_db": np.round(beam_db, 3).tolist(),
            "peak_angle_deg": round(az_peak, 3),
            "output_time_ms": np.round(waveform_time_ms, 6).tolist(),
            "output_waveform": np.round(beam_output[waveform_indexes], 6).tolist(),
        }

        return {
            "type": "frame",
            "frame": frame,
            "t": round(time_s, 4),
            "az_true": round(az_true, 3),
            "az_peak": round(az_peak, 3),
            "el": round(elevation, 3),
            "range_km": round(range_km, 4),
            "bearing_deg": round(bearing_deg, 6),
            "array_heading": round(array_heading, 3),
            "array_curvature": round(array_curvature, 5),
            "snr": round(float(snr), 3),
            "detected": bool(detected),
            "event": bool(event),
            "curve": np.round(beam_db, 3).tolist(),
            "scan_angles_deg": np.round(scan_angles, 3).tolist(),
            "frequencies_hz": np.round(frequencies[useful_indexes], 3).tolist(),
            "spectrum_db": np.round(spectrum_db, 3).tolist(),
            "sample_rate_hz": round(float(sample_rate), 3),
            "tone_frequency_hz": round(float(tone_frequency), 3),
            "radial_speed_mps": round(radial_speed_mps, 6),
            "doppler_hz": round(doppler_hz, 6),
            "selected_element": selected_element,
            "signal": signal_domains,
            "channel": channel,
            "beam": beam,
            "propagation": propagation,
            "noise": noise_metadata,
        }


async def serve_array_fusion_websocket(
    websocket: Any,
    jobs_dir: str | Path | None = None,
    profile_provider: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
    authorize_job: Callable[[str], bool] | None = None,
) -> None:
    """接受一个 FastAPI WebSocket，直到客户端断开为止。"""

    await websocket.accept()
    session = ArrayFusionSession(jobs_dir=jobs_dir)
    try:
        await websocket.send_text(json.dumps(session.config.acknowledgement(), ensure_ascii=False))
        while True:
            raw_message = await websocket.receive_text()
            try:
                payload = json.loads(raw_message)
                if not isinstance(payload, dict):
                    raise ValueError("消息必须是 JSON 对象")
                message_type = str(payload.get("type") or "frame").lower()
                if authorize_job is not None:
                    referenced: list[str] = []
                    for key in ("bellhop_job_id", "job_id", "ray_job_id", "arrival_job_id"):
                        value = payload.get(key)
                        if value:
                            referenced.append(str(value))
                    snapshots = payload.get("snapshots")
                    if isinstance(snapshots, list):
                        for item in snapshots:
                            if isinstance(item, Mapping):
                                value = item.get("bellhop_job_id") or item.get("job_id")
                                if value:
                                    referenced.append(str(value))
                    if any(not authorize_job(job) for job in dict.fromkeys(referenced)):
                        raise ValueError("无权访问指定的声场任务。")
                if message_type == "config":
                    response = session.configure(payload)
                elif message_type in ("motion_config", "motion_start"):
                    response = session.configure_motion(payload)
                elif message_type in ("bellhop_series_bind", "field_series_bind", "snapshot_series_bind"):
                    response = await asyncio.to_thread(session.bind_bellhop_series, payload)
                elif message_type in ("field", "field_req", "bellhop_bind"):
                    job_id = payload.get("bellhop_job_id") or payload.get("job_id")
                    if job_id:
                        response = await asyncio.to_thread(
                            session.bind_bellhop,
                            str(job_id),
                            str(payload.get("ray_job_id") or "").strip() or None,
                            str(payload.get("arrival_job_id") or "").strip() or None,
                        )
                    elif session.field_timeline is not None:
                        response = session.field_timeline.acknowledgement()
                    elif session.field is not None:
                        response = session.field.field_response()
                    else:
                        raise ValueError("field_req 需要 bellhop_job_id")
                elif message_type in ("profile", "profile_req"):
                    requested_job = payload.get("bellhop_job_id") or payload.get("job_id")
                    if requested_job or session.field is not None:
                        response = await asyncio.to_thread(
                            session.sound_speed_profile,
                            {**payload, "bellhop_job_id": requested_job or ""},
                        )
                    elif profile_provider is not None:
                        try:
                            response = await asyncio.to_thread(profile_provider, payload)
                        except Exception as exc:
                            response = session.sound_speed_profile(payload)
                            response["fallback_reason"] = str(exc)
                    else:
                        response = session.sound_speed_profile(payload)
                elif message_type in ("static_channel", "static_channel_req"):
                    requested_job = payload.get("bellhop_job_id") or payload.get("job_id")
                    requested_ray_job = str(payload.get("ray_job_id") or "").strip() or None
                    requested_arrival_job = (
                        str(payload.get("arrival_job_id") or "").strip() or None
                    )
                    if requested_job and (
                        session.field_timeline is not None
                        or session.field is None
                        or session.field.job_id != str(requested_job)
                        or str(session.field.ray_job_id or session.field.job_id)
                        != str(requested_ray_job or requested_job)
                        or session.field.arrival_job_id != requested_arrival_job
                    ):
                        await asyncio.to_thread(
                            session.bind_bellhop,
                            str(requested_job),
                            requested_ray_job,
                            requested_arrival_job,
                        )
                    response = await asyncio.to_thread(session.static_channel, payload)
                elif message_type == "ping":
                    response = {"type": "pong", "t": payload.get("t")}
                elif message_type in ("motion_pause", "motion_seek"):
                    response = {
                        "type": "motion_state_ack",
                        "state": "paused" if message_type == "motion_pause" else "seeked",
                        "t": payload.get("t", 0.0),
                    }
                else:
                    requested_job = payload.get("bellhop_job_id") or payload.get("job_id")
                    if requested_job and (
                        session.field_timeline is not None
                        or session.field is None or session.field.job_id != str(requested_job)
                    ):
                        await asyncio.to_thread(session.bind_bellhop, str(requested_job))
                    # 单帧矩阵规模受配置约束，128 阵元的典型计算约为几十毫秒。
                    # 直接在当前会话中完成可避免默认线程池退出异常，并保持帧严格有序。
                    response = (
                        session.process_motion_frame(payload)
                        if message_type == "motion_frame"
                        else session.process_frame(payload)
                    )
            except (TypeError, ValueError, json.JSONDecodeError, OSError, RuntimeError) as exc:
                response = {"type": "error", "message": str(exc)}
            await websocket.send_text(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
    except (WebSocketDisconnect, BrokenPipeError, ConnectionResetError, OSError):
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
