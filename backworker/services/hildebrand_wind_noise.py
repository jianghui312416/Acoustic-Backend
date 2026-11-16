"""Hildebrand 等（2021）的风生海洋噪声基线模型。

这是作者公开参考代码 ``OceanNoiseModel.m`` 的 Python 实现。该模型返回
*接收声压谱级*，因此它属于接收点基线模型，不得再次通过 Bellhop/KRAKEN
进行传播计算。
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any

import numpy as np


REFERENCE_WIND_SPEEDS_MPS = np.asarray(
    [1.0, 2.5, 4.5, 6.7, 9.4, 12.3, 15.5, 19.0, 22.6, 26.5, 30.5],
    dtype=np.float64,
)
MODEL_MIN_FREQUENCY_HZ = 10.0
MODEL_MAX_FREQUENCY_HZ = 100000.0
VALIDATED_DEPTH_MAX_M = 1400.0


def _absorption_db_per_km(
    frequency_khz: np.ndarray,
    depth_m: float,
    temperature_c: float = 10.0,
    salinity_psu: float = 35.0,
    ph: float = 8.0,
) -> np.ndarray:
    """采用 Ainslie 和 McColm（1998）方法，与参考 MATLAB 函数一致。"""
    f = np.asarray(frequency_khz, dtype=np.float64)
    f1 = 0.78 * math.sqrt(salinity_psu / 35.0) * math.exp(temperature_c / 26.0)
    f2 = 42.0 * math.exp(temperature_c / 17.0)
    fsq = np.square(f)
    alpha1 = fsq * 0.106 * math.exp((ph - 8.0) / 0.56) * f1 / (fsq + f1 * f1)
    alpha2 = (
        0.52
        * (1.0 + temperature_c / 43.0)
        * (salinity_psu / 35.0)
        * math.exp(-depth_m / 6000.0)
        * fsq
        * f2
        / (fsq + f2 * f2)
    )
    alpha3 = 0.00049 * fsq * math.exp(-(temperature_c / 27.0 + depth_m / 17000.0))
    return alpha1 + alpha2 + alpha3


def _depth_absorption_correction_db(frequency_hz: np.ndarray, depth_m: float) -> np.ndarray:
    """MATLAB 代码使用的角度平均深度/吸收修正。"""
    if depth_m <= 0.0:
        return np.zeros_like(frequency_hz, dtype=np.float64)
    alpha = _absorption_db_per_km(np.asarray(frequency_hz, dtype=np.float64) / 1000.0, depth_m)
    log_eah = -alpha * depth_m / 1000.0 / 10.0 * math.log(10.0)
    theta = np.linspace(0.0, math.pi / 2.0, 91, dtype=np.float64)
    cosine = np.cos(theta)
    sine = np.sin(theta)
    geometric_weight = cosine * sine
    secant = np.divide(1.0, cosine, out=np.full_like(cosine, np.inf), where=cosine > 0.0)
    exponent = log_eah[:, None] * secant[None, :]
    attenuation = np.exp(np.clip(exponent, -745.0, 0.0))
    numerator = np.sum(geometric_weight[None, :] * attenuation, axis=1)
    denominator = float(np.sum(geometric_weight))
    return 10.0 * np.log10(np.maximum(numerator / denominator, np.finfo(np.float64).tiny))


def _reference_grid() -> np.ndarray:
    # OceanNoiseModel.m 在 10—100 Hz 范围内以 10 Hz 为间隔取点，此后以 100 Hz
    # 为间隔取点。对于重复的 100 Hz 点，使用高频段分支；论文图中 100 Hz 及以上
    # 频率使用的也是该分支。
    return np.concatenate((np.arange(10.0, 100.0, 10.0), np.arange(100.0, 100100.0, 100.0)))


@lru_cache(maxsize=32)
def _reference_spectra(depth_m: float) -> tuple[np.ndarray, np.ndarray]:
    """返回指定接收深度下的 11 条 MATLAB 参考风噪声曲线。"""
    frequency = _reference_grid()
    spectra = np.empty((REFERENCE_WIND_SPEEDS_MPS.size, frequency.size), dtype=np.float64)
    offset0 = 50.0 + 2.8 * math.exp(-depth_m / 600.0)

    low = frequency < 100.0
    low_frequency_khz = frequency[low] / 1000.0
    low_offsets = np.asarray([6.4, 1.5, -0.5, 3.2, 1.83, 2.4, 2.18, 2.27, 2.54, 2.72, 2.8])
    low_base_shift = np.asarray([-8.0, -8.0, -8.0] + [-18.0] * 8)
    low_wind_coeff = np.asarray([10.0, 20.0, 20.0, 27.5] + [28.75] * 7)
    for wind_index, wind_speed in enumerate(REFERENCE_WIND_SPEEDS_MPS):
        spectra[wind_index, low] = (
            offset0
            + low_base_shift[wind_index]
            + low_offsets[wind_index]
            + low_wind_coeff[wind_index] * math.log10(float(wind_speed))
            - 4.0 * np.log10(low_frequency_khz)
        )

    high_frequency = frequency[~low]
    high_frequency_khz = high_frequency / 1000.0
    matlab_index = high_frequency / 100.0
    depth_for_m = min(depth_m, 1000.0)
    m_factor = np.zeros_like(high_frequency)
    above_15khz = matlab_index >= 150.0
    m_factor[above_15khz] = (
        (1.0 - depth_for_m / 1000.0)
        * 3.0
        * (150.0 - np.minimum(matlab_index[above_15khz], 400.0))
        / 1600.0
    )

    # 每行对应一条参考风噪声曲线。各列描述 MATLAB 中的四个频段：
    # 100—400 Hz、500—900 Hz、1—4 kHz 和不低于 4.1 kHz。
    segment_offsets = np.asarray([
        [5.4, 0.3, 0.1, 2.0],
        [-0.14, -4.7, -5.0, -3.0],
        [-2.81, -6.8, -7.0, -4.0],
        [-9.82, -10.1, -10.4, -4.25],
        [-11.9, -10.0, -10.3, -4.5],
        [-12.10, -10.12, -10.3, -4.4],
        [-13.02, -10.3, -10.3, -4.25],
        [-13.54, -10.3, -10.3, -3.9],
        [-14.06, -10.3, -10.3, -3.6],
        [-14.58, -10.3, -10.3, -3.15],
        [-15.2, -10.3, -10.3, -2.85],
    ], dtype=np.float64)
    wind_coefficients = np.asarray([
        [10.0, 10.0, 10.0, 10.0],
        [20.0, 20.0, 20.0, 20.0],
        [20.0, 20.0, 20.0, 20.0],
        [27.5, 25.0, 25.0, 20.0],
        [28.75, 25.0, 25.0, 20.0],
        [28.75, 25.0, 25.0, 20.0],
        [28.75, 25.0, 25.0, 20.0],
        [28.75, 25.0, 25.0, 20.0],
        [28.75, 25.0, 25.0, 20.0],
        [28.75, 25.0, 25.0, 20.0],
        [28.75, 25.0, 25.0, 20.0],
    ], dtype=np.float64)
    frequency_coefficients = np.asarray([
        [3.0, -10.5, -10.4, -13.5],
        [2.3, -10.7, -10.2, -13.5],
        [1.6, -10.0, -8.7, -13.5],
        [0.9, -7.0, -10.2, -13.5],
        [0.2, -6.0, -12.5, -14.0],
        [-0.5, -6.0, -13.7, -14.5],
        [-1.2, -5.0, -14.5, -14.5],
        [-1.9, -6.0, -14.5, -14.5],
        [-2.6, -6.0, -14.5, -14.5],
        [-3.3, -6.0, -14.5, -14.5],
        [-4.0, -6.0, -14.5, -14.5],
    ], dtype=np.float64)
    segment = np.select(
        [matlab_index <= 4.0, matlab_index <= 9.0, matlab_index <= 40.0],
        [0, 1, 2],
        default=3,
    ).astype(np.int64)
    log_frequency = np.log10(high_frequency_khz)
    log_wind = np.log10(REFERENCE_WIND_SPEEDS_MPS)

    nfac_a = np.asarray([0.0, 0.0, 0.0, 0.04, 0.08, 0.17, 0.24, 0.6, 1.0, 1.6, 1.8])
    nfac_b = np.asarray([0.0, 0.0, 0.0, 0.3, 0.3, 0.7, 1.2, 1.8, 2.2, 2.5, 2.7])
    nfac_start = np.asarray([np.inf, np.inf, np.inf, 201.0, 201.0, 101.0, 62.0, 41.0, 41.0, 41.0, 41.0])
    for wind_index in range(REFERENCE_WIND_SPEEDS_MPS.size):
        local_wind_coefficient = wind_coefficients[wind_index, segment].copy()
        local_offset = segment_offsets[wind_index, segment].copy()
        nonlinear = matlab_index >= nfac_start[wind_index]
        if np.any(nonlinear):
            nfac = (
                depth_m / 1000.0 + nfac_a[wind_index] * math.exp(-depth_m / 400.0)
            ) * nfac_b[wind_index] * (matlab_index[nonlinear] - nfac_start[wind_index]) / 1600.0
            local_wind_coefficient[nonlinear] = 20.0 * (1.0 - nfac)
        # 公开 MATLAB 代码中的三个分支会在非线性风速阈值处将拟合偏移量
        # 调整数百分之一分贝。
        if wind_index == 3:
            local_offset[matlab_index >= 201.0] = -4.3
        elif wind_index == 5:
            local_offset[matlab_index >= 101.0] = -4.35
        elif wind_index == 6:
            local_offset[matlab_index >= 62.0] = -4.15
        spectra[wind_index, ~low] = (
            offset0
            + local_offset
            + local_wind_coefficient * log_wind[wind_index]
            + frequency_coefficients[wind_index, segment] * (1.0 - m_factor) * log_frequency
        )

    spectra += _depth_absorption_correction_db(frequency, depth_m)[None, :]
    return frequency, spectra


def hildebrand2021_psd_db(
    frequency_hz: Any,
    wind_speed_mps: float,
    receiver_depth_m: float,
) -> np.ndarray:
    """计算接收风噪声 PSD，单位为 dB re 1 µPa²/Hz。

    公开参考实现定义了 11 条风速曲线。曲线间的数值按 log10（风速）进行线性
    插值，这与模型中的对数风速项及作者图件一致。频率则在模型采样点之间按
    对数频率插值。
    """
    frequencies = np.asarray(frequency_hz, dtype=np.float64)
    if frequencies.ndim == 0:
        frequencies = frequencies.reshape(1)
    if np.any(~np.isfinite(frequencies)) or np.any(frequencies < MODEL_MIN_FREQUENCY_HZ) or np.any(
        frequencies > MODEL_MAX_FREQUENCY_HZ
    ):
        raise ValueError(
            f"Hildebrand 2021 模型频率必须位于 {MODEL_MIN_FREQUENCY_HZ:g} 到 "
            f"{MODEL_MAX_FREQUENCY_HZ:g} Hz。"
        )
    wind = float(wind_speed_mps)
    depth = float(receiver_depth_m)
    if not math.isfinite(wind) or wind < 0.0:
        raise ValueError("Hildebrand 2021 模型风速不能为负数。")
    if not math.isfinite(depth) or depth < 0.0:
        raise ValueError("Hildebrand 2021 模型接收深度不能为负数。")

    model_frequency, reference = _reference_spectra(depth)
    clipped_wind = float(np.clip(wind, REFERENCE_WIND_SPEEDS_MPS[0], REFERENCE_WIND_SPEEDS_MPS[-1]))
    log_reference_wind = np.log10(REFERENCE_WIND_SPEEDS_MPS)
    log_wind = math.log10(clipped_wind)
    wind_curve = np.asarray(
        [np.interp(log_wind, log_reference_wind, reference[:, index]) for index in range(reference.shape[1])]
    )
    return np.interp(np.log(frequencies), np.log(model_frequency), wind_curve)


def model_metadata(wind_speed_mps: float, receiver_depth_m: float) -> dict[str, Any]:
    wind = float(wind_speed_mps)
    depth = float(receiver_depth_m)
    return {
        "name": "Hildebrand2021",
        "paper": "An empirical model for wind-generated ocean noise",
        "doi": "10.1121/10.0005430",
        "output": "received_pressure_spectrum_level",
        "units": "dB re 1 uPa^2/Hz",
        "windInterpolation": "linear_level_on_log10_wind_speed",
        "frequencyInterpolation": "linear_level_on_log_frequency",
        "referenceWindSpeedRangeMps": [
            float(REFERENCE_WIND_SPEEDS_MPS[0]),
            float(REFERENCE_WIND_SPEEDS_MPS[-1]),
        ],
        "windSpeedClipped": not REFERENCE_WIND_SPEEDS_MPS[0] <= wind <= REFERENCE_WIND_SPEEDS_MPS[-1],
        "windSpeedUsedMps": float(np.clip(wind, REFERENCE_WIND_SPEEDS_MPS[0], REFERENCE_WIND_SPEEDS_MPS[-1])),
        "validatedDepthMaxM": VALIDATED_DEPTH_MAX_M,
        "depthExtrapolated": depth > VALIDATED_DEPTH_MAX_M,
    }
