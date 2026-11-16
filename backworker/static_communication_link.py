#!/usr/bin/env python3
"""静态信道下可归因的完整数字通信链路仿真。

链路支持 BPSK/QPSK、单载波/OFDM、未编码/Turbo 编码。Turbo 译码器采用
迭代 Log-MAP（BCJR）最大后验概率算法。信道在一次研究中保持完全不变；默认
提供平坦信道和固定多径信道两种配置。

本模块刻意使用成对受控实验，而不是只画若干任意曲线：

* 调制贡献：固定波形与编码，只比较 BPSK/QPSK；
* 波形贡献：固定调制与编码，只比较单载波/OFDM；
* 编码贡献：固定调制与波形，只比较未编码/Turbo；

所有曲线横轴均为包含编码冗余、OFDM 循环前缀和填充开销的信息比特 Eb/N0，
从而避免把额外发射能量误当作编码或 OFDM 增益。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MODEL_VERSION = "static_bpsk_qpsk_ofdm_turbo_logmap_v2"


@dataclass(frozen=True)
class ChannelConfig:
    """固定离散复基带信道；整个 BER 扫描期间抽头不随时间改变。"""

    name: str = "static_multipath"
    taps_real: tuple[float, ...] = (1.0, 0.0, 0.0, 0.48, 0.0, 0.0, 0.27)
    taps_imag: tuple[float, ...] = (0.0, 0.0, 0.0, 0.24, 0.0, 0.0, -0.18)


@dataclass(frozen=True)
class OfdmConfig:
    fft_size: int = 64
    cyclic_prefix_samples: int = 8
    # 0 表示兼容旧配置：除导频外全部 FFT 频点用于数据。
    data_subcarriers: int = 0
    pilot_subcarriers: int = 0


@dataclass(frozen=True)
class FrameConfig:
    """通信帧开销；前导以波形符号数计，保护间隔以采样点计。"""

    preamble_symbols: int = 0
    header_bits: int = 0
    guard_samples: int = 0


@dataclass(frozen=True)
class TurboConfig:
    iterations: int = 6
    interleaver_seed: int = 20261021
    llr_clip: float = 60.0


@dataclass(frozen=True)
class SimulationConfig:
    ebn0_db: tuple[float, ...] = (-2.0, 0.0, 2.0, 4.0, 6.0, 8.0)
    information_bits_per_frame: int = 512
    frames_per_point: int = 16
    random_seed: int = 20261021
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    ofdm: OfdmConfig = field(default_factory=OfdmConfig)
    frame: FrameConfig = field(default_factory=FrameConfig)
    turbo: TurboConfig = field(default_factory=TurboConfig)


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    modulation: str
    waveform: str
    coding: str


DEFAULT_SCENARIOS = (
    Scenario("baseline", "BPSK / 单载波 / 未编码", "BPSK", "single_carrier", "none"),
    Scenario("modulation", "QPSK / 单载波 / 未编码", "QPSK", "single_carrier", "none"),
    Scenario("waveform", "BPSK / OFDM / 未编码", "BPSK", "ofdm", "none"),
    Scenario("coding", "BPSK / 单载波 / Turbo", "BPSK", "single_carrier", "turbo"),
    # 两条补充曲线令三类技术都能在完全相同的其余条件下成对比较。
    Scenario("qpsk_ofdm_uncoded", "QPSK / OFDM / 未编码", "QPSK", "ofdm", "none"),
    Scenario("full", "QPSK / OFDM / Turbo", "QPSK", "ofdm", "turbo"),
)


def scenarios_with_selection(
    modulation: str,
    waveform: str,
    coding: str,
) -> tuple[tuple[Scenario, ...], str]:
    """返回归因基准场景，并保证用户重点方案也在结果中。"""

    modulation = str(modulation).upper()
    waveform = str(waveform).lower()
    coding = str(coding).lower()
    if modulation not in {"BPSK", "QPSK"}:
        raise ValueError("modulation 只能是 BPSK 或 QPSK。")
    if waveform not in {"single_carrier", "ofdm"}:
        raise ValueError("waveform 只能是 single_carrier 或 ofdm。")
    if coding not in {"none", "turbo"}:
        raise ValueError("coding 只能是 none 或 turbo。")
    for scenario in DEFAULT_SCENARIOS:
        if (scenario.modulation, scenario.waveform, scenario.coding) == (
            modulation,
            waveform,
            coding,
        ):
            return DEFAULT_SCENARIOS, scenario.key
    selected = Scenario(
        "selected",
        f"{modulation} / {'OFDM' if waveform == 'ofdm' else '单载波'} / "
        f"{'Turbo' if coding == 'turbo' else '未编码'}",
        modulation,
        waveform,
        coding,
    )
    return (*DEFAULT_SCENARIOS, selected), selected.key


def channel_config_from_array_fusion(
    static_channel: Mapping[str, Any],
    *,
    element_index: int = 0,
    communication_sample_rate_hz: float | None = None,
    maximum_memory_samples: int = 8192,
) -> tuple[ChannelConfig, dict[str, Any]]:
    """把 ``array_fusion_service.static_channel`` 的稀疏 CIR 转成仿真抽头。

    可传完整 ``static_channel_resp``，也可传前端裁剪后的 ``element`` 对象。
    依据真实超额时延在通信采样率上重新量化，而不是沿用显示波形的采样索引。
    """

    if not isinstance(static_channel, Mapping):
        raise ValueError("static_channel 必须是 array_fusion_service 的响应对象。")
    motion = static_channel.get("motion", {})
    if isinstance(motion, Mapping) and motion.get("time_invariant") is False:
        raise ValueError("完整通信静态研究只能使用 time_invariant=true 的静态信道。")
    elements = static_channel.get("elements")
    if isinstance(elements, Sequence) and not isinstance(elements, (str, bytes)):
        if not 0 <= element_index < len(elements):
            raise ValueError("selected_element 超出静态阵元范围。")
        element = elements[element_index]
    else:
        element = static_channel.get("element")
        if isinstance(element, Mapping):
            element_index = int(element.get("element_index", element_index))
    if not isinstance(element, Mapping):
        raise ValueError("静态信道响应缺少可用的 element/elements。")
    impulse = element.get("impulse_response")
    if not isinstance(impulse, Mapping):
        raise ValueError("所选阵元缺少 impulse_response。")
    sparse_taps = impulse.get("taps")
    if not isinstance(sparse_taps, Sequence) or not sparse_taps:
        raise ValueError("所选阵元的 impulse_response.taps 为空。")
    source_rate = float(
        impulse.get("sample_rate_hz")
        or static_channel.get("impulse_response", {}).get("sample_rate_hz")
        or 0.0
    )
    sample_rate = float(communication_sample_rate_hz or source_rate)
    if not math.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("communication_sample_rate_hz 必须为正数。")
    if source_rate > 0.0 and sample_rate > source_rate:
        raise ValueError("通信采样率不能高于静态 CIR 的采样率。")

    indexed: list[tuple[int, complex, float]] = []
    for tap in sparse_taps:
        if not isinstance(tap, Mapping):
            continue
        delay = float(tap.get("excess_delay_s", 0.0))
        if not math.isfinite(delay) or delay < 0.0:
            raise ValueError("静态 CIR 包含无效的超额时延。")
        index = int(round(delay * sample_rate))
        gain = complex(float(tap.get("gain_real", 0.0)), float(tap.get("gain_imag", 0.0)))
        if math.isfinite(gain.real) and math.isfinite(gain.imag):
            indexed.append((index, gain, delay))
    if not indexed:
        raise ValueError("静态 CIR 没有有限复数抽头。")
    memory = max(item[0] for item in indexed)
    if memory > maximum_memory_samples:
        raise ValueError(
            f"通信采样率下信道记忆为 {memory} 点，超过安全上限 {maximum_memory_samples}；"
            "请降低通信采样率或缩短信道时延窗。"
        )
    dense = np.zeros(memory + 1, dtype=np.complex128)
    for index, gain, _delay in indexed:
        dense[index] += gain
    nonzero = np.flatnonzero(np.abs(dense) > 0.0)
    if nonzero.size == 0:
        raise ValueError("静态 CIR 合并后能量为零。")
    dense = dense[: int(nonzero[-1]) + 1]
    job_id = str(static_channel.get("bellhop_job_id") or "unknown")
    metadata = {
        "source": "array_fusion_service.static_channel",
        "bellhop_job_id": job_id,
        "element_index": element_index,
        "transmission_loss_db": element.get("transmission_loss_db"),
        "path_count": len(indexed),
        "source_impulse_sample_rate_hz": source_rate,
        "communication_sample_rate_hz": sample_rate,
        "maximum_excess_delay_s": max(item[2] for item in indexed),
        "modeled_memory_samples": int(dense.size - 1),
        "absolute_path_loss_removed_for_ebn0_sweep": True,
    }
    return (
        ChannelConfig(
            name=f"array_fusion:{job_id}:element_{element_index}",
            taps_real=tuple(float(value) for value in np.real(dense)),
            taps_imag=tuple(float(value) for value in np.imag(dense)),
        ),
        metadata,
    )


def _tuple_float(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} 必须是非空数组。")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} 只能包含有限数值。")
    return result


def load_config(path: Path | None = None) -> SimulationConfig:
    if path is None:
        return SimulationConfig()
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("配置文件根节点必须是 JSON 对象。")
    channel_raw = raw.get("channel", {})
    ofdm_raw = raw.get("ofdm", {})
    frame_raw = raw.get("frame", {})
    turbo_raw = raw.get("turbo", {})
    channel = ChannelConfig(
        name=str(channel_raw.get("name", ChannelConfig.name)),
        taps_real=_tuple_float(channel_raw.get("taps_real", ChannelConfig.taps_real), "channel.taps_real"),
        taps_imag=_tuple_float(channel_raw.get("taps_imag", ChannelConfig.taps_imag), "channel.taps_imag"),
    )
    config = SimulationConfig(
        ebn0_db=_tuple_float(raw.get("ebn0_db", SimulationConfig.ebn0_db), "ebn0_db"),
        information_bits_per_frame=int(raw.get("information_bits_per_frame", SimulationConfig.information_bits_per_frame)),
        frames_per_point=int(raw.get("frames_per_point", SimulationConfig.frames_per_point)),
        random_seed=int(raw.get("random_seed", SimulationConfig.random_seed)),
        channel=channel,
        ofdm=OfdmConfig(
            fft_size=int(ofdm_raw.get("fft_size", OfdmConfig.fft_size)),
            cyclic_prefix_samples=int(ofdm_raw.get("cyclic_prefix_samples", OfdmConfig.cyclic_prefix_samples)),
            data_subcarriers=int(ofdm_raw.get("data_subcarriers", OfdmConfig.data_subcarriers)),
            pilot_subcarriers=int(ofdm_raw.get("pilot_subcarriers", OfdmConfig.pilot_subcarriers)),
        ),
        frame=FrameConfig(
            preamble_symbols=int(frame_raw.get("preamble_symbols", FrameConfig.preamble_symbols)),
            header_bits=int(frame_raw.get("header_bits", FrameConfig.header_bits)),
            guard_samples=int(frame_raw.get("guard_samples", FrameConfig.guard_samples)),
        ),
        turbo=TurboConfig(
            iterations=int(turbo_raw.get("iterations", TurboConfig.iterations)),
            interleaver_seed=int(turbo_raw.get("interleaver_seed", TurboConfig.interleaver_seed)),
            llr_clip=float(turbo_raw.get("llr_clip", TurboConfig.llr_clip)),
        ),
    )
    validate_config(config)
    return config


def validate_config(config: SimulationConfig) -> None:
    if config.information_bits_per_frame < 16:
        raise ValueError("information_bits_per_frame 至少为 16。")
    if config.frames_per_point < 1:
        raise ValueError("frames_per_point 至少为 1。")
    if len(config.channel.taps_real) != len(config.channel.taps_imag):
        raise ValueError("channel.taps_real 与 channel.taps_imag 长度必须相同。")
    if config.ofdm.fft_size < 8:
        raise ValueError("ofdm.fft_size 至少为 8。")
    if not 0 <= config.ofdm.cyclic_prefix_samples < config.ofdm.fft_size:
        raise ValueError("OFDM 循环前缀必须满足 0 <= CP < FFT 点数。")
    if config.ofdm.data_subcarriers < 0 or config.ofdm.pilot_subcarriers < 0:
        raise ValueError("OFDM 数据与导频子载波数不能为负数。")
    data_subcarriers = config.ofdm.data_subcarriers or (
        config.ofdm.fft_size - config.ofdm.pilot_subcarriers
    )
    if data_subcarriers < 1:
        raise ValueError("OFDM 至少需要 1 个数据子载波。")
    if data_subcarriers + config.ofdm.pilot_subcarriers > config.ofdm.fft_size:
        raise ValueError("OFDM 数据与导频子载波总数不能超过 FFT 点数。")
    if config.frame.preamble_symbols < 0:
        raise ValueError("frame.preamble_symbols 不能为负数。")
    if config.frame.header_bits < 0:
        raise ValueError("frame.header_bits 不能为负数。")
    if config.frame.guard_samples < 0:
        raise ValueError("frame.guard_samples 不能为负数。")
    if config.turbo.iterations < 1:
        raise ValueError("Turbo 迭代次数至少为 1。")
    if config.turbo.llr_clip <= 0:
        raise ValueError("Turbo LLR 截断值必须为正数。")
    taps = channel_taps(config.channel)
    if not np.any(np.abs(taps) > 0):
        raise ValueError("静态信道抽头不能全为零。")


def channel_taps(config: ChannelConfig) -> np.ndarray:
    taps = np.asarray(config.taps_real, dtype=float) + 1j * np.asarray(config.taps_imag, dtype=float)
    energy = float(np.sum(np.abs(taps) ** 2))
    if not math.isfinite(energy) or energy <= 0.0:
        raise ValueError("静态信道能量必须为正数。")
    return taps.astype(np.complex128) / math.sqrt(energy)


def _rsc_trellis() -> tuple[np.ndarray, np.ndarray]:
    """约束长度 3、反馈多项式 7、前馈多项式 5 的 RSC 网格。"""

    next_state = np.zeros((4, 2), dtype=np.int8)
    parity = np.zeros((4, 2), dtype=np.uint8)
    for state in range(4):
        memory_0 = (state >> 1) & 1
        memory_1 = state & 1
        for bit in (0, 1):
            feedback = bit ^ memory_0 ^ memory_1
            parity[state, bit] = feedback ^ memory_1
            next_state[state, bit] = (feedback << 1) | memory_0
    return next_state, parity


RSC_NEXT_STATE, RSC_PARITY = _rsc_trellis()


def rsc_encode(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    parity = np.empty(bits.size, dtype=np.uint8)
    state = 0
    for index, bit in enumerate(bits):
        value = int(bit)
        parity[index] = RSC_PARITY[state, value]
        state = int(RSC_NEXT_STATE[state, value])
    return parity


def make_interleaver(length: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(length)


def turbo_encode(bits: np.ndarray, interleaver: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if interleaver.shape != (bits.size,):
        raise ValueError("Turbo 交织器长度必须等于信息比特数。")
    parity_1 = rsc_encode(bits)
    parity_2 = rsc_encode(bits[interleaver])
    return np.concatenate((bits, parity_1, parity_2))


def _log_map_siso(
    systematic_llr: np.ndarray,
    parity_llr: np.ndarray,
    apriori_llr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """RSC 分量码的精确 Log-MAP/BCJR 软输入软输出译码。"""

    systematic_llr = np.asarray(systematic_llr, dtype=float)
    parity_llr = np.asarray(parity_llr, dtype=float)
    apriori_llr = np.asarray(apriori_llr, dtype=float)
    length = systematic_llr.size
    alpha = np.full((length + 1, 4), -np.inf, dtype=float)
    beta = np.full((length + 1, 4), -np.inf, dtype=float)
    alpha[0, 0] = 0.0
    # 未加尾比特，终态未知，因此所有终态等概率。
    beta[length, :] = 0.0

    gamma = np.empty((length, 4, 2), dtype=float)
    for time in range(length):
        for state in range(4):
            for bit in (0, 1):
                parity = int(RSC_PARITY[state, bit])
                gamma[time, state, bit] = 0.5 * (
                    (1.0 - 2.0 * bit) * (systematic_llr[time] + apriori_llr[time])
                    + (1.0 - 2.0 * parity) * parity_llr[time]
                )

    for time in range(length):
        for state in range(4):
            if not np.isfinite(alpha[time, state]):
                continue
            for bit in (0, 1):
                target = int(RSC_NEXT_STATE[state, bit])
                candidate = alpha[time, state] + gamma[time, state, bit]
                alpha[time + 1, target] = np.logaddexp(alpha[time + 1, target], candidate)
        alpha[time + 1] -= np.max(alpha[time + 1])

    for time in range(length - 1, -1, -1):
        for state in range(4):
            value = -np.inf
            for bit in (0, 1):
                target = int(RSC_NEXT_STATE[state, bit])
                value = np.logaddexp(value, gamma[time, state, bit] + beta[time + 1, target])
            beta[time, state] = value
        beta[time] -= np.max(beta[time])

    posterior = np.empty(length, dtype=float)
    for time in range(length):
        metric_0 = -np.inf
        metric_1 = -np.inf
        for state in range(4):
            for bit in (0, 1):
                target = int(RSC_NEXT_STATE[state, bit])
                metric = alpha[time, state] + gamma[time, state, bit] + beta[time + 1, target]
                if bit == 0:
                    metric_0 = np.logaddexp(metric_0, metric)
                else:
                    metric_1 = np.logaddexp(metric_1, metric)
        posterior[time] = metric_0 - metric_1
    extrinsic = posterior - systematic_llr - apriori_llr
    return posterior, extrinsic


def turbo_decode(
    code_llr: np.ndarray,
    interleaver: np.ndarray,
    iterations: int = 6,
    llr_clip: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    code_llr = np.asarray(code_llr, dtype=float).reshape(-1)
    length = interleaver.size
    if code_llr.size != 3 * length:
        raise ValueError("Turbo 码 LLR 长度必须等于信息块长度的 3 倍。")
    systematic = np.clip(code_llr[:length], -llr_clip, llr_clip)
    parity_1 = np.clip(code_llr[length : 2 * length], -llr_clip, llr_clip)
    parity_2 = np.clip(code_llr[2 * length :], -llr_clip, llr_clip)
    apriori_1 = np.zeros(length, dtype=float)
    posterior_2 = np.zeros(length, dtype=float)
    for _ in range(iterations):
        _posterior_1, extrinsic_1 = _log_map_siso(systematic, parity_1, apriori_1)
        apriori_2 = np.clip(extrinsic_1[interleaver], -llr_clip, llr_clip)
        posterior_2, extrinsic_2 = _log_map_siso(
            systematic[interleaver], parity_2, apriori_2
        )
        apriori_1 = np.empty(length, dtype=float)
        apriori_1[interleaver] = np.clip(extrinsic_2, -llr_clip, llr_clip)
    posterior = np.empty(length, dtype=float)
    posterior[interleaver] = posterior_2
    return (posterior < 0.0).astype(np.uint8), posterior


def map_bits(bits: np.ndarray, modulation: str) -> tuple[np.ndarray, int]:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    modulation = modulation.upper()
    if modulation == "BPSK":
        return (1.0 - 2.0 * bits.astype(float)).astype(np.complex128), 0
    if modulation == "QPSK":
        padding = bits.size % 2
        if padding:
            bits = np.pad(bits, (0, 1))
        pairs = bits.reshape(-1, 2)
        symbols = ((1.0 - 2.0 * pairs[:, 0]) + 1j * (1.0 - 2.0 * pairs[:, 1])) / math.sqrt(2.0)
        return symbols.astype(np.complex128), padding
    raise ValueError(f"不支持的调制方式：{modulation}")


def symbols_to_llr(
    observed: np.ndarray,
    channel: np.ndarray | complex,
    noise_variance: np.ndarray | float,
    modulation: str,
) -> np.ndarray:
    observed = np.asarray(observed, dtype=np.complex128)
    channel = np.asarray(channel, dtype=np.complex128)
    variance = np.maximum(np.asarray(noise_variance, dtype=float), 1e-15)
    matched = np.conj(channel) * observed
    if modulation.upper() == "BPSK":
        return (4.0 * np.real(matched) / variance).reshape(-1)
    if modulation.upper() == "QPSK":
        scale = 2.0 * math.sqrt(2.0) / variance
        llr = np.empty(2 * observed.size, dtype=float)
        llr[0::2] = scale * np.real(matched)
        llr[1::2] = scale * np.imag(matched)
        return llr
    raise ValueError(f"不支持的调制方式：{modulation}")


def _ofdm_carrier_layout(
    fft_size: int,
    data_subcarriers: int | None = None,
    pilot_subcarriers: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """返回确定性的频域数据/导频索引，未使用频点自然成为 DC/边缘保护带。"""

    pilot_count = int(pilot_subcarriers)
    data_count = int(data_subcarriers or (fft_size - pilot_count))
    active_count = data_count + pilot_count
    if data_count < 1 or pilot_count < 0 or active_count > fft_size:
        raise ValueError("OFDM 子载波配置无效。")
    if active_count == fft_size:
        active = np.arange(fft_size, dtype=int)
    else:
        # 先使用最靠近基带中心的正负频率，保留 DC 与频带边缘作为保护频点。
        candidates = sorted(
            range(1, fft_size), key=lambda index: (min(index, fft_size - index), index)
        )
        active = np.asarray(candidates[:active_count], dtype=int)
    if pilot_count:
        pilot_positions = np.linspace(0, active_count - 1, pilot_count, dtype=int)
        pilot_mask = np.zeros(active_count, dtype=bool)
        pilot_mask[pilot_positions] = True
        pilot_indexes = active[pilot_mask]
        data_indexes = active[~pilot_mask][:data_count]
    else:
        pilot_indexes = np.asarray([], dtype=int)
        data_indexes = active[:data_count]
    return data_indexes, pilot_indexes


def ofdm_modulate(
    symbols: np.ndarray,
    fft_size: int,
    cp_samples: int,
    data_subcarriers: int | None = None,
    pilot_subcarriers: int = 0,
) -> tuple[np.ndarray, int]:
    symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    data_indexes, pilot_indexes = _ofdm_carrier_layout(
        fft_size, data_subcarriers, pilot_subcarriers
    )
    data_count = int(data_indexes.size)
    padding = (-symbols.size) % data_count
    padded = np.pad(symbols, (0, padding))
    blocks = padded.reshape(-1, data_count)
    grid = np.zeros((blocks.shape[0], fft_size), dtype=np.complex128)
    grid[:, data_indexes] = blocks
    if pilot_indexes.size:
        pilot_sequence = 1.0 - 2.0 * (
            (np.arange(blocks.shape[0])[:, None] + np.arange(pilot_indexes.size)[None, :]) % 2
        )
        grid[:, pilot_indexes] = pilot_sequence
    useful = np.fft.ifft(grid, axis=1, norm="ortho")
    if cp_samples:
        framed = np.concatenate((useful[:, -cp_samples:], useful), axis=1)
    else:
        framed = useful
    return framed.reshape(-1), padding


def ofdm_demodulate_llr(
    received: np.ndarray,
    taps: np.ndarray,
    noise_variance: float,
    block_count: int,
    fft_size: int,
    cp_samples: int,
    modulation: str,
    data_subcarriers: int | None = None,
    pilot_subcarriers: int = 0,
) -> np.ndarray:
    block_size = fft_size + cp_samples
    needed = block_count * block_size
    received = np.pad(received[:needed], (0, max(0, needed - received.size)))
    blocks = received.reshape(block_count, block_size)
    useful = blocks[:, cp_samples:]
    observed = np.fft.fft(useful, axis=1, norm="ortho")
    response = np.fft.fft(taps, n=fft_size)
    data_indexes, _pilot_indexes = _ofdm_carrier_layout(
        fft_size, data_subcarriers, pilot_subcarriers
    )
    return symbols_to_llr(
        observed[:, data_indexes].reshape(-1),
        np.tile(response[data_indexes], block_count),
        noise_variance,
        modulation,
    )


def frame_resource_accounting(
    config: SimulationConfig, scenario: Scenario
) -> dict[str, int | float]:
    """按选定调制、编码与 OFDM 网格计算一帧真实资源和开销。"""

    information_bits = int(config.information_bits_per_frame)
    coded_bits = information_bits * (3 if scenario.coding == "turbo" else 1)
    bits_per_symbol = 2 if scenario.modulation.upper() == "QPSK" else 1
    payload_symbols = int(math.ceil(coded_bits / bits_per_symbol))
    header_symbols = int(math.ceil(config.frame.header_bits / bits_per_symbol))
    if scenario.waveform == "ofdm":
        data_subcarriers = config.ofdm.data_subcarriers or (
            config.ofdm.fft_size - config.ofdm.pilot_subcarriers
        )
        ofdm_symbol_samples = config.ofdm.fft_size + config.ofdm.cyclic_prefix_samples
        payload_blocks = int(math.ceil(payload_symbols / data_subcarriers))
        header_blocks = int(math.ceil(header_symbols / data_subcarriers)) if header_symbols else 0
        payload_samples = payload_blocks * ofdm_symbol_samples
        header_samples = header_blocks * ofdm_symbol_samples
        preamble_samples = config.frame.preamble_symbols * ofdm_symbol_samples
    else:
        data_subcarriers = 1
        payload_blocks = payload_symbols
        payload_samples = payload_symbols
        header_samples = header_symbols
        preamble_samples = config.frame.preamble_symbols
    total_samples = (
        payload_samples + header_samples + preamble_samples + config.frame.guard_samples
    )
    overhead_samples = total_samples - payload_samples
    return {
        "information_bits": information_bits,
        "coded_bits": coded_bits,
        "bits_per_symbol": bits_per_symbol,
        "payload_symbols": payload_symbols,
        "payload_blocks": payload_blocks,
        "payload_samples": payload_samples,
        "header_samples": header_samples,
        "preamble_samples": preamble_samples,
        "guard_samples": config.frame.guard_samples,
        "overhead_samples": overhead_samples,
        "total_samples": total_samples,
        "data_subcarriers": data_subcarriers,
        "pilot_subcarriers": config.ofdm.pilot_subcarriers if scenario.waveform == "ofdm" else 0,
        "guard_subcarriers": (
            config.ofdm.fft_size - data_subcarriers - config.ofdm.pilot_subcarriers
            if scenario.waveform == "ofdm" else 0
        ),
        "overhead_fraction": overhead_samples / max(1, total_samples),
    }


def apply_static_channel(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """按非零抽头叠加固定 FIR，避免稀疏长时延信道的无效密集卷积。"""

    signal = np.asarray(signal, dtype=np.complex128).reshape(-1)
    taps = np.asarray(taps, dtype=np.complex128).reshape(-1)
    output = np.zeros(signal.size + taps.size - 1, dtype=np.complex128)
    for index in np.flatnonzero(np.abs(taps) > 0.0):
        output[index : index + signal.size] += taps[index] * signal
    return output


def _transmit_receive_llr(
    coded_bits: np.ndarray,
    information_bit_count: int,
    scenario: Scenario,
    taps: np.ndarray,
    ebn0_db: float,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    symbols, modulation_padding_bits = map_bits(coded_bits, scenario.modulation)
    ofdm_padding_symbols = 0
    if scenario.waveform == "ofdm":
        transmitted, ofdm_padding_symbols = ofdm_modulate(
            symbols,
            config.ofdm.fft_size,
            config.ofdm.cyclic_prefix_samples,
            config.ofdm.data_subcarriers or None,
            config.ofdm.pilot_subcarriers,
        )
        data_subcarriers = config.ofdm.data_subcarriers or (
            config.ofdm.fft_size - config.ofdm.pilot_subcarriers
        )
        block_count = (symbols.size + ofdm_padding_symbols) // data_subcarriers
    elif scenario.waveform == "single_carrier":
        transmitted = symbols
        block_count = 0
    else:
        raise ValueError(f"不支持的波形：{scenario.waveform}")

    # Eb 包含编码、调制、填充、导频、CP，以及可调前导/帧头/帧间保护开销。
    resources = frame_resource_accounting(config, scenario)
    payload_energy = float(np.sum(np.abs(transmitted) ** 2))
    average_payload_power = payload_energy / max(1, transmitted.size)
    overhead_energy = average_payload_power * float(resources["overhead_samples"])
    energy_per_information_bit = (payload_energy + overhead_energy) / information_bit_count
    noise_variance = energy_per_information_bit / (10.0 ** (ebn0_db / 10.0))
    clean = apply_static_channel(transmitted, taps)
    noise = math.sqrt(noise_variance / 2.0) * (
        rng.standard_normal(clean.size) + 1j * rng.standard_normal(clean.size)
    )
    received = clean + noise

    if scenario.waveform == "ofdm":
        llr = ofdm_demodulate_llr(
            received,
            taps,
            noise_variance,
            block_count,
            config.ofdm.fft_size,
            config.ofdm.cyclic_prefix_samples,
            scenario.modulation,
            config.ofdm.data_subcarriers or None,
            config.ofdm.pilot_subcarriers,
        )
    else:
        # 单载波接收机仅做主径相干检测；固定晚到径作为可复现 ISI。这样与
        # 带足够 CP 的 OFDM 对照能显示波形抵抗静态多径的贡献。
        main_tap = taps[0]
        isi_variance = float(np.sum(np.abs(taps[1:]) ** 2))
        llr = symbols_to_llr(
            received[: symbols.size], main_tap, noise_variance + isi_variance, scenario.modulation
        )
    llr = llr[: coded_bits.size]
    details = {
        "energy_per_information_bit": energy_per_information_bit,
        "noise_variance_complex": noise_variance,
        "modulation_padding_bits": float(modulation_padding_bits),
        "ofdm_padding_symbols": float(ofdm_padding_symbols),
        "frame_overhead_samples": float(resources["overhead_samples"]),
        "frame_total_samples": float(resources["total_samples"]),
    }
    return llr, details


def simulate_frame(
    information_bits: np.ndarray,
    scenario: Scenario,
    taps: np.ndarray,
    ebn0_db: float,
    config: SimulationConfig,
    rng: np.random.Generator,
    interleaver: np.ndarray,
) -> tuple[int, dict[str, float]]:
    if scenario.coding == "turbo":
        coded_bits = turbo_encode(information_bits, interleaver)
    elif scenario.coding == "none":
        coded_bits = information_bits
    else:
        raise ValueError(f"不支持的编码方式：{scenario.coding}")
    llr, details = _transmit_receive_llr(
        coded_bits,
        information_bits.size,
        scenario,
        taps,
        ebn0_db,
        config,
        rng,
    )
    if scenario.coding == "turbo":
        decoded, _posterior = turbo_decode(
            llr,
            interleaver,
            iterations=config.turbo.iterations,
            llr_clip=config.turbo.llr_clip,
        )
    else:
        decoded = (llr < 0.0).astype(np.uint8)
    return int(np.count_nonzero(decoded != information_bits)), details


def _seed(base: int, *parts: Any) -> int:
    value = str(base) + "|" + "|".join(str(part) for part in parts)
    # Python 的 hash 带进程随机盐；这里用稳定的字节滚动哈希保证可复现。
    result = 2166136261
    for byte in value.encode("utf-8"):
        result = ((result ^ byte) * 16777619) & 0xFFFFFFFF
    return result


def run_study(
    config: SimulationConfig,
    scenarios: Sequence[Scenario] = DEFAULT_SCENARIOS,
) -> dict[str, Any]:
    validate_config(config)
    taps = channel_taps(config.channel)
    interleaver = make_interleaver(
        config.information_bits_per_frame, config.turbo.interleaver_seed
    )
    points: list[dict[str, Any]] = []
    for scenario in scenarios:
        for ebn0_db in config.ebn0_db:
            rng = np.random.default_rng(_seed(config.random_seed, scenario.key, ebn0_db))
            errors = 0
            total_bits = 0
            detail: dict[str, float] = {}
            for _frame_index in range(config.frames_per_point):
                bits = rng.integers(
                    0, 2, size=config.information_bits_per_frame, dtype=np.uint8
                )
                frame_errors, detail = simulate_frame(
                    bits, scenario, taps, ebn0_db, config, rng, interleaver
                )
                errors += frame_errors
                total_bits += bits.size
            ber = errors / total_bits
            points.append(
                {
                    "scenario": scenario.key,
                    "label": scenario.label,
                    "modulation": scenario.modulation,
                    "waveform": scenario.waveform,
                    "coding": scenario.coding,
                    "ebn0_db": float(ebn0_db),
                    "bit_errors": errors,
                    "total_bits": total_bits,
                    "ber": ber,
                    "zero_error_95pct_upper_bound": 3.0 / total_bits if errors == 0 else None,
                    **detail,
                }
            )
    attribution = build_attribution(points, config)
    return {
        "model": MODEL_VERSION,
        "assumptions": {
            "channel": "fixed deterministic complex FIR; taps do not vary with frame or Eb/N0",
            "channel_knowledge": "perfect at the receiver",
            "single_carrier_receiver": "coherent main-path detector; delayed paths remain as ISI",
            "ofdm_receiver": "CP removal and perfect-CSI per-data-subcarrier soft demodulation; reserved pilots are overhead",
            "turbo_code": "rate 1/3 parallel concatenated RSC(7,5), deterministic interleaver",
            "turbo_decoder": "iterative exact Log-MAP/BCJR maximum-a-posteriori decoding",
            "ebn0_accounting": "information-bit Eb includes coding, padding, pilots, OFDM CP, preamble, header, and frame guard energy",
        },
        "config": asdict(config),
        "channel": {
            "name": config.channel.name,
            "normalized_taps_real": np.real(taps).tolist(),
            "normalized_taps_imag": np.imag(taps).tolist(),
            "energy": float(np.sum(np.abs(taps) ** 2)),
            "memory_samples": int(taps.size - 1),
            "ofdm_cp_covers_channel": config.ofdm.cyclic_prefix_samples >= taps.size - 1,
        },
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "frame_structure": {
            "configured": asdict(config.frame),
            "per_scenario": {
                scenario.key: frame_resource_accounting(config, scenario)
                for scenario in scenarios
            },
        },
        "points": points,
        "attribution": attribution,
    }


def _effective_ber(row: Mapping[str, Any], *, conservative: bool = True) -> float:
    """返回 BER 曲线值；覆盖计算用上界，点估计用 Jeffreys 平滑避免假零值。"""

    ber = float(row.get("ber", math.nan))
    if math.isfinite(ber) and ber > 0.0:
        return min(0.5, ber)
    if not conservative:
        errors = int(row.get("bit_errors", 0) or 0)
        total = int(row.get("total_bits", 0) or 0)
        if total > 0:
            return min(0.5, (errors + 0.5) / (total + 1.0))
    upper = float(row.get("zero_error_95pct_upper_bound", math.nan))
    return min(0.5, upper) if math.isfinite(upper) and upper > 0.0 else math.nan


def _selected_ber_curve(
    study: Mapping[str, Any], selected_scenario: str, *, conservative: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    rows = sorted(
        (
            row
            for row in study.get("points", ())
            if str(row.get("scenario")) == str(selected_scenario)
        ),
        key=lambda row: float(row.get("ebn0_db", 0.0)),
    )
    x = np.asarray([float(row["ebn0_db"]) for row in rows], dtype=np.float64)
    y = np.asarray(
        [_effective_ber(row, conservative=conservative) for row in rows],
        dtype=np.float64,
    )
    valid = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    x, y = x[valid], y[valid]
    if x.size:
        # Monte-Carlo 抖动不应导致距离增加时预测 BER 反而系统性下降。
        y = np.minimum.accumulate(y)
    return x, y


def _ber_at_ebn0(ebn0_db: np.ndarray | float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    values = np.asarray(ebn0_db, dtype=np.float64)
    if x.size == 0:
        return np.full_like(values, np.nan, dtype=np.float64)
    log_ber = np.interp(values, x, np.log10(np.maximum(y, 1.0e-15)))
    return np.power(10.0, log_ber)


def _required_ebn0_for_ber(
    threshold: float, x: np.ndarray, y: np.ndarray
) -> tuple[float | None, str]:
    qualifying = np.flatnonzero(y <= threshold)
    if not qualifying.size:
        return None, "ber_threshold_not_reached_in_sweep"
    index = int(qualifying[0])
    if index == 0:
        return float(x[0]), "at_or_below_sweep_minimum"
    x0, x1 = float(x[index - 1]), float(x[index])
    y0, y1 = math.log10(float(y[index - 1])), math.log10(float(y[index]))
    target = math.log10(threshold)
    fraction = 1.0 if abs(y1 - y0) < 1.0e-15 else (target - y0) / (y1 - y0)
    return x0 + float(np.clip(fraction, 0.0, 1.0)) * (x1 - x0), "interpolated"


def _depth_profile(sound_field: Mapping[str, Any], receiver_depth_m: float) -> tuple[np.ndarray, np.ndarray]:
    ranges = np.asarray(sound_field.get("range_km", ()), dtype=np.float64).reshape(-1)
    depths = np.asarray(sound_field.get("depth_m", ()), dtype=np.float64).reshape(-1)
    raw_grid = sound_field.get("tl_db", ())
    try:
        grid = np.asarray(
            [[math.nan if value is None else float(value) for value in row] for row in raw_grid],
            dtype=np.float64,
        )
    except (TypeError, ValueError):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    if grid.shape == (ranges.size, depths.size):
        grid = grid.T
    if grid.shape != (depths.size, ranges.size) or not ranges.size or not depths.size:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    profile = np.full(ranges.size, np.nan, dtype=np.float64)
    for index in range(ranges.size):
        column = grid[:, index]
        valid = np.isfinite(depths) & np.isfinite(column)
        if np.count_nonzero(valid) == 1:
            profile[index] = float(column[valid][0])
        elif np.count_nonzero(valid) >= 2:
            valid_depths = depths[valid]
            valid_values = column[valid]
            order = np.argsort(valid_depths)
            if valid_depths[order][0] <= receiver_depth_m <= valid_depths[order][-1]:
                profile[index] = float(
                    np.interp(receiver_depth_m, valid_depths[order], valid_values[order])
                )
    order = np.argsort(ranges)
    return ranges[order], profile[order]


def build_current_link_metrics(
    study: Mapping[str, Any],
    static_channel: Mapping[str, Any],
    selected_scenario: str,
    *,
    ber_threshold: float = 1.0e-3,
) -> dict[str, Any]:
    """把归一化 BER 扫描与当前物理 TL/噪声连接成覆盖性能指标。"""

    if not math.isfinite(ber_threshold) or not 0.0 < ber_threshold < 0.5:
        raise ValueError("ber_threshold 必须位于 0 到 0.5 之间。")
    scenarios = {
        str(row.get("key")): row for row in study.get("scenarios", ()) if isinstance(row, Mapping)
    }
    scenario = scenarios.get(str(selected_scenario))
    if not isinstance(scenario, Mapping):
        raise ValueError("找不到重点通信方案。")
    config = study.get("config", {})
    sample_rate_hz = float(config.get("communication_sample_rate_hz", 0.0) or 0.0)
    if sample_rate_hz <= 0.0:
        source = study.get("static_channel_source", {})
        sample_rate_hz = float(source.get("communication_sample_rate_hz", 0.0) or 0.0)
    if sample_rate_hz <= 0.0:
        raise ValueError("缺少通信采样率，无法计算链路速率。")
    modulation = str(scenario.get("modulation", "BPSK")).upper()
    waveform = str(scenario.get("waveform", "single_carrier")).lower()
    coding = str(scenario.get("coding", "none")).lower()
    bits_per_symbol = 2.0 if modulation == "QPSK" else 1.0
    code_rate = 1.0 / 3.0 if coding == "turbo" else 1.0
    ofdm = config.get("ofdm", {})
    fft_size = int(ofdm.get("fft_size", 1))
    cp_samples = int(ofdm.get("cyclic_prefix_samples", 0))
    scenario_resources = (
        study.get("frame_structure", {}).get("per_scenario", {}).get(str(selected_scenario), {})
    )
    total_frame_samples = int(scenario_resources.get("total_samples", 0) or 0)
    if total_frame_samples > 0:
        active_information_rate_bps = (
            sample_rate_hz
            * float(config.get("information_bits_per_frame", 1))
            / total_frame_samples
        )
    else:
        # 兼容旧结果文件：旧模型没有显式帧结构和子载波资源表。
        waveform_efficiency = (
            fft_size / max(1, fft_size + cp_samples) if waveform == "ofdm" else 1.0
        )
        active_information_rate_bps = (
            sample_rate_hz * bits_per_symbol * code_rate * waveform_efficiency
        )
    gating = static_channel.get("signal", {}).get("gating", {})
    duty_cycle = float(gating.get("duty_cycle", 1.0) or 1.0)
    duty_cycle = float(np.clip(duty_cycle, 0.0, 1.0))
    net_information_rate_bps = active_information_rate_bps * duty_cycle

    element = static_channel.get("element", {})
    current_tl_db = float(element.get("transmission_loss_db", math.nan))
    noise = static_channel.get("noise", {})
    source_level_db = float(noise.get("source_level_db", 190.0))
    ambient_noise_db = float(noise.get("ambient_noise_db", 78.0))
    receiver_gain_db = float(
        noise.get("receiver_sensitivity_db", noise.get("sensitivity_db", 0.0))
    )
    current_snr_db = source_level_db - current_tl_db - ambient_noise_db + receiver_gain_db
    ebn0_offset_db = 10.0 * math.log10(sample_rate_hz / max(active_information_rate_bps, 1.0e-12))
    current_ebn0_db = current_snr_db + ebn0_offset_db

    curve_x, curve_y = _selected_ber_curve(study, selected_scenario, conservative=True)
    estimate_x, estimate_y = _selected_ber_curve(
        study, selected_scenario, conservative=False
    )
    required_ebn0_db, threshold_status = _required_ebn0_for_ber(
        ber_threshold, curve_x, curve_y
    )
    current_ber = float(_ber_at_ebn0(current_ebn0_db, estimate_x, estimate_y))
    current_ber_upper = float(_ber_at_ebn0(current_ebn0_db, curve_x, curve_y))
    bits_per_frame = int(config.get("information_bits_per_frame", 1))
    outage_probability = (
        1.0 - math.exp(bits_per_frame * math.log1p(-min(current_ber, 1.0 - 1.0e-15)))
        if math.isfinite(current_ber)
        else math.nan
    )
    effective_goodput_bps = net_information_rate_bps * (
        1.0 - outage_probability if math.isfinite(outage_probability) else math.nan
    )
    outage_probability_upper = (
        1.0 - math.exp(bits_per_frame * math.log1p(-min(current_ber_upper, 1.0 - 1.0e-15)))
        if math.isfinite(current_ber_upper)
        else math.nan
    )

    array = static_channel.get("array", {})
    receiver_depth_m = float(array.get("depth_m", 0.0) or 0.0)
    ranges_km, tl_profile_db = _depth_profile(
        static_channel.get("sound_field", {}), receiver_depth_m
    )
    finite = np.isfinite(ranges_km) & np.isfinite(tl_profile_db)
    ranges_km, tl_profile_db = ranges_km[finite], tl_profile_db[finite]
    profile_ebn0_db = (
        source_level_db - tl_profile_db - ambient_noise_db + receiver_gain_db + ebn0_offset_db
    )
    profile_ber = _ber_at_ebn0(profile_ebn0_db, curve_x, curve_y)
    maximum_distance_km: float | None = None
    coverage_limited = False
    if required_ebn0_db is not None and ranges_km.size:
        qualifying = np.flatnonzero(profile_ebn0_db >= required_ebn0_db)
        if qualifying.size:
            last = int(qualifying[-1])
            maximum_distance_km = float(ranges_km[last])
            coverage_limited = last == ranges_km.size - 1
            if not coverage_limited and profile_ebn0_db[last + 1] < required_ebn0_db:
                x0, x1 = float(ranges_km[last]), float(ranges_km[last + 1])
                y0, y1 = float(profile_ebn0_db[last]), float(profile_ebn0_db[last + 1])
                if abs(y1 - y0) > 1.0e-12:
                    fraction = (required_ebn0_db - y0) / (y1 - y0)
                    maximum_distance_km = x0 + float(np.clip(fraction, 0.0, 1.0)) * (x1 - x0)

    curve_step = max(1, int(math.ceil(max(1, ranges_km.size) / 160)))
    distance_curve = [
        {
            "range_km": round(float(distance), 6),
            "transmission_loss_db": round(float(tl), 6),
            "ebn0_db": round(float(ebn0), 6),
            "estimated_ber": float(ber),
        }
        for distance, tl, ebn0, ber in zip(
            ranges_km[::curve_step],
            tl_profile_db[::curve_step],
            profile_ebn0_db[::curve_step],
            profile_ber[::curve_step],
        )
    ]
    def finite_or_none(value: float | None) -> float | None:
        return float(value) if value is not None and math.isfinite(float(value)) else None

    return {
        "definition": "maximum range whose estimated BER is at or below the selected threshold",
        "ber_threshold": float(ber_threshold),
        "current_range_km": finite_or_none(float(array.get("center_range_km", math.nan))),
        "current_transmission_loss_db": finite_or_none(current_tl_db),
        "current_snr_db": finite_or_none(current_snr_db),
        "current_ebn0_db": finite_or_none(current_ebn0_db),
        "current_ber": finite_or_none(current_ber),
        "current_ber_95pct_upper_bound": finite_or_none(current_ber_upper),
        "outage_probability": finite_or_none(outage_probability),
        "outage_probability_95pct_upper_bound": finite_or_none(outage_probability_upper),
        "outage_definition": "probability of at least one residual bit error in an information frame",
        "active_information_rate_bps": active_information_rate_bps,
        "net_information_rate_bps": net_information_rate_bps,
        "effective_goodput_bps": finite_or_none(effective_goodput_bps),
        "frame_total_samples": total_frame_samples or None,
        "frame_overhead_fraction": finite_or_none(
            float(scenario_resources.get("overhead_fraction", math.nan))
        ),
        "data_subcarriers": int(scenario_resources.get("data_subcarriers", 0) or 0),
        "pilot_subcarriers": int(scenario_resources.get("pilot_subcarriers", 0) or 0),
        "guard_subcarriers": int(scenario_resources.get("guard_subcarriers", 0) or 0),
        "duty_cycle": duty_cycle,
        "required_ebn0_db": required_ebn0_db,
        "threshold_status": threshold_status,
        "maximum_distance_km": maximum_distance_km,
        "coverage_limited_by_field": coverage_limited,
        "field_limit_km": float(ranges_km[-1]) if ranges_km.size else None,
        "distance_curve": distance_curve,
        "ber_estimation": "log-linear interpolation; current BER uses a Jeffreys-smoothed point estimate while maximum range uses the conservative zero-error 95% 3/N upper bound",
    }


def build_ber_diagnostics(
    study: Mapping[str, Any],
    selected_scenario: str,
    link_metrics: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """给出与当前参数和接收机实现直接对应的 BER 解释。"""

    scenarios = {
        str(row.get("key")): row
        for row in study.get("scenarios", ())
        if isinstance(row, Mapping)
    }
    scenario = scenarios.get(str(selected_scenario), {})
    config = study.get("config", {})
    channel = study.get("channel", {})
    diagnostics: list[dict[str, str]] = []
    waveform = str(scenario.get("waveform", "single_carrier"))
    coding = str(scenario.get("coding", "none"))
    memory = int(channel.get("memory_samples", 0) or 0)
    cp_samples = int(config.get("ofdm", {}).get("cyclic_prefix_samples", 0) or 0)
    if waveform == "ofdm" and cp_samples < memory:
        diagnostics.append({
            "severity": "high",
            "title": "循环前缀短于信道记忆",
            "detail": f"CP 为 {cp_samples} 点，静态 CIR 记忆为 {memory} 点，仍存在块间干扰；建议 CP 至少设为 {memory} 点。",
        })
    elif waveform == "ofdm":
        diagnostics.append({
            "severity": "good",
            "title": "循环前缀已覆盖多径",
            "detail": f"CP {cp_samples} 点覆盖 {memory} 点信道记忆；当前 OFDM 高 BER 主要不是 CP 不足造成的。",
        })
    elif memory > 0:
        diagnostics.append({
            "severity": "high",
            "title": "单载波晚到径形成 ISI",
            "detail": "当前单载波接收机只对主径做相干检测，没有时域均衡，所有晚到路径都作为码间干扰。",
        })

    frames = int(config.get("frames_per_point", 1) or 1)
    bits = int(config.get("information_bits_per_frame", 1) or 1)
    observed_bits = frames * bits
    upper = 3.0 / max(1, observed_bits)
    if observed_bits < 10_000:
        diagnostics.append({
            "severity": "medium",
            "title": "统计样本偏少",
            "detail": f"每个 Eb/N0 仅统计 {observed_bits} 个信息比特；即使零误码，95% BER 上界仍为 {upper:.2e}。门限低于该值时应增加每点帧数。",
        })

    resources = study.get("frame_structure", {}).get("per_scenario", {}).get(
        str(selected_scenario), {}
    )
    overhead = float(resources.get("overhead_fraction", 0.0) or 0.0)
    if overhead > 0.2:
        diagnostics.append({
            "severity": "medium",
            "title": "短帧的固定开销较高",
            "detail": f"前导、帧头、保护间隔与 OFDM 填充占本帧 {overhead * 100:.1f}%；同一信息比特 Eb/N0 口径下，这些能量不会被误算成编码增益。",
        })
    if coding == "turbo" and bits < 512:
        diagnostics.append({
            "severity": "medium",
            "title": "Turbo 码块较短",
            "detail": f"当前码块只有 {bits} bit，交织增益受限；可先提高到 1024 bit，并保持 6–8 次 Log-MAP 迭代。",
        })

    if link_metrics:
        current = link_metrics.get("current_ebn0_db")
        required = link_metrics.get("required_ebn0_db")
        if current is not None and required is not None and float(current) < float(required):
            diagnostics.insert(0, {
                "severity": "high",
                "title": "当前链路预算未达到 BER 门限",
                "detail": f"当前 Eb/N0 为 {float(current):.2f} dB，达到所选 BER 门限约需 {float(required):.2f} dB。",
            })
    diagnostics.append({
        "severity": "info",
        "title": "当前采用理想同步与完美信道知识",
        "detail": "仿真中的导频计入开销，但暂未引入同步误差和信道估计误差；当前 BER 偏高不能归因于估计误差。",
    })
    return diagnostics[:5]


def _point_map(points: Sequence[Mapping[str, Any]]) -> dict[tuple[str, float], Mapping[str, Any]]:
    return {(str(row["scenario"]), float(row["ebn0_db"])): row for row in points}


def build_attribution(
    points: Sequence[Mapping[str, Any]], config: SimulationConfig
) -> dict[str, Any]:
    mapping = _point_map(points)
    pairs = (
        ("modulation", "baseline", "modulation", "仅调制变化：QPSK 相对 BPSK（单载波、未编码）"),
        ("waveform", "baseline", "waveform", "仅波形变化：OFDM 相对单载波（BPSK、未编码）"),
        ("coding", "qpsk_ofdm_uncoded", "full", "仅编码变化：Turbo 相对未编码（QPSK、OFDM）"),
    )
    comparisons: list[dict[str, Any]] = []
    for effect, reference, changed, description in pairs:
        rows = []
        for ebn0 in config.ebn0_db:
            left = mapping[(reference, float(ebn0))]
            right = mapping[(changed, float(ebn0))]
            # 零误码点使用三事件法则上界，避免虚假的无限增益。
            left_ber = float(left["ber"] or left["zero_error_95pct_upper_bound"])
            right_ber = float(right["ber"] or right["zero_error_95pct_upper_bound"])
            rows.append(
                {
                    "ebn0_db": float(ebn0),
                    "reference_ber": float(left["ber"]),
                    "changed_ber": float(right["ber"]),
                    "ber_ratio_changed_over_reference": right_ber / left_ber,
                }
            )
        comparisons.append(
            {
                "effect": effect,
                "description": description,
                "reference_scenario": reference,
                "changed_scenario": changed,
                "points": rows,
            }
        )
    return {
        "method": "one-factor-at-a-time paired comparison at identical information-bit Eb/N0",
        "comparisons": comparisons,
        "interpretation": [
            "平坦静态信道中，Gray BPSK 与 QPSK 的理论 BER 基本相同；QPSK 的主要收益是频谱效率。",
            "固定多径下，CP 足够长时 OFDM 消除块间干扰并把频率选择性信道拆成平坦子信道。",
            "Turbo 曲线相对未编码曲线的下降才是信道编码增益；低迭代次数或短码块会限制该增益。",
        ],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_value(row: Mapping[str, Any]) -> float:
    value = float(row["ber"])
    if value > 0.0:
        return value
    return float(row["zero_error_95pct_upper_bound"])


def _configure_chinese_font() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def write_plots(result: Mapping[str, Any], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_chinese_font()
    points = result["points"]
    scenario_rows = result["scenarios"]
    paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    for scenario in scenario_rows:
        rows = sorted(
            (row for row in points if row["scenario"] == scenario["key"]),
            key=lambda row: row["ebn0_db"],
        )
        axis.semilogy(
            [row["ebn0_db"] for row in rows],
            [_plot_value(row) for row in rows],
            "o-",
            linewidth=1.7,
            markersize=4,
            label=scenario["label"],
        )
    axis.set_xlabel("信息比特 Eb/N0（dB）")
    axis.set_ylabel("误比特率 BER")
    axis.set_title("静态信道完整通信链路 BER 对比")
    axis.grid(True, which="both", alpha=0.28)
    axis.legend(fontsize=8)
    path = output_dir / "ber_all_scenarios.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True, sharey=True)
    mapping = {row["key"]: row["label"] for row in scenario_rows}
    for axis, comparison in zip(axes, result["attribution"]["comparisons"]):
        for key in (comparison["reference_scenario"], comparison["changed_scenario"]):
            rows = sorted(
                (row for row in points if row["scenario"] == key),
                key=lambda row: row["ebn0_db"],
            )
            axis.semilogy(
                [row["ebn0_db"] for row in rows],
                [_plot_value(row) for row in rows],
                "o-",
                label=mapping[key],
            )
        axis.set_title(comparison["effect"].capitalize())
        axis.set_xlabel("Eb/N0（dB）")
        axis.grid(True, which="both", alpha=0.28)
        axis.legend(fontsize=7)
    axes[0].set_ylabel("BER")
    figure.suptitle("单因素受控对照：性能变化来自哪里")
    path = output_dir / "ber_attribution.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    taps = np.asarray(result["channel"]["normalized_taps_real"]) + 1j * np.asarray(
        result["channel"]["normalized_taps_imag"]
    )
    response = np.fft.fft(taps, n=512)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    axes[0].stem(np.arange(taps.size), np.abs(taps) ** 2, basefmt=" ")
    axes[0].set_xlabel("离散时延（采样）")
    axes[0].set_ylabel("归一化功率")
    axes[0].set_title("固定信道冲激响应")
    axes[0].grid(True, alpha=0.25)
    frequency = np.linspace(0.0, 1.0, response.size, endpoint=False)
    axes[1].plot(frequency, 20.0 * np.log10(np.maximum(np.abs(response), 1e-8)))
    axes[1].set_xlabel("归一化频率")
    axes[1].set_ylabel("幅频响应（dB）")
    axes[1].set_title("静态频率选择性")
    axes[1].grid(True, alpha=0.25)
    path = output_dir / "static_channel.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)
    return paths


def write_report(result: Mapping[str, Any], output_dir: Path) -> Path:
    channel = result["channel"]
    config = result["config"]
    lines = [
        "# 静态信道完整通信链路 BER 研究",
        "",
        "## 链路结构",
        "",
        "信息比特 → 可选 Turbo(1/3) → BPSK/QPSK → 单载波/OFDM → 固定复数 FIR 信道 + AWGN → 软解调 → 可选 Log-MAP Turbo 译码 → BER。",
        "",
        "## 可解释性设计",
        "",
        "- 调制贡献：单载波、未编码条件下对比 BPSK 与 QPSK。",
        "- 波形贡献：BPSK、未编码条件下对比单载波与 OFDM。",
        "- 编码贡献：QPSK、OFDM 条件下对比未编码与 Turbo。",
        "- 横轴按信息比特 Eb/N0 统一计量，已计入码率、循环前缀和填充开销。",
        "- 仿真零误码点在绘图时采用 95% 置信上界 3/N，不画成虚假的 0。",
        "",
        "## 当前配置",
        "",
        f"- 信道：{channel['name']}，固定 {channel['memory_samples'] + 1} 个抽头，记忆长度 {channel['memory_samples']} 个采样。",
        f"- OFDM：FFT={config['ofdm']['fft_size']}，CP={config['ofdm']['cyclic_prefix_samples']} 个采样，覆盖信道：{channel['ofdm_cp_covers_channel']}。",
        f"- Turbo：RSC(7,5)，码率 1/3，Log-MAP 迭代 {config['turbo']['iterations']} 次。",
        f"- 每点：{config['frames_per_point']} 帧 × {config['information_bits_per_frame']} 信息比特。",
        "",
        "## 结果文件",
        "",
        "- `ber_all_scenarios.png`：完整方案总览。",
        "- `ber_attribution.png`：调制、波形、编码三个单因素对照。",
        "- `static_channel.png`：固定信道的冲激响应与频率选择性。",
        "- `ber_results.csv`：每个场景、每个 Eb/N0 的误码统计。",
        "- `result.json`：完整配置、假设、结果与归因数据。",
        "",
        "## 读图原则",
        "",
        "在平坦信道中，BPSK 与 Gray QPSK 在相同 Eb/N0 下的 BER 应接近；QPSK 的优势主要是单位符号携带两比特。固定多径中，OFDM 的收益取决于 CP 是否覆盖信道记忆。Turbo 与未编码曲线的差距才可归因于信道编码和迭代 MAP 译码。",
    ]
    path = output_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_results(result: Mapping[str, Any], output_dir: Path, plot: bool = True) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    csv_path = output_dir / "ber_results.csv"
    _write_csv(csv_path, result["points"])
    report_path = write_report(result, output_dir)
    paths = [result_path, csv_path, report_path]
    if plot:
        paths.extend(write_plots(result, output_dir))
    return paths


def example_config() -> dict[str, Any]:
    return asdict(SimulationConfig())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="静态信道 BPSK/QPSK、单载波/OFDM、未编码/Turbo(Log-MAP) BER 对比"
    )
    parser.add_argument("--config", type=Path, help="JSON 配置文件")
    parser.add_argument("--output-dir", type=Path, default=Path("output/static_communication"))
    parser.add_argument("--no-plot", action="store_true", help="不生成 PNG 图")
    parser.add_argument("--write-example-config", type=Path, help="写出示例配置后退出")
    args = parser.parse_args()
    if args.write_example_config:
        args.write_example_config.parent.mkdir(parents=True, exist_ok=True)
        args.write_example_config.write_text(
            json.dumps(example_config(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"已写出示例配置：{args.write_example_config}")
        return 0
    config = load_config(args.config)
    result = run_study(config)
    paths = write_results(result, args.output_dir, plot=not args.no_plot)
    print(f"静态通信链路研究完成，共写出 {len(paths)} 个文件：{args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
