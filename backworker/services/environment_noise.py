"""基于改造后 Wenz 经验曲线的海洋环境噪声谱计算。"""

import numpy as np

# 海况等级换算为代表性风速，单位 m/s。
_WIND_SPEED_MPS = np.asarray(
    [0.0, 1.0, 2.45, 4.4, 6.7, 9.35, 12.3, 15.5, 18.95, 22.6, 26.4, 30.5, 35.0],
    dtype=float,
)

# 降雨等级换算为代表性雨强，单位 mm/h。接口传入的是等级，不是实测雨强，
# 因此这里使用工程近似映射。
_RAIN_RATE_MM_H = np.asarray([0.0, 0.25, 1.0, 2.5, 7.5, 20.0, 50.0], dtype=float)


def wind_speed_mps_from_level(wind_level):
    """把 0—12 级海况/风级转换为代表性 10 米风速。"""
    level = int(wind_level)
    if level < 0 or level >= _WIND_SPEED_MPS.size:
        raise ValueError("wind_level 必须位于 0 到 12 级。")
    return float(_WIND_SPEED_MPS[level])




def rain_noise_psd_db(freq_hz, rain_rate_mm_h):
    """返回连续雨强对应的经验降雨噪声功率谱级。

    ``rain_rate_mm_h`` 可以是标量或数组，且必须严格大于零。这里的
    结果是经验参考 PSD，并不是已标定的单位面积声源级。
    """
    frequency = np.asarray(freq_hz, dtype=float)
    rain_rate = np.asarray(rain_rate_mm_h, dtype=float)
    if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError("频率必须是正有限数。")
    if np.any(~np.isfinite(rain_rate)) or np.any(rain_rate <= 0.0):
        raise ValueError("雨强必须是正有限数，单位为 mm/h。")

    level_at_5khz = 50.0 + 17.0 * np.log10(rain_rate)
    peak_freq = 13500.0
    peak_level = level_at_5khz + 15.0 * np.log2(peak_freq / 5000.0)
    return np.where(
        frequency < peak_freq,
        level_at_5khz + 15.0 * np.log2(frequency / 5000.0),
        peak_level - 7.0 * np.log2(frequency / peak_freq),
    )


def rain_surface_source_level_db(rain_rate_mm_h):
    """返回单位海面面积的降雨噪声源级。

    采用柳云峰等（2022）式(8)引用的 Urick 关系：
    ``SLR = 51 + 10 log10(R)``，``R`` 的单位为 mm/h。该源级用于
    500–6400 Hz 的海面非均匀分布源传播叠加，不是接收点谱级。
    """
    rain_rate = np.asarray(rain_rate_mm_h, dtype=float)
    if np.any(~np.isfinite(rain_rate)) or np.any(rain_rate <= 0.0):
        raise ValueError("雨强必须是正有限数，单位为 mm/h。")
    return 51.0 + 10.0 * np.log10(rain_rate)


def _rain_noise_db(freq_hz, rain_level):
    """按旧接口的降雨等级计算降雨噪声。"""
    if rain_level <= 0:
        return None
    return rain_noise_psd_db(freq_hz, float(_RAIN_RATE_MM_H[rain_level]))


def coates_stojanovic_shipping_psd_db(freq_hz, shipping_activity):
    """Coates/Stojanovic 对 Wenz 远距离航运噪声的经验参数化。

    ``shipping_activity`` 为 0—1 的航运活动因子，频率输入单位 Hz，返回
    dB re 1 uPa^2/Hz。闭式表达式中的频率使用 kHz。
    """
    frequency = np.asarray(freq_hz, dtype=float)
    activity = float(shipping_activity)
    if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError("频率必须是正有限数。")
    if not np.isfinite(activity) or not 0.0 <= activity <= 1.0:
        raise ValueError("shipping_activity 必须位于 0 到 1。")
    frequency_khz = frequency / 1000.0
    return (
        40.0
        + 20.0 * (activity - 0.5)
        + 26.0 * np.log10(frequency_khz)
        - 60.0 * np.log10(frequency_khz + 0.03)
    )


def coates_stojanovic_wind_psd_db(freq_hz, wind_speed_mps):
    """Coates/Stojanovic 风生表面噪声经验谱，直接使用连续风速。"""
    frequency = np.asarray(freq_hz, dtype=float)
    wind_speed = float(wind_speed_mps)
    if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError("频率必须是正有限数。")
    if not np.isfinite(wind_speed) or wind_speed < 0.0:
        raise ValueError("wind_speed_mps 必须是非负有限数。")
    frequency_khz = frequency / 1000.0
    return (
        50.0
        + 7.5 * np.sqrt(wind_speed)
        + 20.0 * np.log10(frequency_khz)
        - 40.0 * np.log10(frequency_khz + 0.4)
    )


def component_levels_db(freq_hz, ship_level, wind_level, rain_level):
    """返回各噪声分量谱级；wind_level=None 时由调用方提供传播后的风分量。"""
    freq_hz = np.asarray(freq_hz, dtype=float)
    freq_khz = freq_hz / 1000.0

    # 改造后的 Wenz 曲线各分量都是功率谱级，单位 dB re 1 uPa^2/Hz。
    # 分贝 dB 不能直接相加，所以先转成功率，最后再转回 dB。
    shipping_factor = (float(ship_level) - 1.0) / 4.0

    turbulence_db = 17.0 - 30.0 * np.log10(freq_khz)
    shipping_db = coates_stojanovic_shipping_psd_db(freq_hz, shipping_factor)
    thermal_db = -15.0 + 20.0 * np.log10(freq_khz)
    result = {
        "turbulence": turbulence_db,
        "shipping": shipping_db,
        "thermal": thermal_db,
    }
    if wind_level is not None:
        wind_mps = wind_speed_mps_from_level(wind_level)
        result["wind"] = coates_stojanovic_wind_psd_db(freq_hz, wind_mps)
    rain_db = _rain_noise_db(freq_hz, int(rain_level))
    if rain_db is not None:
        result["rain"] = rain_db
    return result


def total_from_component_levels(components):
    """把多个 dB 功率谱级在线性域求和。"""
    component_powers = [
        np.power(10.0, np.asarray(values, dtype=float) / 10.0)
        for values in components.values()
    ]
    total_db = 10.0 * np.log10(np.sum(component_powers, axis=0))
    if not np.all(np.isfinite(total_db)):
        raise ValueError("噪声谱计算产生了无效数值。")
    return total_db


def build_response(params):
    freq_hz = np.linspace(
        params["minFreq"], params["maxFreq"], params["numPoints"], dtype=float
    )
    components = component_levels_db(
        freq_hz,
        params["shipDensityLevel"],
        params["windSpeed"],
        params["rainIntensity"],
    )

    total_db = total_from_component_levels(components)

    return {
        "freq": np.round(freq_hz, 6).tolist(),
        "data": np.round(total_db, 6).tolist(),
    }
