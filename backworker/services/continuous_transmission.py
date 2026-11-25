"""Waypoint transmission generator. Every yielded block has absolute timestamps."""
import json
import math
import multiprocessing
import secrets
import shutil
import struct
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.signal import firwin2, lfilter, fftconvolve

from .waypoint_channel import WaypointChannel, build_waypoints


WAYPOINT_WORKER_COUNT = 4


class IncrementalWavWriter:
    def __init__(self, path, sample_rate):
        self._stream = open(path, 'wb')
        self._data_bytes = 0
        self._stream.write(b'RIFF')
        self._stream.write(struct.pack('<I', 0))
        self._stream.write(b'WAVEfmt ')
        self._stream.write(struct.pack('<IHHIIHH', 16, 1, 1, int(sample_rate),
                                       int(sample_rate)*2, 2, 16))
        self._stream.write(b'data')
        self._stream.write(struct.pack('<I', 0))
        self._stream.flush()

    def write(self, samples):
        pcm = np.rint(np.clip(samples, -1, 1)*32767).astype('<i2').tobytes()
        self._stream.write(pcm)
        self._data_bytes += len(pcm)
        self._update_header()

    def _update_header(self):
        self._stream.seek(4)
        self._stream.write(struct.pack('<I', 36 + self._data_bytes))
        self._stream.seek(40)
        self._stream.write(struct.pack('<I', self._data_bytes))
        self._stream.seek(0, 2)
        self._stream.flush()

    def close(self):
        if self._stream is not None:
            self._update_header()
            self._stream.close()
            self._stream = None


class StreamingFilter:
    def __init__(self, coefficients):
        self.b = coefficients
        self.state = np.zeros(len(coefficients)-1)

    def __call__(self, samples):
        output, self.state = lfilter(self.b, [1.], samples, zi=self.state)
        return output


class StreamingMonitorNormalizer:
    """Make low-level ADC samples audible without changing stored chain data.

    The ADC-normalized stream is often tens of dB below digital full scale.  If
    it is written directly to a 16-bit monitor WAV it is quantized mostly to
    zero.  This stateful gain stages only the listening copy; NPZ samples,
    spectrograms, ADC codes and calibrated pressure remain untouched.
    """

    def __init__(self, target_peak=.8, maximum_gain_db=100.):
        self.target_peak = float(target_peak)
        self.maximum_gain = 10. ** (float(maximum_gain_db) / 20.)
        self.gain = None

    @property
    def gain_db(self):
        gain = 1. if self.gain is None else max(float(self.gain), 1e-30)
        return 20. * math.log10(gain)

    def __call__(self, samples):
        values = np.asarray(samples, dtype=float)
        if not len(values):
            return values.copy()
        magnitude = np.abs(values)
        # A high percentile avoids a single impulse controlling an entire
        # two-second block while retaining the relative level within it.
        reference = float(np.percentile(magnitude, 99.9))
        desired = min(self.maximum_gain, self.target_peak / max(reference, 1e-12))
        if self.gain is None:
            start_gain = desired
        else:
            start_gain = float(self.gain)
            # Reduce gain quickly when a louder block arrives; restore it more
            # slowly to prevent audible pumping between Waypoint blocks.
            if desired < start_gain:
                desired = max(desired, start_gain / 16.)
            else:
                desired = min(desired, start_gain * 2.)
        self.gain = desired
        gain_curve = np.geomspace(max(start_gain, 1e-30), max(desired, 1e-30), len(values))
        # Soft limiting prevents a transient from hard-clipping the monitor.
        return np.tanh(values * gain_curve)


def _calculate_waypoint(index, point, config, jobs_dir, core, output):
    """Calculate and persist one independent Bellhop/analytic snapshot."""
    from .. import signal_transmission_service as s
    station = {
        **config,
        'source_lon': point['lon'],
        'source_lat': point['lat'],
        'source_depth_m': point['depth_m'],
        'allow_empty_arrivals': True,
    }
    response = (
        s._bellhop_multipath(station, core)
        if config['propagation_model'] == 'bellhop'
        else s._analytic_multipath(station)
    )
    records = response.get('raw_arrivals')
    taps = []
    for record in records or response['impulse_response']:
        phase = math.radians(record['phase_deg'])
        taps.append({
            **record,
            'gain_real': record['amplitude'] * math.cos(phase),
            'gain_imag': record['amplitude'] * math.sin(phase),
        })
    raw_path = response.get('raw_arrival_file')
    if raw_path:
        shutil.copy2(
            Path(jobs_dir) / response['job_id'] / raw_path,
            output / f'waypoint_{index:04d}.arr',
        )
    elif config['propagation_model'] == 'bellhop':
        raise ValueError('缺少原始.arr文件，无法建立可追溯信道。')
    snapshot = {**point, 'arrivals': taps, 'bellhop': response}
    (output / f'waypoint_{index:04d}.json').write_text(
        json.dumps(snapshot, ensure_ascii=False), encoding='utf-8'
    )
    return snapshot


def _calculate_waypoint_process(task):
    """Process entry point; each worker owns an isolated NetCDF native state."""
    index, point, config, jobs_dir, output = task
    core = None
    if config['propagation_model'] == 'bellhop':
        # Import inside the child so netCDF4 and its C DLL are never shared by
        # concurrent Waypoint calculations in the backend process.
        from .. import backend_worker as core
    snapshot = _calculate_waypoint(
        index, point, config, jobs_dir, core, Path(output)
    )
    return index, snapshot


def _calculate_waypoints(
    points, config, jobs_dir, core, output, progress, cancelled,
    process_worker=_calculate_waypoint_process,
):
    """Run at most four isolated Waypoint processes, preserving route order."""
    del core  # Bellhop core is imported independently inside every child.
    snapshots = [None] * len(points)
    worker_count = min(WAYPOINT_WORKER_COUNT, len(points))
    executor = ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context('spawn'),
    )
    futures = {
        executor.submit(
            process_worker,
            (index, point, config, str(jobs_dir), str(output)),
        ): index
        for index, point in enumerate(points)
    }
    completed = 0
    try:
        for future in as_completed(futures):
            index = futures[future]
            if cancelled.is_set():
                return None
            try:
                result_index, snapshot = future.result()
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(
                    f'Waypoint {index + 1}/{len(points)} 计算失败：{exc}'
                ) from exc
            if result_index != index:
                raise RuntimeError(
                    f'Waypoint并行结果索引异常：期望 {index}，实际 {result_index}'
                )
            if snapshot is None:
                return None
            snapshots[index] = snapshot
            completed += 1
            progress(
                'waypoint',
                round(50 * completed / len(points)),
                f'已完成Waypoint {completed}/{len(points)}（最多{worker_count}路并发）',
            )
    finally:
        if cancelled.is_set():
            for future in futures:
                future.cancel()
        # Running children finish their isolated job directories before the
        # pool exits; queued tasks are cancelled after a disconnect/failure.
        executor.shutdown(wait=True, cancel_futures=True)
    if any(snapshot is None for snapshot in snapshots):
        return None
    return snapshots


def run(request, jobs_dir, core, progress, cancelled):
    from .. import signal_transmission_service as s
    config = s._normalize_request(request)
    fs = config['sample_rate']
    route_points = config['source_route']
    spacing = s._number(request, ('waypoint_spacing_m',), 50, 'Waypoint间距', minimum=1, maximum=1000)
    speed = s._number(request, ('vessel_speed_kn',), 4, '航速', minimum=.1, maximum=60)
    fixed_source = len(route_points) == 1
    if not fixed_source:
        points = build_waypoints(route_points, speed, spacing)
        calculation_points = points
        public_points = points
    else:
        start_point = {**route_points[0], 'emission_time_s': 0}
        points = [start_point,
                  {**start_point, 'emission_time_s': config['duration_s']}]
        # A fixed source needs two time knots for channel interpolation, but
        # both knots share one physical location and one propagation result.
        calculation_points = points[:1]
        public_points = points[:1]
    if len(points) > 1000 or points[-1]['emission_time_s'] > 7200:
        raise ValueError('航迹超过1000个Waypoint或2小时，请缩短航迹。')
    job_id = secrets.token_hex(6)
    output = Path(jobs_dir)/job_id/'continuous'
    output.mkdir(parents=True, exist_ok=True)
    # 保存原始请求，工作区恢复时可按同一参数重放连续链路结果。
    try:
        (output/'request.json').write_text(
            json.dumps(request, ensure_ascii=False), encoding='utf-8'
        )
    except (TypeError, ValueError):
        pass
    progress(
        'waypoint', 0,
        f'开始计算{len(calculation_points)}个Waypoint（最多{min(WAYPOINT_WORKER_COUNT, len(calculation_points))}路并发）',
    )
    snapshots = _calculate_waypoints(
        calculation_points, config, jobs_dir, core, output, progress, cancelled
    )
    if snapshots is None or cancelled.is_set():
        return
    if fixed_source:
        snapshots = [snapshots[0], {
            **snapshots[0],
            'emission_time_s': config['duration_s'],
        }]
    progress('waypoint_match', 52, 'Waypoint全部完成，正在统一匹配相邻传播路径')
    channel = WaypointChannel(snapshots)
    end = points[-1]['emission_time_s']
    tail = max((float(a['delay_s']) for w in snapshots for a in w['arrivals']), default=0)
    (output/'paths.json').write_text(json.dumps(channel.tracks, ensure_ascii=False), encoding='utf-8')
    config['source_route'] = points
    s._resolve_wenz_noise_spectra(config, core, progress)
    # Persistent FIR states prevent block resets. This is a causal finite-length
    # approximation to the configured frequency responses, with explicit latency.
    taps_count = 1025
    frequency = np.linspace(0, fs/2, 2049)
    def noise_filter(spectrum):
        if spectrum:
            db = np.interp(np.log(np.maximum(frequency, 10)), np.log(spectrum['frequency_hz']), spectrum['psd_db'])
        else:
            db = np.full(frequency.shape, config['noise_level_db'])
        response = np.sqrt(fs/2 * 1e-12 * 10**(db/10))
        return StreamingFilter(firwin2(taps_count, frequency, response, fs=fs))
    receiver_noise = noise_filter(config.get('receiver_noise_spectrum'))
    source_noise_coefficients = [noise_filter(p.get('noise_spectrum')).b for p in points]
    source_white_history = np.zeros(taps_count-1)
    rng_rx = np.random.default_rng(config['random_seed'])
    rng_tx = np.random.default_rng(config['random_seed']+1)
    sensitivity = s._hydrophone_sensitivity_response_db(frequency, config)
    response = np.where(np.isfinite(sensitivity), 1e6*10**(sensitivity/20)*10**(config['preamp_gain_db']/20), 0)
    hydrophone = StreamingFilter(firwin2(taps_count, frequency, response, fs=fs))
    inverse = np.divide(1., response, out=np.zeros_like(response), where=response > 0)
    calibration = StreamingFilter(firwin2(taps_count, frequency, inverse, fs=fs))
    frequencies = np.asarray(config['frequencies_hz'])
    levels = config.get('source_band_levels_db')
    weights = np.ones(len(frequencies)) if levels is None else 10**((np.asarray(levels)-max(levels))/20)
    peaks = math.sqrt(2)*1e-6*10**(config['source_level_db']/20)*weights/np.linalg.norm(weights)
    def source(times):
        values = np.zeros(times.shape, complex)
        for f, amplitude in zip(frequencies, peaks):
            values += -1j*amplitude*np.exp(2j*np.pi*f*times)
        return values * ((times >= 0) & (times <= end))
    if config['signal_type'] != 'CW':
        raise ValueError('连续Waypoint链路当前仅支持CW/多频信号。')
    block_size = min(fs*2, s.MAX_SAMPLE_COUNT)
    count = math.ceil((end+tail+2*(taps_count-1)/fs)*fs)
    manifest = dict(job_id=job_id, request_id=config['request_id'], duration_s=count/fs,
                    waypoints=[{k: v for k, v in p.items() if k != 'noise_spectrum'} for p in public_points],
                    spacing_m=spacing, filter_latency_s=(taps_count-1)/fs,
                    path_count=len(channel.tracks), blocks=[],
                    waypoint_worker_count=min(WAYPOINT_WORKER_COUNT, len(calculation_points)),
                    fixed_source=fixed_source)
    manifest.update(channel_model='emission-time matched paths with local PCHIP',
                    reference_frequency_hz=float(np.mean(frequencies)),
                    gain_model='single-frequency Bellhop gain applied to transmitted band',
                    delay_gate_s=0.1, angle_gate_deg=20., fir_taps=taps_count,
                    sample_rate=fs, source_duration_s=end,
                    config={k: v for k, v in config.items() if k not in ('source_route', 'receiver_noise_spectrum')})
    source_wav = IncrementalWavWriter(output/'source.wav', fs)
    receiver_wav = IncrementalWavWriter(output/'receiver.wav', fs)
    receiver_monitor = StreamingMonitorNormalizer()
    try:
        for start in range(0, count, block_size):
            if cancelled.is_set():
                return
            t = np.arange(start, min(count, start+block_size))/fs
            tx = source(t).real
            rx = channel.receive(t, source).real
            if config['noise_enabled']:
                rx += receiver_noise(rng_rx.normal(size=len(t)))
                white = np.r_[source_white_history, rng_tx.normal(size=len(t))]
                source_white_history = white[-(taps_count-1):].copy()
                knots = np.asarray([p['emission_time_s'] for p in points])
                left = np.clip(np.searchsorted(knots, t, side='right')-1, 0, len(knots)-2)
                fraction = np.clip((t-knots[left])/(knots[left+1]-knots[left]), 0, 1)
                for i in np.unique(np.r_[left, left+1]):
                    shaped = fftconvolve(white, source_noise_coefficients[i], mode='valid')
                    weight = np.where(left == i, 1-fraction, 0) + np.where(left+1 == i, fraction, 0)
                    tx += weight*shaped
            voltage = hydrophone(rx)
            scale = config['adc_full_scale_v']
            max_code = (1 << (config['adc_bits']-1))-1
            codes = np.rint(np.clip(voltage/scale, -1, 1)*max_code).astype(np.int64)
            pressure = calibration(codes/max_code*scale)
            source_wav.write(tx/max(float(np.sum(peaks)), 1e-30))
            receiver_wav.write(receiver_monitor(codes/max_code))
            filename = f'block_{start:012d}.npz'
            np.savez(output/filename, source_pressure_pa=tx, received_pressure_pa=rx,
                     adc_codes=codes, adc_calibrated_pressure_pa=pressure, sample_start=start, sample_rate=fs)
            spectra = {}
            for name, values in [('source_pressure', tx), ('adc_calibrated_pressure', pressure)]:
                spec = s._spectrogram_payload(values*1e6, fs, 'dB re 1 µPa²/Hz')
                spec['time_s'] = [v+start/fs for v in spec['time_s']]
                spec['continuous'] = True
                keep = np.linspace(0, len(spec['time_s'])-1, min(32, len(spec['time_s'])), dtype=int)
                spec['time_s'] = [spec['time_s'][i] for i in keep]
                spec['psd_db'] = [[row[i] for i in keep] for row in spec['psd_db']]
                del spec['dbfs_per_hz']
                spectra[name] = spec
            result = dict(type='chunk', continuous=True, request_id=config['request_id'], job_id=job_id,
                          sample_start=start, sample_end=start+len(t), available_until_s=(start+len(t))/fs,
                          duration_s=count/fs, source_duration_s=end, fixed_source=fixed_source,
                          config={k: v for k, v in config.items() if k not in ('source_route', 'receiver_noise_spectrum')},
                          waveforms={'source_pressure_pa': s._decimate_waveform(t, tx),
                                     'adc_calibrated_pressure_pa': s._decimate_waveform(t, pressure)},
                          spectrograms=spectra, metrics={'adc_peak_code': int(np.max(np.abs(codes))),
                          'adc_clipping_percent': float(np.mean(np.abs(voltage)>scale)*100)},
                          waypoints=manifest['waypoints'],
                          files={'source_monitor_wav': f'/file/{job_id}/continuous/source.wav?v={start+len(t)}',
                                 'receiver_monitor_wav': f'/file/{job_id}/continuous/receiver.wav?v={start+len(t)}'})
            for waveform in result['waveforms'].values():
                waveform['continuous'] = True
            manifest['blocks'].append(filename)
            yield result
        (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
        yield dict(type='stream_complete', request_id=config['request_id'], job_id=job_id,
                   duration_s=count/fs, manifest=f'/file/{job_id}/continuous/manifest.json',
                   files={'source_monitor_wav': f'/file/{job_id}/continuous/source.wav',
                          'receiver_monitor_wav': f'/file/{job_id}/continuous/receiver.wav'})
    finally:
        source_wav.close()
        receiver_wav.close()
