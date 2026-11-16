"""Emission-time path tracking and retarded-time channel evaluation.

This module does not infer identity from Bellhop record ordering. Units are
seconds, degrees and linear complex pressure gain.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import linear_sum_assignment


def build_waypoints(route, speed_kn, spacing_m=50.0):
    """Sample a geographic polyline by distance, retaining turns and endpoint."""
    if not np.isfinite(speed_kn) or speed_kn <= 0 or not np.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError('Speed and Waypoint spacing must be positive')
    if len(route) < 2:
        raise ValueError('A moving route requires at least two points')
    coordinates = np.asarray([[p['lon'], p['lat']] for p in route], float)
    if not np.all(np.isfinite(coordinates)) or np.any(np.abs(coordinates[:, 1]) > 90) or np.any(np.abs(coordinates[:, 0]) > 180):
        raise ValueError('Invalid route coordinates')
    radians = np.deg2rad(coordinates)
    lon, lat = radians.T
    vectors = np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))
    angles = np.arctan2(np.linalg.norm(np.cross(vectors[:-1], vectors[1:]), axis=1),
                        np.sum(vectors[:-1]*vectors[1:], axis=1))
    if np.any(angles > np.pi - 1e-6):
        raise ValueError('Antipodal route segment is ambiguous')
    distances = np.r_[0., np.cumsum(6371008.8*angles)]
    total = distances[-1]
    if total <= 0:
        raise ValueError('Moving route has zero length')
    targets = np.unique(np.r_[np.arange(0, total, spacing_m), distances])
    result = []
    for distance in targets:
        i = min(len(route)-2, np.searchsorted(distances, distance, side='right')-1)
        span = distances[i+1]-distances[i]
        fraction = (distance-distances[i])/span if span else 0
        angle = angles[i]
        vector = (np.sin((1-fraction)*angle)*vectors[i] + np.sin(fraction*angle)*vectors[i+1])/np.sin(angle) if angle > 1e-12 else vectors[i]
        depth = (1-fraction)*float(route[i].get('depth_m', 6)) + fraction*float(route[i+1].get('depth_m', 6))
        result.append(dict(lon=float(np.rad2deg(np.arctan2(vector[1], vector[0]))),
                           lat=float(np.rad2deg(np.arctan2(vector[2], np.hypot(vector[0], vector[1])))),
                           depth_m=depth, distance_m=float(distance),
                           emission_time_s=float(distance/(speed_kn*1852/3600))))
    return result


def match_paths(left, right, delay_gate_s=0.1, angle_gate_deg=20.0):
    """Gated assignment with explicit unmatched choices (birth/death)."""
    n, m = len(left), len(right)
    if not n or not m:
        return [], list(range(n)), list(range(m))
    cost = np.full((n + m, n + m), 1e6)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if any(a.get(k) != b.get(k) for k in ('top_bounces', 'bottom_bounces')):
                continue
            delay = abs(float(a['delay_s']) - float(b['delay_s'])) / delay_gate_s
            differences = []
            for key in ('source_angle_deg', 'arrival_angle_deg'):
                if a.get(key) is not None and b.get(key) is not None:
                    differences.append(abs(float(a[key]) - float(b[key])) / angle_gate_deg)
            if delay > 1 or any(d > 1 for d in differences):
                continue
            cost[i, j] = (delay ** 2 + sum(d*d for d in differences)) / (1 + len(differences))
    cost[:n, m:] = 0.6
    cost[n:, :m] = 0.6
    cost[n:, m:] = 0
    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(i), int(j)) for i, j in zip(rows, cols)
             if i < n and j < m and cost[i, j] < 1.2]
    return pairs, sorted(set(range(n)) - {i for i, _ in pairs}), sorted(set(range(m)) - {j for _, j in pairs})


def track_paths(waypoints, **gates):
    """Return persistent path histories while retaining original arrival records."""
    tracks = []
    previous = []
    active = {}
    for w, waypoint in enumerate(waypoints):
        arrivals = waypoint['arrivals']
        pairs, _, born = match_paths(previous, arrivals, **gates)
        next_active = {}
        for i, j in pairs:
            identity = active[i]
            tracks[identity].append((w, arrivals[j]))
            next_active[j] = identity
        for j in born:
            next_active[j] = len(tracks)
            tracks.append([(w, arrivals[j])])
        previous, active = arrivals, next_active
    return tracks


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x*x*(3 - 2*x)


class WaypointChannel:
    """Local shape-preserving cubic paths and a monotonic arrival-time inverse.

    source(te) must return the analytic transmitted signal at arbitrary emission
    times. Real pressure is the real part of receive(). No block looping occurs.
    """
    def __init__(self, waypoints, **gates):
        self.times = np.asarray([w['emission_time_s'] for w in waypoints], float)
        if len(self.times) < 2 or not np.all(np.isfinite(self.times)) or np.any(np.diff(self.times) <= 0):
            raise ValueError('Waypoint emission times must be finite and strictly increasing')
        self.tracks = track_paths(waypoints, **gates)
        self.paths = []
        for identity, track in enumerate(self.tracks):
            indices = [i for i, _ in track]
            records = [r for _, r in track]
            t = self.times[indices]
            delay = np.asarray([r['delay_s'] for r in records], float)
            gain = np.asarray([complex(r['gain_real'], r['gain_imag']) for r in records])
            if not np.all(np.isfinite(delay)) or np.any(delay < 0) or not np.all(np.isfinite(gain)):
                raise ValueError('Invalid raw arrival delay or gain')
            start, stop = t[0], t[-1]
            # An unobserved birth/death is approximated over the adjacent interval.
            # Keep delay constant on that interval; only the amplitude fades.
            if indices[0] > 0:
                t = np.r_[self.times[indices[0]-1], t]
                delay = np.r_[delay[0], delay]
                gain = np.r_[gain[0], gain]
            if indices[-1] < len(self.times)-1:
                t = np.r_[t, self.times[indices[-1]+1]]
                delay = np.r_[delay, delay[-1]]
                gain = np.r_[gain, gain[-1]]
            spline = PchipInterpolator(t, delay)
            # Check exact extrema of each quadratic derivative, not only knots.
            for k, dt in enumerate(np.diff(t)):
                a, b, c = 3*spline.c[0, k], 2*spline.c[1, k], spline.c[2, k]
                probes = [0., dt]
                if a != 0 and 0 < -b/(2*a) < dt:
                    probes.append(-b/(2*a))
                if min(1 + a*x*x + b*x + c for x in probes) <= 0:
                    raise ValueError('Non-monotonic arrival time; refine Waypoints or reject path match')
            self.paths.append(dict(id=identity, t=t, start=start, stop=stop, delay=spline,
                                   amplitude=PchipInterpolator(t, np.abs(gain)),
                                   phase=PchipInterpolator(t, np.unwrap(np.angle(gain)))))

    def receive(self, receiver_times, source):
        tr = np.asarray(receiver_times, float)
        result = np.zeros(tr.shape, complex)
        for path in self.paths:
            lo_t, hi_t = path['t'][0], path['t'][-1]
            delay = path['delay']
            valid = (tr >= lo_t + delay(lo_t)) & (tr <= hi_t + delay(hi_t))
            if not np.any(valid):
                continue
            target = tr[valid]
            lo, hi = np.full(target.shape, lo_t), np.full(target.shape, hi_t)
            # Retarded-time inversion is independent of output block boundaries.
            for _ in range(48):
                mid = (lo + hi)/2
                below = mid + delay(mid) < target
                lo = np.where(below, mid, lo)
                hi = np.where(below, hi, mid)
            te = (lo + hi)/2
            envelope = np.ones(te.shape)
            if path['start'] > lo_t:
                envelope *= _smoothstep((te-lo_t)/(path['start']-lo_t))
            if path['stop'] < hi_t:
                envelope *= _smoothstep((hi_t-te)/(hi_t-path['stop']))
            gain = path['amplitude'](te) * np.exp(1j*path['phase'](te))
            result[valid] += envelope * gain * source(te)
        return result
