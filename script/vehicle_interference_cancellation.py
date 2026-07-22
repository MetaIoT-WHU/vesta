#!/usr/bin/env python3
"""GNSS carrier-phase vehicle interference cancellation.

Expects raw receiver measurements in the JSON (no offline clock or motion
pre-correction). Vehicle motion-induced interference and clock drift are
estimated online from observations via least squares.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from constants import GnssConstants

ROOT = _SCRIPT_DIR.parent
DEFAULT_DATA = ROOT / "data" / "demo_data.json"
DEFAULT_FIG_DIR = ROOT / "figures"
DISPLAY_SATS = ["E29", "G27"]  # satellites plotted in the open-source demo

# Observation grid
FS_HZ = 20  # Hz; sync_time step = 1000 / FS_HZ ms

# Estimation quality gates
CN0_MIN = 35.0   # dB-Hz
ADR_MAX = 0.01   # m; max |delta_phase_d| per satellite
MIN_SATS = 4     # min satellites per epoch for LS solve

# Phase display smoothing
SMOOTH_ADR = 20  # phase smoothing window for display (samples)


def band_of(sat: str) -> str:
    if sat.startswith("B"):
        return "B1"
    if sat.startswith("G"):
        return "L1"
    if sat.startswith("E"):
        return "E1"
    raise ValueError(sat)


def m2r_for_sat(sat: str) -> float:
    band = band_of(sat)
    if band == "B1":
        return GnssConstants.M2R_B1
    if band == "L1":
        return GnssConstants.M2R_L1
    return GnssConstants.M2R_E1


def freq_param(band: str):
    if band == "L1":
        return GnssConstants.WAVE_LENGTH_L1, "G", GnssConstants.FREQID_L1_ID
    if band == "B1":
        return GnssConstants.WAVE_LENGTH_B1, "B", GnssConstants.FREQID_B1_ID
    return GnssConstants.WAVE_LENGTH_E1, "E", GnssConstants.FREQID_E1_ID


def parse_band(raw: dict, sync_time: np.ndarray, band: str) -> dict:
    """Build per-band matrices: rows = sync_time epochs, cols = satellites."""
    wl, head, fid = freq_param(band)
    time_idx = np.asarray(raw["time_idx"], dtype=int)   # index into sync_time
    svid = np.asarray(raw["svid"], dtype=object)        # satellite id strings
    flag = np.asarray(raw["freq_flag"])                 # receiver frequency id
    adr = np.asarray(raw["adr"], dtype=float)           # accumulated carrier cycles
    cn0 = np.asarray(raw["cn0"], dtype=float)           # C/N0 (dB-Hz)
    # freq_flag alone does not separate B1/L1 (both 0); svid prefix disambiguates
    mask = (flag == fid) & np.array([str(s).startswith(head) for s in svid])
    if not np.any(mask):
        return {}
    time_idx, svid, adr, cn0 = time_idx[mask], svid[mask], adr[mask], cn0[mask]
    times = sync_time[np.unique(time_idx)]
    sats = np.unique(svid)
    t_idx = {int(i): k for k, i in enumerate(np.unique(time_idx))}
    sat_idx = {s: i for i, s in enumerate(sats)}
    adr_m = np.full((len(times), len(sats)), np.nan)
    cn0m = np.full((len(times), len(sats)), np.nan)
    for ti, s, a, c in zip(time_idx, svid, adr, cn0):
        row = t_idx[int(ti)]
        col = sat_idx[s]
        adr_m[row, col] = a * wl  # accumulated carrier phase, meters
        cn0m[row, col] = c        # C/N0, dB-Hz
    return {
        "sync_time": times,   # epoch times (ms)
        "svid": sats,         # satellite ids for columns
        "adr_m": adr_m,       # [epoch, sat] phase in meters
        "cn0": cn0m,          # [epoch, sat] C/N0 in dB-Hz
    }


def parse_satinfo(raw: dict) -> dict:
    sync_time = np.asarray(raw["sync_time"], dtype=float)
    time_idx = np.asarray(raw["time_idx"], dtype=int)
    svid = np.asarray(raw["svid"], dtype=object)
    az = np.asarray(raw["az"], dtype=float)
    el = np.asarray(raw["el"], dtype=float)
    n_epoch = len(sync_time)
    max_sat = max(int(np.sum(time_idx == i)) for i in range(n_epoch))
    sv = np.full((n_epoch, max_sat), "", dtype=object)
    azm = np.full((n_epoch, max_sat), np.nan)
    elm = np.full((n_epoch, max_sat), np.nan)
    for i in range(n_epoch):
        ix = np.where(time_idx == i)[0]
        n = len(ix)
        sv[i, :n] = svid[ix]
        azm[i, :n] = az[ix]
        elm[i, :n] = el[ix]
    return {"sync_time": sync_time, "svid": sv, "az": azm, "el": elm}  # az/el in degrees


def load_measurement(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    sync_time = np.asarray(raw["sync_time"], dtype=float)
    return {
        "sync_time": sync_time,
        "sensing_mes": {b: parse_band(raw["sensing_mes"], sync_time, b) for b in ("B1", "L1", "E1")},
        "reference_mes": {b: parse_band(raw["reference_mes"], sync_time, b) for b in ("B1", "L1", "E1")},
        "sat_geometry": parse_satinfo(raw["sat_geometry"]),
    }


def geometry_epoch(mes: dict, t_ms: float) -> int:
    """Map an observation time to the nearest geometry epoch (1 Hz grid)."""
    times = np.asarray(mes["sat_geometry"]["sync_time"], dtype=float)
    return int(np.argmin(np.abs(times - t_ms)))


def sat_az_el(mes: dict, sat: str, geo_idx: int) -> Tuple[float, float]:
    info = mes["sat_geometry"]
    c = np.where(info["svid"][geo_idx] == sat)[0]
    if len(c) == 0:
        raise KeyError(f"{sat} not visible at geometry epoch {geo_idx}")
    c = c[0]
    return float(info["az"][geo_idx, c]), float(info["el"][geo_idx, c])


def sat_signal(band_mes: dict, sat: str) -> dict:
    """Per-satellite time series used by estimation / plotting."""
    d = band_mes[band_of(sat)]
    j = np.where(d["svid"] == sat)[0][0]
    adr_m = d["adr_m"][:, j]          # accumulated carrier phase (m)
    # delta_phase = diff(adr_m)  [m / sample]
    delta_phase = np.diff(adr_m)
    delta_phase = np.append(delta_phase, delta_phase[-1])  # pad for per-epoch LS
    return {
        "adr_m": adr_m,
        "cn0": d["cn0"][:, j],        # C/N0 (dB-Hz)
        "delta_phase": delta_phase,
    }


def dir_vec(az: float, el: float) -> np.ndarray:
    azr, elr = np.deg2rad(az), np.deg2rad(el)
    return np.array([np.cos(elr) * np.sin(azr), np.cos(elr) * np.cos(azr), np.sin(elr)])


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    k = np.ones(w)
    return np.convolve(x, k, "same") / np.convolve(np.ones_like(x), k, "same")


def compensate_phase(phase_diff: np.ndarray, rot_effect: np.ndarray, clock_bias: np.ndarray) -> np.ndarray:
    """Remove cumulative rotation + clock effects from differential phase."""
    return smooth(phase_diff - np.cumsum(rot_effect + clock_bias), SMOOTH_ADR)


def select_sats(mes: dict, geo_idx: int) -> np.ndarray:
    row = mes["sat_geometry"]["svid"][geo_idx]
    row = row[row != ""]
    out = []
    for b in ("B1", "L1", "E1"):
        if not mes["sensing_mes"][b] or not mes["reference_mes"][b]:
            continue
        c = np.intersect1d(mes["sensing_mes"][b]["svid"], mes["reference_mes"][b]["svid"])
        out.extend(np.intersect1d(c, row).tolist())
    return np.unique(np.array(out, dtype=object))


def time_grid(mes: dict) -> np.ndarray:
    sync_time = np.asarray(mes["sync_time"], dtype=float)
    step_ms = 1000.0 / FS_HZ
    grid = np.arange(sync_time[0], sync_time[-1] + 1e-9, step_ms)
    if len(grid) != len(sync_time) or not np.allclose(grid, sync_time):
        raise ValueError(f"sync_time must be uniform {FS_HZ} Hz grid, got {len(sync_time)} samples")
    return sync_time


def solve_rotation_clock_ls(los: np.ndarray, obs: np.ndarray, ix: np.ndarray) -> Optional[np.ndarray]:
    """L2 least squares: obs_i ~= los_i . rotation_vec + clock_bias."""
    if len(ix) < MIN_SATS:
        return None
    a = np.column_stack([los[ix], np.ones(len(ix))])
    b = obs[ix]
    try:
        return np.linalg.solve(a.T @ a, a.T @ b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(a, b, rcond=None)[0]


def los_at_geometry(mes: dict, sats: np.ndarray, geo_idx: int) -> np.ndarray:
    los = np.full((len(sats), 3), np.nan)
    for i, s in enumerate(sats):
        try:
            az, el = sat_az_el(mes, s, geo_idx)
            los[i] = dir_vec(az, el)
        except KeyError:
            pass
    return los


def estimate(mes, sats, grid: np.ndarray):
    """Per-epoch LS for vehicle motion and clock drift."""
    n = len(grid)
    cn0m = np.zeros((len(sats), n))
    # Phi_d = Phi_ref - Phi_sig;  delta_phase_d = diff(Phi_d)  [m / sample]
    delta_phase_d = np.zeros((len(sats), n))
    for i, s in enumerate(sats):
        sig = sat_signal(mes["sensing_mes"], s)
        ref = sat_signal(mes["reference_mes"], s)
        cn0m[i] = sig["cn0"]                 # sensing C/N0 (dB-Hz)
        delta_phase_d[i] = ref["delta_phase"] - sig["delta_phase"]

    mask = (cn0m >= CN0_MIN) & (np.abs(delta_phase_d) <= ADR_MAX)
    gated = delta_phase_d * mask
    rotation_vec = np.zeros((n, 3))
    clock_bias = np.zeros(n)
    for t in range(n):
        geo_idx = geometry_epoch(mes, grid[t])
        los_t = los_at_geometry(mes, sats, geo_idx)
        ix = np.array(
            [i for i in np.where(mask[:, t])[0] if np.all(np.isfinite(los_t[i]))],
            dtype=int,
        )
        x = solve_rotation_clock_ls(los_t, gated[:, t], ix)
        if x is None:
            continue
        rotation_vec[t] = x[:3]
        clock_bias[t] = x[3]
    return rotation_vec, clock_bias


def save_line(x, y, path: Path, ylabel: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9.8, 7.35))
    plt.plot(x, y, linewidth=2, color=GnssConstants.COLOR_BLUE)
    plt.xlabel("Samples")
    plt.ylabel(ylabel)
    plt.grid(axis="y")
    plt.gca().tick_params(labelsize=32)
    plt.tight_layout()
    plt.savefig(path, format="svg")
    plt.close()
    print(path)


def plot_phase(mes, rotation_vec, clock_bias, grid: np.ndarray, sat: str, out_dir: Path):
    """Plot phase before/after cancellation for one satellite."""
    n = len(grid)
    sig = sat_signal(mes["sensing_mes"], sat)
    ref = sat_signal(mes["reference_mes"], sat)
    phase_diff = ref["adr_m"] - sig["adr_m"]
    los = np.zeros((n, 3))
    for t in range(n):
        geo_idx = geometry_epoch(mes, grid[t])
        az, el = sat_az_el(mes, sat, geo_idx)
        los[t] = dir_vec(az, el)
    rot_effect = np.einsum("ij,ij->i", rotation_vec, los)
    compensated = compensate_phase(phase_diff, rot_effect, clock_bias)
    phase_display = smooth(phase_diff, SMOOTH_ADR)
    m2r = m2r_for_sat(sat)
    x = np.arange(1, n + 1)
    before = m2r * (phase_display - phase_display[0])
    after = m2r * (compensated - compensated[0])
    save_line(x, after, out_dir / "phase_after_compensation.svg", "Carrier Phase (rad)")
    save_line(x, before, out_dir / "phase_before_compensation.svg", "Carrier Phase (rad)")


def main():
    parser = argparse.ArgumentParser(description="Vehicle interference cancellation")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="measurement JSON path")
    parser.add_argument("--output", type=Path, default=DEFAULT_FIG_DIR, help="output directory for phase plots")
    args = parser.parse_args()

    mes = load_measurement(args.data)
    grid = time_grid(mes)
    geo_idx = geometry_epoch(mes, grid[0])
    sats = select_sats(mes, geo_idx)
    missing = [sat for sat in DISPLAY_SATS if sat not in sats]
    if missing:
        raise SystemExit(
            f"display satellite(s) {', '.join(missing)} not in dataset "
            f"(available: {', '.join(map(str, sats))})"
        )

    rotation_vec, clock_bias = estimate(mes, sats, grid)
    print(f"data: {args.data}")
    for sat in DISPLAY_SATS:
        out_dir = args.output / sat
        print(f"display satellite: {sat} -> {out_dir}")
        plot_phase(mes, rotation_vec, clock_bias, grid, sat, out_dir)


if __name__ == "__main__":
    main()
