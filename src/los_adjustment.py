# src/los_adjustment.py
"""
Line-of-Sight (LoS) adjustment module for kinematic/dynamic use cases.

Instead of the grid-based simulative approach (adjustment.py), this module
works with REAL satellite azimuth/elevation angles from an observation file.
It computes PCC corrections by interpolating the ANTEX grid at the actual
satellite positions and solves the normal equations epoch-by-epoch.

Core algorithm follows the reference implementation described in:
- Bilinear PCV interpolation (AZEL case)
- PCO projection onto line-of-sight
- PCC = -PCO_los + PCV

Reference:
    Kröger, J. (2025): Systematische Untersuchung von GNSS-Antennenkalibrierungen
    und deren Einfluss auf geodätische Parameter. Dissertation, Leibniz Universität
    Hannover.
"""

import numpy as np
import csv
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from .adjustment import _get_pcv_data
from .geodesy import iono_free_coeffs


# =============================================================================
# Signal Resolution Helper (shared logic with adjustment.py)
# =============================================================================

SYSTEM_MAP = {
    'G': 'GPS', 'E': 'Galileo', 'R': 'GLONASS',
    'C': 'BDS', 'J': 'QZSS', 'S': 'SBAS'
}


def _resolve_signal(signal: str, antex_data_1: dict, antenna_type_1: str):
    """
    Resolves a signal code into (system, sig1_code, sig2_code).
    sig2_code is None for single-frequency, set for iono-free combinations.
    Mirrors the logic in adjustment._calculate_delta_pcc_grid().
    """
    system = None
    sig1_code = None
    sig2_code = None

    if signal.startswith("IF_"):
        parts = signal.split('_')
        if len(parts) == 3:
            sig1_code = parts[1]
            sig2_code = parts[2]
            sys_char = sig1_code[0].upper()
            system = SYSTEM_MAP.get(sys_char)
            if system is None:
                raise ValueError(f"Unknown system char '{sys_char}' in signal {signal}")
        else:
            raise ValueError(f"Invalid IF-LC signal format: {signal}")

    elif signal == 'G3':
        system = 'GPS'
        sig1_code = 'G01'
        has_g02 = _get_pcv_data(antex_data_1, antenna_type_1, 'GPS', 'G02')[0] is not None
        has_g05 = _get_pcv_data(antex_data_1, antenna_type_1, 'GPS', 'G05')[0] is not None
        if has_g02:
            sig2_code = 'G02'
        elif has_g05:
            sig2_code = 'G05'
        else:
            raise ValueError("GPS G3 (IF-LC) requested but neither G02 nor G05 found.")

    elif signal in ['E00', 'E0']:
        system = 'Galileo'
        sig1_code = 'E01'
        sig2_code = 'E05'

    else:
        sys_char = signal[0].upper() if signal else None
        system = SYSTEM_MAP.get(sys_char)
        sig1_code = signal

    if system is None:
        raise ValueError(f"Could not determine system for signal {signal}")

    return system, sig1_code, sig2_code


# =============================================================================
# Observation File Parser (CSV format)
# =============================================================================

def parse_satellite_observations(filepath: str) -> Dict[datetime, List[Tuple[str, float, float]]]:
    """
    Parses a satellite observation CSV file.

    Expected CSV format:
        epoch_utc,prn,elevation_deg,azimuth_deg[,heading_deg]
        2024-01-01T00:00:00,G01,34.7,112.3
        2024-01-01T00:00:00,E05,21.2,278.9,...

    Args:
        filepath: Path to the CSV observation file.

    Returns:
        Dictionary mapping epoch (datetime) to list of (prn, azimuth_deg, elevation_deg) tuples.
        If a heading_deg column exists, it's accessible via a second return value.
    """
    epochs = {}
    headings = {}  # Optional per-epoch heading

    with open(filepath, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        # Normalize column names (strip whitespace, lowercase)
        if reader.fieldnames:
            reader.fieldnames = [name.strip().lower() for name in reader.fieldnames]

        has_heading = 'heading_deg' in (reader.fieldnames or [])

        for row in reader:
            try:
                epoch_str = row['epoch_utc'].strip()
                prn = row['prn'].strip().upper()
                el_deg = float(row['elevation_deg'].strip())
                az_deg = float(row['azimuth_deg'].strip())

                # Parse epoch — support multiple formats
                for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S',
                            '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%d %H:%M:%S.%f'):
                    try:
                        epoch_dt = datetime.strptime(epoch_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    print(f"  Warning: Could not parse epoch '{epoch_str}', skipping row.")
                    continue

                if epoch_dt not in epochs:
                    epochs[epoch_dt] = []

                epochs[epoch_dt].append((prn, az_deg, el_deg))

                if has_heading and 'heading_deg' in row and row['heading_deg'].strip():
                    headings[epoch_dt] = float(row['heading_deg'].strip())

            except (KeyError, ValueError) as e:
                print(f"  Warning: Skipping malformed row: {e}")
                continue

    print(f"  Parsed {len(epochs)} epochs, {sum(len(v) for v in epochs.values())} total observations.")
    return epochs, headings


# =============================================================================
# ATT File Parser (Attitude: Yaw/Pitch/Roll)
# =============================================================================

def _doy_sod_to_datetime(year: int, doy: int, sod: float) -> datetime:
    """
    Converts year, day-of-year, and second-of-day to a datetime object.

    Args:
        year: 4-digit year (e.g., 2023).
        doy: Day of year (1-366).
        sod: Second of day (0.0-86400.0).

    Returns:
        datetime object (UTC).
    """
    base = datetime(year, 1, 1) + timedelta(days=doy - 1)
    return base + timedelta(seconds=sod)


def parse_att_file(filepath: str) -> Tuple[Dict[float, Tuple[float, float, float]], int, int]:
    """
    Parses a kinematic attitude (.ATT) file.

    File format (fixed-width, 2-line header):
        STATION NAME   |YEAR|DOY| SECOND  |  YAW [rad]   | PITCH [rad]  |  ROLL [rad]
        ******************|****|***|*********|**************|**************|**************
        0046               2023  94 35840.100  -1.2526676992   0.0111642165   0.0092117583

    Args:
        filepath: Path to the .ATT file.

    Returns:
        Tuple of:
        - Dict mapping second-of-day (float) to (yaw_rad, pitch_rad, roll_rad)
        - year (int)
        - doy (int)
    """
    attitude_data = {}
    year = None
    doy = None
    station_ids = set()

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    # Skip 2-line header
    data_lines = lines[2:] if len(lines) > 2 else []

    for line in data_lines:
        line = line.rstrip()
        if not line or line.startswith('*'):
            continue

        try:
            # Fixed-width parsing based on observed format
            # Station: cols 0-18, Year: 19-23, DOY: 24-27, SOD: 27-37,
            # Yaw: 37-52, Pitch: 52-67, Roll: 67-82
            parts = line.split()
            if len(parts) < 7:
                continue

            # Track station ID
            station_id = parts[0].strip()
            station_ids.add(station_id)

            year_val = int(parts[1])
            doy_val = int(parts[2])
            sod = float(parts[3])
            yaw_rad = float(parts[4])
            pitch_rad = float(parts[5])
            roll_rad = float(parts[6])

            if year is None:
                year = year_val
                doy = doy_val

            # Key by second-of-day (0.1s precision)
            sod_key = round(sod, 1)
            attitude_data[sod_key] = (yaw_rad, pitch_rad, roll_rad)

        except (ValueError, IndexError):
            continue

    if len(station_ids) > 1:
        print(f"  WARNING: Multiple station IDs found in ATT file: {sorted(station_ids)}")
        print(f"           Timestamps from different stations may overlap!")

    print(f"  ATT: Parsed {len(attitude_data)} attitude records "
          f"(station(s)={sorted(station_ids)}, year={year}, doy={doy}, "
          f"SOD range: {min(attitude_data.keys()):.1f}-{max(attitude_data.keys()):.1f})")
    return attitude_data, year, doy


# =============================================================================
# KIN File Parser (Kinematic Trajectory: X/Y/Z)
# =============================================================================

def parse_kin_file(filepath: str) -> Tuple[Dict[float, Tuple[float, float, float]], int, int]:
    """
    Parses a kinematic trajectory (.KIN) file.

    File format (fixed-width, 2-line header):
        STATION NAME   |YEAR|DOY| SECOND  |    X [m]     |    Y [m]     |    Z [m]     | ...
        ******************|****|***|*********|**************|...
        0046               2023  94 37560.000   3849501.1901    649930.7711   5026986.9393  ...

    Args:
        filepath: Path to the .KIN file.

    Returns:
        Tuple of:
        - Dict mapping second-of-day (float) to (x_m, y_m, z_m)
        - year (int)
        - doy (int)
    """
    trajectory_data = {}
    year = None
    doy = None
    station_ids = set()

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    # Skip 2-line header
    data_lines = lines[2:] if len(lines) > 2 else []

    for line in data_lines:
        line = line.rstrip()
        if not line or line.startswith('*'):
            continue

        try:
            parts = line.split()
            if len(parts) < 7:
                continue

            # Track station ID
            station_id = parts[0].strip()
            station_ids.add(station_id)

            year_val = int(parts[1])
            doy_val = int(parts[2])
            sod = float(parts[3])
            x_m = float(parts[4])
            y_m = float(parts[5])
            z_m = float(parts[6])

            if year is None:
                year = year_val
                doy = doy_val

            sod_key = round(sod, 1)
            trajectory_data[sod_key] = (x_m, y_m, z_m)

        except (ValueError, IndexError):
            continue

    if len(station_ids) > 1:
        print(f"  WARNING: Multiple station IDs found in KIN file: {sorted(station_ids)}")
        print(f"           Timestamps from different stations may overlap!")

    print(f"  KIN: Parsed {len(trajectory_data)} trajectory records "
          f"(station(s)={sorted(station_ids)}, year={year}, doy={doy}, "
          f"SOD range: {min(trajectory_data.keys()):.1f}-{max(trajectory_data.keys()):.1f})")
    return trajectory_data, year, doy


# =============================================================================
# Satellite List Parser (exampleSatelliteList.txt format)
# =============================================================================

def parse_satellite_list(filepath: str) -> Dict[datetime, List[str]]:
    """
    Parses a satellite list file in the per-epoch format.

    Format (tab-delimited):
        # YYYY-MM-DD HH:MM:SS
        # YYYY-DOY-SOD
        2026-03-04\t19:00:00\t11\tG01,G02,G05,...

    Args:
        filepath: Path to the satellite list file.

    Returns:
        Dict mapping datetime to list of PRN strings.
    """
    sat_list = {}

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) < 4:
                continue

            try:
                date_str = parts[0].strip()
                time_str = parts[1].strip()
                # parts[2] = count (we derive it from the PRN list)
                prn_str = parts[3].strip()

                epoch_dt = datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M:%S')
                prns = [p.strip().upper() for p in prn_str.split(',') if p.strip()]

                sat_list[epoch_dt] = prns
            except (ValueError, IndexError):
                continue

    print(f"  SatList: Parsed {len(sat_list)} epochs.")
    return sat_list


# =============================================================================
# Euler Angle Rotation (v1.1: replaces heading-only rotation)
# =============================================================================

def apply_euler_rotation(az_deg: float, el_deg: float,
                         yaw_rad: float = 0.0, pitch_rad: float = 0.0,
                         roll_rad: float = 0.0) -> Tuple[float, float]:
    """
    Transforms sky-fixed (az, el) into the antenna body frame using full
    Euler rotation R = Rz(yaw) * Ry(pitch) * Rx(roll).

    For a static antenna with heading only (pitch=roll=0), this reduces to
    az_body = az - yaw, el_body = el (backward compatible with v1.0).

    For kinematic cases, the full rotation matrix is applied to the
    line-of-sight unit vector.

    Steps:
        1. Convert (az, el) -> unit vector e_NEU in sky frame
        2. Build R = Rz(yaw) * Ry(pitch) * Rx(roll)
        3. Transform: e_body = R^T * e_NEU   (inverse rotation)
        4. Extract (az_body, el_body) from e_body

    Args:
        az_deg: Sky-fixed satellite azimuth in degrees.
        el_deg: Sky-fixed satellite elevation in degrees.
        yaw_rad: Platform yaw (heading) in radians.
        pitch_rad: Platform pitch in radians (default 0.0).
        roll_rad: Platform roll in radians (default 0.0).

    Returns:
        Tuple of (az_body_deg, el_body_deg) in the antenna body frame.
    """
    # For the common case (yaw-only, no pitch/roll) use the fast path
    if abs(pitch_rad) < 1e-10 and abs(roll_rad) < 1e-10:
        yaw_deg = np.degrees(yaw_rad)
        az_body = (az_deg - yaw_deg) % 360.0
        return az_body, el_deg

    # Full Euler rotation path
    az_rad = np.radians(az_deg)
    el_rad = np.radians(el_deg)

    # Unit vector in NEU (North-East-Up) sky frame
    e_n = np.cos(az_rad) * np.cos(el_rad)
    e_e = np.sin(az_rad) * np.cos(el_rad)
    e_u = np.sin(el_rad)
    e_sky = np.array([e_n, e_e, e_u])

    # Build rotation matrices (NEU convention)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)

    # Rz(yaw) — rotation about Up axis
    Rz = np.array([[ cy, sy, 0],
                   [-sy, cy, 0],
                   [  0,  0, 1]])

    # Ry(pitch) — rotation about East axis
    Ry = np.array([[cp, 0, -sp],
                   [ 0, 1,   0],
                   [sp, 0,  cp]])

    # Rx(roll) — rotation about North axis
    Rx = np.array([[1,  0,   0],
                   [0,  cr,  sr],
                   [0, -sr,  cr]])

    # Combined: R = Rz * Ry * Rx (sky-to-body rotation)
    R = Rz @ Ry @ Rx
    e_body = R @ e_sky

    # Extract body-frame azimuth and elevation
    el_body_rad = np.arcsin(np.clip(e_body[2], -1.0, 1.0))
    az_body_rad = np.arctan2(e_body[1], e_body[0])

    az_body_deg = np.degrees(az_body_rad) % 360.0
    el_body_deg = np.degrees(el_body_rad)

    return az_body_deg, el_body_deg


def apply_heading_rotation(az_deg: float, heading_deg: float) -> float:
    """
    Rotates azimuth by platform heading for kinematic cases.
    Backward-compatible wrapper around apply_euler_rotation().

    Args:
        az_deg: Sky-fixed satellite azimuth in degrees.
        heading_deg: Platform heading (yaw) in degrees clockwise from North.

    Returns:
        Rotated azimuth in degrees [0, 360).
    """
    az_body, _ = apply_euler_rotation(az_deg, 0.0, yaw_rad=np.radians(heading_deg))
    return az_body


# =============================================================================
# Kinematic Data Merger
# =============================================================================

def merge_kinematic_data(
    obs_epochs: Dict[datetime, List[Tuple[str, float, float]]],
    att_data: Optional[Dict[float, Tuple[float, float, float]]] = None,
    att_year: Optional[int] = None,
    att_doy: Optional[int] = None,
    tolerance_sec: float = 0.05
) -> Dict[datetime, Tuple[float, float, float]]:
    """
    Time-matches attitude data (ATT) to observation epochs.

    For each observation epoch, finds the nearest ATT record within the
    specified tolerance and returns the Euler angles for that epoch.

    Args:
        obs_epochs: Dict {epoch_dt: [(prn, az, el), ...]} from CSV parser.
        att_data: Dict {sod: (yaw_rad, pitch_rad, roll_rad)} from ATT parser.
        att_year: Year from ATT file header.
        att_doy: DOY from ATT file header.
        tolerance_sec: Maximum time difference for matching (default 0.05s).

    Returns:
        Dict mapping epoch_dt to (yaw_rad, pitch_rad, roll_rad).
        Epochs without a matching ATT record are omitted.
    """
    if att_data is None or not att_data:
        return {}

    epoch_attitudes = {}
    att_sods = sorted(att_data.keys())
    att_sod_array = np.array(att_sods)

    matched = 0
    unmatched = 0

    for epoch_dt in obs_epochs:
        # Convert epoch datetime to second-of-day
        epoch_sod = epoch_dt.hour * 3600 + epoch_dt.minute * 60 + epoch_dt.second
        epoch_sod += epoch_dt.microsecond / 1e6
        epoch_sod = round(epoch_sod, 1)

        # Find nearest ATT record
        idx = np.searchsorted(att_sod_array, epoch_sod)

        best_sod = None
        best_diff = float('inf')

        for candidate_idx in [idx - 1, idx, idx + 1]:
            if 0 <= candidate_idx < len(att_sod_array):
                diff = abs(att_sod_array[candidate_idx] - epoch_sod)
                if diff < best_diff:
                    best_diff = diff
                    best_sod = att_sod_array[candidate_idx]

        if best_sod is not None and best_diff <= tolerance_sec:
            epoch_attitudes[epoch_dt] = att_data[best_sod]
            matched += 1
        else:
            unmatched += 1

    print(f"  ATT merge: {matched} epochs matched, {unmatched} unmatched "
          f"(tolerance={tolerance_sec}s)")
    return epoch_attitudes


# =============================================================================
# PCV Bilinear Interpolation (Port of MATLAB PCV_interpRxPCV.m, lines 325-353)
# =============================================================================

def interpolate_pcv(pcv_grid: np.ndarray, az_deg: float, zen_deg: float,
                    dazi: float, dzen: float) -> float:
    """
    Bilinear interpolation of PCV value at a specific (azimuth, zenith) point.

    Direct Python port of the AZEL interpolation case from MATLAB:
        - Grid indices: floor-based (1-indexed in MATLAB → 0-indexed here)
        - Fractional distances: p (azimuth), q (zenith)
        - 4-corner bilinear formula

    Args:
        pcv_grid: 2D PCV array, shape (n_azi, n_zen).
                  n_azi = 360/dazi + 1 (0° to 360° inclusive).
                  n_zen = 90/dzen + 1 (0° to 90° inclusive).
        az_deg: Satellite azimuth in degrees [0, 360).
        zen_deg: Satellite zenith angle in degrees [0, 90].
        dazi: Azimuth grid spacing in degrees (e.g., 5.0).
        dzen: Zenith grid spacing in degrees (e.g., 5.0).

    Returns:
        Interpolated PCV value in mm.
    """
    # Clamp inputs to valid range
    az_deg = az_deg % 360.0
    zen_deg = max(0.0, min(zen_deg, 90.0))

    # Grid indices (0-indexed, equivalent to MATLAB's floor()/dstep + 1 shifted)
    i_az = int(np.floor(az_deg / dazi))
    i_zen = int(np.floor(zen_deg / dzen))

    n_azi, n_zen = pcv_grid.shape

    # Clamp indices to valid range
    max_az_idx = n_azi - 2   # Need i_az+1 to be valid
    max_zen_idx = n_zen - 2  # Need i_zen+1 to be valid
    i_az = min(i_az, max_az_idx)
    i_zen = min(i_zen, max_zen_idx)

    # Fractional distances
    p = (az_deg - i_az * dazi) / dazi          # azimuth fraction
    q = (zen_deg - i_zen * dzen) / dzen        # zenith fraction

    # Clamp fractions to [0, 1]
    p = max(0.0, min(p, 1.0))
    q = max(0.0, min(q, 1.0))

    # Bilinear interpolation (4 corners)
    pcv = ((1 - p) * (1 - q) * pcv_grid[i_az,     i_zen]
         + (1 - p) *    q    * pcv_grid[i_az,     i_zen + 1]
         +    p    * (1 - q) * pcv_grid[i_az + 1, i_zen]
         +    p    *    q    * pcv_grid[i_az + 1, i_zen + 1])

    return float(pcv)


# =============================================================================
# PCO Projection onto Line-of-Sight (Port of MATLAB lines 360-368)
# =============================================================================

def compute_pco_los(pco_neu_mm: np.ndarray, az_rad: float, el_rad: float) -> float:
    """
    Projects the PCO vector onto the line-of-sight direction.

    Unit vector from station to satellite in NEU frame:
        e_N = cos(az) · cos(el)
        e_E = sin(az) · cos(el)
        e_U = sin(el)

    PCO_los = pco_N·e_N + pco_E·e_E + pco_U·e_U

    Args:
        pco_neu_mm: PCO vector [N, E, U] in mm.
        az_rad: Satellite azimuth in radians.
        el_rad: Satellite elevation in radians.

    Returns:
        PCO projected onto line-of-sight in mm.
    """
    e_n = np.cos(az_rad) * np.cos(el_rad)
    e_e = np.sin(az_rad) * np.cos(el_rad)
    e_u = np.sin(el_rad)

    return float(pco_neu_mm[0] * e_n + pco_neu_mm[1] * e_e + pco_neu_mm[2] * e_u)


# =============================================================================
# PCC at a Single Angle
# =============================================================================

def compute_pcc_at_angle(pco_mm: np.ndarray, pcv_grid: np.ndarray,
                         grid_step: float, az_deg: float, el_deg: float) -> float:
    """
    Computes the scalar Phase Center Correction at a single (az, el) point.

    PCC = -PCO_los + PCV

    Args:
        pco_mm: PCO vector [N, E, U] in mm.
        pcv_grid: 2D PCV grid array (n_azi x n_zen), values in mm.
        grid_step: Grid spacing in degrees (same for az and zen).
        az_deg: Satellite azimuth in degrees.
        el_deg: Satellite elevation in degrees.

    Returns:
        PCC value in mm.
    """
    zen_deg = 90.0 - el_deg
    az_rad = np.radians(az_deg)
    el_rad = np.radians(el_deg)

    pco_los = compute_pco_los(pco_mm, az_rad, el_rad)
    pcv_val = interpolate_pcv(pcv_grid, az_deg, zen_deg, grid_step, grid_step)

    return -pco_los + pcv_val


# =============================================================================
# Delta PCC for One Satellite (l-value)
# =============================================================================

def compute_delta_pcc_at_angle(
    antex_data_1: dict, antex_data_2: Optional[dict],
    antenna_type_1: str, antenna_type_2: Optional[str],
    signal: str, az_deg: float, el_deg: float,
    grid_step_1: float, grid_step_2: float
) -> float:
    """
    Computes the observation value l = PCC_1 - PCC_2 for a single satellite.

    Handles single-frequency and iono-free linear combinations.

    Args:
        antex_data_1, antex_data_2: Parsed ANTEX file data (dict).
        antenna_type_1, antenna_type_2: Resolved antenna type keys.
        signal: Signal code (e.g., 'G01', 'IF_G01_G02').
        az_deg: Satellite azimuth in degrees.
        el_deg: Satellite elevation in degrees.
        grid_step_1, grid_step_2: ANTEX grid spacing for each file.

    Returns:
        Delta PCC value in mm.
    """
    system, sig1_code, sig2_code = _resolve_signal(signal, antex_data_1, antenna_type_1)

    def _pcc_single(antex_data, antenna_type, sig_code, grid_step):
        """Get PCC for a single frequency at one angle."""
        pco, pcv = _get_pcv_data(antex_data, antenna_type, system, sig_code)
        if pco is None:
            raise ValueError(f"Missing ANTEX data for {antenna_type}/{system}/{sig_code}")
        return compute_pcc_at_angle(pco, pcv, grid_step, az_deg, el_deg)

    # --- PCC for File 1 ---
    if sig2_code is None:
        # Single frequency
        pcc_1 = _pcc_single(antex_data_1, antenna_type_1, sig1_code, grid_step_1)
    else:
        # Iono-free linear combination
        pcc_1_f1 = _pcc_single(antex_data_1, antenna_type_1, sig1_code, grid_step_1)
        pcc_1_f2 = _pcc_single(antex_data_1, antenna_type_1, sig2_code, grid_step_1)
        alpha, beta = iono_free_coeffs(sig1_code, sig2_code)
        pcc_1 = alpha * pcc_1_f1 + beta * pcc_1_f2

    # --- PCC for File 2 (or zero in single-file mode) ---
    if antex_data_2 is None:
        pcc_2 = 0.0
    elif sig2_code is None:
        pco2, pcv2 = _get_pcv_data(antex_data_2, antenna_type_2, system, sig1_code)
        if pco2 is None:
            pcc_2 = 0.0
        else:
            pcc_2 = compute_pcc_at_angle(pco2, pcv2, grid_step_2, az_deg, el_deg)
    else:
        pco2_f1, pcv2_f1 = _get_pcv_data(antex_data_2, antenna_type_2, system, sig1_code)
        pco2_f2, pcv2_f2 = _get_pcv_data(antex_data_2, antenna_type_2, system, sig2_code)
        pcc_2_f1 = compute_pcc_at_angle(pco2_f1, pcv2_f1, grid_step_2, az_deg, el_deg) if pco2_f1 is not None else 0.0
        pcc_2_f2 = compute_pcc_at_angle(pco2_f2, pcv2_f2, grid_step_2, az_deg, el_deg) if pco2_f2 is not None else 0.0
        alpha, beta = iono_free_coeffs(sig1_code, sig2_code)
        pcc_2 = alpha * pcc_2_f1 + beta * pcc_2_f2

    return pcc_1 - pcc_2


# =============================================================================
# Design Matrix Row
# =============================================================================

def build_design_row(az_deg: float, el_deg: float, sat_system: str,
                     unique_systems: list, tropo_model: str,
                     gradient_active: bool = False) -> np.ndarray:
    """
    Builds a single row of the design matrix A for one satellite observation.

    Mirrors _create_A_grids() from adjustment.py but for a single (az, el) point.

    Parameters in order: [North, East, Up, Clock_Sys1, Clock_Sys2, ..., Tropo, Tropo_Gn, Tropo_Ge]

    Args:
        az_deg: Satellite azimuth in degrees.
        el_deg: Satellite elevation in degrees.
        sat_system: System name of this satellite (e.g., 'GPS').
        unique_systems: Sorted list of all system names in the processing.
        tropo_model: Troposphere model name (or 'None').
        gradient_active: Whether troposphere gradient parameters are included.

    Returns:
        1D numpy array — one row of the design matrix.
    """
    az_rad = np.radians(az_deg)
    el_rad = np.radians(el_deg)
    zen_rad = np.pi / 2.0 - el_rad

    sin_zen = np.sin(zen_rad)
    cos_zen = np.cos(zen_rad)
    sin_az = np.sin(az_rad)
    cos_az = np.cos(az_rad)
    sin_el = np.sin(el_rad)

    # Position components (per Kröger 2025, Fig 3.6)
    a_N = -sin_zen * cos_az
    a_E = -sin_zen * sin_az
    a_U = -cos_zen

    row = [a_N, a_E, a_U]

    # Clock columns — 1 for this satellite's system, 0 for others
    for sys_name in unique_systems:
        row.append(1.0 if sys_name == sat_system else 0.0)

    # Troposphere
    tropo_active = tropo_model != 'None'
    if tropo_active:
        sin_el_safe = max(sin_el, np.sin(np.deg2rad(0.1)))

        if tropo_model == '1/sin(Elevation)':
            a_tropo = 1.0 / sin_el_safe
        elif tropo_model in ('GMF', 'VMF', 'VMF1'):
            # GMF continued fraction. NOTE: gridded TU Wien VMF1 (station/epoch aw)
            # is applied in the grid-based adjustment (adjustment.py); in this LoS
            # path VMF uses the GMF shape as a close stand-in.
            a_val = 2.53e-5
            b_val = 5.49e-3
            c_val = 1.14e-3
            den_c = sin_el_safe + c_val
            den_b = sin_el_safe + b_val / (den_c + 1e-9)
            den_a = sin_el_safe + a_val / (den_b + 1e-9)
            num_a = 1.0 + a_val / (1.0 + b_val / (1.0 + c_val))
            a_tropo = num_a / (den_a + 1e-9)
        else:
            a_tropo = 1.0 / sin_el_safe

        row.append(a_tropo)

        # Troposphere gradients (Chen & Herring 1997)
        if gradient_active:
            cos_el = np.cos(el_rad)
            C_gradient = 0.0007
            gradient_denom = sin_el ** 2 + C_gradient
            gradient_base = cos_el / gradient_denom
            row.append(gradient_base * cos_az)  # North gradient
            row.append(gradient_base * sin_az)  # East gradient

    return np.array(row, dtype=float)


# =============================================================================
# Weighting
# =============================================================================

def build_weight(el_deg: float, weighting_model: str) -> float:
    """
    Computes the scalar weight for a single satellite observation.

    Uses standard P (not double-weighted as in the grid approach).

    Args:
        el_deg: Satellite elevation in degrees.
        weighting_model: 'sin', 'sin2', or 'equal'.

    Returns:
        Scalar weight value.
    """
    el_rad = np.radians(el_deg)
    sin_el = max(np.sin(el_rad), 1e-6)

    if weighting_model == 'sin':
        return np.sqrt(sin_el)
    elif weighting_model == 'sin2':
        return sin_el
    else:
        return 1.0


# =============================================================================
# Main LoS Adjustment Engine
# =============================================================================

def perform_los_adjustment(config: dict, antex_data_1: dict,
                           antex_data_2: Optional[dict],
                           obs_epochs: Dict[datetime, List[Tuple[str, float, float]]],
                           epoch_headings: Optional[Dict[datetime, float]] = None,
                           epoch_attitudes: Optional[Dict[datetime, Tuple[float, float, float]]] = None
                           ) -> tuple:
    """
    Performs least-squares adjustment using real line-of-sight satellite angles.

    For each epoch, builds design matrix rows and accumulates the normal equations:
        N += AᵀPA    (per epoch)
        n += AᵀPl    (per epoch)

    Then solves: N · x̂ = n

    Args:
        config: Processing configuration dictionary.
        antex_data_1: Parsed ANTEX data for file 1.
        antex_data_2: Parsed ANTEX data for file 2 (None for single-file mode).
        obs_epochs: Dict {epoch_dt: [(prn, az_deg, el_deg), ...]}.
        epoch_headings: Optional dict {epoch_dt: heading_deg}.
        epoch_attitudes: Optional dict {epoch_dt: (yaw_rad, pitch_rad, roll_rad)}
            from ATT file. When present, full Euler rotation is used.

    Returns:
        Tuple of (results_vec, param_names, all_azimuths, all_elevations, all_sat_ids)
        matching the signature expected by the rest of the pipeline.
    """
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"LINE-OF-SIGHT ADJUSTMENT")
    print(f"{'='*60}")
    print(f"  Epochs: {len(obs_epochs)}")
    print(f"  Total observations: {sum(len(v) for v in obs_epochs.values())}")

    signals_to_process = config['signals']
    grid_step_1 = config.get('grid_step_1', 5.0)
    grid_step_2 = config.get('grid_step_2', 5.0)
    weighting_model = config.get('weighting', 'sin').lower()
    tropo_model = config.get('tropo_model', 'None')
    tropo_active = tropo_model != 'None'
    gradient_active = config.get('gradient_active', False)
    heading_deg = config.get('heading_deg', 0.0)

    antenna_type_1 = config['antenna_type']
    antenna_type_2 = config.get('_antenna_type_file2')

    # --- Parameter setup (mirrors adjustment.py lines 468-517) ---
    unique_systems = set()
    for sig in signals_to_process:
        if not sig:
            continue
        sys_char = None
        if sig.startswith('IF_'):
            parts = sig.split('_')
            if len(parts) >= 2 and len(parts[1]) > 0:
                sys_char = parts[1][0].upper()
        elif len(sig) > 0:
            sys_char = sig[0].upper()
        if sys_char and sys_char in SYSTEM_MAP:
            unique_systems.add(SYSTEM_MAP[sys_char])

    if not unique_systems:
        unique_systems = {'GPS'}
    unique_systems = sorted(list(unique_systems))

    param_names = ['North', 'East', 'Up']
    clock_indices = {}
    for sys_name in unique_systems:
        clock_indices[sys_name] = len(param_names)
        param_names.append(f'Clock_{sys_name}')

    if tropo_active:
        param_names.append('Tropo')
    if gradient_active and tropo_active:
        param_names.append('Tropo_Gn')
        param_names.append('Tropo_Ge')

    num_params = len(param_names)
    print(f"  Parameters ({num_params}): {param_names}")

    # --- Accumulate normal equations ---
    N_total = np.zeros((num_params, num_params))
    n_total = np.zeros(num_params)

    all_azimuths = []
    all_elevations = []
    all_sat_ids = []

    # Per-epoch results storage (for epoch dropdown display)
    epoch_results = []  # List of (epoch_dt, results_vec, n_sats)

    skipped_epochs = 0
    processed_epochs = 0
    total_obs_used = 0

    sorted_epochs = sorted(obs_epochs.keys())

    # --- Optional resampling (thin high-rate data) ---
    resample_sec = config.get('los_resample_sec')
    if resample_sec and resample_sec > 0 and len(sorted_epochs) > 1:
        original_count = len(sorted_epochs)
        resampled = [sorted_epochs[0]]
        last_kept = sorted_epochs[0]
        for ep in sorted_epochs[1:]:
            if (ep - last_kept).total_seconds() >= resample_sec:
                resampled.append(ep)
                last_kept = ep
        sorted_epochs = resampled
        # Also filter obs_epochs to match
        obs_epochs = {ep: obs_epochs[ep] for ep in sorted_epochs}
        print(f"  Resampled: {original_count} -> {len(sorted_epochs)} epochs "
              f"(interval={resample_sec}s)")

    # --- Optional obstruction mask (horizon imported from RINEX-Masker) ---
    # Applied to the observed, sky-fixed angles: the obstruction belongs to the
    # station's surroundings, not to the platform, so it is filtered before any
    # attitude rotation.
    obstruction_mask = config.get('obstruction_mask_obj')
    if obstruction_mask is not None:
        obs_before = sum(len(v) for v in obs_epochs.values())
        filtered = {}
        for ep, sat_list in obs_epochs.items():
            kept = [(prn, az, el) for prn, az, el in sat_list
                    if not obstruction_mask.is_obstructed(az, el)]
            if kept:
                filtered[ep] = kept
        obs_epochs = filtered
        sorted_epochs = sorted(obs_epochs.keys())
        obs_after = sum(len(v) for v in obs_epochs.values())
        print(f"  Obstruction mask: {obs_before - obs_after} of {obs_before} observations "
              f"removed, {obs_after} kept ({obstruction_mask.summary()})")
        if not sorted_epochs:
            raise ValueError(
                "The obstruction mask removed every observation. Check that the mask's "
                "north reference matches the observation file, and that its elevations "
                "are not higher than the whole visible sky."
            )

    # Determine if we have full Euler rotation data
    has_euler = epoch_attitudes is not None and len(epoch_attitudes) > 0
    if has_euler:
        print(f"  Using full Euler rotation from ATT data ({len(epoch_attitudes)} epochs)")
    else:
        print(f"  Using heading-only rotation (heading={heading_deg:.1f} deg)")

    for epoch_dt in sorted_epochs:
        sat_list = obs_epochs[epoch_dt]

        # Get rotation for this epoch
        epoch_yaw_rad = np.radians(heading_deg)  # default
        epoch_pitch_rad = 0.0
        epoch_roll_rad = 0.0

        if has_euler and epoch_dt in epoch_attitudes:
            epoch_yaw_rad, epoch_pitch_rad, epoch_roll_rad = epoch_attitudes[epoch_dt]
        elif epoch_headings and epoch_dt in epoch_headings:
            epoch_yaw_rad = np.radians(epoch_headings[epoch_dt])

        # Process each signal — accumulate per epoch
        for signal in signals_to_process:
            if not signal:
                continue

            # Determine which system this signal belongs to
            sys_char = None
            if signal.startswith('IF_'):
                parts = signal.split('_')
                if len(parts) >= 2 and len(parts[1]) > 0:
                    sys_char = parts[1][0].upper()
            elif len(signal) > 0:
                sys_char = signal[0].upper()
            sig_system = SYSTEM_MAP.get(sys_char, 'GPS')

            # Filter satellites to only those matching this signal's system
            system_prefix = sys_char if sys_char else 'G'
            epoch_sats = [(prn, az, el) for prn, az, el in sat_list
                          if prn[0].upper() == system_prefix]

            if len(epoch_sats) < num_params:
                skipped_epochs += 1
                continue

            # Build per-epoch matrices
            A_epoch = np.zeros((len(epoch_sats), num_params))
            P_epoch = np.zeros(len(epoch_sats))
            l_epoch = np.zeros(len(epoch_sats))

            for k, (prn, az, el) in enumerate(epoch_sats):
                # Apply Euler rotation (reduces to heading-only when pitch=roll=0)
                az_rotated, el_rotated = apply_euler_rotation(
                    az, el, epoch_yaw_rad, epoch_pitch_rad, epoch_roll_rad
                )

                # Design matrix row
                A_epoch[k, :] = build_design_row(
                    az_rotated, el_rotated, sig_system, unique_systems,
                    tropo_model, gradient_active
                )

                # Weight
                P_epoch[k] = build_weight(el_rotated, weighting_model)

                # Observation (Delta PCC)
                try:
                    l_epoch[k] = compute_delta_pcc_at_angle(
                        antex_data_1, antex_data_2,
                        antenna_type_1, antenna_type_2,
                        signal, az_rotated, el_rotated,
                        grid_step_1, grid_step_2
                    )
                except Exception as e:
                    print(f"  Warning: Failed PCC for {prn} at az={az:.1f} el={el:.1f}: {e}")
                    l_epoch[k] = 0.0
                    P_epoch[k] = 0.0  # Zero weight = excluded

                # Collect for skyplot output
                all_azimuths.append(az)
                all_elevations.append(el)
                all_sat_ids.append(prn)

            # Accumulate: N += AᵀPA, n += AᵀPl
            P_diag = np.diag(P_epoch)
            AtP = A_epoch.T @ P_diag
            N_total += AtP @ A_epoch
            n_total += AtP @ l_epoch

            total_obs_used += len(epoch_sats)

            # --- Per-epoch solution (for epoch dropdown) ---
            try:
                N_epoch = AtP @ A_epoch
                n_epoch = AtP @ l_epoch
                if not np.any(np.abs(np.diag(N_epoch)) < 1e-12):
                    x_epoch = np.linalg.solve(N_epoch, n_epoch)
                    if not np.isnan(x_epoch).any():
                        epoch_results.append((epoch_dt, x_epoch, len(epoch_sats)))
            except (np.linalg.LinAlgError, Exception):
                pass  # Epoch underdetermined — skip silently

        processed_epochs += 1

    print(f"  Processed epochs: {processed_epochs}")
    print(f"  Skipped epochs (underdetermined): {skipped_epochs}")
    print(f"  Total observations used: {total_obs_used}")
    print(f"  Per-epoch solutions: {len(epoch_results)}")

    if total_obs_used == 0:
        print("  ERROR: No observations could be processed.")
        return None, param_names, None, None, None, []

    # --- Solve normal equations ---
    try:
        if np.any(np.abs(np.diag(N_total)) < 1e-12):
            print("  ERROR: Near-zero diagonal in N. Singular system.")
            return None, param_names, None, None, None, []

        results_vec = np.linalg.solve(N_total, n_total)

    except np.linalg.LinAlgError:
        print("  ERROR: Normal matrix is singular. Cannot solve.")
        return None, param_names, None, None, None, []
    except Exception as e:
        print(f"  ERROR solving system: {e}")
        return None, param_names, None, None, None, []

    elapsed = time.time() - start_time

    if np.isnan(results_vec).any():
        print("  ERROR: Results contain NaN values.")
        return None, param_names, None, None, None, []

    print(f"\n  [OK] LoS Adjustment complete in {elapsed:.2f}s")
    print(f"  Results: {np.round(results_vec, 4)}")

    return results_vec, param_names, all_azimuths, all_elevations, all_sat_ids, epoch_results

