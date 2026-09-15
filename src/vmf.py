# src/vmf.py
"""
Gridded Vienna Mapping Function (VMF1) support for PCC-Explorer.

Downloads the operational TU Wien VMF1 grid products (6-hourly, 2.0 deg lat x
2.5 deg lon), interpolates the wet mapping coefficient ``aw`` to a station
location/epoch, and evaluates the VMF1 wet mapping function.

In PCC-Explorer's least-squares adjustment the estimated troposphere parameter
is the zenith *wet* delay, so the design-matrix column uses the **wet** mapping
function ``mf_w(E)``. For VMF1 the wet continued-fraction coefficients ``b`` and
``c`` are constants (Boehm et al., 2006), so only the grid-supplied ``aw`` is
needed:

    mf_w(E) = (1 + aw/(1 + bw/(1 + cw)))
              -------------------------------------------------
              (sin E + aw/(sin E + bw/(sin E + cw)))

    bw = 0.00146,  cw = 0.04391

Reference:
    Boehm, J., Werl, B., Schuh, H. (2006): Troposphere mapping functions for GPS
    and very long baseline interferometry from European Centre for Medium-Range
    Weather Forecasts operational analysis data. JGR, 111, B02406.

Data source:
    https://vmf.geo.tuwien.ac.at/trop_products/GRID/2.5x2/VMF1/VMF1_OP/
"""
from __future__ import annotations

import os
import math
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

import numpy as np

try:  # NASA-style download helper already used elsewhere in the project
    import urllib.request as _urlreq
except Exception:  # pragma: no cover
    _urlreq = None

# VMF1 wet continued-fraction constants (Boehm et al., 2006).
BW_WET = 0.00146
CW_WET = 0.04391

VMF1_BASE_URL = ("https://vmf.geo.tuwien.ac.at/trop_products/GRID/2.5x2/"
                 "VMF1/VMF1_OP/{year}/VMFG_{ymd}.H{hh:02d}")

# In-memory cache of parsed grids, keyed by (date_str, hour) -> parsed dict.
_GRID_CACHE: Dict[Tuple[str, int], dict] = {}


# =============================================================================
# Download + parse
# =============================================================================
def download_vmf1_grid(epoch: datetime, vmf_dir: str, timeout: int = 60) -> Optional[str]:
    """Download one 6-hourly VMF1 grid file (cached). Returns local path or None.

    Args:
        epoch: UTC epoch; only the date and the 6-hourly hour (0/6/12/18) are used.
        vmf_dir: Local cache directory (created if missing).
        timeout: HTTP timeout in seconds.
    """
    hh = (epoch.hour // 6) * 6
    ymd = epoch.strftime("%Y%m%d")
    fname = f"VMFG_{ymd}.H{hh:02d}"
    os.makedirs(vmf_dir, exist_ok=True)
    local = os.path.join(vmf_dir, fname)
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return local

    url = VMF1_BASE_URL.format(year=epoch.year, ymd=ymd, hh=hh)
    if _urlreq is None:
        return None
    try:
        with _urlreq.urlopen(url, timeout=timeout) as resp:
            data = resp.read()
        if not data:
            return None
        with open(local, "wb") as f:
            f.write(data)
        return local
    except Exception as exc:  # pragma: no cover - network failure path
        print(f"  [VMF] download failed for {url}: {exc}")
        return None


def parse_vmf1_grid(path: str) -> dict:
    """Parse a VMF1 grid file into ascending-ordered lat/lon axes and 2-D grids.

    Returns a dict with keys ``lat`` (nlat,), ``lon`` (nlon,), ``ah`` and ``aw``
    each shaped (nlat, nlon), with latitude and longitude ascending.
    """
    raw = np.loadtxt(path, comments="!")  # cols: lat lon ah aw zhd zwd
    lat_vals = raw[:, 0]
    lon_vals = raw[:, 1]
    lats = np.unique(lat_vals)            # ascending
    lons = np.unique(lon_vals)            # ascending (0 .. 360)
    nlat, nlon = lats.size, lons.size

    lat_idx = np.searchsorted(lats, lat_vals)
    lon_idx = np.searchsorted(lons, lon_vals)
    ah = np.full((nlat, nlon), np.nan)
    aw = np.full((nlat, nlon), np.nan)
    ah[lat_idx, lon_idx] = raw[:, 2]
    aw[lat_idx, lon_idx] = raw[:, 3]
    return {"lat": lats, "lon": lons, "ah": ah, "aw": aw}


def _get_grid(epoch: datetime, vmf_dir: str) -> Optional[dict]:
    """Return parsed grid for the 6-hourly slot containing ``epoch`` (cached)."""
    hh = (epoch.hour // 6) * 6
    key = (epoch.strftime("%Y%m%d"), hh)
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]
    slot = epoch.replace(hour=hh, minute=0, second=0, microsecond=0)
    path = download_vmf1_grid(slot, vmf_dir)
    if not path:
        return None
    grid = parse_vmf1_grid(path)
    _GRID_CACHE[key] = grid
    return grid


# =============================================================================
# Interpolation
# =============================================================================
def _bilinear(lats: np.ndarray, lons: np.ndarray, field: np.ndarray,
              lat: float, lon: float) -> float:
    """Bilinear interpolation of ``field`` (nlat, nlon) at (lat, lon).

    ``lats``/``lons`` ascending; ``lon`` is normalised to [0, 360).
    """
    lon = lon % 360.0
    i = int(np.clip(np.searchsorted(lats, lat) - 1, 0, lats.size - 2))
    j = int(np.clip(np.searchsorted(lons, lon) - 1, 0, lons.size - 2))
    la0, la1 = lats[i], lats[i + 1]
    lo0, lo1 = lons[j], lons[j + 1]
    fy = 0.0 if la1 == la0 else (lat - la0) / (la1 - la0)
    fx = 0.0 if lo1 == lo0 else (lon - lo0) / (lo1 - lo0)
    f00, f01 = field[i, j], field[i, j + 1]
    f10, f11 = field[i + 1, j], field[i + 1, j + 1]
    return float((f00 * (1 - fx) + f01 * fx) * (1 - fy)
                 + (f10 * (1 - fx) + f11 * fx) * fy)


def aw_at(epoch: datetime, lat_deg: float, lon_deg: float, vmf_dir: str) -> Optional[float]:
    """Wet mapping coefficient ``aw`` at a station, time-interpolated between the
    two bracketing 6-hourly VMF1 grids."""
    hh0 = (epoch.hour // 6) * 6
    lower = epoch.replace(hour=hh0, minute=0, second=0, microsecond=0)
    upper = lower + timedelta(hours=6)
    g0 = _get_grid(lower, vmf_dir)
    g1 = _get_grid(upper, vmf_dir)
    if g0 is None and g1 is None:
        return None
    if g1 is None:
        g1, upper = g0, lower
    if g0 is None:
        g0, lower = g1, upper
    a0 = _bilinear(g0["lat"], g0["lon"], g0["aw"], lat_deg, lon_deg)
    a1 = _bilinear(g1["lat"], g1["lon"], g1["aw"], lat_deg, lon_deg)
    span = (upper - lower).total_seconds()
    w = 0.0 if span == 0 else (epoch - lower).total_seconds() / span
    return a0 * (1 - w) + w * a1


def station_daily_aw(date: datetime, lat_deg: float, lon_deg: float,
                     vmf_dir: str) -> Optional[float]:
    """Mean ``aw`` for a station over the four 6-hourly epochs of ``date``.

    A single representative value is adequate for the adjustment's troposphere
    design column (aw varies slowly and largely cancels in GAL-GPS differences).
    """
    day = date.replace(hour=0, minute=0, second=0, microsecond=0)
    vals = []
    for hh in (0, 6, 12, 18):
        a = aw_at(day + timedelta(hours=hh), lat_deg, lon_deg, vmf_dir)
        if a is not None and np.isfinite(a):
            vals.append(a)
    return float(np.mean(vals)) if vals else None


# =============================================================================
# Mapping function
# =============================================================================
def vmf1_wet_mapping(aw: float, elevation_deg: float) -> float:
    """VMF1 wet mapping function value at a given elevation angle (degrees)."""
    e = math.radians(max(elevation_deg, 0.1))
    sin_e = math.sin(e)
    num = 1.0 + aw / (1.0 + BW_WET / (1.0 + CW_WET))
    den = sin_e + aw / (sin_e + BW_WET / (sin_e + CW_WET))
    return num / den
