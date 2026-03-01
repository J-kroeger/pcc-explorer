# src/geodesy.py
"""
Geodetic transformations and satellite position calculations.
"""

import numpy as np
from datetime import datetime, timedelta
import math
from scipy.interpolate import RegularGridInterpolator
import pyproj
from typing import Tuple

# --- Constants ---
WGS84_A = 6378137.0         # WGS84 semi-major axis
WGS84_F = 1 / 298.257223563 # WGS84 flattening
WGS84_E2 = WGS84_F * (2 - WGS84_F) # WGS84 first eccentricity squared
GM = 3.986005e14            # Earth gravitational constant (WGS84) [m^3/s^2]
OMEGA_E = 7.2921151467e-5   # Earth rotation rate (WGS84) [rad/s]
C = 299792458.0             # Speed of light [m/s]

# --- Coordinate Transformations ---

# Create a pyproj transformer for WGS84 ECEF <-> Ellipsoidal
ecef_proj = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla_proj = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
transformer_ecef_to_lla = pyproj.Transformer.from_proj(ecef_proj, lla_proj, always_xy=True)
transformer_lla_to_ecef = pyproj.Transformer.from_proj(lla_proj, ecef_proj, always_xy=True)

def ell_to_ecef(lat_rad: float, lon_rad: float, h: float) -> np.ndarray:
    """
    Converts ellipsoidal coordinates (lat, lon, h) to ECEF (x, y, z).
    Uses pyproj for a robust implementation.
    """
    lon_deg = np.degrees(lon_rad)
    lat_deg = np.degrees(lat_rad)
    x, y, z = transformer_lla_to_ecef.transform(lon_deg, lat_deg, h)
    return np.array([x, y, z])

def ecef_to_ell(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Converts ECEF (x, y, z) to ellipsoidal coordinates (lat, lon, h).
    Uses pyproj for a robust implementation.
    """
    lon_deg, lat_deg, h = transformer_ecef_to_lla.transform(x, y, z)
    return np.radians(lat_deg), np.radians(lon_deg), h

def ecef_to_topocentric(station_ecef: np.ndarray, target_ecef: np.ndarray) -> Tuple[float, float]:
    """
    Calculates azimuth and elevation from a station to a target (e.g., satellite)
    given their ECEF coordinates.
    """
    lat_rad, lon_rad, _ = ecef_to_ell(station_ecef[0], station_ecef[1], station_ecef[2])
    vec_ecef = target_ecef - station_ecef
    
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    R = np.array([
        [-sin_lon,           cos_lon,             0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon,  cos_lat],
        [ cos_lat * cos_lon,  cos_lat * sin_lon,  sin_lat]
    ])
    
    vec_neu = R.dot(vec_ecef)
    n, e, u = vec_neu[0], vec_neu[1], vec_neu[2]
    horizontal_dist = np.sqrt(n**2 + e**2)
    
    if horizontal_dist < 1e-9:
        azimuth_rad = 0.0
        elevation_rad = np.pi / 2.0
    else:
        azimuth_rad = np.arctan2(n, e)
        elevation_rad = np.arctan2(u, horizontal_dist)
        
    azimuth_deg = np.degrees(azimuth_rad)
    azimuth_deg = azimuth_deg % 360.0
    
    return azimuth_deg, np.degrees(elevation_rad)

# --- Satellite Position ---

def compute_satellite_position(eph: dict, t_gps_sow: float) -> np.ndarray:
    """
    Computes satellite ECEF position from broadcast ephemeris at a given time.
    """
    t_k = t_gps_sow - eph['Toe']
    if t_k > 302400: t_k -= 604800
    if t_k < -302400: t_k += 604800
        
    n_0 = np.sqrt(GM / (eph['sqrt_A']**6))
    n = n_0 + eph['Delta_n']
    M_k = eph['M0'] + n * t_k
    
    E_k = M_k
    for _ in range(10):
        E_new = M_k + eph['e'] * np.sin(E_k)
        if abs(E_new - E_k) < 1e-12:
            break
        E_k = E_new
    
    v_k = np.arctan2(np.sqrt(1 - eph['e']**2) * np.sin(E_k), np.cos(E_k) - eph['e'])
    Phi_k = v_k + eph['omega']
    
    du_k = eph['Cus'] * np.sin(2 * Phi_k) + eph['Cuc'] * np.cos(2 * Phi_k)
    dr_k = eph['Crs'] * np.sin(2 * Phi_k) + eph['Crc'] * np.cos(2 * Phi_k)
    di_k = eph['Cis'] * np.sin(2 * Phi_k) + eph['Cic'] * np.cos(2 * Phi_k)
    
    u_k = Phi_k + du_k
    r_k = (eph['sqrt_A']**2) * (1 - eph['e'] * np.cos(E_k)) + dr_k
    i_k = eph['i0'] + di_k + eph['i_dot'] * t_k
    
    x_prime_k = r_k * np.cos(u_k)
    y_prime_k = r_k * np.sin(u_k)
    
    Omega_k = eph['Omega0'] + (eph['Omega_dot'] - OMEGA_E) * t_k - OMEGA_E * eph['Toe']
    
    x = x_prime_k * np.cos(Omega_k) - y_prime_k * np.cos(i_k) * np.sin(Omega_k)
    y = x_prime_k * np.sin(Omega_k) + y_prime_k * np.cos(i_k) * np.cos(Omega_k)
    z = y_prime_k * np.sin(i_k)
    
    return np.array([x, y, z])

# --- Iono-free Combination ---

def iono_free_coeffs(sig1_code: str, sig2_code: str) -> Tuple[float, float]:
    """
    Calculates the ionosphere-free linear combination coefficients (alpha, beta).
    """
    FREQ_MAP = {
        # GPS
        'G01': 1575.42e6, # L1
        'G02': 1227.60e6, # L2
        'G05': 1176.45e6, # L5
        
        # GLONASS (Using k=0 nominal frequencies)
        'R01': 1602.0e6,   # G1
        'R02': 1246.0e6,   # G2
        'R03': 1202.025e6, # G3
        'R04': 1600.995e6, # G1a (L1OC) - Varies, but using standard if needed
        'R06': 1248.06e6,  # G2a (L2C)  - Varies
        
        # Galileo
        'E01': 1575.42e6,  # E1
        'E05': 1176.45e6,  # E5a
        'E06': 1278.75e6,  # E6
        'E07': 1207.14e6,  # E5b
        'E08': 1191.795e6, # E5 (AltBOC)
        
        # BeiDou
        'C02': 1561.098e6, # B1I
        'C01': 1575.42e6,  # B1C
        'C07': 1207.14e6,  # B2I / B2b
        'C06': 1268.52e6,  # B3I
        'C05': 1176.45e6,  # B2a
        'C08': 1191.795e6  # B2 (AltBOC)
    }
    
    f1 = FREQ_MAP.get(sig1_code)
    f2 = FREQ_MAP.get(sig2_code)
    
    if not f1 or not f2:
        # Fallback for GLONASS if exact code not found (e.g. if code is just 'R1')
        # But our system uses strict 3-char codes (R01).
        raise ValueError(f"Frequency not defined for signals {sig1_code} or {sig2_code}")
        
    gamma = (f1 / f2)**2
    alpha = gamma / (gamma - 1)
    beta = -1.0 / (gamma - 1)
    
    return alpha, beta

# --- PCC Grid Calculation ---

def compute_pcc(pco_mm: np.ndarray, pcv_grid_mm: np.ndarray, pcv_grid_step: float, 
                target_grid_shape: tuple, target_grid_step: float) -> np.ndarray:
    """
    Computes the full PCC grid (PCO + PCV) for a specific signal,
    resampling PCV if necessary.
    """
    
    target_azi_steps, target_zen_steps = target_grid_shape
    target_azi_coords = np.linspace(0.0, 360.0, target_azi_steps)
    target_zen_coords = np.linspace(0.0, 90.0, target_zen_steps)
    
    az_mesh_rad, zen_mesh_rad = np.meshgrid(
        np.radians(target_azi_coords), 
        np.radians(target_zen_coords), 
        indexing='ij'
    )

    e_n_grid = np.sin(zen_mesh_rad) * np.cos(az_mesh_rad)
    e_e_grid = np.sin(zen_mesh_rad) * np.sin(az_mesh_rad)
    e_u_grid = np.cos(zen_mesh_rad)

    pco_n, pco_e, pco_u = pco_mm[0], pco_mm[1], pco_mm[2]
    pco_dot_e = (pco_n * e_n_grid) + (pco_e * e_e_grid) + (pco_u * e_u_grid)
    
    pcv_to_use = None
    if pcv_grid_step == target_grid_step:
        pcv_to_use = pcv_grid_mm
    else:
        # --- Pass target_grid_shape to the helper ---
        pcv_to_use = _resample_pcv_grid(
            pcv_grid_mm, pcv_grid_step, 
            target_azi_coords, target_zen_coords,
            target_grid_shape 
        )
        if pcv_to_use is None:
             raise ValueError("PCV grid resampling failed.")
             
    if pcv_to_use.shape != target_grid_shape:
        # Handle NOAZI case - either 1D vector or 2D array with shape (1, n)
        is_noazi_1d = pcv_to_use.ndim == 1 and pcv_to_use.shape[0] <= target_zen_steps
        is_noazi_2d = pcv_to_use.ndim == 2 and pcv_to_use.shape[0] == 1
        # AZ-dependent grid with different shape - needs 2D resampling
        is_az_grid_diff_shape = pcv_to_use.ndim == 2 and pcv_to_use.shape[0] > 1
        
        if is_noazi_1d:
            print(f"  Info: Broadcasting 1D NOAZI PCV ({pcv_to_use.shape}) to shape {target_grid_shape}")
            # Interpolate zenith if different step count
            if pcv_to_use.shape[0] != target_zen_steps:
                old_zen_coords = np.linspace(0.0, 90.0, pcv_to_use.shape[0])
                new_zen_coords = np.linspace(0.0, 90.0, target_zen_steps)
                pcv_to_use = np.interp(new_zen_coords, old_zen_coords, pcv_to_use)
            pcv_to_use = np.tile(pcv_to_use, (target_azi_steps, 1))
            
        elif is_noazi_2d:
            print(f"  Info: Broadcasting 2D NOAZI PCV ({pcv_to_use.shape}) to shape {target_grid_shape}")
            noazi_vector = pcv_to_use[0, :]  # Extract the single row
            # Interpolate zenith if different step count
            if noazi_vector.shape[0] != target_zen_steps:
                old_zen_coords = np.linspace(0.0, 90.0, noazi_vector.shape[0])
                new_zen_coords = np.linspace(0.0, 90.0, target_zen_steps)
                noazi_vector = np.interp(new_zen_coords, old_zen_coords, noazi_vector)
            pcv_to_use = np.tile(noazi_vector, (target_azi_steps, 1))
            
        elif is_az_grid_diff_shape:
            # Resample AZ-dependent grid to target shape using 2D interpolation
            print(f"  Info: Resampling AZ PCV grid ({pcv_to_use.shape}) to shape {target_grid_shape}")
            old_azi_steps, old_zen_steps = pcv_to_use.shape
            old_azi_coords = np.linspace(0.0, 360.0, old_azi_steps)
            old_zen_coords = np.linspace(0.0, 90.0, old_zen_steps)
            
            # Ensure grid wraps at azimuth (360 = 0)
            if old_azi_steps > 1 and not np.isclose(old_azi_coords[-1], 360.0):
                # Add wrap-around row
                pcv_wrapped = np.vstack([pcv_to_use, pcv_to_use[0:1, :]])
                old_azi_coords = np.append(old_azi_coords, 360.0)
            else:
                pcv_wrapped = pcv_to_use
            
            try:
                from scipy.interpolate import RegularGridInterpolator
                interpolator = RegularGridInterpolator(
                    (old_azi_coords, old_zen_coords), pcv_wrapped,
                    method='linear', bounds_error=False, fill_value=0.0
                )
                
                target_azi_mesh, target_zen_mesh = np.meshgrid(target_azi_coords, target_zen_coords, indexing='ij')
                points = np.array([target_azi_mesh.ravel(), target_zen_mesh.ravel()]).T
                pcv_to_use = interpolator(points).reshape(target_grid_shape)
            except Exception as e:
                print(f"  Warning: Resampling failed: {e}")
                raise ValueError(f"PCV grid shape mismatch. Expected {target_grid_shape}, got {pcv_to_use.shape}")
        else:
            raise ValueError(f"PCV grid shape mismatch. Expected {target_grid_shape}, got {pcv_to_use.shape}")

    # PCC = -PCO·e + PCV (sign convention per reference implementation)
    pcc_grid = -pco_dot_e + pcv_to_use
    
    return pcc_grid


def _resample_pcv_grid(pcv_grid: np.ndarray, old_step: float, 
                       target_azi_coords: np.ndarray, 
                       target_zen_coords: np.ndarray,
                       target_grid_shape: tuple) -> np.ndarray:
    """
    Resamples a PCV grid to a new target grid definition using linear interpolation.
    Handles NOAZI and AZ-dependent grids.
    """
    
    old_azi_steps, old_zen_steps = pcv_grid.shape
    old_zen_coords = np.linspace(0.0, 90.0, old_zen_steps)
    is_noazi = old_azi_steps == 1
    
    pcv_for_interp = pcv_grid
    
    if is_noazi:
        pcv_vector = pcv_grid.squeeze()
        try:
             # Use numpy.interp for 1D, fill with endpoints
             interpolated_vector = np.interp(target_zen_coords, old_zen_coords, pcv_vector, 
                                             left=pcv_vector[0], right=pcv_vector[-1])
        except Exception as e:
             print(f"Error during 1D (NOAZI) interpolation: {e}")
             return None
        
        # Tile this 1D vector across all azimuths
        resampled_grid = np.tile(interpolated_vector, (len(target_azi_coords), 1))
        return resampled_grid
    
    else:
        # --- Handle AZ-dependent grid ---
        if old_azi_steps == int(round(360.0 / old_step)) + 1:
            old_azi_coords = np.linspace(0.0, 360.0, old_azi_steps)
            azi_coords_for_interp = old_azi_coords
        else:
            old_azi_coords = np.linspace(0.0, 360.0 - old_step, old_azi_steps)
            if old_azi_coords.size > 0:
                 # Stack the first azimuth row at the end (360 deg = 0 deg)
                 pcv_for_interp = np.vstack([pcv_grid, pcv_grid[0, :]])
                 azi_coords_for_interp = np.append(old_azi_coords, 360.0)
            else: 
                 azi_coords_for_interp = old_azi_coords

    try:
         pcv_clean = np.nan_to_num(pcv_for_interp, nan=0.0)
         interpolator = RegularGridInterpolator(
             (azi_coords_for_interp, old_zen_coords), pcv_clean,
             method='linear', bounds_error=False, fill_value=0.0 # Use 0 for points outside interp range
         )
    except ValueError as e:
         print(f"Error creating interpolator during resampling: {e}")
         print(f"  PCV shape: {pcv_clean.shape}")
         print(f"  Azi coords: {azi_coords_for_interp.shape}, Zen coords: {old_zen_coords.shape}")
         return None

    target_azi_mesh, target_zen_mesh = np.meshgrid(target_azi_coords, target_zen_coords, indexing='ij')
    points_to_interpolate = np.array([target_azi_mesh.ravel(), target_zen_mesh.ravel()]).T
    
    resampled_values_flat = interpolator(points_to_interpolate)
    
    # --- Use the passed-in target_grid_shape ---
    resampled_grid = resampled_values_flat.reshape(target_grid_shape)
    
    return resampled_grid