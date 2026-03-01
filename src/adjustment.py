# src/adjustment.py
"""
Performs the least-squares adjustment using the gridded simulative approach
from Kröger (2025), Section 3.3.2.

Reference:
    Kröger, J. (2025): Systematische Untersuchung von GNSS-Antennenkalibrierungen
    und deren Einfluss auf geodätische Parameter. Dissertation, Leibniz Universität
    Hannover.

This file contains:
- perform_adjustment: The main "engine" that solves the normal equations
  using pre-computed 2D grids.
- compute_m_matrix: The function to generate the observation density grid (P_2).
- Helper functions for grid creation and ANTEX data interpolation.
"""
import numpy as np
from datetime import timedelta, datetime
import time
from scipy.interpolate import RegularGridInterpolator
import re

# Import necessary functions
from .geodesy import (ecef_to_topocentric, iono_free_coeffs, ecef_to_ell,
                    compute_satellite_position, compute_pcc)
from .orbit_utils import interpolate_orbit
from .time_utils import datetime_to_gps_sow

# Constants
C = 299792458.0  # Speed of light in m/s

# --- Helper Functions for ANTEX Parsing ---

def _get_pcv_data(antex_data: dict, antenna_type: str, system: str, signal: str):
    """Safely retrieves PCO and PCV data, returning None if not found."""
    pco = antex_data.get('pco', {}).get(antenna_type, {}).get(system, {}).get(signal)
    pcv = antex_data.get('pcv', {}).get(antenna_type, {}).get(system, {}).get(signal)
    
    if pco is None or pcv is None:
        return None, None
    try:
        pco_float = np.array(pco, dtype=float)
        pcv_array = np.array(pcv, dtype=object) # Read as object first
        pcv_float = np.full(pcv_array.shape, np.nan, dtype=float) # Initialize with NaN
        non_empty_mask = pcv_array != None
        # Helper to check if element is a number
        valid_float_mask = np.vectorize(lambda x: isinstance(x, (int, float, np.number)))(pcv_array[non_empty_mask])
        
        combined_mask = np.full(pcv_array.shape, False, dtype=bool)
        combined_mask[non_empty_mask] = valid_float_mask
        
        pcv_float[combined_mask] = pcv_array[combined_mask].astype(float)
        pcv_float = np.nan_to_num(pcv_float, nan=0.0)

        if pco_float.shape != (3,):
             print(f"  Warning: Unexpected PCO shape {pco_float.shape} for {system}/{signal}. Expected (3,).")
        if pcv_float.ndim != 2:
            print(f"  Warning: Unexpected PCV dimension {pcv_float.ndim} for {system}/{signal}. Expected 2.")

        return pco_float, pcv_float
    except Exception as e:
        print(f"  Warning: Could not convert PCO/PCV to float for {system}/{signal}. Error: {e}")
        return None, None


def _create_pcv_interpolator(pcv_grid: np.ndarray, grid_step: float):
    """Creates a RegularGridInterpolator for a PCV grid."""
    if pcv_grid is None or pcv_grid.size == 0:
        return None
    if not isinstance(pcv_grid, np.ndarray) or pcv_grid.ndim != 2:
        return None

    azi_steps, zen_steps = pcv_grid.shape

    if azi_steps == int(round(360.0 / grid_step)) + 1: # 0 to 360 inclusive
        azi_coords = np.linspace(0.0, 360.0, azi_steps)
    else: # 0 to 355 exclusive
        azi_coords = np.linspace(0.0, 360.0 - grid_step, azi_steps)

    expected_zen_steps = int(round(90.0 / grid_step)) + 1
    if zen_steps != expected_zen_steps:
         max_zen = min(90.0, (zen_steps - 1) * grid_step)
         zen_coords = np.linspace(0.0, max_zen, zen_steps)
    else:
         zen_coords = np.linspace(0.0, 90.0, zen_steps)

    try:
        pcv_values = np.nan_to_num(pcv_grid.astype(float), nan=0.0)

        interpolator = RegularGridInterpolator(
            (azi_coords, zen_coords), pcv_values,
            method='linear', bounds_error=False, fill_value=np.nan
        )
        return interpolator
    except ValueError:
        return None
    except Exception:
        return None

# --- M-Matrix Calculation ---
def compute_m_matrix(config: dict) -> tuple:
    """
    Computes the M-Matrix (normalized satellite distribution histogram).
    Mimics the logic from MATLAB's compute_M_Matrix.m, including fliplr.
    """
    print("--- Computing P2 Matrix (Satellite Distribution) ---")
    start_m_time = time.time()

    start_time = config['start_time']
    end_time = config['end_time']
    sampling_rate = timedelta(seconds=config['sampling_rate_sec'])
    station_pos = config['position']
    orbit_data = config['orbit_data']
    grid_step = config.get('grid_step_1', 5.0)
    el_mask_active = config.get('elevation_mask_active', False)
    el_mask_angle = config.get('elevation_mask_angle', 0.0) if el_mask_active else 0.0
    az_mask_active = config.get('azimuth_mask_active', False)
    az_mask_values = config.get('azimuth_mask_values', [])
    
    signals_to_process = config.get('signals', [])
    systems_to_process = set()
    for s in signals_to_process:
        if not s: continue
        # Handle IF_ format
        if s.startswith('IF_'):
            # IF_E01_E05 -> 'E'
            parts = s.split('_')
            if len(parts) >= 2 and len(parts[1]) > 0:
                sys_char = parts[1][0]
                if sys_char in ['G', 'R', 'E', 'C']:
                    systems_to_process.add(sys_char)
        else:
            # Standard G01, E01 etc.
            if s[0] in ['G', 'R', 'E', 'C']:
                systems_to_process.add(s[0])
    
    if not systems_to_process:
        print("  Warning: No systems (G,R,E,C) found in signals list. Defaulting to 'G'.")
        systems_to_process = {'G'}
        
    available_systems = systems_to_process

    is_final_orbit = 'orbits' in orbit_data
    all_valid_azimuths = []
    all_valid_elevations = []
    all_satellite_ids = []  # Track satellite IDs (e.g., 'G01', 'E05') for skyplot
    current_time = start_time
    total_epochs = int((end_time - start_time) / sampling_rate) + 1 if sampling_rate.total_seconds() > 0 else 1
    processed_epochs = 0

    while current_time <= end_time:
        processed_epochs += 1
        _, current_sow = datetime_to_gps_sow(current_time)

        for system in available_systems:
            if system == 'J': continue
            sats_in_system = []
            if is_final_orbit:
                sats_in_system = orbit_data.get('orbits', {}).get(system, {}).keys()
            else: # Broadcast
                sats_in_system = orbit_data.get(system, {}).keys()

            for prn in sats_in_system:
                sat_pos = None
                if is_final_orbit:
                    orbit_records = orbit_data['orbits'][system].get(prn)
                    if orbit_records: sat_pos = interpolate_orbit(orbit_records, current_time)
                else: # Broadcast ephemeris
                    ephemeris_list = orbit_data[system].get(prn)
                    if ephemeris_list:
                        best_ephem = None
                        valid_eph = [eph for eph in ephemeris_list if 'Toe' in eph]
                        if valid_eph:
                             candidates_before = [eph for eph in valid_eph if eph['Toe'] <= current_sow and current_sow - eph['Toe'] <= 7200]
                             if candidates_before:
                                  best_ephem = max(candidates_before, key=lambda eph: eph['Toe'])
                             else:
                                  candidates_nearby = [eph for eph in valid_eph if abs(current_sow - eph['Toe']) <= 7200]
                                  if candidates_nearby:
                                       best_ephem = min(candidates_nearby, key=lambda eph: abs(current_sow - eph['Toe']))
                        if best_ephem:
                            try:
                                approx_transmit_sow = current_sow - 0.075
                                sat_pos = compute_satellite_position(best_ephem, approx_transmit_sow)
                            except Exception: sat_pos = None

                if sat_pos is None or np.any(np.isnan(sat_pos)): continue

                try: azimuth_deg, elevation_deg = ecef_to_topocentric(station_pos, sat_pos)
                except Exception: continue

                if el_mask_active and elevation_deg < el_mask_angle: continue
                
                azimuth_masked_out = False
                if az_mask_active:
                    current_az_norm = azimuth_deg % 360.0
                    for mask in az_mask_values:
                        az_from, az_to, el_limit = mask
                        in_range = False
                        if az_from <= az_to:
                            if az_from <= current_az_norm <= az_to: in_range = True
                        else: # Wraps around 360
                            if current_az_norm >= az_from or current_az_norm <= az_to: in_range = True
                        if in_range and elevation_deg < el_limit:
                            azimuth_masked_out = True; break
                if azimuth_masked_out: continue

                if elevation_deg >= 0:
                     all_valid_azimuths.append(azimuth_deg % 360.0)
                     all_valid_elevations.append(elevation_deg)
                     all_satellite_ids.append(f"{system}{prn:02d}")  # e.g., 'G01', 'E05'

        if sampling_rate.total_seconds() <= 0: break
        current_time += sampling_rate

    if not all_valid_azimuths:
         print("  Warning: No valid observations found. Cannot compute P2 matrix. Returning None.")
         return None, None, None, None

    az_bins = np.arange(-grid_step / 2.0, 360.0 + grid_step, grid_step)
    el_bins = np.arange(-grid_step / 2.0, 90.0 + grid_step, grid_step)

    try:
        M_hist, _, _ = np.histogram2d(
            all_valid_azimuths,
            all_valid_elevations,
            bins=[az_bins, el_bins]
        )
    except Exception as e:
        print(f"  [ERROR] Creating histogram for P2 matrix: {e}")
        return None, None, None, None

    M_hist = np.fliplr(M_hist)
    
    total_counts = np.sum(M_hist)
    if total_counts == 0:
        print("  Warning: Histogram is empty after potential flip. Returning None.")
        return None, None, None, None

    M_matrix_normalized = M_hist / total_counts

    end_m_time = time.time()
    # print(f"  M-Matrix computation finished in {end_m_time - start_m_time:.2f} seconds.")
    
    return M_matrix_normalized, all_valid_azimuths, all_valid_elevations, all_satellite_ids


# --- A-Grids Creation ---
def _create_A_grids(grid_shape, grid_step, tropo_model, config):
    """
    Creates the 2D design matrices (A-grids) for each parameter.
    Signs are matched to the reference implementation (see Kröger, 2025, Fig 3.6).
    """
    azi_steps, zen_steps = grid_shape
    
    if azi_steps == int(round(360.0 / grid_step)) + 1: 
        az_coords = np.linspace(0.0, 360.0, azi_steps)
    else:
        az_coords = np.linspace(0.0, 360.0 - grid_step, azi_steps)
        
    zen_coords = np.linspace(0.0, 90.0, zen_steps)
    
    az_mesh_rad, zen_mesh_rad = np.meshgrid(np.radians(az_coords), np.radians(zen_coords), indexing='ij')

    sin_zen = np.sin(zen_mesh_rad)
    cos_zen = np.cos(zen_mesh_rad)
    sin_az = np.sin(az_mesh_rad)
    cos_az = np.cos(az_mesh_rad)
    
    # Per Kröger (2025), Fig 3.6:
    A_N_grid = -sin_zen * cos_az 
    A_E_grid = -sin_zen * sin_az 
    A_U_grid = -cos_zen          
    
    A_Clk_grid = np.ones(grid_shape)
    
    A_Tropo_grid = np.zeros(grid_shape)
    A_Tropo_Gn_grid = np.zeros(grid_shape)  # North gradient
    A_Tropo_Ge_grid = np.zeros(grid_shape)  # East gradient
    
    if tropo_model != 'None':
        el_mesh_rad = np.pi/2.0 - zen_mesh_rad
        sin_el = np.sin(el_mesh_rad)
        sin_el_safe = np.maximum(sin_el, np.sin(np.deg2rad(0.1)))

        if tropo_model == '1/sin(Elevation)':
             A_Tropo_grid = 1.0 / sin_el_safe
        elif tropo_model == 'GMF':
             # GMF (simplified/inline)
             a = 2.53e-5; b = 5.49e-3; c = 1.14e-3
             den_c = sin_el_safe + c
             den_b = sin_el_safe + b / (den_c + 1e-9) 
             den_a = sin_el_safe + a / (den_b + 1e-9)
             num_a = 1.0 + a / (1.0 + b / (1.0 + c))
             A_Tropo_grid = num_a / (den_a + 1e-9)
        else: 
             A_Tropo_grid = 1.0 / sin_el_safe
        
        # Troposphere gradient matrices: Chen & Herring (1997)
        # North: cos(Az) * cos(el) / (sin²(el) + C)
        # East:  sin(Az) * cos(el) / (sin²(el) + C)
        # C = 0.0007 (water vapor component, scale height ~3 km)
        C_gradient = 0.0007
        cos_el = np.cos(el_mesh_rad)
        gradient_denom = sin_el**2 + C_gradient
        gradient_base = cos_el / gradient_denom
        A_Tropo_Gn_grid = gradient_base * cos_az  # North gradient component
        A_Tropo_Ge_grid = gradient_base * sin_az  # East gradient component
             
    return A_N_grid, A_E_grid, A_U_grid, A_Clk_grid, A_Tropo_grid, A_Tropo_Gn_grid, A_Tropo_Ge_grid


# --- P1-Grid (Weighting) ---
def _create_P1_grid(grid_shape, grid_step, weighting_model):
    """
    Creates the 2D observation weighting grid (P1).
    """
    _, zen_steps = grid_shape
    
    zen_coords = np.linspace(0.0, 90.0, zen_steps)
    zen_mesh_rad = np.radians(zen_coords)
    
    el_mesh_rad = np.pi/2.0 - zen_mesh_rad
    sin_el = np.sin(el_mesh_rad)
    
    P1_vector = np.ones(zen_steps)
    if weighting_model == 'sin':
        P1_vector = np.sqrt(sin_el)  # P_1 = sqrt(sin(e))
    elif weighting_model == 'sin2':
        P1_vector = sin_el           # P_1 = sqrt(sin(e)^2) = sin(e)
        
    P1_vector = np.maximum(P1_vector, 1e-6)
    P1_grid = np.tile(P1_vector, (grid_shape[0], 1))
    
    return P1_grid


# --- Calculate Delta PCC Grid (l-vector) ---
def _calculate_delta_pcc_grid(config, antex_data_1, antex_data_2, grid_shape, grid_step_1, grid_step_2, signal):
    """
    Calculates the (l) vector: the Delta-PCC grid for a specific signal.
    """
    antenna_type_1 = config['antenna_type']
    antenna_type_2 = config['_antenna_type_file2']
    
    system = None
    sig1_code = None
    sig2_code = None # For iono-free
    
    system = None
    sig1_code = None
    sig2_code = None # For iono-free
    
    # Check for IF-LC format: "IF_G01_G02"
    if signal.startswith("IF_"):
        parts = signal.split('_')
        if len(parts) == 3:
            # Format: IF_G01_G02
            sig1_code = parts[1]
            sig2_code = parts[2]
            
            # Determine system from first char
            sys_char = sig1_code[0].upper()
            if sys_char == 'G': system = 'GPS'
            elif sys_char == 'R': system = 'GLONASS'
            elif sys_char == 'E': system = 'Galileo'
            elif sys_char == 'C': system = 'BDS'
            elif sys_char == 'J': system = 'QZSS'
            else: raise ValueError(f"Unknown system char '{sys_char}' in signal {signal}")
            
        else:
             raise ValueError(f"Invalid IF-LC signal format: {signal}. Expected IF_Code1_Code2")
             
    # Legacy Fallback for "G3" (Keep for backward compatibility if needed, or remove)
    elif signal == 'G3':
        system = 'GPS'; sig1_code = 'G01';
        # Check combination based on File 1 ONLY
        has_g02_1 = _get_pcv_data(antex_data_1, antenna_type_1, 'GPS', 'G02')[0] is not None
        has_g05_1 = _get_pcv_data(antex_data_1, antenna_type_1, 'GPS', 'G05')[0] is not None
        if has_g02_1: sig2_code = 'G02'
        elif has_g05_1: sig2_code = 'G05'
        else: raise ValueError("GPS G3 (IF-LC) requested but neither G02 nor G05 found in File 1.")

    elif signal in ['E00', 'E0']:
        system = 'Galileo'; sig1_code = 'E01'; sig2_code = 'E05' # Default to E1/E5a

    else:
        # Standard Single Frequency
        if signal.startswith('G'): system = 'GPS'
        elif signal.startswith('E'): system = 'Galileo'
        elif signal.startswith('R'): system = 'GLONASS'
        elif signal.startswith('C'): system = 'BDS'
        elif signal.startswith('J'): system = 'QZSS'
        sig1_code = signal
        
    if system is None:
        raise ValueError(f"Could not determine system for signal {signal}")

    # --- Get PCC Grid 1 ---
    if sig2_code is None: # Single frequency
        pco1, pcv1 = _get_pcv_data(antex_data_1, antenna_type_1, system, sig1_code)
        if pco1 is None: raise ValueError(f"Missing ANTEX 1 data for {antenna_type_1}/{system}/{sig1_code}")
        PCC_1_grid = compute_pcc(pco1, pcv1, grid_step_1, grid_shape, grid_step_1)
    else: # Iono-free
        pco1_f1, pcv1_f1 = _get_pcv_data(antex_data_1, antenna_type_1, system, sig1_code)
        pco1_f2, pcv1_f2 = _get_pcv_data(antex_data_1, antenna_type_1, system, sig2_code)
        
        if pco1_f1 is None: raise ValueError(f"Missing {sig1_code} in File 1 for {signal}")
        if pco1_f2 is None: raise ValueError(f"Missing {sig2_code} in File 1 for {signal}")
        
        PCC_1_f1 = compute_pcc(pco1_f1, pcv1_f1, grid_step_1, grid_shape, grid_step_1)
        PCC_1_f2 = compute_pcc(pco1_f2, pcv1_f2, grid_step_1, grid_shape, grid_step_1)
        
        alpha, beta = iono_free_coeffs(sig1_code, sig2_code)
        PCC_1_grid = alpha * PCC_1_f1 + beta * PCC_1_f2
        
    # --- Get PCC Grid 2 (skip if single-file mode) ---
    if antex_data_2 is None:
        # Single-file mode: return absolute PCC values (no delta)
        PCC_2_grid = np.zeros(grid_shape)
    elif sig2_code is None: # Single frequency
        pco2, pcv2 = _get_pcv_data(antex_data_2, antenna_type_2, system, sig1_code)
        if pco2 is None: 
            PCC_2_grid = np.zeros(grid_shape)
        else:
            PCC_2_grid = compute_pcc(pco2, pcv2, grid_step_2, grid_shape, grid_step_1)
    else: # Iono-free
        pco2_f1, pcv2_f1 = _get_pcv_data(antex_data_2, antenna_type_2, system, sig1_code)
        pco2_f2, pcv2_f2 = _get_pcv_data(antex_data_2, antenna_type_2, system, sig2_code)
        
        if pco2_f1 is None:
            PCC_2_f1 = np.zeros(grid_shape)
        else:
            PCC_2_f1 = compute_pcc(pco2_f1, pcv2_f1, grid_step_2, grid_shape, grid_step_1)
            
        if pco2_f2 is None:
            PCC_2_f2 = np.zeros(grid_shape)
        else:
            PCC_2_f2 = compute_pcc(pco2_f2, pcv2_f2, grid_step_2, grid_shape, grid_step_1)

        alpha, beta = iono_free_coeffs(sig1_code, sig2_code)
        PCC_2_grid = alpha * PCC_2_f1 + beta * PCC_2_f2
        
    l_grid = PCC_1_grid - PCC_2_grid

    return l_grid


# --- Main Adjustment Function ---

def perform_adjustment(config: dict, antex_data_1: dict, antex_data_2: dict, 
                       m_matrix: np.ndarray) -> tuple:
    """
    Performs least squares adjustment using the gridded simulative approach.
    """
    start_adj_time = time.time()
    
    signals_to_process = config['signals']
    grid_step_1 = config.get('grid_step_1', 5.0)
    grid_step_2 = config.get('grid_step_2', 5.0)
    weighting_model = config.get('weighting', 'sin').lower()
    tropo_model = config.get('tropo_model', 'None')
    tropo_active = tropo_model != 'None'
    
    grid_shape = m_matrix.shape
    P_2_grid = m_matrix 
    
    # --- 2. Parameter Setup ---
    # Detect unique GNSS systems from signal codes
    system_map = {'G': 'GPS', 'E': 'Galileo', 'R': 'GLONASS', 'C': 'BDS', 'J': 'QZSS', 'S': 'SBAS'}
    unique_systems = set()
    unique_systems = set()
    for sig in signals_to_process:
        if not sig: continue
        
        sys_char = None
        if sig.startswith('IF_'):
             parts = sig.split('_')
             if len(parts) >= 2 and len(parts[1]) > 0:
                 sys_char = parts[1][0].upper()
        elif len(sig) > 0:
             sys_char = sig[0].upper()
             
        if sys_char and sys_char in system_map:
            unique_systems.add(system_map[sys_char])
    
    # Default to GPS if no systems found
    if not unique_systems:
        unique_systems = {'GPS'}
    
    # Sort for consistent ordering
    unique_systems = sorted(list(unique_systems))
    
    param_names = ['North', 'East', 'Up']
    
    # Add clock parameter for each system
    clock_indices = {}  # Maps system name to parameter index
    for sys_name in unique_systems:
        clock_indices[sys_name] = len(param_names)
        param_names.append(f'Clock_{sys_name}')
    
    tropo_col_index = -1
    if tropo_active:
        tropo_col_index = len(param_names)
        param_names.append('Tropo')
    
    # Add troposphere gradient parameters if enabled
    gradient_active = config.get('gradient_active', False)
    gradient_n_col_index = -1
    gradient_e_col_index = -1
    if gradient_active and tropo_active:
        gradient_n_col_index = len(param_names)
        param_names.append('Tropo_Gn')  # North gradient
        gradient_e_col_index = len(param_names)
        param_names.append('Tropo_Ge')  # East gradient
        
    num_params = len(param_names)

    # --- 3. Create Grids ---
    try:
        A_N_grid, A_E_grid, A_U_grid, A_Clk_grid, A_Tropo_grid, A_Tropo_Gn_grid, A_Tropo_Ge_grid = _create_A_grids(
            grid_shape, grid_step_1, tropo_model, config
        )
        P1_grid = _create_P1_grid(grid_shape, grid_step_1, weighting_model)
        
    except Exception as e:
        print(f"  [ERROR] Failed to create geometry/weighting grids: {e}")
        return None, param_names, m_matrix, None, None

    # --- 4. Build Normal Equations ---
    N_bar_total = np.zeros((num_params, num_params))
    n_bar_total = np.zeros(num_params)

    signals_processed_count = 0
    
    for signal in signals_to_process:
        try:
            l_grid = _calculate_delta_pcc_grid(
                config, antex_data_1, antex_data_2, 
                grid_shape, grid_step_1, grid_step_2, 
                signal
            )
            
            P1_x_P2 = P1_grid * P_2_grid
            l_x_P1 = l_grid * P1_grid

            # Build A_grids for this signal with correct clock
            # Determine which system this signal belongs to
            # Determine which system this signal belongs to
            sig_sys_char = 'G'
            if signal.startswith('IF_'):
                 parts = signal.split('_')
                 if len(parts) >= 2 and len(parts[1]) > 0:
                     sig_sys_char = parts[1][0].upper()
            elif signal and len(signal) > 0:
                 sig_sys_char = signal[0].upper()
            
            sig_system = system_map.get(sig_sys_char, 'GPS')
            
            # Create A_grids list: [N, E, U, Clk_Sys1, Clk_Sys2, ..., Tropo, Tropo_Gn, Tropo_Ge]
            A_grids_signal = [A_N_grid, A_E_grid, A_U_grid]
            
            # Add clock columns - ones for this signal's system, zeros for others
            for sys_name in unique_systems:
                if sys_name == sig_system:
                    A_grids_signal.append(A_Clk_grid)  # This signal contributes to its clock
                else:
                    A_grids_signal.append(np.zeros(grid_shape))  # No contribution to other clocks
            
            # Add tropo if active
            if tropo_active:
                A_grids_signal.append(A_Tropo_grid)
            
            # Add tropo gradients if active
            if gradient_active and tropo_active:
                A_grids_signal.append(A_Tropo_Gn_grid)
                A_grids_signal.append(A_Tropo_Ge_grid)

            N_bar_signal = np.zeros((num_params, num_params))
            n_bar_signal = np.zeros(num_params)

            for i in range(num_params):
                A_i_x_P1 = A_grids_signal[i] * P1_grid
                n_i_grid = A_i_x_P1 * P_2_grid * l_x_P1
                n_bar_signal[i] = np.sum(n_i_grid)

            for i in range(num_params):
                A_i_x_P1 = A_grids_signal[i] * P1_grid
                for j in range(i, num_params): 
                    A_j_x_P1 = A_grids_signal[j] * P1_grid
                    N_ij_grid = A_i_x_P1 * P_2_grid * A_j_x_P1
                    N_ij_val = np.sum(N_ij_grid)
                    
                    N_bar_signal[i, j] = N_ij_val
                    if i != j:
                        N_bar_signal[j, i] = N_ij_val
                    
            N_bar_total += N_bar_signal
            n_bar_total += n_bar_signal
            signals_processed_count += 1

        except Exception as e:
            print(f"  ⚠️ WARNING: Failed to process signal {signal}. Skipping. Error: {e}")
            import traceback
            traceback.print_exc()
            
    if signals_processed_count == 0:
        print("\n  ❌ ERROR: No signals could be processed. Adjustment failed.")
        return None, param_names, m_matrix, None, None
        
    # --- 5. Solve Normal Equations ---
    try:
         cond_N = np.linalg.cond(N_bar_total); 
         if np.any(np.abs(np.diag(N_bar_total)) < 1e-12):
             print("  ❌ ERROR: Near-zero diagonal elements found in N_bar_total. Singular.")
             return None, param_names, m_matrix, None, None

         results_vec = np.linalg.solve(N_bar_total, n_bar_total)
         
    except np.linalg.LinAlgError: 
        print("  ❌ ERROR: Normal matrix N_bar_total is singular. Cannot solve.")
        return None, param_names, m_matrix, None, None
    except Exception as e: 
        print(f"  ❌ ERROR solving linear system: {e}")
        return None, param_names, m_matrix, None, None

    end_adj_time = time.time()
    
    if np.isnan(results_vec).any():
        print("\n❌ PROCESSING FAILED: Final mapped results contain NaN values.")
        return None, param_names, m_matrix, None, None
    
    return results_vec, param_names, m_matrix, None, None



