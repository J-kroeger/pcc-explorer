# src/processing_core.py
"""
Core logic for processing GNSS data and performing the adjustment.
Orchestrates the gridded simulative approach:
1. Computes the M-Matrix (observation density).
2. Calls the gridded adjustment function.
"""

import numpy as np
import os
import re
from datetime import datetime, timedelta

from .data_io import (read_antex_file, download_and_unzip_broadcast_file,
                    read_broadcast_file, download_final_orbit_file,
                    read_final_orbit_file, save_results_to_txt)

from .adjustment import perform_adjustment, compute_m_matrix
from .orbit_utils import interpolate_orbit
from .time_utils import datetime_to_gps_sow
from .geodesy import ell_to_ecef  # Needed for global analysis loop


def _get_antex_grid_step(antex_data: dict, antenna_type: str) -> float:
    """Safely retrieves the grid step from ANTEX metadata, defaulting to 5.0."""
    grid_step = 5.0
    if antex_data and 'metadata' in antex_data:
        metadata_for_antenna = antex_data['metadata'].get(antenna_type, {})
        grid_step = metadata_for_antenna.get('dzen', 5.0)
    
    if not (0.1 <= grid_step <= 10.0):
        # Fallback for unrealistic steps
        grid_step = 5.0
    return grid_step





def _resolve_antenna_key(antex_data: dict, requested_name: str) -> str:
    """
    Finds the actual key in the ANTEX data for a requested antenna name.
    Tries exact match first, then matches by 'Type' (ignoring Serial).
    """
    # 1. Try exact match
    if requested_name in antex_data.get('pco', {}):
        return requested_name
        
    # 2. Try canonical match (ignore whitespace differences)
    req_canon = ' '.join(requested_name.split()).upper()
    for key in antex_data.get('pco', {}).keys():
        if ' '.join(key.split()).upper() == req_canon:
            return key
            
    # 3. Try matching by TYPE only (from metadata)
    metadata = antex_data.get('metadata', {})
    for key, info in metadata.items():
        ant_type = info.get('type', '').strip()
        if ' '.join(ant_type.split()).upper() == req_canon:
            return key
            
    return None


# --- NEW: Signal/System Validation ---
SYSTEM_NAMES = {
    'G': 'GPS',
    'E': 'Galileo',
    'R': 'GLONASS',
    'C': 'BDS/BeiDou',
    'J': 'QZSS',
    'S': 'SBAS'
}


def validate_signals_in_antex(antex_data: dict, antenna_type: str, signals: list) -> list:
    """
    Validates that all requested signals exist in the ANTEX data for the given antenna.
    
    Args:
        antex_data: Parsed ANTEX file data
        antenna_type: The resolved antenna type key
        signals: List of signal codes (e.g., ['G01', 'E01', 'G3', 'IF_G01_G02'])
    
    Returns:
        List of human-readable descriptions of missing systems/signals.
        Empty list if all signals are available.
    """
    if not signals:
        return []
    
    missing = []
    pco_data = antex_data.get('pco', {}).get(antenna_type, {})
    
    # Helper to check a single frequency (e.g., "G01")
    def check_single_freq(freq_code):
        if not freq_code: return None
        sys_char = freq_code[0]
        # Map char to ANTEX system key
        if sys_char == 'G': sys_key = 'GPS'
        elif sys_char == 'R': sys_key = 'GLONASS'
        elif sys_char == 'E': sys_key = 'Galileo'
        elif sys_char == 'C': sys_key = 'BDS' # Some use Beidou, parser usually standardizes to BDS? 
        # Actually our parser might use 'BDS' or 'BeiDou'. Let's check internal structure if needed.
        # Based on typical ANTEX parsers: G->GPS, R->GLONASS, E->Galileo, C->BDS or Beidou.
        # Let's try to be robust.
        elif sys_char == 'J': sys_key = 'QZSS'
        elif sys_char == 'S': sys_key = 'SBAS'
        else: sys_key = SYSTEM_NAMES.get(sys_char, "Unknown")

        # Try to find the system dict
        # The parser (read_antex_file) usually uses specific keys. 
        # Let's check what's actually in pco_data.
        # It usually matches ANTEX satellite system codes.
        
        # fallback: check all keys if direct lookup fails
        tgt_data = pco_data.get(sys_key)
        if tgt_data is None:
             # Try alternative names for C/BDS
             if sys_char == 'C':
                 tgt_data = pco_data.get('BeiDou') or pco_data.get('COMPASS')
        
        if tgt_data is None:
             return f"{sys_key} (system not in file)"
        
        if freq_code not in tgt_data:
             return f"{sys_key} {freq_code}"
        return None

    # Group signals by system
    for signal in signals:
        if not signal:
            continue
            
        # Handle New Dynamic IF codes: IF_SysFreq1_SysFreq2
        if signal.startswith('IF_'):
            # format: IF_G01_G02
            parts = signal.split('_')
            if len(parts) == 3:
                freq1 = parts[1]
                freq2 = parts[2]
                
                err1 = check_single_freq(freq1)
                if err1: missing.append(f"{err1} (required for {signal})")
                
                err2 = check_single_freq(freq2)
                if err2: missing.append(f"{err2} (required for {signal})")
            else:
                 missing.append(f"Invalid format {signal}")
            continue

        # Handle Legacy Codes
        if signal == 'G3':
            # Iono-free GPS - needs G01 and (G02 or G05)
            gps_data = pco_data.get('GPS', {})
            has_g01 = 'G01' in gps_data
            has_g02_or_g05 = 'G02' in gps_data or 'G05' in gps_data
            if not has_g01 or not has_g02_or_g05:
                missing.append(f"GPS IF-LC (requires G01 and G02/G05)")
        elif signal in ['E00', 'E0']:
            # Iono-free Galileo - needs E01 and E05
            gal_data = pco_data.get('Galileo', {})
            has_e01 = 'E01' in gal_data
            has_e05 = 'E05' in gal_data
            if not has_e01 or not has_e05:
                missing.append(f"Galileo IF-LC (requires E01 and E05)")
        elif signal == 'R3': # Legacy R3 implied R01/R02
             err1 = check_single_freq("R01")
             err2 = check_single_freq("R02")
             if err1: missing.append(err1)
             if err2: missing.append(err2)
        else:
            # Single frequency signal
            err = check_single_freq(signal)
            if err: missing.append(err)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_missing = []
    for item in missing:
        if item not in seen:
            seen.add(item)
            unique_missing.append(item)
    
    return unique_missing


def run_processing_pipeline(config: dict, progress_callback=None) -> tuple:
    """
    Runs the full processing pipeline for a single time period.
    """
    if progress_callback is None:
        progress_callback = lambda current, total, msg: None

    total_steps = 6
    current_step = 0

    print(f"\n{'='*60}\nPROCESSING PIPELINE STARTED\n{'='*60}")

    try:
        # --- 1. Load ANTEX Files (Optimized) ---
        current_step += 1
        progress_callback(current_step, total_steps, "Loading ANTEX files")
        
        # Check if single-file mode (absolute PCC, no comparison)
        is_single_file_mode = config.get('analysis_mode') == 'single_file'
        
        # Check if pre-loaded data exists in config to avoid re-reading files
        if 'antex_data_1_obj' in config and 'antex_data_2_obj' in config:
            # print("  Using pre-loaded ANTEX data from memory.") 
            antex_data_1 = config['antex_data_1_obj']
            antex_data_2 = config['antex_data_2_obj']
        elif is_single_file_mode:
            # Single file mode - only load File 1
            if not config.get('antex_file_1'):
                 raise ValueError("ANTEX file path missing in config.")
            antex_data_1 = read_antex_file(config['antex_file_1'])
            antex_data_2 = None  # No File 2 for single-file mode
            print("  [Single File Mode] Analyzing absolute PCC values")
        else:
            if not config.get('antex_file_1') or not config.get('antex_file_2'):
                 raise ValueError("ANTEX file paths missing in config.")

            antex_data_1 = read_antex_file(config['antex_file_1'])
            antex_data_2 = read_antex_file(config['antex_file_2'])

        if not antex_data_1:
             raise ValueError("Failed to parse ANTEX file 1.")
        if not is_single_file_mode and not antex_data_2:
             raise ValueError("Failed to parse ANTEX file 2.")

        # --- 2. Antenna Validation & Normalization ---
        config_antenna_type_raw = config.get('antenna_type', '')
        
        original_name1 = _resolve_antenna_key(antex_data_1, config_antenna_type_raw)
        
        # For single-file mode, skip File 2 validation
        if is_single_file_mode:
            original_name2 = None
        else:
            original_name2 = config.get('_antenna_type_file2')
            if not original_name2:
                 original_name2 = _resolve_antenna_key(antex_data_2, config_antenna_type_raw)

        if not original_name1:
            file1_path = config.get('antex_file_1', 'File 1')
            available_1 = list(antex_data_1.get('pco', {}).keys())[:5]  # Show first 5
            raise ValueError(
                f"Antenna '{config_antenna_type_raw}' not found in:\n"
                f"  {file1_path}\n"
                f"  Available antennas: {available_1}{'...' if len(antex_data_1.get('pco', {})) > 5 else ''}"
            )
        
        if not is_single_file_mode and not original_name2:
            file2_path = config.get('antex_file_2', 'File 2')
            available_2 = list(antex_data_2.get('pco', {}).keys())[:5]  # Show first 5
            raise ValueError(
                f"Antenna '{config_antenna_type_raw}' not found in:\n"
                f"  {file2_path}\n"
                f"  Available antennas: {available_2}{'...' if len(antex_data_2.get('pco', {})) > 5 else ''}"
            )

        print(f"  File 1 Antenna Key: '{original_name1}'")
        if not is_single_file_mode:
            print(f"  File 2 Antenna Key: '{original_name2}'")
        
        # --- Check for serial number mismatch (only in comparison mode) ---
        if not is_single_file_mode:
            meta1 = antex_data_1.get('metadata', {}).get(original_name1, {})
            meta2 = antex_data_2.get('metadata', {}).get(original_name2, {})
            serial1 = meta1.get('serial', 'NONE')
            serial2 = meta2.get('serial', 'NONE')
            type1 = meta1.get('type', '')
            type2 = meta2.get('type', '')
            
            if type1 != type2:
                # Different antenna TYPES - major warning
                print(f"  [WARNING] Different antenna types!")
                print(f"            File 1 Type: '{type1}'")
                print(f"            File 2 Type: '{type2}'")
                print(f"            Comparison will proceed, but results may reflect calibration differences.")
            elif serial1 != serial2:
                # Same type but different serial numbers
                print(f"  [WARNING] Serial number mismatch!")
                print(f"            File 1 Serial: '{serial1}'")
                print(f"            File 2 Serial: '{serial2}'")
                print(f"            Comparison will proceed, but results may reflect calibration differences.")

        # Update config with resolved names
        config['antenna_type'] = original_name1
        config['_antenna_type_file2'] = original_name2
        
        config['grid_step_1'] = _get_antex_grid_step(antex_data_1, original_name1)
        config['grid_step_2'] = _get_antex_grid_step(antex_data_2, original_name2) if antex_data_2 else 5.0

        # --- Validate that requested signals exist in ANTEX files ---
        requested_signals = config.get('signals', [])
        if not requested_signals:
            raise ValueError(
                "No signal selected!\n\n"
                "Please select at least one signal (e.g., G01, E01, IF_G01_G02) before running the analysis."
            )
        missing_signals = validate_signals_in_antex(antex_data_1, original_name1, requested_signals)
        if missing_signals:
            missing_list = ', '.join(missing_signals)
            # Get list of available signals for this antenna
            pco_data = antex_data_1.get('pco', {}).get(original_name1, {})
            available_signals = []
            for system, signals_dict in pco_data.items():
                if isinstance(signals_dict, dict):
                    for sig in signals_dict.keys():
                        available_signals.append(f"{sig}")
            available_str = ', '.join(sorted(available_signals)) if available_signals else 'None found'
            raise ValueError(
                f"The following signals are not available in ANTEX File 1 for antenna '{original_name1}':\n"
                f"  ❌ Missing: {missing_list}\n\n"
                f"Available signals for this antenna:\n"
                f"  ✓ {available_str}\n\n"
                f"Please select signals that are present in the ANTEX file."
            )

        # --- 3. Load Orbit Data (Optimized) ---
        current_step += 1
        progress_callback(current_step, total_steps, "Loading Orbit Data")
        
        # Check for pre-loaded orbit data
        if 'orbit_data_obj' in config:
            orbit_data = config['orbit_data_obj']
        else:
            orbit_time = config['start_time']
            orbit_type = config['orbit_type']
            orbit_dir = config['orbit_dir']
            orbit_data = None

            if orbit_type == 'broadcast':
                path = download_and_unzip_broadcast_file(orbit_time, orbit_dir)
                if path: orbit_data = read_broadcast_file(path)
            elif orbit_type == 'final':
                path = download_final_orbit_file(orbit_time, orbit_dir)
                if path: orbit_data = read_final_orbit_file(path)

            if not orbit_data:
                raise ConnectionError(f"Failed to load valid orbit data for {orbit_time.date()}")

        config['orbit_data'] = orbit_data

        # --- 4. Compute M-Matrix ---
        current_step += 1
        progress_callback(current_step, total_steps, "Computing P2 Matrix")
        
        m_matrix, az, el, sat_ids = compute_m_matrix(config)
        if m_matrix is None:
            raise ValueError("P2 matrix computation failed (no valid satellites found).")

        # --- 5. Run Adjustment ---
        current_step += 1
        progress_callback(current_step, total_steps, "Running Adjustment")
        
        results, param_names, _, _, _ = perform_adjustment(
            config, antex_data_1, antex_data_2, m_matrix
        )

        if results is None or np.any(np.isnan(results)):
             raise ValueError("Adjustment returned invalid results (NaN/None).")
        
        # --- Save to Text File ---
        # Only save if NOT in global/timeline mode (to avoid many individual files)
        if not config.get('is_global_run', False) and not config.get('is_timeline_run', False):
            try:
                output_dir = os.path.join(os.getcwd(), 'results')
                filename_base = f"Analysis_{original_name1.replace(' ', '_')}"
                save_results_to_txt(output_dir, filename_base, param_names, results, config)
            except Exception as e:
                print(f"Warning: Could not save text report: {e}")

        # --- 6. Finalize ---
        current_step += 1
        progress_callback(current_step, total_steps, "Finalizing")
        print(f"\n[OK] Pipeline Complete. Results: {np.round(results, 4)}")

        return results, param_names, m_matrix, az, el, sat_ids

    except Exception as e:
        print(f"\n[ERROR] PIPELINE ERROR: {e}")
        progress_callback(total_steps, total_steps, f"Error: {e}")
        return None, None, None, None, None, None


def run_timeline_pipeline(config: dict, progress_callback=None):
    """
    Runs the processing pipeline for a date range with configurable intervals.
    Supports both daily and sub-daily (hourly, 3-hourly) intervals.
    OPTIMIZED: Reads ANTEX files only ONCE.
    """
    start_date = config['start_date'].date()
    end_date = config['end_date'].date()
    
    # Get interval from config (in minutes) - default to 1440 (24 hours = daily)
    interval_minutes = config.get('interval_min', 1440)
    
    # Determine if this is sub-daily analysis
    is_subdaily = interval_minutes < 1440  # Less than 24 hours
    
    if progress_callback is None:
        progress_callback = lambda c, t, m: None

    results_map = {}
    param_names = None
    timestamps = []  # Changed from 'dates' to handle datetime objects too

    # Pre-load ANTEX files
    print(f"\nRunning Time Series: {start_date} to {end_date}")
    print(f"Interval: {interval_minutes} minutes ({'sub-daily' if is_subdaily else 'daily'})")

    print("Pre-loading ANTEX files for time series analysis...")
    try:
        # Handle single file mode (absolute PCC analysis)
        is_single_file_mode = config.get('analysis_mode') == 'single_file'
        
        if not config.get('antex_file_1'):
            raise ValueError("ANTEX file 1 path missing in config.")
        
        if not is_single_file_mode and not config.get('antex_file_2'):
            raise ValueError("ANTEX file 2 path missing in config (required for comparison mode).")

        preloaded_antex_1 = read_antex_file(config['antex_file_1'])
        preloaded_antex_2 = read_antex_file(config['antex_file_2']) if not is_single_file_mode else None
        
        if not preloaded_antex_1:
            raise ValueError("Failed to pre-load ANTEX file 1.")
        if not is_single_file_mode and not preloaded_antex_2:
            raise ValueError("Failed to pre-load ANTEX file 2.")
            
    except Exception as e:
        print(f"❌ Time Series Setup Error: {e}")
        return None, None, None

    if is_subdaily:
        # --- Sub-daily processing (intervals < 24 hours) ---
        interval_delta = timedelta(minutes=interval_minutes)
        
        # Calculate total time periods
        # Start from 00:00 on start_date
        start_datetime = datetime.combine(start_date, datetime.min.time())
        # End at 00:00 on the day AFTER end_date (so we include end_date fully)
        end_datetime = datetime.combine(end_date + timedelta(days=1), datetime.min.time())
        
        total_periods = int((end_datetime - start_datetime).total_seconds() / (interval_minutes * 60))
        print(f"Total periods to process: {total_periods}")
        
        current_time = start_datetime
        period_idx = 0
        
        while current_time < end_datetime:
            period_idx += 1
            period_end = current_time + interval_delta
            
            progress_callback(period_idx, total_periods, f"Processing {current_time.strftime('%Y-%m-%d %H:%M')}")
            
            period_config = config.copy()
            period_config['start_time'] = current_time
            period_config['end_time'] = min(period_end, end_datetime)
            
            # Pass the pre-loaded ANTEX objects
            period_config['antex_data_1_obj'] = preloaded_antex_1
            period_config['antex_data_2_obj'] = preloaded_antex_2
            # Prevent saving individual per-period files in timeline mode
            period_config['is_timeline_run'] = True

            res, params, _, _, _, _ = run_processing_pipeline(period_config)

            timestamps.append(current_time)
            if res is not None:
                results_map[current_time] = res
                if not param_names: param_names = params
            else:
                results_map[current_time] = np.full(len(param_names) if param_names else 5, np.nan)
            
            current_time = period_end
            
    else:
        # --- Daily processing (original behavior) ---
        num_days = (end_date - start_date).days + 1
        print(f"Days to process: {num_days}")
        
        for i in range(num_days):
            curr_date = start_date + timedelta(days=i)
            progress_callback(i + 1, num_days, f"Processing {curr_date}")
            
            day_config = config.copy()
            day_config['start_time'] = datetime.combine(curr_date, datetime.min.time())
            day_config['end_time'] = datetime.combine(curr_date, datetime.max.time())
            
            # Pass the pre-loaded objects
            day_config['antex_data_1_obj'] = preloaded_antex_1
            day_config['antex_data_2_obj'] = preloaded_antex_2
            # Prevent saving individual per-period files in timeline mode
            day_config['is_timeline_run'] = True

            res, params, _, _, _, _ = run_processing_pipeline(day_config)

            timestamps.append(curr_date)
            if res is not None:
                results_map[curr_date] = res
                if not param_names: param_names = params
            else:
                results_map[curr_date] = np.full(len(param_names) if param_names else 5, np.nan)

    results_matrix = np.array([results_map[t] for t in timestamps])
    return timestamps, results_matrix, param_names


def run_folder_pipeline(config: dict, progress_callback=None):
    """
    Processes all ANTEX files in a folder against a reference file.
    """
    folder_path = config.get('folder_path')
    ref_file = config.get('antex_file_2')

    if not folder_path or not ref_file:
        raise ValueError("Folder path or reference file missing.")

    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.atx')])
    results_dict = {}
    param_names = None

    print(f"\nRunning Folder Analysis: {len(files)} files in {folder_path}")

    print("Pre-loading Reference ANTEX file...")
    ref_antex_data = read_antex_file(ref_file)

    failed_files = []
    
    for i, fname in enumerate(files):
        progress_callback(i + 1, len(files), f"Processing {fname}")
        
        full_path = os.path.join(folder_path, fname)
        if os.path.abspath(full_path) == os.path.abspath(ref_file):
            continue

        file_config = config.copy()
        file_config['antex_file_1'] = full_path
        file_config['analysis_mode'] = 'folder_pair'
        
        file_config['antex_data_2_obj'] = ref_antex_data
        
        # Find common antennas between folder file and reference file
        # Read the file to get available antenna keys
        try:
            file_antex_data = read_antex_file(full_path)
            if file_antex_data and file_antex_data.get('pco'):
                folder_antennas = set(file_antex_data.get('pco', {}).keys())
                ref_antennas = set(ref_antex_data.get('pco', {}).keys())
                
                # Find common antennas (exist in both files) — exact match first
                common_antennas = folder_antennas.intersection(ref_antennas)
                
                if common_antennas:
                    # Use the first common antenna (exact key match)
                    selected_antenna = sorted(list(common_antennas))[0]
                    file_config['antenna_type'] = selected_antenna
                    file_config['_antenna_type_file2'] = selected_antenna
                    file_config['antex_data_1_obj'] = file_antex_data
                    print(f"  [OK] Exact antenna match: '{selected_antenna}'")
                else:
                    # Fallback: match by antenna TYPE only (first 20 chars),
                    # ignoring serial number differences
                    type_match_f1 = None
                    type_match_ref = None
                    for fa in sorted(folder_antennas):
                        fa_type = fa[:20].strip()
                        for ra in sorted(ref_antennas):
                            ra_type = ra[:20].strip()
                            if fa_type == ra_type:
                                type_match_f1 = fa
                                type_match_ref = ra
                                break
                        if type_match_f1:
                            break
                    
                    if type_match_f1 and type_match_ref:
                        file_config['antenna_type'] = type_match_f1
                        file_config['_antenna_type_file2'] = type_match_ref
                        file_config['antex_data_1_obj'] = file_antex_data
                        print(f"  [OK] Type match: '{type_match_f1}' <-> '{type_match_ref}'")
                    else:
                        # No match at all - skip this file
                        print(f"  [SKIP] No matching antenna type between {fname} and reference")
                        print(f"         Folder file antennas: {sorted(list(folder_antennas))[:3]}...")
                        print(f"         Reference antennas: {sorted(list(ref_antennas))[:3]}...")
                        continue
            else:
                print(f"  [WARNING] Failed to parse {fname}")
                continue
        except Exception as e:
            print(f"  [ERROR] Could not read {fname}: {e}")
            continue

        try:
            res, params, _, _, _, _ = run_processing_pipeline(file_config)

            if res is not None:
                results_dict[fname] = res
                if not param_names: param_names = params
            else:
                results_dict[fname] = np.full(len(param_names) if param_names else 5, np.nan)
        except Exception as e:
            # Log which file failed and continue with other files
            error_msg = str(e).split('\n')[0]  # First line of error
            print(f"  [ERROR] Failed to process {fname}: {error_msg}")
            failed_files.append((fname, str(e)))
            results_dict[fname] = np.full(len(param_names) if param_names else 5, np.nan)
    
    # Report summary of failed files
    if failed_files:
        print(f"\n[WARNING] {len(failed_files)} file(s) failed to process:")
        for fname, err in failed_files:
            print(f"  - {fname}: {err.split(chr(10))[0]}")

    return results_dict, param_names


# --- NEW: Phase 4 Global Analysis Pipeline ---
def run_global_pipeline(config: dict, progress_callback=None):
    """
    Runs the analysis over a global grid of coordinates.
    Optimized to load ANTEX and Orbit files only ONCE.
    """
    if progress_callback is None:
        progress_callback = lambda c, t, m: None

    lat_min = config.get('lat_min', -90)
    lat_max = config.get('lat_max', 90)
    lon_min = config.get('lon_min', -180)
    lon_max = config.get('lon_max', 180)
    step = config.get('global_grid_step', 15.0)

    # Generate grid points
    lats = np.arange(lat_min, lat_max + 0.1, step)
    lons = np.arange(lon_min, lon_max + 0.1, step)
    
    total_points = len(lats) * len(lons)
    print(f"\nRunning Global Analysis: {len(lats)}x{len(lons)} = {total_points} points.")

    # --- 1. Optimization: Pre-load Data ONCE ---
    print("Pre-loading ANTEX and Orbit files...")
    
    # ANTEX
    antex_1 = read_antex_file(config['antex_file_1'])
    
    # Handle Single File mode for Global Analysis
    if config.get('analysis_mode') == 'single_file':
        antex_2 = None
    else:
        antex_2 = read_antex_file(config['antex_file_2'])
    
    # Orbit (Download once for the start date)
    orbit_time = config['start_time']
    orbit_dir = config['orbit_dir']
    orbit_type = config['orbit_type']
    orbit_data = None
    
    if orbit_type == 'broadcast':
        path = download_and_unzip_broadcast_file(orbit_time, orbit_dir)
        if path: orbit_data = read_broadcast_file(path)
    elif orbit_type == 'final':
        path = download_final_orbit_file(orbit_time, orbit_dir)
        if path: orbit_data = read_final_orbit_file(path)

    if not orbit_data:
        raise ConnectionError("Failed to load orbit data for global analysis.")
        
    # --- 2. Grid Loop ---
    # Store results in a 3D array (lat, lon, params)
    # We don't know num_params yet, so we'll init after first success
    results_grid = None 
    param_names = None

    count = 0
    
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            count += 1
            progress_callback(count, total_points, f"Processing Lat:{lat} Lon:{lon}")
            
            # Prepare config for this point
            point_config = config.copy()
            point_config['position'] = ell_to_ecef(np.deg2rad(lat), np.deg2rad(lon), 0) # Height 0
            point_config['position_type'] = 'ECEF'
            
            # Inject pre-loaded data
            point_config['antex_data_1_obj'] = antex_1
            point_config['antex_data_2_obj'] = antex_2
            point_config['orbit_data_obj'] = orbit_data
            
            # Flag to prevent saving individual text files
            point_config['is_global_run'] = True

            # Run (suppress print output to keep console clean)
            res, params, _, _, _, _ = run_processing_pipeline(point_config)
            
            if res is not None:
                if results_grid is None:
                    # Initialize grid on first success
                    param_names = params
                    results_grid = np.full((len(lats), len(lons), len(params)), np.nan)
                
                results_grid[i, j, :] = res

    print("\nGlobal Analysis Complete.")
    return lons, lats, results_grid, param_names