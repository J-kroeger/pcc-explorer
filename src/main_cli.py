# src/main_cli.py
import argparse
import sys
import os
import numpy as np
from datetime import datetime

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.data_io import read_config_file, read_antex_file
    from src.processing_core import (
        run_processing_pipeline,
        run_timeline_pipeline,
        run_folder_pipeline,
        run_global_pipeline
    )
    from src.geodesy import ell_to_ecef
    from src.input_validation import ConfigValidator
except ImportError as e:
    print(f"[ERROR] Import Error: {e}")
    sys.exit(1)

def console_progress_callback(current, total, message):
    """Simple progress bar for the console."""
    percent = 100 * (current / float(total)) if total > 0 else 0
    bar_length = 40
    filled_length = int(bar_length * current // total) if total > 0 else 0
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    # \r allows overwriting the line
    sys.stdout.write(f'\r|{bar}| {percent:.1f}% - {message}')
    sys.stdout.flush()
    if current == total:
        print() # Newline on completion

def parse_config_types(raw_config):
    """
    Converts string values from the text file into Python types 
    required by processing_core (datetime, float, int, list).
    """
    config = raw_config.copy()
    
    # 1. Parse Dates & Times
    fmt_dt = "%Y-%m-%d %H:%M"
    fmt_d = "%Y-%m-%d"
    
    # Helper to combine separate date/time fields if they exist
    if 'start_date' in config and 'start_time' in config:
        # Check if they are already datetime objects (rare) or strings
        s_date = config['start_date']
        s_time = config['start_time']
        # If running timeline, we might interpret start_date as the date object
        try:
            config['start_time'] = datetime.strptime(f"{s_date} {s_time}", fmt_dt)
        except ValueError:
            # Fallback for Timeline mode where we just need the date object
            config['start_date'] = datetime.strptime(s_date, fmt_d)
            
    if 'end_date' in config and 'end_time' in config:
        e_date = config['end_date']
        e_time = config['end_time']
        try:
            config['end_time'] = datetime.strptime(f"{e_date} {e_time}", fmt_dt)
        except ValueError:
            config['end_date'] = datetime.strptime(e_date, fmt_d)

    # 2. Parse Numbers
    for key in ['sampling_rate_sec', 'interval_min']:
        if key in config: config[key] = int(config[key])
        
    float_keys = [
        'latitude', 'longitude', 'height', 
        'x_coordinate', 'y_coordinate', 'z_coordinate',
        'elevation_mask_angle', 'lat_min', 'lat_max', 
        'lon_min', 'lon_max', 'global_grid_step'
    ]
    for key in float_keys:
        if key in config: config[key] = float(config[key])

    # 3. Parse Lists (Azimuth masks, signals)
    if 'signals' in config and isinstance(config['signals'], str):
        # Convert comma-separated string to list
        config['signals'] = [s.strip() for s in config['signals'].split(',')]
        
    # Azimuth masks usually come as a complex string in text files or skipped.
    # For CLI, we assume they might be passed as a list of tuples if manually constructed,
    # or we might need to parse a string representation. 
    # For now, if it's not a list, initialize empty.
    if 'azimuth_mask_values' not in config:
        config['azimuth_mask_values'] = []

    # 4. Handle Position (Ellipsoidal -> ECEF conversion if needed)
    pos_type = config.get('position_type', 'Ellipsoidal')
    if pos_type in ['Ellipsoidal', 'Map'] and 'latitude' in config:
        lat = np.deg2rad(config['latitude'])
        lon = np.deg2rad(config['longitude'])
        h = config['height']
        config['position'] = ell_to_ecef(lat, lon, h)
    elif pos_type == 'ECEF' and 'x_coordinate' in config:
        config['position'] = np.array([config['x_coordinate'], config['y_coordinate'], config['z_coordinate']])
        
    return config

def main():
    parser = argparse.ArgumentParser(description="PCC-Explorer Command Line Interface")
    parser.add_argument("config_file", help="Path to the configuration parameter file (.txt)")
    parser.add_argument("--mode", choices=['single', 'timeline', 'global', 'folder'], 
                        help="Override analysis mode (optional)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config_file):
        print(f"[ERROR] Config file not found: {args.config_file}")
        sys.exit(1)
        
    print(f"--- PCC-Explorer CLI ---")
    print(f"Loading configuration: {args.config_file}")
    
    # 1. Read Raw Config
    try:
        raw_config = read_config_file(args.config_file)
        # 2. Convert Types
        config = parse_config_types(raw_config)
    except Exception as e:
        print(f"[ERROR] Configuration Error: {e}")
        sys.exit(1)

    # 3. Determine Mode
    mode = args.mode if args.mode else config.get('analysis_mode', 'two_files')
    
    # Smart detection for global/timeline if not explicitly set
    if args.mode is None:
        if config.get('lat_min') is not None and config.get('lon_min') is not None:
            mode = 'global'
        elif config.get('start_date') and config.get('end_date') and 'start_time' not in config:
             # Heuristic: if start_time is missing but start_date exists, it's likely timeline
             mode = 'timeline'
    
    print(f"Mode: {mode}")

    # 4. Run Pipeline
    try:
        if mode == 'global':
            run_global_pipeline(config, progress_callback=console_progress_callback)
        elif mode == 'timeline':
            run_timeline_pipeline(config, progress_callback=console_progress_callback)
        elif mode == 'whole_folder' or mode == 'folder':
            run_folder_pipeline(config, progress_callback=console_progress_callback)
        else:
            # Default: Single Analysis (two_files)
            # Pipeline returns 6 values (results, params, m_matrix, az, el, sat_ids)
            run_processing_pipeline(config, progress_callback=console_progress_callback)
            
    except Exception as e:
        print(f"\n[ERROR] Execution Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()