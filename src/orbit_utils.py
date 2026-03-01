# src/orbit_utils.py
"""
Orbit interpolation and satellite ground track utilities.
"""
import numpy as np
from datetime import datetime, timedelta
from .geodesy import ecef_to_ell, compute_satellite_position
from .time_utils import datetime_to_gps_sow

def interpolate_orbit(orbit_records: list, target_time: datetime) -> np.ndarray:
    """
    Interpolates satellite position from a list of (time, position) records using linear interpolation.
    
    Args:
        orbit_records: List of (datetime, position_array) tuples
        target_time: Time for which to interpolate position
        
    Returns:
        np.ndarray: Interpolated ECEF position [x, y, z] or None if insufficient data
    """
    if not orbit_records or len(orbit_records) < 2:
        return None

    before_record = None
    after_record = None
    
    # Find bounding records
    for record in orbit_records:
        if record[0] <= target_time:
            before_record = record
        elif record[0] > target_time:
            after_record = record
            break

    if not before_record or not after_record:
        return None

    t_before, pos_before = before_record
    t_after, pos_after = after_record

    total_time_diff = (t_after - t_before).total_seconds()
    target_time_diff = (target_time - t_before).total_seconds()

    if total_time_diff == 0:
        return pos_before

    # Linear interpolation
    weight = target_time_diff / total_time_diff
    interpolated_pos = pos_before + weight * (pos_after - pos_before)

    return interpolated_pos

def propagate_broadcast_orbit(ephemeris: dict, start_time: datetime, end_time: datetime,
                              station_pos: np.ndarray, step_seconds: int = 300) -> list:
    """
    Propagate satellite orbit using broadcast ephemeris over a time span.
    
    Args:
        ephemeris: Ephemeris record dictionary
        start_time: Start of propagation period
        end_time: End of propagation period
        station_pos: Station ECEF position (for travel time correction)
        step_seconds: Time step for propagation (default 5 minutes)
        
    Returns:
        list: List of (datetime, position_array) tuples
    """
    track_points = []
    current_time = start_time
    
    while current_time <= end_time:
        try:
            _, reception_sow = datetime_to_gps_sow(current_time)
            sat_pos = compute_satellite_position(ephemeris, reception_sow, station_pos)
            
            if sat_pos is not None and not np.any(np.isnan(sat_pos)):
                track_points.append((current_time, sat_pos))
        except Exception as e:
            print(f"Warning: Failed to compute position at {current_time}: {e}")
        
        current_time += timedelta(seconds=step_seconds)
    
    return track_points

def get_ground_tracks(orbit_data: dict, system: str, prn: int, 
                     start_time: datetime = None, end_time: datetime = None,
                     station_pos: np.ndarray = None) -> list:
    """
    Extracts the latitude and longitude track for a single satellite.
    
    This function handles both final orbits (SP3) and broadcast ephemeris.
    For broadcast ephemeris, it propagates the orbit using Keplerian elements.
    
    Args:
        orbit_data: Orbit data dictionary (either SP3 or broadcast format)
        system: GNSS system character ('G', 'R', 'E', 'C')
        prn: Satellite PRN number
        start_time: Optional start time for broadcast propagation
        end_time: Optional end time for broadcast propagation
        station_pos: Station ECEF position (required for broadcast propagation)
        
    Returns:
        list: List of (longitude_deg, latitude_deg) tuples
    """
    tracks = []
    
    try:
        # Determine if this is final orbit (SP3) or broadcast ephemeris data
        is_final_orbit = 'orbits' in orbit_data
        
        if is_final_orbit:
            # Handle SP3 final orbit structure: {'orbits': {'G': {1: [(datetime, pos), ...]}}}
            if system in orbit_data['orbits'] and prn in orbit_data['orbits'][system]:
                records = orbit_data['orbits'][system][prn]
                
                # Process all position records for this satellite
                for time_stamp, pos in records:
                    if pos is not None and len(pos) == 3:
                        try:
                            # Convert ECEF to lat/lon
                            lon_rad, lat_rad, _ = ecef_to_ell(pos[0], pos[1], pos[2])
                            lon_deg, lat_deg = np.rad2deg(lon_rad), np.rad2deg(lat_rad)
                            
                            # Handle longitude wrapping (ensure -180 to +180)
                            if lon_deg > 180:
                                lon_deg -= 360
                            elif lon_deg < -180:
                                lon_deg += 360
                                
                            tracks.append((lon_deg, lat_deg))
                        except Exception as e:
                            print(f"Warning: Could not convert position for {system}{prn:02d}: {e}")
                            continue
        else:
            # Handle broadcast ephemeris structure: {system: {prn: [ephemeris_records]}}
            if system in orbit_data and prn in orbit_data[system]:
                ephemeris_records = orbit_data[system][prn]
                
                if not ephemeris_records:
                    print(f"Warning: No ephemeris records for {system}{prn:02d}")
                    return []
                
                if station_pos is None:
                    # Use Earth center as default (less accurate but functional)
                    station_pos = np.array([0.0, 0.0, 0.0])
                    print(f"Warning: No station position provided for {system}{prn:02d}, using Earth center")
                
                # Determine time range for propagation
                if start_time is None or end_time is None:
                    # Use the epoch time of the ephemeris records
                    epochs = [rec['epoch'] for rec in ephemeris_records]
                    start_time = min(epochs)
                    end_time = max(epochs) + timedelta(hours=2)  # Extend 2 hours past last epoch
                
                # Use the most recent ephemeris for propagation
                latest_ephemeris = max(ephemeris_records, key=lambda x: x['epoch'])
                
                # Propagate the orbit
                track_points = propagate_broadcast_orbit(
                    latest_ephemeris, start_time, end_time, station_pos, step_seconds=300
                )
                
                # Convert ECEF positions to lat/lon
                for time_stamp, pos in track_points:
                    try:
                        lon_rad, lat_rad, _ = ecef_to_ell(pos[0], pos[1], pos[2])
                        lon_deg, lat_deg = np.rad2deg(lon_rad), np.rad2deg(lat_rad)
                        
                        # Handle longitude wrapping
                        if lon_deg > 180:
                            lon_deg -= 360
                        elif lon_deg < -180:
                            lon_deg += 360
                            
                        tracks.append((lon_deg, lat_deg))
                    except Exception as e:
                        print(f"Warning: Could not convert position for {system}{prn:02d}: {e}")
                        continue
    
    except Exception as e:
        print(f"Error processing ground tracks for {system}{prn:02d}: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    if tracks:
        print(f"Generated {len(tracks)} track points for satellite {system}{prn:02d}")
    else:
        print(f"Warning: No track points generated for {system}{prn:02d}")
    
    return tracks

def validate_orbit_data(orbit_data: dict) -> dict:
    """
    Validate and analyze orbit data structure.
    
    Args:
        orbit_data: Orbit data dictionary
        
    Returns:
        dict: Summary of orbit data including systems, satellites, and time spans
    """
    summary = {
        'type': 'unknown',
        'systems': {},
        'total_satellites': 0,
        'time_span': None
    }
    
    if 'orbits' in orbit_data:
        summary['type'] = 'final'
        for system, sats in orbit_data['orbits'].items():
            sat_count = len(sats)
            summary['systems'][system] = sat_count
            summary['total_satellites'] += sat_count
            
            # Get time span
            all_times = []
            for prn, records in sats.items():
                all_times.extend([rec[0] for rec in records])
            if all_times:
                summary['time_span'] = (min(all_times), max(all_times))
    else:
        summary['type'] = 'broadcast'
        for system, sats in orbit_data.items():
            if isinstance(sats, dict):
                sat_count = len(sats)
                summary['systems'][system] = sat_count
                summary['total_satellites'] += sat_count
                
                # Get time span from epochs
                all_epochs = []
                for prn, records in sats.items():
                    all_epochs.extend([rec['epoch'] for rec in records])
                if all_epochs:
                    summary['time_span'] = (min(all_epochs), max(all_epochs))
    
    return summary