# src/input_validation.py
"""
Comprehensive input validation utilities for GNSS analysis configuration.

This module provides robust validation for all user inputs, configuration
parameters, and file inputs to prevent runtime errors and provide clear
feedback to users.
"""

from datetime import datetime, timedelta
import os
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Any


class ValidationError(Exception):
    """Custom exception for validation errors with user-friendly messages."""
    pass


class ConfigValidator:
    """Validates configuration dictionaries for GNSS processing."""

    # Constants for validation
    MIN_SAMPLING_RATE = 1  # seconds  <--- MODIFIED FROM 30 to 1
    MAX_SAMPLING_RATE = 3600  # seconds
    MIN_ELEVATION = 0  # degrees
    MAX_ELEVATION = 90  # degrees
    MIN_EARTH_RADIUS = 6200000  # meters (lowered to allow negative heights)
    MAX_ORBIT_ALTITUDE = 30000000  # meters
    FINAL_ORBIT_DELAY_DAYS = 21
    BROADCAST_DELAY_DAYS = 2
    MAX_TIMELINE_DAYS = 90
    MIN_TIMELINE_DAYS = 1
    MAX_TIME_SPAN_HOURS = 48
    MIN_TIME_SPAN_HOURS = 1

    @staticmethod
    def validate_config(config: Dict[str, Any], analysis_type: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a complete configuration dictionary.

        Args:
            config: Configuration dictionary
            analysis_type: One of 'single', 'timeline', 'global', 'folder'

        Returns:
            tuple: (is_valid, error_message)
                  error_message is None if validation passes
        """
        try:
            # Common validations FIRST (catches missing ANTEX files early)
            ConfigValidator._validate_common_settings(config, analysis_type)

            # Then validate based on analysis type
            if analysis_type == 'folder':
                ConfigValidator._validate_folder_config(config)
            elif analysis_type == 'timeline':
                ConfigValidator._validate_timeline_config(config)
            elif analysis_type == 'global':
                ConfigValidator._validate_global_config(config)
            else:  # single
                ConfigValidator._validate_single_config(config)

            return True, None

        except ValidationError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected validation error: {e}"

    @staticmethod
    def _validate_single_config(config: Dict[str, Any]):
        """Validate configuration for single-point analysis."""
        # Validate ANTEX files
        ConfigValidator._validate_antex_files(config)

        # Validate position
        ConfigValidator._validate_position(config)

        # Validate time range
        ConfigValidator._validate_datetime_range(config)

        # Validate signals
        ConfigValidator._validate_signals(config)

    @staticmethod
    def _validate_timeline_config(config: Dict[str, Any]):
        """Validate configuration for timeline analysis."""
        ConfigValidator._validate_antex_files(config)
        ConfigValidator._validate_position(config)
        ConfigValidator._validate_timeline_dates(config)
        ConfigValidator._validate_signals(config)

    @staticmethod
    def _validate_global_config(config: Dict[str, Any]):
        """Validate configuration for global analysis."""
        ConfigValidator._validate_antex_files(config)
        ConfigValidator._validate_datetime_range(config)
        ConfigValidator._validate_signals(config)

        # Validate grid parameters
        grid_step = config.get('global_grid_step', 15)
        if not isinstance(grid_step, (int, float)) or grid_step <= 0:
            raise ValidationError("Global grid step must be a positive number.")
        if grid_step < 5:
            raise ValidationError("Global grid step should be at least 5 degrees for performance.")
        if grid_step > 45:
            raise ValidationError("Global grid step should not exceed 45 degrees for accuracy.")

    @staticmethod
    def _validate_folder_config(config: Dict[str, Any]):
        """Validate configuration for folder processing."""
        folder_path = config.get('folder_path')

        if not folder_path:
            raise ValidationError("Please select an ANTEX folder.")

        if not os.path.exists(folder_path):
            raise ValidationError(f"Folder not found:\n{folder_path}")

        if not os.path.isdir(folder_path):
            raise ValidationError(f"Path is not a directory:\n{folder_path}")

        # Check if folder contains .atx files
        try:
            atx_files = [f for f in os.listdir(folder_path)
                        if f.lower().endswith('.atx')]
        except PermissionError:
            raise ValidationError(f"Permission denied accessing folder:\n{folder_path}")

        if len(atx_files) == 0:
            raise ValidationError(
                f"No ANTEX (.atx) files found in folder:\n{folder_path}"
            )

        if len(atx_files) < 2:
            raise ValidationError(
                f"Folder must contain at least 2 ANTEX files for comparison.\n"
                f"Found only {len(atx_files)} file(s)."
            )

    @staticmethod
    def _validate_antex_files(config: Dict[str, Any]):
        """Validate ANTEX file inputs."""
        antex_file_1 = config.get('antex_file_1')
        antex_file_2 = config.get('antex_file_2')
        is_single_file_mode = config.get('analysis_mode') == 'single_file'

        if not antex_file_1:
            raise ValidationError("Please select the first ANTEX file.")

        if not os.path.exists(antex_file_1):
            raise ValidationError(f"First ANTEX file not found:\n{antex_file_1}")

        if not os.path.isfile(antex_file_1):
            raise ValidationError(f"First ANTEX path is not a file:\n{antex_file_1}")

        if not antex_file_1.lower().endswith('.atx'):
            raise ValidationError(
                f"First file does not appear to be an ANTEX file (.atx):\n{antex_file_1}"
            )

        # Skip File 2 validation for single-file mode
        if not is_single_file_mode:
            if not antex_file_2:
                raise ValidationError("Please select the second ANTEX file.")

            if not os.path.exists(antex_file_2):
                raise ValidationError(f"Second ANTEX file not found:\n{antex_file_2}")

            if not os.path.isfile(antex_file_2):
                raise ValidationError(f"Second ANTEX path is not a file:\n{antex_file_2}")

            if not antex_file_2.lower().endswith('.atx'):
                raise ValidationError(
                    f"Second file does not appear to be an ANTEX file (.atx):\n{antex_file_2}"
                )

        # Check file sizes (basic sanity check)
        try:
            size1 = os.path.getsize(antex_file_1)
            if size1 < 100:
                raise ValidationError(f"First ANTEX file is too small ({size1} bytes). File may be corrupt.")
            
            if not is_single_file_mode and antex_file_2:
                size2 = os.path.getsize(antex_file_2)
                if size2 < 100:
                    raise ValidationError(f"Second ANTEX file is too small ({size2} bytes). File may be corrupt.")
        except OSError as e:
            raise ValidationError(f"Error checking ANTEX file sizes: {e}")

        # Validate antenna type
        antenna_type = config.get('antenna_type')
        if not antenna_type or not antenna_type.strip():
            raise ValidationError("Please select an antenna type.")

    @staticmethod
    def _validate_position(config: Dict[str, Any]):
        """Validate station position."""
        if 'position' not in config:
            raise ValidationError("Station position is missing from configuration.")

        pos = config['position']

        # Handle both tuples and numpy arrays
        if isinstance(pos, tuple):
            pos = np.array(pos)
            config['position'] = pos  # Update config with numpy array
        elif not isinstance(pos, np.ndarray):
            raise ValidationError("Position must be a numpy array or tuple.")

        if len(pos) != 3:
            raise ValidationError(
                f"Position must have exactly 3 coordinates (X, Y, Z).\n"
                f"Got {len(pos)} values."
            )

        # Check for NaN or infinity
        if np.any(np.isnan(pos)):
            raise ValidationError("Position contains NaN values.")

        if np.any(np.isinf(pos)):
            raise ValidationError("Position contains infinite values.")

        # Check if position is reasonable (on or above Earth's surface)
        pos_magnitude = np.linalg.norm(pos)

        if pos_magnitude < ConfigValidator.MIN_EARTH_RADIUS:
            raise ValidationError(
                f"Position magnitude ({pos_magnitude/1000:.1f} km) is less than Earth's radius.\n"
                f"Please check your coordinates."
            )

        if pos_magnitude > ConfigValidator.MAX_ORBIT_ALTITUDE:
            raise ValidationError(
                f"Position magnitude ({pos_magnitude/1000:.1f} km) is unreasonably large.\n"
                f"Please check your coordinates."
            )

    @staticmethod
    def _validate_signals(config: Dict[str, Any]):
        """Validate signal selection."""
        signals = config.get('signals', [])

        if not signals or len(signals) == 0:
            raise ValidationError("Please select at least one signal/frequency.")

        # Validate signal format
        # Allow:
        # 1. Single frequencies: G01, R02, E05, C02, J01, S01
        # 2. Legacy IF codes: G3, R3, E00, C3
        # 3. New Dynamic IF codes: IF_G01_G02, IF_E01_E05, etc.
        valid_pattern = re.compile(r'^(?:[GRECJSC]\d{2}|G3|R3|E00|C3|IF_[GRECJSC]\d{2}_[GRECJSC]\d{2})$')

        for signal in signals:
            if not isinstance(signal, str):
                raise ValidationError(f"Invalid signal type: {signal}")

            if not valid_pattern.match(signal):
                raise ValidationError(
                    f"Invalid signal format: '{signal}'\n"
                    f"Expected format example: G01, G3, or IF_G01_G02"
                )

    @staticmethod
    def _validate_datetime_range(config: Dict[str, Any]):
        """Validate start_time and end_time."""
        if 'start_time' not in config:
            raise ValidationError("Start time is missing.")

        if 'end_time' not in config:
            raise ValidationError("End time is missing.")

        start = config['start_time']
        end = config['end_time']

        if not isinstance(start, datetime):
            raise ValidationError("Start time must be a datetime object.")

        if not isinstance(end, datetime):
            raise ValidationError("End time must be a datetime object.")

        if start >= end:
            raise ValidationError(
                f"End time must be after start time.\n"
                f"Start: {start.strftime('%Y-%m-%d %H:%M')}\n"
                f"End: {end.strftime('%Y-%m-%d %H:%M')}"
            )

        duration_hours = (end - start).total_seconds() / 3600

        if duration_hours < ConfigValidator.MIN_TIME_SPAN_HOURS:
            raise ValidationError(
                f"Time span should be at least {ConfigValidator.MIN_TIME_SPAN_HOURS} hour.\n"
                f"Current span: {duration_hours:.1f} hours"
            )

        if duration_hours > ConfigValidator.MAX_TIME_SPAN_HOURS:
            # This is a warning, not a hard error
            print(
                f"WARNING: Long time span ({duration_hours:.1f} hours). "
                f"Processing may take a while."
            )

    @staticmethod
    def _validate_timeline_dates(config: Dict[str, Any]):
        """Validate start_date and end_date for timeline analysis."""
        if 'start_date' not in config:
            raise ValidationError("Start date is missing.")

        if 'end_date' not in config:
            raise ValidationError("End date is missing.")

        start = config['start_date']
        end = config['end_date']

        if not isinstance(start, datetime):
            raise ValidationError("Start date must be a datetime object.")

        if not isinstance(end, datetime):
            raise ValidationError("End date must be a datetime object.")

        # Allow same day (inclusive range)
        if start > end:
            raise ValidationError(
                f"End date must be on or after start date.\n"
                f"Start: {start.strftime('%Y-%m-%d')}\n"
                f"End: {end.strftime('%Y-%m-%d')}"
            )

        duration_days = (end - start).days + 1

        if duration_days < ConfigValidator.MIN_TIMELINE_DAYS:
            raise ValidationError("Timeline must span at least one day.")

        if duration_days > ConfigValidator.MAX_TIMELINE_DAYS:
            # Warning, not error
            print(
                f"WARNING: Long timeline ({duration_days} days). "
                f"Processing may take a while."
            )

    @staticmethod
    def _validate_common_settings(config: Dict[str, Any], analysis_type: str):
        """Validate settings common to all analysis types."""

        # Validate sampling rate
        sampling_rate = config.get('sampling_rate_sec')

        if sampling_rate is None:
            raise ValidationError("Sampling rate is missing.")

        try:
            sampling_rate = float(sampling_rate)
        except (TypeError, ValueError):
            raise ValidationError(f"Sampling rate must be a number, got: {sampling_rate}")

        if sampling_rate <= 0:
            raise ValidationError(f"Sampling rate must be positive. Got: {sampling_rate} seconds")

        if sampling_rate < ConfigValidator.MIN_SAMPLING_RATE:
            raise ValidationError(
                f"Sampling rate should be at least {ConfigValidator.MIN_SAMPLING_RATE} seconds "
                f"for reliable results. Got: {sampling_rate} seconds"
            )

        if sampling_rate > ConfigValidator.MAX_SAMPLING_RATE:
            raise ValidationError(
                f"Sampling rate should not exceed {ConfigValidator.MAX_SAMPLING_RATE} seconds (1 hour)."
            )

        # Validate orbit type
        orbit_type = config.get('orbit_type')

        if not orbit_type:
            raise ValidationError("Orbit type is missing.")

        if orbit_type not in ['broadcast', 'final']:
            raise ValidationError(
                f"Invalid orbit type: '{orbit_type}'\n"
                f"Must be either 'broadcast' or 'final'"
            )

        # Check orbit availability based on date
        if analysis_type != 'timeline':
            start_time = config.get('start_time') or config.get('start_date')
            if start_time and orbit_type == 'final':
                days_ago = (datetime.now() - start_time).days
                if days_ago < ConfigValidator.FINAL_ORBIT_DELAY_DAYS:
                    print(
                        f"WARNING: Selected date is only {days_ago} days ago.\n"
                        f"Final orbits may not be available yet (typically {ConfigValidator.FINAL_ORBIT_DELAY_DAYS} day delay).\n"
                        f"Consider using broadcast ephemeris for recent dates."
                    )

        # Validate elevation mask
        if config.get('elevation_mask_active'):
            mask_angle = config.get('elevation_mask_angle')

            if mask_angle is None:
                raise ValidationError("Elevation mask is active but angle is not specified.")

            try:
                mask_angle = float(mask_angle)
            except (TypeError, ValueError):
                raise ValidationError(f"Elevation mask angle must be a number, got: {mask_angle}")

            if not (ConfigValidator.MIN_ELEVATION <= mask_angle <= ConfigValidator.MAX_ELEVATION):
                raise ValidationError(
                    f"Elevation mask angle must be between {ConfigValidator.MIN_ELEVATION} "
                    f"and {ConfigValidator.MAX_ELEVATION} degrees.\n"
                    f"Got: {mask_angle}°"
                )

        # Validate azimuth mask - add default for missing key
        if config.get('azimuth_mask_active', False):
            masks = config.get('azimuth_mask_values', [])

            if not masks:
                raise ValidationError("Azimuth mask is active but no mask values specified.")

            for i, mask in enumerate(masks):
                if len(mask) != 3:
                    raise ValidationError(
                        f"Azimuth mask {i+1} must have 3 values (from, to, elevation).\n"
                        f"Got {len(mask)} values."
                    )

                az_from, az_to, el_mask = mask

                try:
                    az_from = float(az_from)
                    az_to = float(az_to)
                    el_mask = float(el_mask)
                except (TypeError, ValueError):
                    raise ValidationError(f"Azimuth mask {i+1} contains non-numeric values.")

                if not (0 <= az_from <= 360):
                    raise ValidationError(
                        f"Azimuth mask {i+1}: 'from' angle must be between 0 and 360°.\n"
                        f"Got: {az_from}°"
                    )

                if not (0 <= az_to <= 360):
                    raise ValidationError(
                        f"Azimuth mask {i+1}: 'to' angle must be between 0 and 360°.\n"
                        f"Got: {az_to}°"
                    )

                if not (ConfigValidator.MIN_ELEVATION <= el_mask <= ConfigValidator.MAX_ELEVATION):
                    raise ValidationError(
                        f"Azimuth mask {i+1}: elevation must be between "
                        f"{ConfigValidator.MIN_ELEVATION} and {ConfigValidator.MAX_ELEVATION}°.\n"
                        f"Got: {el_mask}°"
                    )

        # Validate weighting model
        weighting = config.get('weighting')
        valid_weightings = ['sin', 'sin2', 'unit']

        if weighting and weighting not in valid_weightings:
            raise ValidationError(
                f"Invalid weighting model: '{weighting}'\n"
                f"Must be one of: {', '.join(valid_weightings)}"
            )

        # Validate troposphere model
        tropo_model = config.get('tropo_model')
        valid_tropo = ['1/sin(Elevation)', 'GMF', 'None']

        if tropo_model and tropo_model not in valid_tropo:
            raise ValidationError(
                f"Invalid troposphere model: '{tropo_model}'\n"
                f"Must be one of: {', '.join(valid_tropo)}"
            )


def validate_coordinate_input(coord_str: str, coord_type: str) -> float:
    """
    Validate and parse a coordinate string.

    Args:
        coord_str: String representation of coordinate
        coord_type: Type of coordinate ('lat', 'lon', 'height', 'x', 'y', 'z')

    Returns:
        float: Parsed coordinate value

    Raises:
        ValidationError: If coordinate is invalid
    """
    try:
        value = float(coord_str)
    except ValueError:
        raise ValidationError(
            f"Invalid {coord_type} value: '{coord_str}'\n"
            f"Must be a number."
        )

    # Validate ranges
    if coord_type == 'lat':
        if not (-90 <= value <= 90):
            raise ValidationError(
                f"Latitude must be between -90 and 90 degrees.\n"
                f"Got: {value}°"
            )
    elif coord_type == 'lon':
        if not (-180 <= value <= 180):
            raise ValidationError(
                f"Longitude must be between -180 and 180 degrees.\n"
                f"Got: {value}°"
            )
    elif coord_type == 'height':
        if not (-1000 <= value <= 10000):
            raise ValidationError(
                f"Height should be between -1000m and 10000m.\n"
                f"Got: {value}m"
            )
    elif coord_type in ['x', 'y', 'z']:
        if abs(value) > 10000000:  # 10,000 km
            raise ValidationError(
                f"ECEF {coord_type.upper()} coordinate seems unreasonable.\n"
                f"Got: {value/1000:.1f} km"
            )

    return value


def validate_datetime_string(date_str: str, time_str: str = None) -> datetime:
    """
    Validate and parse datetime strings.

    Args:
        date_str: Date string in YYYY-MM-DD format
        time_str: Optional time string in HH:MM format

    Returns:
        datetime: Parsed datetime object

    Raises:
        ValidationError: If datetime is invalid
    """
    # Validate date format
    date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    if not date_pattern.match(date_str):
        raise ValidationError(
            f"Invalid date format: '{date_str}'\n"
            f"Expected format: YYYY-MM-DD"
        )

    # Parse date
    try:
        if time_str:
            time_pattern = re.compile(r'^\d{1,2}:\d{2}$')
            if not time_pattern.match(time_str):
                raise ValidationError(
                    f"Invalid time format: '{time_str}'\n"
                    f"Expected format: HH:MM"
                )
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        else:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        raise ValidationError(f"Invalid date/time: {e}")

    # Check if date is reasonable
    if dt.year < 1980:
        raise ValidationError(
            f"Date is before GPS epoch (1980-01-06).\n"
            f"Got: {dt.strftime('%Y-%m-%d')}"
        )

    if dt > datetime.now() + timedelta(days=1):
        raise ValidationError(
            f"Date is in the future.\n"
            f"Got: {dt.strftime('%Y-%m-%d %H:%M')}"
        )

    return dt


def validate_file_path(file_path: str, file_type: str = "file") -> bool:
    """
    Validate a file path.

    Args:
        file_path: Path to validate
        file_type: Type description for error messages

    Returns:
        bool: True if valid

    Raises:
        ValidationError: If path is invalid
    """
    if not file_path:
        raise ValidationError(f"Please specify a {file_type} path.")

    if not os.path.exists(file_path):
        raise ValidationError(f"{file_type} not found:\n{file_path}")

    if not os.path.isfile(file_path):
        raise ValidationError(f"Path is not a file:\n{file_path}")

    # Check read permissions
    if not os.access(file_path, os.R_OK):
        raise ValidationError(f"No read permission for {file_type}:\n{file_path}")

    return True


def validate_directory_path(dir_path: str, dir_type: str = "directory") -> bool:
    """
    Validate a directory path.

    Args:
        dir_path: Path to validate
        dir_type: Type description for error messages

    Returns:
        bool: True if valid

    Raises:
        ValidationError: If path is invalid
    """
    if not dir_path:
        raise ValidationError(f"Please specify a {dir_type} path.")

    if not os.path.exists(dir_path):
        raise ValidationError(f"{dir_type} not found:\n{dir_path}")

    if not os.path.isdir(dir_path):
        raise ValidationError(f"Path is not a directory:\n{dir_path}")

    # Check read and write permissions
    if not os.access(dir_path, os.R_OK):
        raise ValidationError(f"No read permission for {dir_type}:\n{dir_path}")

    if not os.access(dir_path, os.W_OK):
        raise ValidationError(f"No write permission for {dir_type}:\n{dir_path}")

    return True