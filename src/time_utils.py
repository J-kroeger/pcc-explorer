# src/time_utils.py
"""
GPS/UTC time conversion utilities.
"""

import math
from datetime import datetime

#: MJD 0 = 1858-11-17 00:00.
MJD_EPOCH = datetime(1858, 11, 17)


def datetime_to_mjd(dt_obj: datetime) -> float:
    """
    Converts a Python datetime object to Modified Julian Date (MJD).

    Plain day counting from MJD 0, without leap seconds, which is what GPS
    time needs. The day number is exact; within a UTC leap-second day the
    fraction can differ from a UTC-scale conversion by up to one second.

    Args:
        dt_obj: The datetime object to convert.

    Returns:
        The Modified Julian Date as a float.
    """
    return (dt_obj - MJD_EPOCH).total_seconds() / 86400.0


def mjd_to_gps_week(mjd: float) -> tuple:
    """
    Converts a Modified Julian Date to GPS Week and Day of Week.
    """
    # GPS time started at MJD 44244 (Jan 6, 1980)
    gps_days = math.floor(mjd - 44244)
    gps_week = gps_days // 7
    day_of_week = gps_days % 7  # 0=Sunday, ..., 6=Saturday
    return gps_week, day_of_week
# In src/adjustment.py


def datetime_to_gps_sow(dt_obj: datetime) -> tuple:
    """Converts a Python datetime object to GPS week and seconds of the week."""
    gps_epoch = datetime(1980, 1, 6)
    time_diff = dt_obj - gps_epoch
    gps_week = time_diff.days // 7
    seconds_of_week = (time_diff.days % 7) * 86400 + time_diff.seconds + dt_obj.microsecond / 1e6
    return gps_week, seconds_of_week
