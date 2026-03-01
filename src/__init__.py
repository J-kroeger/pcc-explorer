# src/__init__.py

"""
Main package for the Delta PCC Impact Analysis Tool.

This file makes key functions from submodules directly available
at the package level, simplifying imports.
"""

# Import key functions from each module
from .data_io import (
    read_antex_file,
    download_and_unzip_broadcast_file,
    read_broadcast_file,
    download_final_orbit_file,
    read_final_orbit_file
)

from .geodesy import (
    ell_to_ecef,
    ecef_to_topocentric,
    compute_satellite_position,
    compute_pcc
)

from .time_utils import (
    datetime_to_mjd,
    mjd_to_gps_week,
    datetime_to_gps_sow
)

from .adjustment import (
    perform_adjustment,
    compute_m_matrix
)

from .processing_core import (
    run_processing_pipeline,
    run_timeline_pipeline
)

from .plotting import (
    plot_results_bar_chart,
    plot_world_map,
    plot_skyplot,
    plot_timeline
)

print("DPCC Analysis Package initialized.")