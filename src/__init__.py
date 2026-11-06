# src/__init__.py

"""
Main package for the Delta PCC Impact Analysis Tool.

This file makes key functions from submodules directly available
at the package level, simplifying imports.

The names are loaded on first use (PEP 562), so that importing one module,
for example ``from src.data_io import read_antex_file`` in PCC-Viewer and
ATX-Converter, does not also import the processing and plotting modules and
their large dependencies (scipy, pyproj, matplotlib, cartopy).
"""

import importlib

# Package-level name -> submodule that defines it
_EXPORTS = {
    'read_antex_file': 'data_io',
    'download_and_unzip_broadcast_file': 'data_io',
    'read_broadcast_file': 'data_io',
    'download_final_orbit_file': 'data_io',
    'read_final_orbit_file': 'data_io',
    'ell_to_ecef': 'geodesy',
    'ecef_to_topocentric': 'geodesy',
    'compute_satellite_position': 'geodesy',
    'compute_pcc': 'geodesy',
    'datetime_to_mjd': 'time_utils',
    'mjd_to_gps_week': 'time_utils',
    'datetime_to_gps_sow': 'time_utils',
    'perform_adjustment': 'adjustment',
    'compute_m_matrix': 'adjustment',
    'run_processing_pipeline': 'processing_core',
    'run_timeline_pipeline': 'processing_core',
    'plot_results_bar_chart': 'plotting',
    'plot_world_map': 'plotting',
    'plot_skyplot': 'plotting',
    'plot_timeline': 'plotting',
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


print("DPCC Analysis Package initialized.")
