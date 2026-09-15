# src/rinex_bridge.py
"""
RINEX-Masker Integration Bridge for PCC-Explorer.

Converts RINEX observation files into PCC-Explorer's LoS observation CSV format
by computing satellite azimuth/elevation angles from SP3 precise orbits.

Dependencies:
    - RINEX-Masker project (rinex_handler.py, satellite_position.py)
      Must be on sys.path or installed as a package.
"""

import os
import sys
import csv
import json
import time
import numpy as np
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# RINEX-Masker import helper
# ---------------------------------------------------------------------------

_RINEX_MASKER_MODULE = None  # Cached module references
_MASKER_PATH_OVERRIDE = None  # Folder explicitly chosen by the user
_RESOLVED_MASKER_PATH = None  # Where the modules were actually found


def _settings_file() -> str:
    """Path of the small user-settings file (not an analysis config)."""
    base = os.environ.get('APPDATA') or os.path.expanduser('~')
    return os.path.join(base, 'PCC-Explorer', 'settings.json')


#: Modules the bridge imports from RINEX-Masker.
_REQUIRED_MODULES = ('rinex_handler.py', 'satellite_position.py')

#: Sub-folders of a chosen folder that may hold them. '' = the folder itself
#: (a source checkout); 'core' = the folder a packaged RINEX-Masker ships,
#: because the .exe contains no .py files of its own. The rest let the user
#: point at the parent directory or at the unzipped package wrapper.
_MODULE_SUBDIRS = ('', 'core', 'RINEX-Masker', os.path.join('RINEX-Masker', 'core'))


def resolve_masker_module_dir(path: str):
    """
    Return the folder inside `path` that holds the RINEX-Masker modules, or None.

    An earlier version required the two .py files directly in the
    chosen folder. That is true of a source checkout and of NOTHING a user ever
    receives — the released package ships only RINEX-Masker.exe and _internal/,
    with every module compiled into the executable, so no folder a user could
    pick was ever accepted and the browse button could not work no matter how
    good the hint was.

    The packaged tool now carries a plain-source 'core' folder alongside the
    .exe, and this resolver accepts the source layout, the package layout, and
    the parent of either — so any of the folders a user would reasonably pick
    resolves to the same place.
    """
    if not path or not os.path.isdir(path):
        return None
    for sub in _MODULE_SUBDIRS:
        candidate = os.path.normpath(os.path.join(path, sub)) if sub else os.path.normpath(path)
        if not os.path.isdir(candidate):
            continue
        if all(os.path.exists(os.path.join(candidate, m)) for m in _REQUIRED_MODULES):
            return candidate
    return None


def is_rinex_masker_folder(path: str) -> bool:
    """True if `path` is, or contains, a usable RINEX-Masker installation."""
    return resolve_masker_module_dir(path) is not None


def describe_masker_folder_requirement() -> str:
    """One message explaining what to pick, used by every failure path."""
    return (
        "Select the folder where RINEX-Masker is installed - either the folder "
        "holding RINEX-Masker.exe, or the 'core' folder inside it.\n"
        "Do not select '_internal'.\n"
        "If you run RINEX-Masker from source, select the folder containing "
        "rinex_handler.py and satellite_position.py."
    )


def load_saved_masker_path():
    """Return the RINEX-Masker folder saved by a previous session, or None."""
    try:
        with open(_settings_file(), 'r', encoding='utf-8') as fh:
            path = json.load(fh).get('rinex_masker_path')
        return path if is_rinex_masker_folder(path) else None
    except Exception:
        return None


def set_masker_path(path: str, persist: bool = True) -> bool:
    """
    Point the bridge at a specific RINEX-Masker folder.

    The auto-search only knows source-tree sibling layouts, which do not exist on
    an end user's machine, so the user must be able to say where the tool lives.
    Accepts the installation folder or its 'core' sub-folder; what gets stored is
    the folder that actually holds the modules. Returns False if neither the
    chosen folder nor its known sub-folders contain them.
    """
    global _MASKER_PATH_OVERRIDE, _RINEX_MASKER_MODULE
    module_dir = resolve_masker_module_dir(path)
    if module_dir is None:
        return False

    _MASKER_PATH_OVERRIDE = module_dir
    # Drop caches so the new folder wins over anything imported earlier.
    _RINEX_MASKER_MODULE = None
    for mod in ('rinex_handler', 'satellite_position'):
        sys.modules.pop(mod, None)

    if persist:
        try:
            settings_path = _settings_file()
            os.makedirs(os.path.dirname(settings_path), exist_ok=True)
            data = {}
            if os.path.exists(settings_path):
                try:
                    with open(settings_path, 'r', encoding='utf-8') as fh:
                        data = json.load(fh)
                except Exception:
                    data = {}
            data['rinex_masker_path'] = _MASKER_PATH_OVERRIDE
            with open(settings_path, 'w', encoding='utf-8') as fh:
                json.dump(data, fh, indent=2)
        except Exception as e:
            print(f"  Warning: could not save the RINEX-Masker folder: {e}")
    return True


def get_masker_path():
    """Folder the modules were last loaded from, or None if not resolved yet."""
    return _RESOLVED_MASKER_PATH


def _ensure_rinex_masker_available():
    """
    Ensure the RINEX-Masker project modules are importable.
    Searches the user-selected folder first, then known locations, adding to
    sys.path if needed.
    """
    global _RINEX_MASKER_MODULE, _RESOLVED_MASKER_PATH
    if _RINEX_MASKER_MODULE is not None:
        return _RINEX_MASKER_MODULE

    # A folder chosen by the user always wins over the built-in guesses.
    explicit = _MASKER_PATH_OVERRIDE or load_saved_masker_path()

    if not explicit:
        # Try direct import (if already on path or installed)
        try:
            import rinex_handler
            import satellite_position
            _RINEX_MASKER_MODULE = {
                'rinex_handler': rinex_handler,
                'satellite_position': satellite_position,
            }
            _RESOLVED_MASKER_PATH = os.path.dirname(
                getattr(rinex_handler, '__file__', '') or '') or None
            return _RINEX_MASKER_MODULE
        except ImportError:
            pass

    # Search known locations relative to this project
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)

    search_paths = [
        # Sibling folder
        os.path.join(os.path.dirname(project_root), 'RINEX-Masker'),
        # One level up
        os.path.join(project_root, '..', 'RINEX-Masker'),
        # Two levels up
        os.path.join(project_root, '..', '..', 'RINEX-Masker'),
    ]

    # The three paths above only exist in a source checkout. A user has
    # two unzipped packages, so look beside the running executable as well - the
    # same discovery the PCC-Suite launcher does. resolve_masker_module_dir()
    # then descends into 'core', which is where a packaged RINEX-Masker keeps
    # the modules.
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        neighbourhood = [exe_dir, os.path.dirname(exe_dir),
                         os.path.dirname(os.path.dirname(exe_dir))]
        for base in neighbourhood:
            if not os.path.isdir(base):
                continue
            search_paths.append(base)
            try:
                for entry in sorted(os.listdir(base)):
                    if entry.lower().startswith('rinex-masker'):
                        search_paths.append(os.path.join(base, entry))
            except OSError:
                pass

    # Also try reading the shortcut if available
    shortcuts_dir = os.path.join(project_root, 'Side-Projects Shortcuts')
    if os.path.isdir(shortcuts_dir):
        shortcut_path = os.path.join(shortcuts_dir, 'RINEX-Masker - Shortcut.lnk')
        if os.path.exists(shortcut_path):
            try:
                import subprocess
                result = subprocess.run(
                    ['powershell', '-Command',
                     f'(New-Object -ComObject WScript.Shell).CreateShortcut("{shortcut_path}").TargetPath'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    resolved = result.stdout.strip()
                    if resolved and os.path.isdir(resolved):
                        search_paths.insert(0, resolved)
            except Exception:
                pass

    # The user's own choice wins over every guess above, including the resolved
    # shortcut. This insert must stay AFTER the shortcut block, which also
    # inserts at position 0.
    if explicit:
        search_paths.insert(0, explicit)

    seen = set()
    for path in search_paths:
        module_dir = resolve_masker_module_dir(os.path.normpath(path))
        if module_dir is None or module_dir in seen:
            continue
        seen.add(module_dir)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        try:
            import rinex_handler
            import satellite_position
            _RINEX_MASKER_MODULE = {
                'rinex_handler': rinex_handler,
                'satellite_position': satellite_position,
            }
            _RESOLVED_MASKER_PATH = module_dir
            print(f"  RINEX-Masker found at: {module_dir}")
            return _RINEX_MASKER_MODULE
        except ImportError:
            pass

    raise ImportError(
        "Could not find RINEX-Masker.\n\n"
        + describe_masker_folder_requirement() +
        "\n\nUse the 'RINEX-Masker folder...' button next to Convert.\n"
        "Searched:\n" +
        "\n".join(f"  - {os.path.normpath(p)}" for p in search_paths)
    )


# ---------------------------------------------------------------------------
# Core bridge function
# ---------------------------------------------------------------------------

class _Trajectory:
    """Per-epoch rover position, read from a .KIN file.

    In kinematic mode the adjustment must use the user position per epoch, not a
    static one: a different position gives (slightly) different azimuth and
    elevation angles, and so a different impact. The per-epoch position therefore
    has to be applied where the angles are computed, when the observation CSV is
    written; angles computed from a single point would already have flattened the
    rover to a station before the adjustment sees them.

    Measured on the 4 April 2023 experiment (station 0046, 38 km of driving,
    10.2 km between the two furthest epochs): using one static position shifts
    elevation by up to 0.11 deg and azimuth by 0.12 to 0.19 deg for ordinary
    satellites, and up to 1.37 deg for one close to zenith, where a small ground
    displacement swings the azimuth hardest.

    Positions are linearly interpolated in ECEF between the two bracketing samples.
    Over a 0.1 s to 30 s gap the arc is far too short for the straight line to
    matter next to the accuracy of the trajectory itself.
    """

    def __init__(self, sod, xyz, year, doy):
        order = np.argsort(sod)
        self.sod = np.asarray(sod, dtype=float)[order]
        self.xyz = np.asarray(xyz, dtype=float)[order]
        self.year = year
        self.doy = doy
        self.day_start = datetime(year, 1, 1) + timedelta(days=doy - 1)
        self.used = 0
        self.outside = 0

    @classmethod
    def from_kin_file(cls, kin_path: str):
        """Build from a .KIN file, or return None if it holds nothing usable."""
        try:
            from .los_adjustment import parse_kin_file
        except ImportError:
            from los_adjustment import parse_kin_file
        data, year, doy = parse_kin_file(kin_path)
        if not data or year is None:
            print("  [WARNING] KIN file holds no usable trajectory, "
                  "falling back to the static position")
            return None
        keys = sorted(data)
        return cls(keys, [data[k] for k in keys], year, doy)

    def position_at(self, epoch_dt):
        """ECEF position at this epoch, or None if the epoch is outside the file.

        Returning None rather than an extrapolated point is deliberate: a RINEX
        file usually covers more time than the trajectory does, and silently
        extrapolating a car's position is worse than saying so.
        """
        sod = (epoch_dt - self.day_start).total_seconds()
        if sod < self.sod[0] - 0.5 or sod > self.sod[-1] + 0.5:
            self.outside += 1
            return None
        self.used += 1
        return np.array([np.interp(sod, self.sod, self.xyz[:, i]) for i in range(3)])

    def span_m(self) -> float:
        """Straight-line distance between the first and last position."""
        return float(np.linalg.norm(self.xyz[-1] - self.xyz[0]))


def resolve_station_position(header_xyz, station_xyz=None, fallback_station_xyz=None):
    """
    Decide which station position the azimuth/elevation computation uses.

    Returns (xyz_as_list, source_description). Raises ValueError when there is
    no usable position anywhere.

    Order of preference, and it matters:

    1. an explicit `station_xyz` from the caller,
    2. the RINEX file's own APPROX POSITION XYZ,
    3. `fallback_station_xyz`, used ONLY when the header carries none.

    The header must win over anything typed into the GUI, which is why the GUI
    passes its coordinate boxes as `fallback_station_xyz` and never as
    `station_xyz`.

    ⚠ Before the fallback existed, a header without APPROX POSITION simply
    raised and the GUI had no way to answer it, since it passed no position at
    all. That made low-cost and smartphone RINEX files, which commonly omit the
    record, impossible to convert.

    Kept as its own function so the decision can be tested directly rather than
    only through a full conversion.
    """
    def usable(xyz):
        if xyz is None:
            return False
        try:
            return any(float(v) != 0.0 for v in xyz)
        except (TypeError, ValueError):
            return False

    if station_xyz is not None:
        return list(station_xyz), "caller"
    if usable(header_xyz):
        return list(header_xyz), "RINEX header"
    if usable(fallback_station_xyz):
        return list(fallback_station_xyz), "entered by the user (header has none)"
    raise ValueError(
        "RINEX header has no APPROX POSITION XYZ. "
        "Provide station_xyz manually."
    )


def rinex_to_los_observations(
    rinex_path: str,
    sp3_path: str = None,
    station_xyz: list = None,
    sampling_interval: float = 30.0,
    elevation_mask: float = 0.0,
    systems: list = None,
    output_csv: str = None,
    progress_callback=None,
    kin_path: str = None,
    fallback_station_xyz: list = None,
) -> dict:
    """
    Convert a RINEX observation file into PCC-Explorer LoS observations.

    Extracts satellite IDs from each RINEX epoch, computes azimuth/elevation
    from SP3 precise orbits, downsamples to the specified interval, and
    optionally writes the output CSV.

    Args:
        rinex_path: Path to the RINEX observation file (.RNX, .26O, etc.)
        sp3_path: Path to SP3 orbit file. If None, auto-downloads from AIUB FTP.
        station_xyz: [X, Y, Z] ECEF station position in meters.
                     If None, uses APPROX POSITION XYZ from RINEX header.
                     Ignored per epoch when kin_path is given and covers it.
        sampling_interval: Observation sampling interval in seconds (default 30s).
                          Set to 0 to use all epochs from the RINEX file.
        elevation_mask: Minimum elevation angle in degrees (default 0).
        systems: List of GNSS system chars to include (e.g., ['G', 'E']).
                 If None, includes all systems found in RINEX.
        output_csv: Path to write the observation CSV. If None, auto-generates
                    a path next to the RINEX file.
        progress_callback: Optional callable(percent, message) for GUI progress.
        kin_path: Optional .KIN trajectory. When given, each epoch's azimuth and
                  elevation are computed from the rover's position at that epoch
                  instead of from one static point. Epochs the trajectory does
                  not cover fall back to the static position and are counted.

    Returns:
        dict with keys:
            - 'epochs': {datetime: [(prn_str, az_deg, el_deg), ...]}
            - 'csv_path': str path to the written CSV file
            - 'stats': dict of summary statistics
    """
    t_start = time.time()

    # --- Load RINEX-Masker modules ---
    modules = _ensure_rinex_masker_available()
    rinex_handler = modules['rinex_handler']
    sat_pos_module = modules['satellite_position']

    if progress_callback:
        progress_callback(5, "Parsing RINEX file...")

    # --- Parse RINEX header ---
    print(f"\n{'=' * 60}")
    print("RINEX -> PCC-Explorer Bridge")
    print(f"{'=' * 60}")

    header, lines, header_end_idx = rinex_handler.parse_rinex(rinex_path)

    # Station position.
    #
    # Order of preference, and it matters: an explicit station_xyz wins, then the
    # file's own APPROX POSITION XYZ, and only if the header carries none do we
    # fall back to what the caller offers. The header must win over anything
    # typed into the GUI, so the GUI passes its coordinates as
    # `fallback_station_xyz`, never as `station_xyz`.
    #
    # Without the fallback, low-cost and smartphone RINEX files, which often
    # omit APPROX POSITION, could not be converted at all.
    station_xyz, position_source = resolve_station_position(
        header.station_xyz, station_xyz, fallback_station_xyz)
    station_ecef = np.array(station_xyz, dtype=float)
    print(f"  Station ECEF: [{station_ecef[0]:.3f}, {station_ecef[1]:.3f}, {station_ecef[2]:.3f}]  (from: {position_source})")

    # --- Optional per-epoch rover position ---
    trajectory = None
    if kin_path:
        trajectory = _Trajectory.from_kin_file(kin_path)
        if trajectory is not None:
            print(f"  Trajectory: {len(trajectory.sod)} positions, "
                  f"{trajectory.span_m() / 1000.0:.2f} km from first to last epoch")
            print("              angles are computed per epoch at the rover, "
                  "not at the static position")

    # Determine date from RINEX for SP3 auto-download
    rinex_date = header.first_obs_time
    if rinex_date is None:
        # Fallback: parse first epoch
        for epoch in rinex_handler.iterate_epochs(lines, header_end_idx):
            rinex_date = epoch.timestamp
            break
    if rinex_date is None:
        raise ValueError("Could not determine observation date from RINEX file.")

    print(f"  Observation date: {rinex_date.date()}")

    if progress_callback:
        progress_callback(15, "Loading SP3 orbits...")

    # --- Load SP3 orbits ---
    if sp3_path and os.path.exists(sp3_path):
        sp3_data = sat_pos_module.read_sp3(sp3_path)
    else:
        # Auto-download: use PCC-Explorer's orbit directory
        orbit_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'orbit')
        os.makedirs(orbit_dir, exist_ok=True)
        sp3_data = sat_pos_module.load_sp3(rinex_date, download_dir=orbit_dir)

    available_systems = set(sp3_data.get('orbits', {}).keys())
    if systems:
        target_systems = set(systems) & available_systems
    else:
        # Use systems present in both RINEX and SP3
        rinex_systems = set(header.obs_types.keys())
        target_systems = rinex_systems & available_systems

    if not target_systems:
        raise ValueError(
            f"No common GNSS systems between RINEX ({set(header.obs_types.keys())}) "
            f"and SP3 ({available_systems})."
        )
    print(f"  Target systems: {sorted(target_systems)}")

    if progress_callback:
        progress_callback(25, "Processing RINEX epochs...")

    # --- Iterate RINEX epochs, compute az/el ---
    all_epochs = {}
    total_obs = 0
    skipped_epochs = 0
    last_accepted_time = None

    # Pre-count total epochs for progress (approximate from file size and interval)
    rinex_interval = header.interval if header.interval > 0 else 1.0

    epoch_count = 0
    for epoch in rinex_handler.iterate_epochs(lines, header_end_idx):
        epoch_count += 1

        # Downsampling: skip epochs that are too close to the last accepted one
        if sampling_interval > 0 and last_accepted_time is not None:
            dt = (epoch.timestamp - last_accepted_time).total_seconds()
            if dt < sampling_interval - 0.01:  # Small tolerance for floating point
                skipped_epochs += 1
                continue

        # Only process good epochs (flag == 0)
        if epoch.flag != 0:
            continue

        # Extract satellite IDs for target systems
        sat_ids = []
        for sat_obs in epoch.satellites:
            sat_id = sat_obs.sat_id
            if len(sat_id) >= 2 and sat_id[0] in target_systems:
                sat_ids.append(sat_id)

        if not sat_ids:
            continue

        # Where the antenna actually was at this epoch. Without a trajectory, or
        # outside the one given, this stays the single static position.
        epoch_ecef = station_ecef
        if trajectory is not None:
            rover = trajectory.position_at(epoch.timestamp)
            if rover is not None:
                epoch_ecef = rover

        # Compute az/el for all satellites
        epoch_observations = []
        for sat_id in sat_ids:
            azel = sat_pos_module.get_satellite_azel(
                sp3_data, epoch_ecef, epoch.timestamp, sat_id
            )
            if azel is None:
                continue

            az_deg, el_deg = azel

            # Apply elevation mask
            if el_deg < elevation_mask:
                continue

            epoch_observations.append((sat_id, az_deg, el_deg))

        if epoch_observations:
            all_epochs[epoch.timestamp] = epoch_observations
            total_obs += len(epoch_observations)
            last_accepted_time = epoch.timestamp

        # Progress update every 100 epochs
        if progress_callback and epoch_count % 100 == 0:
            pct = min(25 + int(60 * epoch_count / max(1, len(lines) - header_end_idx) * rinex_interval), 85)
            progress_callback(pct, f"Processing epoch {epoch_count}...")

    elapsed_process = time.time() - t_start

    print(f"\n  Extraction complete:")
    print(f"    RINEX epochs scanned: {epoch_count}")
    print(f"    Epochs accepted (after {sampling_interval}s sampling): {len(all_epochs)}")
    print(f"    Total observations: {total_obs}")
    print(f"    Processing time: {elapsed_process:.1f}s")

    if trajectory is not None:
        print(f"    Epochs positioned from the trajectory: {trajectory.used}")
        if trajectory.outside:
            print(f"    [WARNING] {trajectory.outside} epoch(s) lie outside the "
                  f"trajectory and used the static position instead. The RINEX "
                  f"file covers more time than the .KIN file does.")

    if not all_epochs:
        raise ValueError("No valid observations extracted. Check RINEX file, SP3 data, and elevation mask.")

    if progress_callback:
        progress_callback(90, "Writing CSV...")

    # --- Write CSV ---
    if output_csv is None:
        rinex_base = os.path.splitext(os.path.basename(rinex_path))[0]
        output_csv = os.path.join(
            os.path.dirname(rinex_path),
            f"{rinex_base}_los_observations.csv"
        )
        # If RINEX is in a read-only dir, write to PCC-Explorer's data dir
        try:
            with open(output_csv, 'w') as _test:
                pass
            os.remove(output_csv)
        except (PermissionError, OSError):
            output_csv = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'data', f"{rinex_base}_los_observations.csv"
            )

    write_observation_csv(all_epochs, output_csv)

    # --- Summary statistics ---
    all_elevations = [el for sats in all_epochs.values() for _, _, el in sats]
    all_azimuths = [az for sats in all_epochs.values() for _, az, _ in sats]
    sats_per_epoch = [len(sats) for sats in all_epochs.values()]

    stats = {
        'rinex_file': os.path.basename(rinex_path),
        'station': header.marker_name,
        'date': str(rinex_date.date()),
        'rinex_interval': rinex_interval,
        'sampling_interval': sampling_interval,
        'systems': sorted(target_systems),
        'total_epochs': len(all_epochs),
        'total_observations': total_obs,
        'mean_sats_per_epoch': np.mean(sats_per_epoch) if sats_per_epoch else 0,
        'elevation_range': (min(all_elevations), max(all_elevations)) if all_elevations else (0, 0),
        'processing_time_sec': round(time.time() - t_start, 1),
        # Kinematic: how the angles were positioned. 'static' means one point for
        # the whole file, which is what a conversion without a trajectory does.
        'position_mode': 'per-epoch trajectory' if trajectory is not None else 'static',
        # Which source the static position came from, so the GUI can say so
        # instead of leaving the user to guess.
        'position_source': position_source,
        'epochs_from_trajectory': trajectory.used if trajectory is not None else 0,
        'epochs_outside_trajectory': trajectory.outside if trajectory is not None else 0,
        'trajectory_span_km': (round(trajectory.span_m() / 1000.0, 3)
                               if trajectory is not None else 0.0),
    }

    print(f"\n  CSV written: {output_csv}")
    print(f"  Mean satellites/epoch: {stats['mean_sats_per_epoch']:.1f}")
    print(f"  Elevation range: [{stats['elevation_range'][0]:.1f} deg, {stats['elevation_range'][1]:.1f} deg]")

    if progress_callback:
        progress_callback(100, "Done!")

    return {
        'epochs': all_epochs,
        'csv_path': output_csv,
        'stats': stats,
    }


def write_observation_csv(epochs: dict, filepath: str):
    """
    Write epoch satellite data to PCC-Explorer's LoS observation CSV format.

    Format: epoch_utc, prn, elevation_deg, azimuth_deg
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch_utc', 'prn', 'elevation_deg', 'azimuth_deg'])
        for epoch_dt in sorted(epochs.keys()):
            epoch_str = epoch_dt.strftime('%Y-%m-%dT%H:%M:%S')
            for prn, az, el in epochs[epoch_dt]:
                writer.writerow([epoch_str, prn, f'{el:.6f}', f'{az:.6f}'])

    print(f"  Wrote {sum(len(s) for s in epochs.values())} observations "
          f"across {len(epochs)} epochs to {os.path.basename(filepath)}")


# ---------------------------------------------------------------------------
# Quick-check utility
# ---------------------------------------------------------------------------

def check_rinex_masker_available() -> bool:
    """Check if RINEX-Masker modules can be imported. Non-throwing."""
    try:
        _ensure_rinex_masker_available()
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# CLI entry point for standalone testing
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert RINEX observations to PCC-Explorer LoS CSV'
    )
    parser.add_argument('rinex', help='Path to RINEX observation file')
    parser.add_argument('--sp3', help='Path to SP3 orbit file (auto-download if omitted)')
    parser.add_argument('--sampling', type=float, default=30.0,
                        help='Sampling interval in seconds (default: 30)')
    parser.add_argument('--elevation-mask', type=float, default=0.0,
                        help='Minimum elevation angle in degrees (default: 0)')
    parser.add_argument('--systems', nargs='+', default=None,
                        help='GNSS systems to include (e.g., G E R C)')
    parser.add_argument('--output', help='Output CSV path')

    args = parser.parse_args()

    result = rinex_to_los_observations(
        rinex_path=args.rinex,
        sp3_path=args.sp3,
        sampling_interval=args.sampling,
        elevation_mask=args.elevation_mask,
        systems=args.systems,
        output_csv=args.output,
    )

    print(f"\nDone! CSV: {result['csv_path']}")
    print(f"Stats: {result['stats']}")
