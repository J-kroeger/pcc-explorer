# src/plotting.py
"""
Visualization module for PCC-Explorer.
Provides bar charts, time series plots, world map projections, and skyplots
for GNSS antenna PCC impact analysis results.
"""

import matplotlib.pyplot as plt
import numpy as np
import threading
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import GoogleTiles as _ImgTilesBase
from .orbit_utils import get_ground_tracks, validate_orbit_data


def _main_thread_only(what: str) -> bool:
    """
    Guard against building a figure from a background thread.

    A figure created off the main thread gets a Tk figure manager owned by a
    thread that usually dies straight afterwards. Nothing fails at that moment;
    the *next* plt.show() on the main thread walks every open manager, reaches
    the orphan and raises "RuntimeError: main thread is not in main loop" —
    three frames and one thread away from the real mistake.

    Returns True when it is safe to draw. Callers that get False must hand the
    request to the GUI instead (see processing_core's '_pending_plots').
    """
    if threading.current_thread() is threading.main_thread():
        return True
    print(f"  Warning: refusing to draw the {what} from a background thread - "
          f"it would break the next plot window. Draw it from the GUI "
          f"completion callback instead.")
    return False


def cap_major_ticks(ax, max_ticks=12):
    """
    Keep at most `max_ticks` labelled ticks on the x axis of `ax`.

    With one solution per epoch, a long or high-rate run asks for far more tick
    labels than the axis can fit, and they overprint into an unreadable band.
    Rather than switch to a coarser time unit, which changes what the axis
    means, every Nth tick the locator produced is kept, so the spacing stays
    regular and the format stays the same.

    Returns the number of ticks left, so a caller can report what it did.
    """
    import math
    from matplotlib.ticker import FixedLocator
    ticks = list(ax.get_xticks())
    if len(ticks) <= max_ticks:
        return len(ticks)
    step = int(math.ceil(len(ticks) / float(max_ticks)))
    kept = ticks[::step]
    ax.xaxis.set_major_locator(FixedLocator(kept))
    return len(kept)


def plot_timeline(dates: list, results_matrix: np.ndarray, param_names: list, colors: list = None, 
                  y_axis_mode: str = "Auto", y_limits: tuple = None, station_coords: tuple = None, antenna_name: str = None):
    """
    Creates a multi-panel plot showing the impact on each parameter over time.
    Handles both daily and sub-daily (hourly) intervals with appropriate formatting.
    
    Args:
        dates: List of date/datetime objects
        results_matrix: 2D array of results [n_dates x n_params]
        param_names: List of parameter names
        colors: Optional list of colors for each parameter. If None, uses distinct default colors.
        y_axis_mode: "Auto" (default matplotlib), "Symmetric" (centered on 0), "Shared" (all same range), "Custom" (user-defined)
        y_limits: Optional tuple (min, max) for Custom mode
        station_coords: Optional tuple of (latitude, longitude, height) for display
        antenna_name: Optional antenna name for display
    """
    if not _main_thread_only('time series'):
        return
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    
    if len(dates) == 0:
        print("Warning: No dates to plot")
        return
    
    num_params = len(param_names)
    
    # Force white style to prevent black borders/transparent background issues
    plt.style.use('fast')  # Reset to clean style
    # Manually set white background
    fig, axes = plt.subplots(num_params, 1, figsize=(12, max(10, num_params * 2)), sharex=True, facecolor='white')
    
    if num_params == 1:
        axes = [axes]
    
    # Build title with optional subtitle
    title = r'Time Series Analysis of $\Delta$PCC Impact'
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Add subtitle with antenna and coordinates
    subtitle_parts = []
    if antenna_name:
        subtitle_parts.append(f"Antenna: {antenna_name}")
    if station_coords and len(station_coords) >= 2:
        lat, lon = station_coords[0], station_coords[1]
        height = station_coords[2] if len(station_coords) > 2 else None
        coord_str = f"Position: {lat:.3f}°, {lon:.3f}°"
        if height is not None:
            coord_str += f", {height:.1f}m"
        subtitle_parts.append(coord_str)
    
    if subtitle_parts:
        fig.text(0.5, 0.94, " | ".join(subtitle_parts), ha='center', fontsize=10, style='italic')
    
    # Default distinct colors for each parameter
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # blue, orange, green, red, purple
    if colors is None:
        colors = default_colors
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        color = colors[i % len(colors)]
        ax.plot(dates, results_matrix[:, i], marker='o', linestyle='-', markersize=6, color=color)
        ax.set_ylabel(f"{name}\n[mm]", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Apply y-axis mode
    if y_axis_mode == "Symmetric":
        # Each subplot symmetric around 0
        for i, ax in enumerate(axes):
            max_abs = np.max(np.abs(results_matrix[:, i]))
            margin = max_abs * 0.1 if max_abs > 0 else 0.1
            ax.set_ylim(-max_abs - margin, max_abs + margin)
    elif y_axis_mode == "Shared":
        # All subplots share the same y-axis range
        global_max = np.max(np.abs(results_matrix))
        margin = global_max * 0.1 if global_max > 0 else 0.1
        for ax in axes:
            ax.set_ylim(-global_max - margin, global_max + margin)
    elif y_axis_mode == "Custom" and y_limits is not None:
        # User-defined limits
        for ax in axes:
            ax.set_ylim(y_limits[0], y_limits[1])
    
    # --- Detect if data is sub-daily (has time component) ---
    is_subdaily = False
    if len(dates) >= 2:
        # Check if time difference is less than 1 day
        try:
            if hasattr(dates[0], 'hour'):
                # datetime objects
                time_diff = dates[1] - dates[0]
                is_subdaily = time_diff < timedelta(days=1)
            else:
                # date objects (no time component)
                is_subdaily = False
        except Exception:
            is_subdaily = False
    
    # --- Configure x-axis formatting based on data type ---
    if is_subdaily:
        # Sub-daily intervals (hourly, 3-hourly, etc.)
        axes[-1].set_xlabel("Date / Time", fontsize=12)
        
        total_hours = (dates[-1] - dates[0]).total_seconds() / 3600 if len(dates) > 1 else 24
        
        if total_hours <= 24:
            # Single day - show 3-hourly ticks
            axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=3))
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
        elif total_hours <= 72:
            # 2-3 days - show 6-hour ticks
            axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=6))
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
        elif total_hours <= 168:  # 1 week
            # Show 12-hour ticks
            axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=12))
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
        elif total_hours <= 720:  # ~1 month
            # Show daily ticks
            axes[-1].xaxis.set_major_locator(mdates.DayLocator())
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        else:
            # Longer periods - use auto locator with max ticks
            axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=15))
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    else:
        # Daily intervals
        axes[-1].set_xlabel("Date", fontsize=12)
        
        num_days = len(dates)
        if num_days <= 7:
            # For short timelines, show every date
            axes[-1].xaxis.set_major_locator(mdates.DayLocator())
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        elif num_days <= 31:
            # For monthly timelines, show weekly ticks
            axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))  # Mondays only
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        elif num_days <= 180:
            # For half-year, show monthly ticks
            axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        elif num_days <= 730:
            # For 1-2 years, show quarterly ticks
            axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        else:
            # For very long timelines (years), show yearly ticks
            axes[-1].xaxis.set_major_locator(mdates.YearLocator())
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    cap_major_ticks(axes[-1], max_ticks=15)

    # Rotate labels for readability
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_results_bar_chart(param_names: list, results: np.ndarray, colors: list = None, title: str = None):
    """
    Creates a bar chart of the final adjustment results.
    
    Args:
        param_names: List of parameter names
        results: Array of result values
        colors: Optional list of colors to use for bars
        title: Optional custom title for the plot
    """
    if not _main_thread_only('results bar chart'):
        return
    if len(param_names) == 0 or len(results) == 0:
        print("Warning: No results to plot")
        return
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(max(10, len(param_names)), 6))
    
    # Use custom colors if provided, otherwise default viridis
    if colors is not None and len(colors) >= len(param_names):
        bar_colors = colors[:len(param_names)]
    else:
        bar_colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(param_names)))
    bars = ax.bar(param_names, results, color=bar_colors)

    ax.set_ylabel("Impact [mm]", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(r"Impact of $\Delta$PCC on Geodetic Parameters", fontsize=14)
    ax.axhline(0, color='grey', linewidth=0.8)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', 
                va='bottom' if yval >= 0 else 'top', ha='center',
                fontweight='bold', fontsize=9)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def _plot_satellite_tracks(ax, orbit_data: dict, config: dict, max_satellites: int = 10):
    """Helper function to plot satellite ground tracks."""
    if not orbit_data:
        return 0
    
    orbit_summary = validate_orbit_data(orbit_data)
    orbit_type = config.get('orbit_type', 'unknown')
    is_final = orbit_summary['type'] == 'final'
    
    satellites_data = None
    if is_final:
        satellites_data = orbit_data.get('orbits', {}).get('G', {})
    else:
        satellites_data = orbit_data.get('G', {})
    
    if not satellites_data:
        return 0
    
    num_sats_to_plot = min(len(satellites_data), max_satellites)
    colors = plt.cm.Set1(np.linspace(0, 1, num_sats_to_plot))
    
    plotted_count = 0
    station_pos = config.get('position', None)
    start_time = config.get('start_time', None)
    end_time = config.get('end_time', None)
    
    for i, (prn, records) in enumerate(list(satellites_data.items())[:num_sats_to_plot]):
        if not records:
            continue
        try:
            track = get_ground_tracks(
                orbit_data, 'G', prn,
                start_time=start_time,
                end_time=end_time,
                station_pos=station_pos
            )
            if track and len(track) > 1:
                lons_track, lats_track = zip(*track)
                ax.plot(lons_track, lats_track, 
                       color=colors[plotted_count % len(colors)], 
                       linewidth=1.5, alpha=0.8,
                       transform=ccrs.Geodetic(), label=f'GPS-{prn:02d}')
                ax.plot(lons_track[0], lats_track[0], 'go', markersize=5, transform=ccrs.PlateCarree(), alpha=0.8, zorder=10)
                ax.plot(lons_track[-1], lats_track[-1], 'ro', markersize=5, transform=ccrs.PlateCarree(), alpha=0.8, zorder=10)
                plotted_count += 1
        except Exception:
            continue
    return plotted_count


def plot_world_map(lons: np.ndarray, lats: np.ndarray, data_grid: np.ndarray, 
                   parameter_name: str, config: dict, show_tracks: bool = True,
                   colorbar_limits: tuple = None):
    """Generates a high-quality world map.
    
    Args:
        colorbar_limits: Optional tuple (vmin, vmax) for colorbar scaling.
                        If None, uses auto-scaling based on data.
    """
    if not _main_thread_only('world map'):
        return
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, color='gray', alpha=0.5, linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # Apply colorbar limits if provided
    vmin = colorbar_limits[0] if colorbar_limits else None
    vmax = colorbar_limits[1] if colorbar_limits else None

    im = ax.imshow(data_grid, origin='upper', cmap='viridis',
                   extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                   transform=ccrs.PlateCarree(), alpha=0.7,
                   vmin=vmin, vmax=vmax)

    if show_tracks:
        orbit_data = config.get('orbit_data', {})
        plotted_count = _plot_satellite_tracks(ax, orbit_data, config, max_satellites=10)
        if plotted_count > 0:
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=8, framealpha=0.9)

    cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label(f'Impact on {parameter_name} [mm]', fontsize=12)
    
    # Dynamic title based on mode (Absolute vs Delta)
    mode = config.get('analysis_mode', 'two_files')
    if mode == 'single_file':
        title_text = f'Global Absolute PCC Impact on {parameter_name}'
    else:
        # Use double backslash to avoid invalid escape sequence warning
        title_text = f'Global Impact of $\\Delta$PCC on {parameter_name}'
        
    ax.set_title(title_text, fontsize=14, pad=20)
    ax.set_global()
    plt.tight_layout()
    plt.show(block=False)


def plot_skyplot(azimuths: list, elevations: list, m_matrix: np.ndarray = None, masks: list = None, 
                 latitude: float = None, longitude: float = None, height: float = None,
                 satellite_ids: list = None):
    """
    Generates a satellite skyplot showing individual satellite positions.
    Optionally visualizes azimuth masks and satellite constellation info with flags.
    
    Args:
        azimuths: List of azimuth angles in degrees
        elevations: List of elevation angles in degrees
        m_matrix: Optional observation density matrix (unused in simple scatter)
        masks: List of tuples (az_from, az_to, el_limit)
        latitude: Optional latitude in degrees for title
        longitude: Optional longitude in degrees for title
        height: Optional height in meters for title
        satellite_ids: Optional list of satellite IDs (e.g., 'G01', 'E05') for each observation
    """
    if not _main_thread_only('skyplot'):
        return
    if not azimuths or not elevations:
        print("Warning: No satellite positions to plot")
        return
    
    # Country codes instead of flags (for better Windows compatibility)
    CONSTELLATION_FLAGS = {
        'G': '[USA]',  # GPS
        'E': '[EU]',   # Galileo
        'R': '[RUS]',  # GLONASS
        'C': '[CHN]',  # BeiDou
    }
    CONSTELLATION_NAMES = {
        'G': 'GPS',
        'E': 'Galileo', 
        'R': 'GLONASS',
        'C': 'BeiDou',
    }
    
    fig = plt.figure(figsize=(12, 8), num="PCC-Explorer Skyplot")
    ax = fig.add_subplot(111, polar=True)
    
    az_rad = np.deg2rad(np.array(azimuths))
    el_array = np.array(elevations)
    # Polar plot radius is usually distance from center (90)
    # Zenith (90 deg el) = radius 0. Horizon (0 deg el) = radius 90.
    r_coords = 90.0 - el_array
    
    # Create scatter plot
    scatter = ax.scatter(az_rad, r_coords, c=el_array, cmap='viridis', s=10, alpha=0.6, label='Satellites')
    
    # Configure polar plot
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 90)
    ax.set_rgrids([0, 30, 60, 90], labels=['90°', '60°', '30°', 'Horizon'])
    
    # Build title with coordinates if available
    title = "Satellite Visibility Skyplot"
    if latitude is not None and longitude is not None:
        coord_str = f"({latitude:.4f}°, {longitude:.4f}°"
        if height is not None:
            coord_str += f", {height:.1f} m"
        coord_str += ")"
        title += f"\n{coord_str}"
    ax.set_title(title, pad=20, fontsize=14)
    
    # --- DRAW MASKS ---
    if masks:
        for (az_from, az_to, el_limit) in masks:
            # Convert to degrees to radians
            theta1 = np.deg2rad(az_from)
            theta2 = np.deg2rad(az_to)
            
            # Elevation limit < X means we block everything below X.
            # In polar radius: Radius > (90 - X)
            r_inner = 90.0 - el_limit
            r_outer = 90.0
            
            # Handle wrapping (e.g. 300 to 60)
            if theta1 > theta2:
                # Part 1: theta1 to 360 (2pi)
                width1 = (2*np.pi) - theta1
                ax.bar(theta1, r_outer - r_inner, width=width1, bottom=r_inner, 
                       color='red', alpha=0.2, align='edge', edgecolor='red')
                
                # Part 2: 0 to theta2
                width2 = theta2
                ax.bar(0, r_outer - r_inner, width=width2, bottom=r_inner, 
                       color='red', alpha=0.2, align='edge', edgecolor='red')
            else:
                width = theta2 - theta1
                ax.bar(theta1, r_outer - r_inner, width=width, bottom=r_inner, 
                       color='red', alpha=0.2, align='edge', label='Masked Area', edgecolor='red')

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', pad=0.1, shrink=0.7)
    cbar.set_label("Elevation [deg]", fontsize=10)
    
    # --- 1. Top-Right Stats Box ---
    stats_lines = [f"Total observations: {len(azimuths)}"]
    stats_lines.append(f"Mean elevation: {np.mean(elevations):.1f}°")
    
    stats_text = '\n'.join(stats_lines)
    
    # Place stats box at top-right (relative coords)
    plt.gcf().text(0.82, 0.95, stats_text,
                  verticalalignment='top', horizontalalignment='left',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                  fontsize=9, family='monospace', weight='bold')

    # --- 2. Satellite List Below ---
    if satellite_ids:
        unique_sats = sorted(set(satellite_ids))
        constellation_sats = {}
        for sat_id in unique_sats:
            if sat_id and len(sat_id) >= 2:
                system = sat_id[0]
                if system not in constellation_sats:
                    constellation_sats[system] = []
                constellation_sats[system].append(sat_id)
        
        # Build structured text for satellites
        sat_lines = ["Details by System:"]
        sat_lines.append("-" * 25)
        
        for system in ['G', 'E', 'R', 'C']:
            if system in constellation_sats:
                flag = CONSTELLATION_FLAGS.get(system, '')
                name = CONSTELLATION_NAMES.get(system, system)
                sats = constellation_sats[system]
                prn_nums = [s[1:] for s in sats]
                
                # Header line: [USA] GPS (32)
                sat_lines.append(f"{flag} {name} ({len(sats)})")
                
                # PRNs wrapped neatly
                prn_str = ", ".join(prn_nums)
                # Simple manual wrap
                import textwrap
                wrapped_prns = textwrap.wrap(prn_str, width=30) # width in characters
                for line in wrapped_prns:
                    sat_lines.append(f"  {line}")
                sat_lines.append("") # Spacer
        
        sat_text = '\n'.join(sat_lines)
        
        # Place detail box below stats box
        # y=0.88 allows space below the stats box which ends around 0.90
        plt.gcf().text(0.82, 0.88, sat_text,
                      verticalalignment='top', horizontalalignment='left',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
                      fontsize=8, family='monospace')
    
    # adjust layout to make room for text on the right
    plt.subplots_adjust(right=0.8)
    plt.show()


def plot_los_timeline(epoch_results: list, param_names: list, colors: list = None,
                      y_axis_mode: str = "Auto", title: str = None):
    """
    Plots per-epoch LoS adjustment results as a time series with actual UTC timestamps.

    Each parameter gets its own subplot panel with the epoch datetime on the x-axis,
    so every value is shown at its exact timestamp.

    Args:
        epoch_results: List of (epoch_dt, results_vec, n_sats) tuples from
                       perform_los_adjustment().
        param_names: List of parameter names (e.g., ['North', 'East', 'Up', 'Clock_GPS']).
        colors: Optional list of colors for each parameter.
        y_axis_mode: "Auto" (default) or "Symmetric" (centered on 0).
        title: Optional overall title.
    """
    if not _main_thread_only('LoS time series'):
        return
    if not epoch_results:
        print("No per-epoch results to plot.")
        return

    import matplotlib.dates as mdates
    from matplotlib.ticker import AutoMinorLocator

    # Display name mapping
    PARAM_DISPLAY = {
        'North': 'North [mm]', 'East': 'East [mm]', 'Up': 'Up [mm]',
        'Clock_GPS': 'Clock GPS [mm]', 'Clock_GLONASS': 'Clock GLO [mm]',
        'Clock_Galileo': 'Clock GAL [mm]', 'Clock_BDS': 'Clock BDS [mm]',
        'Tropo': 'Tropo [mm]', 'Tropo_Gn': 'Tropo Gn [mm]', 'Tropo_Ge': 'Tropo Ge [mm]'
    }

    # Extract timestamps and result vectors
    timestamps = [ep[0] for ep in epoch_results]
    results_matrix = np.array([ep[1] for ep in epoch_results])
    n_sats_list = [ep[2] for ep in epoch_results]
    n_params = len(param_names)

    # Default colors
    if colors is None:
        colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0',
                  '#FF9800', '#009688', '#E91E63', '#795548']

    # Determine time span for adaptive x-axis formatting
    if len(timestamps) > 1:
        time_span = (timestamps[-1] - timestamps[0]).total_seconds()
    else:
        time_span = 0

    # 2.5 in per panel made a 12 x 12.5 in figure for five
    # parameters, taller than most screens. Shrink that into a smaller window
    # and the fonts, which are fixed in POINTS, swamp the canvas: the y labels
    # ride over the panels and the x axis label is squeezed off the bottom
    # entirely. Cap the height, and let constrained_layout do the spacing.
    #
    # constrained_layout rather than tight_layout because it re-runs on EVERY
    # draw. tight_layout runs once at creation, so a window the user resizes
    # afterwards would keep the spacing computed for the original size.
    fig_h = min(2.5 * n_params, 9.0)
    fig, axes = plt.subplots(n_params, 1, figsize=(12, fig_h),
                             sharex=True, constrained_layout=True)
    if n_params == 1:
        axes = [axes]

    for i, (ax, pname) in enumerate(zip(axes, param_names)):
        color = colors[i % len(colors)]
        display_name = PARAM_DISPLAY.get(pname, pname)
        values = results_matrix[:, i]

        ax.plot(timestamps, values, '-o', color=color, linewidth=1.5,
                markersize=3, alpha=0.8, label=display_name)
        ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5, linestyle='--')

        ax.set_ylabel(display_name, fontsize=10, fontweight='bold')
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which='major', linewidth=0.5, alpha=0.4)
        ax.grid(True, which='minor', linewidth=0.3, alpha=0.2)

        if y_axis_mode == "Symmetric":
            max_abs = max(abs(values.min()), abs(values.max()), 0.1)
            ax.set_ylim(-max_abs * 1.1, max_abs * 1.1)

    # X-axis formatting (adaptive based on time span)
    ax_bottom = axes[-1]
    if time_span < 7200:  # < 2 hours
        ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax_bottom.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(1, int(time_span / 600))))
    elif time_span < 86400:  # < 1 day
        ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_bottom.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(time_span / 28800))))
    else:  # multi-day
        ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        ax_bottom.xaxis.set_major_locator(mdates.AutoDateLocator())

    # None of the branches above bounds the tick count: a high-rate or long run
    # can produce hundreds of labels that overprint into a solid band.
    cap_major_ticks(ax_bottom)

    ax_bottom.set_xlabel('UTC Time', fontsize=11, fontweight='bold')
    # Rotate by hand: fig.autofmt_xdate() calls subplots_adjust, which fights
    # constrained_layout and makes matplotlib drop the managed layout.
    for _lbl in ax_bottom.get_xticklabels():
        _lbl.set_rotation(30)
        _lbl.set_ha('right')

    # Title
    # On the line-of-sight time series both the title and the x axis label
    # could be cropped, and both came from the same two lines. y=1.01 puts the suptitle ABOVE the top edge of the canvas
    # (figure coordinates run 0 to 1), and the tight_layout() below then packs
    # the axes to fill the whole figure without reserving any room for it, so
    # the title was drawn off the paper and the rotated date labels pushed the
    # x-axis label off the bottom. Keep the title inside the canvas and hand
    # tight_layout an explicit rect that leaves a strip for it.
    if not title:
        t0 = timestamps[0].strftime('%Y-%m-%d %H:%M')
        t1 = timestamps[-1].strftime('%H:%M:%S')
        title = (f'LoS Per-Epoch Results ({t0} - {t1}, '
                 f'{len(epoch_results)} epochs)')
    # constrained_layout reserves room for the suptitle itself, so no manual
    # y offset and no tight_layout rect are needed.
    fig.suptitle(title, fontsize=13, fontweight='bold')

    plt.show()


#: Deepest zoom the imagery provider serves reliably. Asking for more returns
#: empty tiles, which is another way to end up with a blank map.
_MAX_TILE_ZOOM = 19

#: Smallest map view, in metres. Below this there is nothing recognisable to
#: look at: the map degenerates into featureless grey tiles.
_MIN_VIEW_M = 120.0


class _AerialTiles(_ImgTilesBase):
    """Esri World Imagery, the closest legal equivalent of a Google Earth view.

    Cartopy ships a GoogleTiles class pointing at Google's own tile servers,
    which their terms do not allow for this. Esri publishes World Imagery for
    exactly this purpose and asks only for the attribution printed on the plot.
    """

    def _image_url(self, tile):
        x, y, z = tile
        return ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}".format(z=z, y=y, x=x))


def _tile_zoom_for_view(view_m, lat_deg, px=1000):
    """Tile zoom whose resolution suits a view of `view_m` metres across `px` pixels."""
    ground_res = view_m / float(px)
    equator_res = 156543.03392 * np.cos(np.deg2rad(lat_deg))
    zoom = int(np.floor(np.log2(max(equator_res / max(ground_res, 1e-6), 1.0))))
    return int(np.clip(zoom, 1, _MAX_TILE_ZOOM))


def _add_track_detail_inset(ax, lats, lons, impacts, lat_c, lon_c,
                            m_per_deg_lat, m_per_deg_lon, track_span_m,
                            norm, cmap):
    """Second panel showing the track in local metres, for very short tracks."""
    north_m = (np.asarray(lats) - lat_c) * m_per_deg_lat
    east_m = (np.asarray(lons) - lon_c) * m_per_deg_lon

    inset = ax.inset_axes([0.66, 0.06, 0.32, 0.32])
    inset.plot(east_m, north_m, '-', color='gray', linewidth=1, alpha=0.6)
    inset.scatter(east_m, north_m, c=impacts, cmap=cmap, norm=norm, s=26,
                  edgecolors='black', linewidths=0.3, zorder=5)
    inset.plot(east_m[0], north_m[0], 'g^', markersize=9, zorder=10)
    inset.plot(east_m[-1], north_m[-1], 'rs', markersize=9, zorder=10)

    pad = max(track_span_m * 0.25, 0.02)
    inset.set_xlim(east_m.min() - pad, east_m.max() + pad)
    inset.set_ylim(north_m.min() - pad, north_m.max() + pad)
    inset.set_aspect('equal', adjustable='box')
    inset.grid(True, alpha=0.3, linewidth=0.5)
    inset.tick_params(labelsize=7)
    inset.set_xlabel('East [m]', fontsize=8)
    inset.set_ylabel('North [m]', fontsize=8)

    span_txt = ('{:.0f} cm'.format(track_span_m * 100) if track_span_m < 1.0
                else '{:.1f} m'.format(track_span_m))
    inset.set_title('Track detail, ' + span_txt + ' total', fontsize=8)
    for spine in inset.spines.values():
        spine.set_edgecolor('#ff3b30')
        spine.set_linewidth(1.4)


def plot_trajectory_map(kin_data: dict, epoch_results: list, param_names: list,
                        kin_year: int = None, kin_doy: int = None,
                        antenna_name: str = None):
    """
    Plots the kinematic trajectory on a map, colored by the impact magnitude.

    The impact is computed as norm(North, East, Up) from the per-epoch
    adjustment results. Each trajectory point is color-coded by this value.

    Uses Cartopy coastlines/borders if available, otherwise falls back to
    a plain matplotlib scatter plot on lat/lon axes.

    Args:
        kin_data: Dict {sod: (x_m, y_m, z_m)} from parse_kin_file().
        epoch_results: List of (epoch_dt, results_vec, n_sats) from
                       perform_los_adjustment().
        param_names: List of parameter names.
        kin_year: Year from KIN file.
        kin_doy: DOY from KIN file.
        antenna_name: Optional antenna name for the title.
    """
    if not _main_thread_only('trajectory map'):
        return
    from .geodesy import ecef_to_ell
    from datetime import datetime, timedelta

    if not kin_data or not epoch_results:
        print("No KIN data or epoch results for trajectory map.")
        return

    # --- Convert ECEF to lat/lon ---
    sods_sorted = sorted(kin_data.keys())
    lats, lons, heights = [], [], []
    for sod in sods_sorted:
        x, y, z = kin_data[sod]
        lat_rad, lon_rad, h = ecef_to_ell(x, y, z)
        lats.append(np.degrees(lat_rad))
        lons.append(np.degrees(lon_rad))
        heights.append(h)

    lats = np.array(lats)
    lons = np.array(lons)

    # --- Match epoch results to KIN timestamps ---
    # Build a SOD lookup for epoch results
    epoch_sods = {}
    for ep_dt, results_vec, n_sats in epoch_results:
        sod = ep_dt.hour * 3600 + ep_dt.minute * 60 + ep_dt.second
        sod += ep_dt.microsecond / 1e6
        epoch_sods[round(sod, 1)] = results_vec

    # Extract N, E, U indices
    try:
        idx_n = param_names.index('North')
        idx_e = param_names.index('East')
        idx_u = param_names.index('Up')
    except ValueError:
        print("Cannot compute impact: N/E/U not in param_names.")
        return

    # Match and compute impact for each KIN point
    impacts = []
    matched_lats = []
    matched_lons = []
    for sod in sods_sorted:
        sod_key = round(sod, 1)
        if sod_key in epoch_sods:
            rv = epoch_sods[sod_key]
            impact = np.sqrt(rv[idx_n]**2 + rv[idx_e]**2 + rv[idx_u]**2)
            impacts.append(impact)
            idx = sods_sorted.index(sod)
            matched_lats.append(lats[idx])
            matched_lons.append(lons[idx])

    if not impacts:
        # Fallback: plot trajectory without coloring
        print("  No epoch results matched to KIN timestamps. Plotting trajectory only.")
        impacts = np.zeros(len(lats))
        matched_lats = lats
        matched_lons = lons

    matched_lats = np.array(matched_lats)
    matched_lons = np.array(matched_lons)
    impacts = np.array(impacts)

    # --- Plot ---
    use_cartopy = True
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        use_cartopy = False

    if use_cartopy:
        fig = plt.figure(figsize=(12, 8))

        # --- View extent -----------------------------------------------------
        # Lowering the view floor to 1e-5 deg (about 1.1 m) cured a map that
        # collapsed to a single dot, but produced the opposite symptom: at
        # that scale there is nothing to see but the flat land polygon and
        # the gridlines, a screen of featureless grey tiles. A useful map
        # needs both, a visible track AND recognisable ground, so the view is
        # kept at least _MIN_VIEW_M wide and a track too small for that scale
        # gets its own detail panel.
        lat_c = float((matched_lats.max() + matched_lats.min()) / 2.0)
        lon_c = float((matched_lons.max() + matched_lons.min()) / 2.0)
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * max(np.cos(np.deg2rad(lat_c)), 1e-6)

        span_lat_m = float(matched_lats.max() - matched_lats.min()) * m_per_deg_lat
        span_lon_m = float(matched_lons.max() - matched_lons.min()) * m_per_deg_lon
        track_span_m = max(span_lat_m, span_lon_m)

        view_m = max(track_span_m * 1.6, _MIN_VIEW_M)
        half_lat = (view_m / 2.0) / m_per_deg_lat
        half_lon = (view_m / 2.0) / m_per_deg_lon
        extent = [lon_c - half_lon, lon_c + half_lon,
                  lat_c - half_lat, lat_c + half_lat]

        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # --- Background: aerial imagery, plain drawing as fallback -----------
        got_imagery = False
        try:
            zoom = _tile_zoom_for_view(view_m, lat_c)
            ax.add_image(_AerialTiles(), zoom)
            got_imagery = True
            print("  Trajectory map: aerial imagery at zoom {}, view {:.0f} m across.".format(
                zoom, view_m))
        except Exception as exc:
            # No internet, or the provider refused. Say so and draw the plain
            # background rather than failing the whole plot.
            print("  Trajectory map: no imagery ({}), using plain background.".format(exc))

        if not got_imagery:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
            ax.add_feature(cfeature.LAND, color='#f0f0f0', alpha=0.5)
            ax.add_feature(cfeature.OCEAN, color='#d4e6f1', alpha=0.3)

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Plot full trajectory line
        ax.plot(lons, lats, '-', color='gray', linewidth=1, alpha=0.4,
                transform=ccrs.PlateCarree(), label='Trajectory')

        # Plot colored scatter
        sc = ax.scatter(matched_lons, matched_lats, c=impacts, cmap='hot_r',
                        s=30, edgecolors='black', linewidths=0.3, alpha=0.9,
                        transform=ccrs.PlateCarree(), zorder=5)

        # Start/end markers
        ax.plot(lons[0], lats[0], 'g^', markersize=12, transform=ccrs.PlateCarree(),
                zorder=10, label='Start')
        ax.plot(lons[-1], lats[-1], 'rs', markersize=12, transform=ccrs.PlateCarree(),
                zorder=10, label='End')

        # --- Detail inset when the track is too small to read on the map -----
        # The shipped example moves 17 cm in total. On a 120 m map that is one
        # pixel, so the epochs get their own panel in local metres.
        if track_span_m < view_m * 0.05:
            _add_track_detail_inset(ax, matched_lats, matched_lons, impacts,
                                    lat_c, lon_c, m_per_deg_lat, m_per_deg_lon,
                                    track_span_m, sc.norm, sc.cmap)
            ax.plot([lon_c], [lat_c], marker='o', markersize=18, markerfacecolor='none',
                    markeredgecolor='#ff3b30', markeredgewidth=1.6,
                    transform=ccrs.PlateCarree(), zorder=9,
                    label='Track (see detail)')

        if got_imagery:
            # Bottom left: the detail inset sits bottom right and its axis
            # label would run into this text.
            ax.text(0.005, 0.005, 'Imagery: Esri World Imagery', transform=ax.transAxes,
                    ha='left', va='bottom', fontsize=7, color='white',
                    bbox=dict(facecolor='black', alpha=0.35, pad=1.5, edgecolor='none'),
                    zorder=20)
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(lons, lats, '-', color='gray', linewidth=1, alpha=0.4, label='Trajectory')
        sc = ax.scatter(matched_lons, matched_lats, c=impacts, cmap='hot_r',
                        s=30, edgecolors='black', linewidths=0.3, alpha=0.9, zorder=5)
        ax.plot(lons[0], lats[0], 'g^', markersize=12, zorder=10, label='Start')
        ax.plot(lons[-1], lats[-1], 'rs', markersize=12, zorder=10, label='End')
        ax.set_xlabel('Longitude [°]')
        ax.set_ylabel('Latitude [°]')
        ax.grid(True, alpha=0.3)

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label(r'$\|\Delta$PCC Impact$\|$ (N,E,U) [mm]', fontsize=11)

    # Title
    title = r'Kinematic Trajectory: $\Delta$PCC Impact'
    if antenna_name:
        title += f'\n{antenna_name}'
    ax.set_title(title, fontsize=14, pad=15)
    # frameon=True is the part that matters. plot_time_series() calls
    # plt.style.use('seaborn-v0_8-whitegrid'), which sets legend.frameon False
    # for the WHOLE process, so any map drawn after a time series lost its
    # legend box and rendered dark labels straight onto the Esri imagery.
    # Nothing here may depend on which plot the user happened to open first.
    ax.legend(loc='upper left', fontsize=9, frameon=True, facecolor='white',
              framealpha=0.85, edgecolor='0.4', labelcolor='black')

    plt.tight_layout()
    plt.show(block=False)

