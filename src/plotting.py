# src/plotting.py
"""
Visualization module for PCC-Explorer.
Provides bar charts, time series plots, world map projections, and skyplots
for GNSS antenna PCC impact analysis results.
"""

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from .orbit_utils import get_ground_tracks, validate_orbit_data


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