# PCC-Explorer

**An open-source software tool to assess the impact of GNSS antenna Phase Center Corrections (PCC) on geodetic parameters.**

PCC-Explorer allows researchers and practitioners to analyze how differences in antenna Phase Center Corrections propagate into estimated geodetic parameters such as coordinates and tropospheric delays.

---

## Table of Contents
- [Citation](#citation)
- [Quick Start](#quick-start)
- [Features & GUI Overview](#features--gui-overview)
  - [Selection of the ANTEX‑Files](#selection-of-the-antex-files)
  - [Antenna Type](#antenna-type)
  - [Analysis Insights](#analysis-insights)
  - [Selection of the User Position](#selection-of-the-user-position)
  - [Selection of GPS Observation Time](#selection-of-gps-observation-time)
  - [Orbit Type](#orbit-type)
  - [Frequency & Signal](#frequency--signal)
  - [Troposphere Mapping Function](#troposphere-mapping-function)
  - [Observation Weighting Model](#observation-weighting-model)
  - [Obstruction Masks](#obstruction-masks)
  - [Plot Settings](#plot-settings)
  - [Analysis Buttons](#analysis-buttons)
  - [Configuration Files](#configuration-files)
  - [Results](#results)
- [Command Line Interface (Batch Processing)](#command-line-interface-batch-processing)
- [File Structure](#file-structure)
- [Contact](#contact)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Citation

If you use this software, please cite:

> Kröger J., Kersten T., Schön S. (2026) PCC-Explorer: An open-source software tool to assess the impact of GNSS antenna phase center corrections on geodetic parameters. GPS Solutions. <https://doi.org/10.1007/s10291-026-02056-2>

---

## Quick Start

```bash
pip install -r requirements.txt
python -m src.main_gui
```

Or use the provided launcher scripts:
- **Windows:** `launch_gui.bat`
- **Linux/macOS:** `launch_gui.sh`

---

## Features and GUI Overview

### Selection of the ANTEX-Files

Choose how to load antenna calibration data:

- **Compare two files:** Load two individual ANTEX files (e.g., chamber vs. robot calibration) and compare the impact of their PCC differences on geodetic parameters.
- **Single file (absolute PCC):** Load a single ANTEX file to analyze the absolute impact of its PCC values.
- **Whole folder:** Select a folder containing multiple ANTEX files and a reference file. The software will compare each file in the folder against the reference file, producing results for all combinations.

### Antenna Type

When loading ANTEX files, the software automatically detects and lists all antenna types contained in each file. For multi-antenna files like `igs20.atx`, you can select the specific antenna type from the dropdown menus for File 1 and File 2 independently. A **Match** indicator shows whether the two selected antennas are the same type.

### Analysis Insights

Displays real-time feedback about your current configuration:
- Whether the selected date is suitable for Final Orbits (precise orbits have a ~21-day delay)
- Automatic suggestions (e.g., switch to Broadcast Ephemeris if the date is too recent)

### Selection of the User Position

Define the station location for the analysis:

- **Ellipsoidal Coordinates:** Enter latitude (deg), longitude (deg), and height (m)
- **ECEF Coordinates:** Enter X, Y, Z in meters
- **Choose on Map:** Click on an interactive map to select the position
- **Area-based calculation (Global):** Define a latitude/longitude grid for global or regional analysis with configurable resolution

### Selection of GPS Observation Time

Configure the observation period:

- **Start/End Date:** The processing date range (YYYY-MM-DD format)
- **Start/End Time:** The time window within each day (HH:MM format)
- **Sample Rate [sec]:** The observation sampling rate in seconds (e.g., 30, 300)
- **Interval [HH:MM]:** For sub-daily analysis, define the interval length (e.g., 03:00 for 3-hour intervals, 24:00 for full-day)

### Orbit Type

Select the GNSS orbit source:

Both options download automatically, trying **several public sources in turn** so that no single server being moved, empty, or busy can break the download:

- **Broadcast Ephemeris:** multi-GNSS broadcast navigation data from BKG (Bundesamt für Kartographie und Geodäsie), then NASA JPL/CDDIS as a last resort.
- **Final Orbits (Precise):** multi-GNSS precise orbits from CODE, tried in this order — **AIUB** (`http://www.aiub.unibe.ch/download/CODE_MGEX/CODE/`, no login), then **IGN** (`igs.ign.fr`, no login), then **NASA CDDIS** (login required, used only if the first two fail). Final products are the most accurate but have a delay of a few days to ~2–3 weeks; orbits are downloaded automatically to the `data/orbit/` folder.

Because the login-free sources (AIUB and IGN) are tried first, a NASA Earthdata login is **no longer required for normal use** — CDDIS is only a last-resort fallback.

**NASA CDDIS Credentials (optional last-resort fallback):**
To enable NASA CDDIS as a fallback orbit source, create a file named `_netrc` (Windows) or `.netrc` (Linux/macOS) in your home directory with the following content:
```
machine urs.earthdata.nasa.gov
    login <your_username>
    password <your_password>
```
Register for a free account at: https://urs.earthdata.nasa.gov/

### Frequency & Signal

Select which GNSS signal(s) to analyze:

- **Single Select Mode:** Choose one signal at a time (Radio buttons)
- **Multi-frequency Mode:** Toggle to select multiple signals simultaneously (Checkboxes)

Available signals depend on the intersection of signals in both loaded ANTEX files. The display uses the convention: G=GPS, R=GLONASS, E=Galileo, C=BDS.

Ionosphere-free linear combinations (IF-LC) are automatically offered when both required signals are available (e.g., GPS IF-LC L1/L2).

### Troposphere Mapping Function

Choose the mapping function for tropospheric delay modeling:

- **1/sin(Elevation):** Standard mapping function
- **GMF:** Global Mapping Function
- **None:** No tropospheric modeling

**Estimate Troposphere Gradients:** Enable this checkbox to estimate North and East tropospheric gradient parameters using the simple gradient model.

### Observation Weighting Model

Select how observations are weighted based on satellite elevation:

- **sin:** Weight = sin(elevation) — standard choice
- **sin²:** Weight = sin²(elevation) — stronger down-weighting of low-elevation observations
- **unit:** Equal weight for all observations regardless of elevation

### Obstruction Masks

Configure elevation and azimuth masks:

- **Elevation mask:** Exclude observations below a specified elevation angle (in degrees)
- **Azimuth mask(s):** Define azimuth sectors to exclude, using format `start-end,elevation` (e.g., 0-45,5 means mask azimuth 0°–45° below 5° elevation)

### Plot Settings

At the bottom of the main window:

- **Color Scheme:** Choose from Distinct, Paper Grey, Grayscale, Warm, or Cool color palettes for bar charts
- **Y-Axis:** Auto (default), Symmetric, Shared, or Custom min/max for time series plots
- **Colorbar (Global):** Auto or Custom min/max for global analysis color scales

### Analysis Buttons

- **Run Single Analysis:** Analyze the impact for a single epoch/day
- **Run Time Series Analysis:** Analyze the impact over a date range, producing time series plots
- **Run Global/Regional Analysis:** Compute the impact for a grid of positions, producing world map plots
- **Show Skyplot:** Visualize satellite visibility for the configured station and time
- **Close All Figure Windows:** Close all open matplotlib plot windows
- **Quit App:** Exit the application

### Configuration Files

- **Load Existing Configuration File:** Load a previously saved configuration to restore all settings
- **Create New Configuration File:** Open a dialog to create and save a configuration file with all current settings

Configuration files are saved in `key=value` format in the `configs/` folder and can be reused for batch processing or to reproduce analyses.

### Results

Analysis results are saved to the `results/` folder as text files. Each result file references the corresponding configuration file used for the analysis.

**Result file format:**
- Lines starting with `#` are comments/headers
- The header includes the analysis date and the referenced configuration file
- Data rows use fixed-width columns:
  - Column 1 (Parameter): Characters 1–20, left-aligned
  - Column 2 (Impact): Characters 24–33, right-aligned, 4 decimal places
- All values are in **millimeters [mm]**

Example:
```
# ==================================================
# PCC-Explorer Analysis Result
# Date: 2026-02-26 09:50:11
# Configuration file:  Analysis_..._config_20260226_095011.txt
# ==================================================
# Parameter              Impact [mm]
North                      0.2508
East                       0.2797
Up                         0.6412
Clock_GPS                  1.1959
Tropo                      0.0170
# ==================================================
```

---

## Command Line Interface (Batch Processing)

PCC-Explorer can also be run from the command line using a configuration file:

```bash
python -m src.main_cli <config_file.txt> [--mode single|timeline|global|folder]
```

- `<config_file.txt>`: Path to a previously saved configuration file
- `--mode`: Optional override for the analysis mode. If omitted, the mode is auto-detected from the configuration file.

Example:
```bash
python -m src.main_cli configs/Analysis_LEIAR25.R4______LEIT_725058_config_20260228_120000.txt --mode single
```

---

## File Structure

```
PCC-Explorer/
├── data/
│   ├── antex/               # ANTEX calibration files
│   └── orbit/               # Downloaded orbit files (auto-populated)
├── configs/                 # Configuration files
├── results/                 # Analysis output files
├── LICENSE                  # GNU GPLv3 full text
├── LICENSE.txt              # Short copyright notice
└── readme.txt               # This file
```

---

## Contact
Dr.-Ing. Johannes Kröger – <kroeger@ife.uni-hannover.de>  

---

## License

This project is licensed under the GNU General Public License v3.0 or later.
See LICENSE and LICENSE.txt

## Acknowledgements

We thank the Center for Orbit Determination in Europe (CODE) for providing high-quality GNSS orbit products, AIUB (University of Bern) and IGN (Institut national de l'information géographique et forestière) for hosting them for public download, the Federal Agency for Cartography and Geodesy (BKG) for providing publicly accessible broadcast ephemeris data, and NASA JPL/CDDIS for providing publicly accessible GNSS data as a fallback source.
