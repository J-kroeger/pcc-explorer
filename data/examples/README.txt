PCC-Explorer - Example Input Files
==================================

Every input format PCC-Explorer accepts, with a small working example of each.
All four files describe the same fictitious scenario: station 0046, 4 April 2023
(day of year 94), 30-second sampling. They are synthetic - use them to check the
file layout and to try the program, not as measurements.

Contents
--------
  example_observation.csv        Line-of-Sight observations   (required for LoS mode)
  example_trajectory.KIN         Kinematic trajectory         (optional)
  example_attitude.ATT           Kinematic attitude           (optional)
  example_obstruction_mask.json  Obstruction mask             (optional)


1. example_observation.csv - Line-of-Sight observations
-------------------------------------------------------
Real satellite directions per epoch. This is what LoS mode analyses instead of
the simulated full-sky grid.

  Format:  comma-separated, one header line, one row per satellite per epoch
  Columns: epoch_utc, prn, elevation_deg, azimuth_deg [, heading_deg]

    epoch_utc       ISO 8601 UTC, e.g. 2023-04-04T10:00:00
    prn             Satellite ID, system letter + 2 digits: G01, E05, R12, C21
    elevation_deg   Elevation above the horizon, 0-90
    azimuth_deg     Azimuth clockwise from North, 0-360
    heading_deg     Optional. Platform heading clockwise from North. Use this
                    for a heading-only kinematic run without a full .ATT file.

  Example:
    epoch_utc,prn,elevation_deg,azimuth_deg
    2023-04-04T10:00:00,G01,45.2,112.3
    2023-04-04T10:00:00,G03,22.8,278.1

  Column names are case-insensitive and surrounding spaces are ignored.

  You do not have to write this file by hand: "Convert -> CSV" in the
  Line-of-Sight section builds it straight from a RINEX observation file
  (this needs RINEX-Masker; use "RINEX-Masker folder..." to say where it is).


2. example_trajectory.KIN - kinematic trajectory
-------------------------------------------------
Station position per epoch, for a moving platform. Fixed-width, two header
lines, then one row per epoch.

  STATION NAME   |YEAR|DOY| SECOND  |    X [m]     |    Y [m]     |    Z [m]

    STATION NAME  Station identifier
    YEAR          4-digit year
    DOY           Day of year, 1-366
    SECOND        Second of day, 0-86400, decimals allowed
    X, Y, Z       ECEF coordinates in metres

  Example row:
    0046               2023  94 36000.000   3849501.1901    649930.7711   5026986.9393

  Only the seven columns above are read. Files written by the IfE workspace
  carry further columns after Z (velocities and accelerations); these are
  ignored, so such a file can be used unchanged.


3. example_attitude.ATT - kinematic attitude
---------------------------------------------
Platform orientation per epoch. Same layout as the .KIN file.

  STATION NAME   |YEAR|DOY| SECOND  |  YAW [rad]   | PITCH [rad]  |  ROLL [rad]

    YAW           Heading, RADIANS, clockwise from North
    PITCH, ROLL   Rotation about the horizontal axes, RADIANS

  Example row:
    0046               2023  94 36000.000  -0.0523598776   0.0034906585   0.0017453293

  Note the unit: these columns are in RADIANS, while the GUI's manual Euler
  angle boxes are in DEGREES. -0.0523598776 rad = -3.0 deg.

  Without an .ATT file the analysis uses the manual Euler angles, or the
  optional heading_deg column of the CSV.


4. example_obstruction_mask.json - obstruction mask
----------------------------------------------------
An azimuth-dependent horizon: the real skyline around the station. Directions
below it are excluded, so the PCC impact is evaluated for the sky the station
actually sees. Produced by RINEX-Masker ("Export Mask (JSON)").

  Format:  pcc-mask-v1 JSON

    uniform_cutoff_deg   Flat cut-off applied at every azimuth
    sectors              Blocked azimuth ranges: az_from, az_to, el_limit
    horizon_profile      Skyline points: azimuth, elevation
                         Interpolated linearly, wrapping across 0/360 deg

  RINEX-Masker's plain-text mask is also accepted:

    # Uniform cutoff: 5.0 deg
    0.0    6.0          <- 2 values  = horizon point (azimuth, elevation)
    20.0   5.5
    90.0 180.0 25.0     <- 3 values  = sector (az_from, az_to, el_limit)

  The example describes an open northern horizon with a tall building to the
  south (elevation rising to 43 deg).

  Tip: the elevation mask and the obstruction mask both remove observations,
  and the flat cut-off usually dominates. To see what the mask itself
  contributes, set the elevation mask to 0. The log reports how many
  directions the mask removed.


Which files do I need?
----------------------
  Grid mode (default)     none of these - the sky is simulated
  LoS, static station     example_observation.csv
  LoS, moving platform    example_observation.csv + .KIN + .ATT
  Any mode, real horizon  add the obstruction mask
