# src/data_io.py
"""
Data input/output module for PCC-Explorer.
Handles ANTEX file parsing, orbit file downloading/reading, configuration
file parsing, and result file writing.
"""

import os
import gzip
import shutil
import subprocess
import traceback
import re
from datetime import datetime, timedelta
from collections import defaultdict
from netrc import netrc
from ftplib import FTP
import time

import numpy as np
import requests

from .ephemeris_parser import read_ephemeris_records
from .time_utils import datetime_to_mjd, mjd_to_gps_week


# ----------------------------
# Antenna name canonicalization
# ----------------------------

def canonicalize_antenna(name: str) -> str:
    """
    Canonicalize antenna names by collapsing internal whitespace and uppercasing.
    Example: "JAVRINGANT_DM    NONE" -> "JAVRINGANT_DM NONE"
    """
    return ' '.join(str(name).split()).upper()


# ----------------------------
# NASA/Cddis helpers (HTTP)
# ----------------------------

def _get_nasa_credentials():
    netrc_path = os.path.join(os.path.expanduser("~"), "_netrc")
    if not os.path.exists(netrc_path):
        return None, None
    try:
        info = netrc(netrc_path)
        auth = info.authenticators("urs.earthdata.nasa.gov")
        if auth:
            login, _, password = auth
            return login, password
    except (FileNotFoundError, TypeError):
        return None, None
    return None, None


def _download_with_requests(url: str, save_path: str):
    username, password = _get_nasa_credentials()
    if not (username and password):
        raise ConnectionError("Could not find NASA credentials in _netrc file.")

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/91.0.4472.124 Safari/537.36'
        )
    }

    with requests.Session() as session:
        session.auth = (username, password)
        response = session.get(url, stream=True, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            raise ConnectionError(
                f"Download failed. Server returned an HTML page instead of a file. URL: {url}"
            )
        print(f"Successfully connected to NASA CDDIS. Downloading file to {save_path}...")
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

# ----------------------------
# Anonymous orbit sources (no login) — used before CDDIS
# ----------------------------

def _http_get_anon(url: str, save_path: str, timeout: int = 90):
    """Anonymous HTTP(S) download (no credentials). Raises on any failure."""
    headers = {'User-Agent': 'PCC-Explorer'}
    with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
        r.raise_for_status()
        if 'text/html' in r.headers.get('Content-Type', ''):
            raise ConnectionError(f"Server returned HTML, not a file: {url}")
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)


def _decompress_gz(local_compressed: str, local_final: str):
    """gunzip local_compressed -> local_final; returns local_final or None."""
    with gzip.open(local_compressed, 'rb') as f_in, open(local_final, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(local_compressed)
    return local_final if os.path.exists(local_final) else None


def _download_from_aiub_http(dt_obj: datetime, download_dir: str) -> str:
    """
    CODE MGEX final orbit via AIUB's new HTTP endpoint (anonymous, no login).
    AIUB moved off FTP in 2026; products now live at
    http://www.aiub.unibe.ch/download/CODE_MGEX/CODE/<year>/<file>
    """
    year_full = dt_obj.strftime('%Y')
    day_of_year = dt_obj.strftime('%j')
    fn = f"COD0MGXFIN_{year_full}{day_of_year}0000_01D_05M_ORB.SP3.gz"
    url = f"http://www.aiub.unibe.ch/download/CODE_MGEX/CODE/{year_full}/{fn}"
    local_c = os.path.join(download_dir, fn)
    local_f = os.path.join(download_dir, fn[:-3])
    print(f"\nAttempting AIUB HTTP: {url}")
    try:
        _http_get_anon(url, local_c)
        return _decompress_gz(local_c, local_f)
    except Exception as e:
        print(f"  AIUB HTTP failed: {e}")
        if os.path.exists(local_c):
            os.remove(local_c)
        return None


def _download_from_ign_ftp(dt_obj: datetime, download_dir: str) -> str:
    """
    Multi-GNSS finals then rapid via IGN's anonymous IGS mirror
    (igs.ign.fr:/pub/igs/products/<week>/). No login. First file found wins.
    """
    year_full = dt_obj.strftime('%Y')
    day_of_year = dt_obj.strftime('%j')
    mjd_val = datetime_to_mjd(dt_obj)
    gps_week, _dow = mjd_to_gps_week(mjd_val)
    remote_dir = f"/pub/igs/products/{gps_week}"

    def _fn(ac, camp):
        return f"{ac}0{camp}_{year_full}{day_of_year}0000_01D_05M_ORB.SP3.gz"

    files = [_fn('COD', 'MGXFIN'), _fn('GRG', 'MGXFIN'),
             _fn('WUM', 'MGXFIN'), _fn('GFZ', 'MGXRAP')]
    print(f"\nAttempting IGN FTP (igs.ign.fr{remote_dir})...")
    ftp = None
    try:
        ftp = FTP('igs.ign.fr', timeout=30)
        ftp.login()  # anonymous
        ftp.cwd(remote_dir)
        for fn in files:
            local_c = os.path.join(download_dir, fn)
            local_f = os.path.join(download_dir, fn[:-3])
            try:
                print(f"  Trying: {fn}...")
                with open(local_c, 'wb') as f:
                    ftp.retrbinary(f"RETR {fn}", f.write)
            except Exception:
                if os.path.exists(local_c):
                    os.remove(local_c)
                continue
            ftp.quit()
            ftp = None
            return _decompress_gz(local_c, local_f)
    except Exception as e:
        print(f"  IGN FTP failed: {e}")
    finally:
        if ftp:
            try:
                ftp.quit()
            except Exception:
                pass
    return None


# ----------------------------
# AIUB/CODE Helpers (FTP)
# ----------------------------

def _download_from_aiub_ftp(dt_obj: datetime, download_dir: str) -> str:
    """
    DEPRECATED shim. AIUB shut down anonymous FTP and moved CODE products to HTTP
    (2026-07); this now delegates to _download_from_aiub_http(). The old FTP code
    below is unreachable and kept only for reference.
    """
    return _download_from_aiub_http(dt_obj, download_dir)
    ftp_host = 'ftp.aiub.unibe.ch'  # noqa - unreachable, legacy FTP path
    
    # Time conversions
    mjd_val = datetime_to_mjd(dt_obj)
    gps_week, day_of_week = mjd_to_gps_week(mjd_val)
    year_full = dt_obj.strftime('%Y')
    day_of_year = dt_obj.strftime('%j')
    
    # Pattern 1: MGEX multi-GNSS products (Priority #1 for Beidou support)
    # Path: /CODE_MGEX/CODE/{year}/
    file_mgex = f"COD0MGXFIN_{year_full}{day_of_year}0000_01D_05M_ORB.SP3.gz"
    dir_mgex = f"CODE_MGEX/CODE/{year_full}"

    # Pattern 2: Standard CODE final (often lacks Beidou)
    # Path: /CODE/{year}/
    file_std_long = f"COD0OPSFIN_{year_full}{day_of_year}0000_01D_05M_ORB.SP3.gz"
    dir_std = f"CODE/{year_full}"
    
    # Pattern 3: Old short format (Legacy)
    file_std_short = f"COD{gps_week:04d}{day_of_week}.EPH.Z"


    
    targets = [
        (file_mgex, dir_mgex),           # Priority 1: CODE MGEX (GRECJ)
        (file_std_long, dir_std),        # Priority 2: CODE Standard Long
        (file_std_short, dir_std)        # Priority 3: CODE Standard Short
    ]
    
    print(f"\nAttempting download from AIUB FTP ({ftp_host})...")
    
    # --- OPTIMIZATION: Check ALL local patterns FIRST before any FTP ---
    # This avoids slow FTP timeouts when files are cached in a different format
    for target_file, remote_dir in targets:
        if target_file.endswith('.Z'):
            local_final = os.path.join(download_dir, target_file[:-2])
        elif target_file.endswith('.gz'):
            local_final = os.path.join(download_dir, target_file[:-3])
        else:
            local_final = os.path.join(download_dir, target_file)

        if os.path.exists(local_final):
            print(f"  Using existing file: {local_final}")
            return local_final
    
    # --- No cached file found, try FTP download ---
    for target_file, remote_dir in targets:
        local_compressed = os.path.join(download_dir, target_file)
        
        if target_file.endswith('.Z'):
            local_final = os.path.join(download_dir, target_file[:-2])
        elif target_file.endswith('.gz'):
            local_final = os.path.join(download_dir, target_file[:-3])
        else:
            local_final = os.path.join(download_dir, target_file)
        
        ftp = None
        try:
            ftp = FTP(ftp_host)
            ftp.login()  # Anonymous login
            ftp.cwd(remote_dir)
            
            print(f"  Trying: {remote_dir}/{target_file}...")
            with open(local_compressed, 'wb') as f:
                ftp.retrbinary(f"RETR {target_file}", f.write)
            
            print("  Download successful. Decompressing...")
            ftp.quit()
            
            # Decompression logic
            if target_file.endswith('.gz'):
                with gzip.open(local_compressed, 'rb') as f_in, open(local_final, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(local_compressed)
                return local_final
                
            elif target_file.endswith('.Z'):
                # Requires 7-Zip (7z) on Windows for .Z files
                try:
                    subprocess.run(["7z", "x", f"-o{download_dir}", "-y", local_compressed],
                                 check=True, capture_output=True)
                    os.remove(local_compressed)
                    
                    extracted_name = target_file[:-2]
                    extracted_path = os.path.join(download_dir, extracted_name)
                    if os.path.exists(extracted_path):
                        return extracted_path
                except (FileNotFoundError, subprocess.CalledProcessError):
                    print("  Warning: 7-Zip failed or not found. Cannot decompress .Z file.")
                    if os.path.exists(local_compressed): os.remove(local_compressed)
                    continue
                    
        except Exception as e:
            print(f"  File not found or error: {e}")
            if os.path.exists(local_compressed): os.remove(local_compressed)
            if ftp: 
                try: ftp.quit()
                except: pass
            continue
            
    print("  AIUB FTP: All file patterns failed.")
    return None


# ----------------------------
# Public Interface
# ----------------------------

def download_and_unzip_broadcast_file(dt_obj: datetime, download_dir: str) -> str:
    """
    Downloads broadcast navigation file.
    Priority 1: BKG FTP (public, no login required)
    Priority 2: NASA CDDIS (requires _netrc credentials)
    """
    year_full = dt_obj.strftime('%Y')
    year_short = dt_obj.strftime('%y')
    day_of_year = dt_obj.strftime('%j')
    year_int = dt_obj.year
    os.makedirs(download_dir, exist_ok=True)

    # --- Priority 1: BKG FTP (igs-ftp.bkg.bund.de) ---
    bkg_host = 'igs-ftp.bkg.bund.de'
    bkg_targets = []

    if year_int >= 2016:
        # Current BKG location (2026-07): broadcast nav moved from /MGEX/BRDC/,
        # which now lags, to /IGS/BRDC/. Try the live location + current names
        # first; the older /MGEX/BRDC/ paths below stay as a fallback.
        igs_dir = f"IGS/BRDC/{year_full}/{day_of_year}"
        bkg_targets.append((f"BRDC00IGS_R_{year_full}{day_of_year}0000_01D_MN.rnx.gz", igs_dir))
        bkg_targets.append((f"BRDM00DLR_S_{year_full}{day_of_year}0000_01D_MN.rnx.gz", igs_dir))
        bkg_targets.append((f"BRDC00WRD_R_{year_full}{day_of_year}0000_01D_MN.rnx.gz", igs_dir))

        # Long filename format (MGEX/BRDC/)
        bkg_dir = f"MGEX/BRDC/{year_full}/{day_of_year}"
        # Pattern 1: BRDC00WRD (current standard on BKG)
        bkg_file_wrd = f"BRDC00WRD_S_{year_full}{day_of_year}0000_01D_MN.rnx.gz"
        bkg_targets.append((bkg_file_wrd, bkg_dir))
        # Pattern 2: BRDM00DLR (alternative/older naming)
        bkg_file_dlr = f"BRDM00DLR_S_{year_full}{day_of_year}0000_01D_MN.rnx.gz"
        bkg_targets.append((bkg_file_dlr, bkg_dir))
        # Pattern 3: Short filename
        bkg_file_short = f"brdm{day_of_year}0.{year_short}p.Z"
        bkg_targets.append((bkg_file_short, bkg_dir))
    if year_int >= 2014 and year_int <= 2020:
        # BRDC_v3 directory (older format)
        bkg_file_v3 = f"brdm{day_of_year}0.{year_short}p.Z"
        bkg_dir_v3 = f"MGEX/BRDC_v3/{year_full}/{day_of_year}"
        bkg_targets.append((bkg_file_v3, bkg_dir_v3))

    # Check for ANY cached file first (from either source)
    for target_file, _ in bkg_targets:
        if target_file.endswith('.gz'):
            local_final = os.path.join(download_dir, target_file[:-3])
        elif target_file.endswith('.Z'):
            local_final = os.path.join(download_dir, target_file[:-2])
        else:
            local_final = os.path.join(download_dir, target_file)
        if os.path.exists(local_final):
            print(f"  Using existing broadcast file: {local_final}")
            return local_final

    # Also check for NASA-format cached file
    nasa_uncompressed = os.path.join(download_dir, f"brdc{day_of_year}0.{year_short}n")
    if os.path.exists(nasa_uncompressed):
        print(f"  Using existing broadcast file: {nasa_uncompressed}")
        return nasa_uncompressed

    # Try BKG FTP download
    if bkg_targets:
        print(f"\nAttempting broadcast download from BKG FTP ({bkg_host})...")
        for target_file, remote_dir in bkg_targets:
            local_compressed = os.path.join(download_dir, target_file)
            if target_file.endswith('.gz'):
                local_final = os.path.join(download_dir, target_file[:-3])
            elif target_file.endswith('.Z'):
                local_final = os.path.join(download_dir, target_file[:-2])
            else:
                local_final = os.path.join(download_dir, target_file)

            ftp = None
            try:
                ftp = FTP(bkg_host)
                ftp.login()  # Anonymous login
                ftp.cwd(remote_dir)
                print(f"  Trying: {remote_dir}/{target_file}...")
                with open(local_compressed, 'wb') as f:
                    ftp.retrbinary(f"RETR {target_file}", f.write)
                print("  Download successful. Decompressing...")
                ftp.quit()

                # Decompress
                if target_file.endswith('.gz'):
                    with gzip.open(local_compressed, 'rb') as f_in, open(local_final, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(local_compressed)
                    return local_final
                elif target_file.endswith('.Z'):
                    try:
                        subprocess.run(["7z", "x", f"-o{download_dir}", "-y", local_compressed],
                                     check=True, capture_output=True)
                        os.remove(local_compressed)
                        if os.path.exists(local_final):
                            return local_final
                    except (FileNotFoundError, subprocess.CalledProcessError):
                        print("  Warning: 7-Zip failed or not found. Cannot decompress .Z file.")
                        if os.path.exists(local_compressed): os.remove(local_compressed)
                        continue
            except Exception as e:
                print(f"  BKG: File not found or error: {e}")
                if os.path.exists(local_compressed): os.remove(local_compressed)
                if ftp:
                    try: ftp.quit()
                    except: pass
                continue

        print("  BKG FTP: All broadcast file patterns failed.")

    # --- Priority 2: NASA CDDIS (Fallback) ---
    print("  Falling back to NASA CDDIS for broadcast file...")
    try:
        compressed_filename = f"brdc{day_of_year}0.{year_short}n.gz"
        url = f"https://cddis.nasa.gov/archive/gnss/data/daily/{year_full}/{day_of_year}/{year_short}n/{compressed_filename}"
        uncompressed_path = os.path.join(download_dir, compressed_filename.replace(".gz", ""))

        compressed_path = os.path.join(download_dir, compressed_filename)
        _download_with_requests(url, compressed_path)

        with gzip.open(compressed_path, 'rb') as f_in, open(uncompressed_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(compressed_path)
        return uncompressed_path
    except Exception as e:
        print(f"  NASA CDDIS fallback also failed: {e}")
        return None


def download_final_orbit_file(dt_obj: datetime, download_dir: str) -> str:
    """
    Downloads final orbit file. 
    Priority 1: AIUB FTP (No login required, CODE products)
    Priority 2: NASA CDDIS (Login required, IGS products)
    """
    os.makedirs(download_dir, exist_ok=True)

    # --- Cache: reuse any known product already downloaded for this day ---
    _yf = dt_obj.strftime('%Y'); _doy = dt_obj.strftime('%j')
    for _p in (f"COD0MGXFIN_{_yf}{_doy}0000_01D_05M_ORB.SP3",
               f"GRG0MGXFIN_{_yf}{_doy}0000_01D_05M_ORB.SP3",
               f"WUM0MGXFIN_{_yf}{_doy}0000_01D_05M_ORB.SP3",
               f"GFZ0MGXRAP_{_yf}{_doy}0000_01D_05M_ORB.SP3"):
        _lf = os.path.join(download_dir, _p)
        if os.path.exists(_lf):
            print(f"Using existing final orbit file: {_lf}")
            return _lf

    # --- Priority 1: anonymous sources, no login (fallback chain) ---
    #   AIUB moved to HTTP (2026-07); IGN is a login-free IGS mirror. Trying
    #   several sources in turn means no single server going down breaks it.
    for _src in (_download_from_aiub_http, _download_from_ign_ftp):
        result_path = _src(dt_obj, download_dir)
        if result_path and os.path.exists(result_path):
            return result_path
        
    print("Falling back to NASA CDDIS (requires _netrc)...")

    # --- Priority 2: NASA CDDIS (HTTP) ---
    mjd_val = datetime_to_mjd(dt_obj)
    gps_week, day_of_week = mjd_to_gps_week(mjd_val)
    year_full = dt_obj.strftime('%Y')
    day_of_year = dt_obj.strftime('%j')

    patterns_to_try = [
        f"COD0MGXFIN_{year_full}{day_of_year}0000_01D_05M_ORB.SP3.gz",
        f"igs{gps_week:04d}{day_of_week}.sp3.Z",
        f"cod{gps_week:04d}{day_of_week}.sp3.Z",
        f"igs{gps_week:04d}{day_of_week}.sp3.gz",
        f"cod{gps_week:04d}{day_of_week}.sp3.gz",
    ]
    base_url = f"https://cddis.nasa.gov/archive/gnss/products/{gps_week:04d}"

    for compressed_filename in patterns_to_try:
        uncompressed_filename = compressed_filename.replace(".gz", "").replace(".Z", "")
        uncompressed_path = os.path.join(download_dir, uncompressed_filename)
        if os.path.exists(uncompressed_path):
            print(f"Using existing final orbit file: {uncompressed_path}")
            return uncompressed_path

        url = f"{base_url}/{compressed_filename}"
        compressed_path = os.path.join(download_dir, compressed_filename)
        try:
            print(f"Attempting to download data from:\n{url}")
            _download_with_requests(url, compressed_path)
            
            if compressed_filename.endswith(".gz"):
                with gzip.open(compressed_path, 'rb') as f_in, open(uncompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            elif compressed_filename.endswith(".Z"):
                subprocess.run(["7z", "x", f"-o{download_dir}", "-y", compressed_path],
                               check=True, capture_output=True)
                
            os.remove(compressed_path)
            print(f"Successfully created: {uncompressed_path}")
            return uncompressed_path
            
        except FileNotFoundError:
            print("7-Zip is not installed or not in PATH. Skipping .Z files.")
            continue
        except Exception as e:
            print(f"Download failed for {url}: {e}")
            if os.path.exists(compressed_path): os.remove(compressed_path)
            continue
            
    return None


# ----------------------------
# ANTEX parsing
# ----------------------------

def _parse_pcv_line(line: str, expected_length: int) -> list:
    """Parse PCV values from a data line (8-character fields starting at column 10)."""
    values = []
    # Loop from column 10, in 8-character steps
    for i in range(10, len(line.rstrip()), 8):
        val_str = line[i:i + 8].strip()
        
        if not val_str:
            values.append(np.nan)
            continue
            
        try:
            # Clean the string: replace common non-ASCII hyphens/minuses
            cleaned_str = val_str.replace('\u2010', '-').replace('\u2013', '-').replace('\u2014', '-')
            values.append(float(cleaned_str))
        except (ValueError, TypeError):
             values.append(np.nan)

    while len(values) < expected_length:
        values.append(np.nan)
        
    return values[:expected_length]


def _get_system_from_code(code: str) -> str:
    """Extract GNSS system from frequency code."""
    if not code:
        return None
    if code.startswith('G'): return 'GPS'
    if code.startswith('R'): return 'GLONASS'
    if code.startswith('E'): return 'Galileo'
    if code.startswith('C'): return 'BDS'
    return None


def read_antex_file(filename: str) -> dict:
    """
    Reads an ANTEX file and extracts PCO and PCV data for all antennas.

    Args:
        filename: Path to the ANTEX (.atx) file.

    Returns:
        dict with keys:
            - 'metadata': dict mapping antenna keys to type/serial info
            - 'pco': nested dict {antenna_key: {system: {freq_code: np.ndarray(3,)}}}
            - 'pcv': nested dict {antenna_key: {system: {freq_code: np.ndarray(azi, zen)}}}
            - 'canonical_map': dict for resolving canonicalized names
    """
    all_pco = defaultdict(lambda: defaultdict(dict))
    all_pcv = defaultdict(lambda: defaultdict(dict))
    metadata = {}

    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"ERROR: Could not read file {filename}: {e}")
        return {'metadata': {}, 'pco': {}, 'pcv': {}, 'canonical_map': {}}

    line_iterator = iter(lines)
    current_antenna_key = None # Stores the unique "Type Serial" key
    num_zenith_steps = 19
    line_num = 0

    for line in line_iterator:
        line_num += 1
        if len(line) < 60:
            continue
        line_label = line[60:].strip() if len(line) > 60 else ""
        line_content = line[0:60] # Use full content for parsing

        # Parse antenna name and serial
        if 'TYPE / SERIAL NO' in line_label:
            # ANTEX format: Type (0-20), Serial (20-40)
            ant_type = line[0:20].strip()
            ant_serial = line[20:40].strip()
            
            # N3: Skip satellite antennas (BLOCK IIR, IIF, IIA, etc., GLONASS, GALILEO, BEIDOU, QZSS, IRNSS)
            satellite_keywords = ['BLOCK', 'GLONASS', 'GALILEO', 'BEIDOU', 'QZSS', 'IRNSS', 'SVN', 'IOV', 'FOC']
            if any(kw in ant_type.upper() for kw in satellite_keywords):
                current_antenna_key = None  # Skip this antenna
                continue
            
            # U2: Create a unique key - use TYPEMEAN instead of NONE for empty serial numbers
            if not ant_serial or ant_serial.upper() == "NONE":
                ant_serial = "TYPEMEAN"
                
            current_antenna_key = f"{ant_type} {ant_serial}".strip()
            
            if current_antenna_key not in metadata:
                metadata[current_antenna_key] = {
                    'type': ant_type,
                    'serial': ant_serial
                }

        elif 'START OF ANTENNA' in line_label:
            pass

        elif ('START OF FREQUENCY' in line_label) and current_antenna_key:
            freq_code = line_content.strip().split()[0] if line_content.strip() else ""

            system = _get_system_from_code(freq_code)
            if not system:
                continue

            current_pcv_rows = []

            while True:
                try:
                    data_line = next(line_iterator)
                    line_num += 1
                    if len(data_line) < 60:
                        continue
                    data_label = data_line[60:].strip() if len(data_line) > 60 else ""
                    data_content = data_line[0:60]

                    if 'END OF FREQUENCY' in data_label:
                        break

                    # PCO line
                    if 'NORTH / EAST / UP' in data_label:
                        try:
                            pco_vals = np.array([
                                float(data_content[0:10]),
                                float(data_content[10:20]),
                                float(data_content[20:30])
                            ], dtype=float)
                            
                            all_pco[current_antenna_key][system][freq_code] = pco_vals
                        except (ValueError, IndexError) as e:
                            if 'zeros.atx' in filename.lower():
                                pco_vals = np.array([0.0, 0.0, 0.0], dtype=float)
                                all_pco[current_antenna_key][system][freq_code] = pco_vals
                            else:
                                print(f" Warning: Could not parse PCO at line {line_num}: {e}")

                    # PCV rows (NOAZI or azimuth-dependent)
                    # NOAZI lines contain non-azimuth-dependent PCV values
                    # Azimuth-dependent lines start with the azimuth angle
                    elif data_content.strip().startswith('NOAZI') or (data_content[0:10].strip() and 
                           not any(lbl in data_label for lbl in ['NORTH', 'EAST', 'UP', 'START', 'END', 'COMMENT'])):
                        pcv_values = _parse_pcv_line(data_line, num_zenith_steps)
                        if pcv_values:
                            current_pcv_rows.append(pcv_values)

                except StopIteration:
                    break

            if current_pcv_rows:
                pcv_array = np.array(current_pcv_rows, dtype=float)
                all_pcv[current_antenna_key][system][freq_code] = pcv_array

        elif 'ZEN1 / ZEN2 / DZEN' in line_label and current_antenna_key:
            try:
                zen1 = float(line[2:8])
                zen2 = float(line[8:14])
                dzen = float(line[14:20])
                num_zenith_steps = int((zen2 - zen1) / dzen) + 1
                metadata[current_antenna_key]['zenith_steps'] = num_zenith_steps
                metadata[current_antenna_key]['dzen'] = dzen
            except (ValueError, KeyError):
                pass

    all_antennas = set(list(all_pco.keys()) + list(all_pcv.keys()))

    # Create canonical map (Normalizing spaces/case)
    canonical_map = {}
    for name in all_antennas:
        canonical_map[canonicalize_antenna(name)] = name

    return {
        'metadata': metadata,
        'pco': all_pco,
        'pcv': all_pcv,
        'canonical_map': canonical_map,
    }


# ----------------------------
# RINEX broadcast reader
# ----------------------------

def read_broadcast_file(filename: str) -> dict:
    """
    Reads a RINEX v2/v3 broadcast navigation file and returns ephemeris data.
    """
    broadcast_data = {}
    print(f"\nParsing broadcast file: {filename}")
    if not os.path.exists(filename):
        print(f"ERROR: Broadcast file does not exist: {filename}")
        return {}
    try:
        for encoding in ['ascii', 'utf-8', 'latin-1']:
            try:
                with open(filename, 'r', encoding=encoding, errors='ignore') as f:
                    lines = f.readlines()
                break
            except UnicodeDecodeError:
                continue
        else:
            print("ERROR: Could not read file with any supported encoding")
            return {}

        header_end_index = 0
        for idx, line in enumerate(lines):
            if "END OF HEADER" in line:
                header_end_index = idx + 1
                break
        if header_end_index == 0:
            print("ERROR: 'END OF HEADER' not found in file")
            return {}

        # Detect RINEX version from header
        rinex_version = 2
        for line in lines[:header_end_index]:
            if 'RINEX VERSION' in line:
                try:
                    ver = float(line[:9].strip())
                    if ver >= 3.0:
                        rinex_version = 3
                except ValueError:
                    pass
                break

        lines_to_parse = lines[header_end_index:]
        records_parsed = 0
        i = 0
        while i < len(lines_to_parse):
            line = lines_to_parse[i]
            if not line.strip():
                i += 1
                continue
            try:
                if rinex_version >= 3:
                    # RINEX v3: First char is system (G/R/E/C/J/S), then 2-digit PRN
                    sys_char = line[0].upper()
                    if sys_char not in ('G', 'R', 'E', 'C', 'J', 'S'):
                        i += 1
                        continue
                    try:
                        prn = int(line[1:3].strip())
                    except ValueError:
                        i += 1
                        continue
                    system = sys_char
                    # Only process GPS for now (ephemeris propagator is GPS-only)
                    if system != 'G':
                        i += 8  # Skip non-GPS records (8 lines per record)
                        continue
                    if prn < 1 or prn > 32:
                        i += 1
                        continue
                else:
                    # RINEX v2: Plain PRN number, GPS only
                    try:
                        prn_str = line[0:3].strip()
                        prn = int(prn_str)
                    except ValueError:
                        i += 1
                        continue
                    system = 'G'
                    if prn < 1 or prn > 32:
                        i += 1
                        continue

                if i + 8 > len(lines_to_parse):
                    break
                record_lines = lines_to_parse[i: i + 8]
                
                try:
                    epoch, params = read_ephemeris_records(record_lines)
                    if system not in broadcast_data:
                        broadcast_data[system] = {}
                    if prn not in broadcast_data[system]:
                        broadcast_data[system][prn] = []
                    broadcast_data[system][prn].append({'epoch': epoch, **params})
                    records_parsed += 1
                except Exception:
                    pass
                i += 8
            except (ValueError, IndexError):
                i += 1
                continue

        print(f"✓ Successfully parsed {records_parsed} ephemeris records")
        return broadcast_data
    except Exception as e:
        print(f"Error in read_broadcast_file: {e}")
        return {}


# ----------------------------
# SP3 final orbit reader
# ----------------------------

def read_final_orbit_file(filename: str) -> dict:
    """
    Reads an SP3 precise orbit file and returns satellite positions.

    Args:
        filename: Path to the SP3 (.sp3) orbit file.

    Returns:
        dict with key 'orbits' mapping system chars to PRN dicts,
        each containing a list of (datetime, np.ndarray) position tuples.
    """
    orbit_data = {}
    current_epoch = None
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('*'):
                try:
                    year, mo, day, hr, mn, sec_full = int(line[3:7]), int(line[8:10]), int(line[11:13]), int(line[14:16]), int(line[17:19]), float(line[20:31])
                    sec, micro = int(sec_full), int(round((sec_full - int(sec_full)) * 1e6))
                    current_epoch = datetime(year, mo, day, hr, mn, sec, micro)
                except (ValueError, IndexError):
                    continue
            elif line.startswith('P') and current_epoch is not None:
                try:
                    system_char = line[1]
                    prn = int(line[2:4])
                    x = float(line[4:18]) * 1000.0
                    y = float(line[18:32]) * 1000.0
                    z = float(line[32:46]) * 1000.0
                    if abs(x) < 999999000.0:
                        if system_char not in orbit_data:
                            orbit_data[system_char] = {}
                        if prn not in orbit_data[system_char]:
                            orbit_data[system_char] = orbit_data.get(system_char, {})
                            orbit_data[system_char][prn] = []
                        orbit_data[system_char][prn].append((current_epoch, np.array([x, y, z], dtype=float)))
                except (ValueError, IndexError):
                    continue
                        
    # Build summary string
    summary_parts = []
    for sys_code, sats in orbit_data.items():
        sys_name = {'G':'GPS', 'R':'GLONASS', 'E':'Galileo', 'C':'BDS', 'J':'QZSS'}.get(sys_code, sys_code)
        summary_parts.append(f"{len(sats)} {sys_name}")
    
    summary_str = ", ".join(summary_parts) if summary_parts else "None"
    print(f"Parsed SP3 file: Found satellites: {summary_str}")
    return {'orbits': orbit_data}


# ----------------------------
# Config reader
# ----------------------------

def read_config_file(filepath: str) -> dict:
    """
    Parses a configuration file in key=value format.

    Args:
        filepath: Path to the configuration (.txt) file.

    Returns:
        dict of parsed configuration parameters with type-converted values
        (datetimes, floats, booleans, lists as appropriate).
    """
    try:
        # First, try German format (backward compatibility)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            first_lines = [f.readline() for _ in range(5)]
            content_sample = ''.join(first_lines)
            is_german = any(keyword in content_sample for keyword in [
                'ANTEX_Art', 'ANTEX-Ordner', 'StartTag', 'StartMonat', 'StartJahr',
                'Frequenz', 'Mappingfunktion', 'Gewichtung', 'Elevationswert'
            ])
            if is_german:
                print("German format configuration file detected — not supported in this version.")
                return {}
    except Exception:
        pass

    # Standard English/Internal format
    config = {}
    if not os.path.exists(filepath):
        return {}
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                stripped = line.strip()
                # Skip comments and empty lines
                if not stripped or stripped.startswith('#') or stripped.startswith(';'):
                    continue
                
                if '=' in stripped:
                    # Split only on first '=' to handle paths with '=' in them
                    key, value = stripped.split('=', 1)
                elif ':' in stripped:
                    key, value = stripped.split(':', 1)
                else:
                    continue

                key = key.strip()
                # Remove inline comments (naive approach)
                if '#' in value:
                    value = value.split('#')[0]
                value = value.strip()

                # Conversion logic
                if key in ['start_date', 'end_date', 'start_time', 'end_time']:
                    config[key] = value
                elif key in ['elevation_mask_active', 'azimuth_mask_active', 'gradient_active']:
                    config[key] = value.lower() == 'true'
                elif key in ['latitude', 'longitude', 'height', 'elevation_mask_angle', 'lat_min', 'lat_max', 'lon_min', 'lon_max', 'global_grid_step']:
                    config[key] = float(value) if value else 0.0
                elif key == 'sampling_rate_sec':
                    config[key] = int(float(value)) if value else 0
                elif key == 'interval_min':
                    # Handle both HH:MM format (from GUI save) and integer minutes
                    if ':' in value:
                        config[key] = value  # Keep as string, loader handles it
                    else:
                        config[key] = int(float(value)) if value else 0
                elif key == 'position_type':
                    config[key] = value
                elif key == 'signals':
                    # Handle both formats: "G01,G02" and "['G01', 'G02']"
                    cleaned = value.strip().strip("[]")
                    signals_list = [s.strip().strip("'\"") for s in cleaned.split(',') if s.strip().strip("'\"")]
                    config[key] = signals_list
                else:
                    config[key] = value

    except Exception as e:
        print(f"Error parsing config file: {e}")
        return {}

    # Post-process DateTimes if needed (Standard format handling)
    try:
        if 'start_date' in config and 'start_time' in config and ' ' not in str(config['start_time']):
             # If separate date/time strings
             try:
                 config['start_time'] = datetime.strptime(f"{config['start_date']} {config['start_time']}", '%Y-%m-%d %H:%M')
                 # del config['start_date'] # Keep it if needed elsewhere
             except: pass
             
        if 'end_date' in config and 'end_time' in config and ' ' not in str(config['end_time']):
             try:
                 config['end_time'] = datetime.strptime(f"{config['end_date']} {config['end_time']}", '%Y-%m-%d %H:%M')
                 # del config['end_date']
             except: pass
    except ValueError:
        pass

    # Post-process azimuth masks: parse 'start-end,elev;start-end,elev' format into list of tuples
    if 'azimuth_masks' in config and config['azimuth_masks']:
        raw_az = config['azimuth_masks'].strip()
        if raw_az:
            az_masks_list = []
            for part in raw_az.split(';'):
                part = part.strip()
                if not part:
                    continue
                try:
                    # New format: "start-end,elev"
                    if ',' in part:
                        range_str, elev_str = part.rsplit(',', 1)
                        az_from_str, az_to_str = range_str.split('-', 1)
                        az_masks_list.append((float(az_from_str), float(az_to_str), float(elev_str)))
                    else:
                        # Legacy format: "start-end" (default elevation 0)
                        az_from_str, az_to_str = part.split('-', 1)
                        az_masks_list.append((float(az_from_str), float(az_to_str), 0.0))
                except (ValueError, IndexError):
                    print(f"  Warning: Could not parse azimuth mask part: '{part}'")
            if az_masks_list:
                config['azimuth_mask_values'] = az_masks_list
                config['azimuth_mask_active'] = True

    return config


def resolve_antenna_name(parsed_antex: dict, requested: str) -> str:
    requested_canon = canonicalize_antenna(requested)
    canon_map = parsed_antex.get('canonical_map', {})
    if requested_canon in canon_map:
        return canon_map[requested_canon]
    if requested in parsed_antex.get('pco', {}) or requested in parsed_antex.get('pcv', {}):
        return requested
    raise ValueError(f"Antenna '{requested}' not found in ANTEX (after canonicalization).")


def get_antenna_data(parsed_antex: dict, antenna_name: str) -> dict:
    name = resolve_antenna_name(parsed_antex, antenna_name)
    pco = parsed_antex['pco'].get(name, {})
    pcv = parsed_antex['pcv'].get(name, {})
    return {'name': name, 'pco': pco, 'pcv': pcv}


# ----------------------------
# NEW: Result Export (Phase 2)
# ----------------------------

def save_results_to_txt(output_dir: str, filename_base: str, param_names: list, results: np.ndarray, config: dict = None):
    """
    Saves the analysis results to a formatted text file.

    Args:
        output_dir: Directory to save the result file.
        filename_base: Base name for the output file (antenna/analysis identifier).
        param_names: List of parameter names (e.g., ['North', 'East', 'Up', ...]).
        results: numpy array of result values corresponding to param_names.
        config: Optional dict of configuration settings to save alongside results.

    Returns:
        Path to the saved result file, or None if an error occurred.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sanitize filename to remove invalid characters (Windows: / \ : * ? " < > |)
        safe_filename_base = filename_base.replace('/', '_').replace('\\', '_').replace(':', '_')
        safe_filename_base = safe_filename_base.replace('*', '_').replace('?', '_').replace('"', '_')
        safe_filename_base = safe_filename_base.replace('<', '_').replace('>', '_').replace('|', '_')
        
        filename = f"{safe_filename_base}_results_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)

        # Save config file first (to configs/ folder) so we can reference it
        config_filename = ""
        if config:
            config_filename = f"{safe_filename_base}_config_{timestamp}.txt"
            # Save config to configs/ folder (sibling of results/)
            project_root = os.path.dirname(output_dir)  # results/ -> project root
            configs_dir = os.path.join(project_root, 'configs')
            os.makedirs(configs_dir, exist_ok=True)
            config_filepath = os.path.join(configs_dir, config_filename)
            with open(config_filepath, 'w') as cf:
                cf.write("# PCC-Explorer Configuration File\n")
                cf.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                for key, value in config.items():
                    # Skip complex objects (numpy arrays, None, etc.)
                    if key == 'position' and not isinstance(value, (str, int, float)):
                        continue  # Skip numpy array position
                    if value is None:
                        continue
                    # Handle datetime objects → format as string
                    if isinstance(value, datetime):
                        if 'date' in key:
                            cf.write(f"{key}={value.strftime('%Y-%m-%d')}\n")
                        else:
                            cf.write(f"{key}={value.strftime('%Y-%m-%d %H:%M')}\n")
                    # Handle lists → comma-separated
                    elif isinstance(value, list):
                        cf.write(f"{key}={','.join(str(v) for v in value)}\n")
                    elif isinstance(value, (str, int, float, bool)):
                        cf.write(f"{key}={value}\n")
            print(f"Config saved to: {config_filepath}")

        # Write result file
        with open(filepath, 'w') as f:
            f.write("# ==================================================\n")
            f.write(f"# PCC-Explorer Analysis Result\n")
            f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if config_filename:
                f.write(f"# Configuration file:  {config_filename}\n")
            f.write("# ==================================================\n")
            
            f.write(f"# {'Parameter':<20}   {'Impact [mm]':<15}\n")
            
            for name, value in zip(param_names, results):
                f.write(f"{name:<20}   {value:>10.4f}\n")
                
            f.write("# \n")
            f.write("# ==================================================\n")

        print(f"Results saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving text file: {e}")
        return None


def save_timeline_results_to_txt(output_dir: str, dates: list, results_matrix, param_names: list, antenna_name: str = None, config_info: dict = None) -> str:
    """
    Saves the consolidated timeline analysis results to a single formatted text file.
    Uses the same header format as single analysis results for consistency.
    
    Args:
        output_dir: Directory to save the file
        dates: List of date or datetime objects for each row
        results_matrix: 2D numpy array with shape (num_periods, num_params)
        param_names: List of parameter names (e.g., ['North', 'East', 'Up', 'Clock'])
        antenna_name: Optional antenna name+serial to include in filename
        config_info: Optional dict with config details
    
    Returns:
        Path to the saved file, or None if error occurred
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build filename with antenna name if provided (matching single analysis format)
        if antenna_name:
            safe_name = antenna_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            safe_name = safe_name.replace(':', '_').replace('*', '_').replace('?', '_')
            safe_name = safe_name.replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
            filename = f"TimeSeries_{safe_name}_{timestamp}.txt"
        else:
            filename = f"TimeSeries_results_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        # Detect if this is sub-daily analysis
        is_subdaily = hasattr(dates[0], 'hour') if dates else False

        # Build config filename reference
        config_filename = filename.replace("TimeSeries_", "TimeSeries_config_").replace("_results_", "_config_")

        with open(filepath, 'w') as f:
            f.write("# ==================================================\n")
            f.write("# PCC-Explorer Time Series Analysis Result\n")
            f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if is_subdaily:
                f.write(f"# Period: {dates[0].strftime('%Y-%m-%d %H:%M')} to {dates[-1].strftime('%Y-%m-%d %H:%M')} ({len(dates)} intervals)\n")
            else:
                f.write(f"# Period: {dates[0]} to {dates[-1]} ({len(dates)} days)\n")
            f.write(f"# Configuration file:  TimeSeries_{safe_name if antenna_name else 'results'}_config_{timestamp}.txt\n")
            f.write("# ==================================================\n")
            
            # Column header
            header = f"{'# Date':<14} {'Time':<8}"
            for name in param_names:
                header += f"   {name:>12}"
            f.write(header + "\n")
            
            # Data rows
            for i, dt in enumerate(dates):
                if is_subdaily:
                    date_str = dt.strftime("%Y-%m-%d")
                    time_str = dt.strftime("%H:%M")
                else:
                    date_str = dt.strftime("%Y-%m-%d") if hasattr(dt, 'strftime') else str(dt)
                    time_str = "00:00"
                
                row = f"{date_str:<14} {time_str:<8}"
                for j, name in enumerate(param_names):
                    value = results_matrix[i, j] if i < len(results_matrix) else float('nan')
                    if np.isnan(value):
                        row += f"   {'N/A':>12}"
                    else:
                        row += f"   {value:>12.4f}"
                f.write(row + "\n")
            
            f.write("# \n")
            f.write("# ==================================================\n")

        print(f"✓ Timeline results saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving timeline results: {e}")
        traceback.print_exc()
        return None