# local_orbits.py
"""
Local orbit files: find, read and evaluate orbit products without a network.

Used when a program runs in "local orbit files" mode, for example inside a
protected network that cannot reach the IGS data servers. The user selects a
folder that holds daily orbit files; this module finds the right file for a
day and reads it.

Accepted files:
- SP3 precise, rapid, ultra-rapid or predicted orbits (SP3-c and SP3-d);
- RINEX navigation files, versions 2, 3 and 4, including mixed files.
Both may be gzip-compressed (``.gz``). Unix-compressed files (``.Z``) are
reported as unsupported, because reading them needs an external program.

Search order for one day:
1. File names. The folder is listed once and every file whose standard IGS
   name (long or short form) contains the day becomes a candidate. Only these
   candidates are opened, in order of preference, and only their header (plus
   the first navigation records) is read to confirm type, systems and day.
2. File contents, only when step 1 finds no usable file. Every other file is
   opened and its header is read; the file name does not matter here.

Broadcast ephemerides are read for GPS (LNAV) and Galileo (I/NAV and F/NAV).
Other systems are skipped record by record, using each record's real length.

The module depends on the standard library and numpy only, so that every
program can carry an identical copy of it.
"""

from __future__ import annotations

import gzip
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

# Gravitational constants of the broadcast models [m^3/s^2]. GPS uses the
# WGS84 value, Galileo the GTRF value; the Earth rotation rate is the same.
GM = {'G': 3.986005e14, 'E': 3.986004418e14}
OMEGA_E = 7.2921151467e-5           # [rad/s]

#: Systems for which broadcast ephemerides can be evaluated.
BROADCAST_SYSTEMS = frozenset({'G', 'E'})

GPS_EPOCH = datetime(1980, 1, 6)
SECONDS_PER_WEEK = 604800.0

# Ephemeris selection window: the newest record not older than this is used,
# else the nearest one within the same distance (same rule as PCC-Explorer).
EPHEMERIS_MAX_AGE_S = 7200.0

_UNSUPPORTED_SUFFIXES = ('.z', '.zip', '.bz2', '.7z', '.rar', '.tar')
_MAX_SCAN_BYTES = 300 * 1024 * 1024  # content scan skips larger files


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def open_text(path: str):
    """Open a text file for reading; ``.gz`` files are decompressed on the fly."""
    if path.lower().endswith('.gz'):
        return gzip.open(path, 'rt', encoding='ascii', errors='ignore')
    return open(path, 'r', encoding='ascii', errors='ignore')


def gps_week_and_sow(t: datetime) -> Tuple[int, float]:
    """GPS week and seconds of week for a time given in GPS time."""
    seconds = (t - GPS_EPOCH).total_seconds()
    week = int(seconds // SECONDS_PER_WEEK)
    return week, seconds - week * SECONDS_PER_WEEK


def _rinex_float(text: str) -> float:
    """RINEX number with 'D' or 'E' exponent; an empty field reads as 0.0."""
    text = text.strip().replace('D', 'E').replace('d', 'E')
    if not text:
        return 0.0
    return float(text)


def _strip_gz(name: str) -> str:
    return name[:-3] if name.lower().endswith('.gz') else name


def is_unsupported_compression(name: str) -> bool:
    """True for archive formats this module cannot open (``.Z``, ``.zip`` ...)."""
    return name.lower().endswith(_UNSUPPORTED_SUFFIXES)


# ---------------------------------------------------------------------------
# File descriptions
# ---------------------------------------------------------------------------

@dataclass
class OrbitFile:
    """One orbit file, as far as its name and header describe it."""

    path: str
    kind: str                       # 'sp3' or 'nav'
    rank: int = 9                   # lower is better within one kind
    systems: Set[str] = field(default_factory=set)
    mixed: bool = False             # navigation file declared as mixed ('M')
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    interval: float = 300.0         # SP3 epoch spacing [s]
    day: Optional[date] = None      # navigation files: day of most records
    version: str = ''
    found_by: str = 'name'          # 'name' or 'content'

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    def covers(self, when: datetime, until: Optional[datetime] = None) -> bool:
        """
        True if the file holds orbits from ``when`` to ``until``. Without
        ``until`` the rest of the day of ``when`` is required, so a daily SP3
        file of the previous day (which ends at 00:00) does not count.
        """
        if self.kind == 'sp3':
            if self.start is None or self.end is None:
                return False
            day_end = datetime(when.year, when.month, when.day, 23, 59, 59)
            need_end = min(until, day_end) if until is not None else day_end
            # One SP3 step of tolerance at the end: a daily file usually ends
            # one step before midnight.
            return (self.start - timedelta(seconds=1) <= when
                    and need_end <= self.end + timedelta(seconds=self.interval))
        return self.day == when.date()

    def can_serve(self, systems: Set[str]) -> bool:
        """True if the file can deliver orbits for every system in ``systems``."""
        if self.kind == 'sp3':
            return set(systems) <= self.systems
        if not set(systems) <= BROADCAST_SYSTEMS:
            return False
        return self.mixed or set(systems) <= self.systems


# Long IGS names. Products carry a 10-character name (centre, version, project,
# type), e.g. COD0MGXFIN_20220070000_01D_05M_ORB.SP3; data files a 9-character
# station or source name plus a source letter, e.g.
# BRDC00IGS_R_20262420000_01D_MN.rnx
_LONG_NAME = re.compile(
    r'^(?P<id>[A-Z0-9]{10}|[A-Z0-9]{9}_[A-Z])_(?P<year>\d{4})(?P<doy>\d{3})\d{4}_\d{2}[A-Z]_'
    r'(?:\d{2}[A-Z]_)?(?P<content>[A-Z]{2,3})\.(?P<fmt>SP3|RNX)$', re.IGNORECASE)
# Short navigation names, e.g. brdc0070.22n, brdm2420.26p, abcd007a.22n
_SHORT_NAV = re.compile(
    r'^[A-Z0-9]{4}(?P<doy>\d{3})[0-9A-X]\.(?P<yy>\d{2})(?P<type>[NLP])$', re.IGNORECASE)
# Short SP3 names, e.g. igs21922.sp3, cod21922.eph, igu21922_12.sp3
_SHORT_SP3 = re.compile(
    r'^(?P<ac>[A-Z]{3})(?P<week>\d{4})(?P<dow>\d)(?:_\d{2})?\.(?:SP3|EPH|SP3C|SP3D)$',
    re.IGNORECASE)

_SP3_TYPE_RANK = {'FIN': 0, 'RAP': 1, 'ULT': 2, 'PRD': 3}
_SP3_SHORT_RANK = {'igs': 0, 'cod': 0, 'com': 0, 'gfz': 0, 'esa': 0, 'jpl': 0,
                   'igr': 1, 'gbm': 1, 'igu': 2}
_NAV_CONTENT_SYSTEMS = {'MN': None, 'GN': {'G'}, 'EN': {'E'}}


def classify_by_name(name: str, day: date) -> Optional[OrbitFile]:
    """
    Describe a file from its name alone, if the name is a standard IGS orbit
    name for ``day``. Returns None for any other name. Nothing is opened.
    """
    base = _strip_gz(name)
    week, sow = gps_week_and_sow(datetime(day.year, day.month, day.day))
    dow = int(sow // 86400)

    m = _LONG_NAME.match(base)
    if m:
        if int(m['year']) != day.year or int(m['doy']) != day.timetuple().tm_yday:
            return None
        content = m['content'].upper()
        if m['fmt'].upper() == 'SP3' and content == 'ORB':
            rank = _SP3_TYPE_RANK.get(m['id'][7:10].upper(), 4)
            return OrbitFile(path=name, kind='sp3', rank=rank)
        if m['fmt'].upper() == 'RNX' and content in _NAV_CONTENT_SYSTEMS:
            systems = _NAV_CONTENT_SYSTEMS[content]
            return OrbitFile(path=name, kind='nav', rank=0 if systems is None else 1,
                             systems=set(systems or ()), mixed=systems is None)
        return None

    m = _SHORT_NAV.match(base)
    if m:
        if int(m['doy']) != day.timetuple().tm_yday or int(m['yy']) != day.year % 100:
            return None
        nav_type = m['type'].lower()
        return OrbitFile(path=name, kind='nav', rank=0 if nav_type == 'p' else 1,
                         systems={'n': {'G'}, 'l': {'E'}, 'p': set()}[nav_type],
                         mixed=nav_type == 'p')

    m = _SHORT_SP3.match(base)
    if m:
        if int(m['week']) != week or int(m['dow']) != dow:
            return None
        return OrbitFile(path=name, kind='sp3',
                         rank=_SP3_SHORT_RANK.get(m['ac'].lower(), 4))
    return None


# ---------------------------------------------------------------------------
# Header readers (cheap: a few dozen lines per file)
# ---------------------------------------------------------------------------

_NAV_RECORD_V3 = re.compile(r'^([GRECJIS])(\d{2}) (\d{4}) (\d{2}) (\d{2}) (\d{2}) (\d{2}) (\d{2})')
_NAV_RECORD_V2 = re.compile(r'^([ \d]\d) ([ \d]\d) ([ \d]\d) ([ \d]\d) ([ \d]\d) ([ \d]\d) ([ \d]\d\.\d)')


def _read_sp3_header(path: str, info: OrbitFile) -> bool:
    """Fill start, end and systems from an SP3 header. False if not SP3."""
    with open_text(path) as f:
        first = f.readline()
        if len(first) < 40 or first[0] != '#' or first[1] not in 'abcd':
            return False
        try:
            start = datetime(int(first[3:7]), int(first[8:10]), int(first[11:13]),
                             int(first[14:16]), int(first[17:19]), int(float(first[20:31])))
            n_epochs = int(first[32:39])
            second = f.readline()
            interval = float(second[24:38])
        except (ValueError, IndexError):
            return False
        systems: Set[str] = set()
        for _ in range(400):
            line = f.readline()
            if not line or line.startswith('*'):
                break
            if line.startswith('+ '):
                for k in range(9, 60, 3):
                    sat = line[k:k + 3]
                    if len(sat) == 3 and sat[0].isalpha() and sat[1:].strip().isdigit() \
                            and int(sat[1:]) > 0:
                        systems.add(sat[0])
    info.kind = 'sp3'
    info.version = first[1]
    info.start = start
    info.end = start + timedelta(seconds=interval * max(n_epochs - 1, 0))
    info.interval = interval if interval > 0 else 300.0
    info.systems = systems
    return True


def _read_nav_header(path: str, info: OrbitFile, sample_records: int = 400) -> bool:
    """
    Fill version, systems and day from a RINEX navigation header plus the
    first navigation records. False if the file is not a navigation file.
    """
    with open_text(path) as f:
        first = f.readline()
        if 'RINEX VERSION / TYPE' not in first:
            return False
        try:
            version = float(first[:9])
        except ValueError:
            return False
        file_type = first[20:21].upper()
        if file_type != 'N':
            return False
        sys_char = first[40:41].upper() if version >= 3 else 'G'
        for _ in range(1000):
            line = f.readline()
            if not line:
                return False
            if 'END OF HEADER' in line:
                break
        else:
            return False

        dates: Counter = Counter()
        seen: Set[str] = set()
        pattern = _NAV_RECORD_V3 if version >= 3 else _NAV_RECORD_V2
        while len(dates) < 1 or sum(dates.values()) < sample_records:
            line = f.readline()
            if not line:
                break
            m = pattern.match(line)
            if not m:
                continue
            try:
                if version >= 3:
                    seen.add(m.group(1))
                    d = date(int(m.group(3)), int(m.group(4)), int(m.group(5)))
                else:
                    yy = int(m.group(2))
                    d = date(yy + (1900 if yy >= 80 else 2000), int(m.group(3)), int(m.group(4)))
            except ValueError:
                continue
            dates[d] += 1
    if not dates:
        return False
    info.kind = 'nav'
    info.version = f"{version:.2f}"
    info.day = dates.most_common(1)[0][0]
    info.mixed = sys_char == 'M'
    if sys_char == 'M':
        info.systems = seen
    elif sys_char in ('G', 'E', ' ', ''):
        info.systems = {'G'} if sys_char in ('G', ' ', '') else {'E'}
    else:
        info.systems = {sys_char}
    return True


def read_header(path: str, found_by: str = 'content',
                hint: Optional[OrbitFile] = None) -> Optional[OrbitFile]:
    """
    Identify an orbit file from its content (header only). Returns None if the
    file is neither SP3 nor a RINEX navigation file, or cannot be read.
    """
    info = OrbitFile(path=path, kind=hint.kind if hint else '', found_by=found_by,
                     rank=hint.rank if hint else 9)
    try:
        if _read_sp3_header(path, info):
            if hint is None:
                info.rank = 4
            return info
        if _read_nav_header(path, info):
            if hint is None:
                info.rank = 0 if info.mixed else 1
            return info
    except (OSError, EOFError, UnicodeError, gzip.BadGzipFile):
        return None
    return None


# ---------------------------------------------------------------------------
# Folder search
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Outcome of one folder search, with everything needed for a clear message."""

    match: Optional[OrbitFile]
    folders: List[str]
    opened: List[str] = field(default_factory=list)
    rejected: List[Tuple[str, str]] = field(default_factory=list)
    unsupported: List[str] = field(default_factory=list)
    content_scan: bool = False

    def failure_message(self, when: datetime, systems: Iterable[str], prefer: str) -> str:
        """Explain to the user why no file was found and what to do."""
        day = when.date()
        systems = sorted(systems)
        lines = [
            f"No usable orbit file for {day} (day of year {day.timetuple().tm_yday:03d}) "
            f"in the local orbit folder.",
            "",
            "Searched: " + "; ".join(self.folders),
            f"Needed: orbits for {', '.join(systems) or 'the selected signals'} "
            f"({'precise SP3 preferred' if prefer == 'precise' else 'navigation file preferred'}).",
        ]
        if not set(systems) <= BROADCAST_SYSTEMS:
            lines.append("Broadcast navigation files can serve GPS and Galileo only; "
                         "other systems need an SP3 file.")
        if self.rejected:
            lines.append("")
            lines.append("Files found but not usable:")
            for path, reason in self.rejected[:8]:
                lines.append(f"  {os.path.basename(path)}: {reason}")
            if len(self.rejected) > 8:
                lines.append(f"  ... and {len(self.rejected) - 8} more")
        if self.unsupported:
            lines.append("")
            lines.append("Compressed files that cannot be read directly "
                         "(decompress them first, or use .gz):")
            for path in self.unsupported[:5]:
                lines.append(f"  {os.path.basename(path)}")
        yy = day.year % 100
        doy = day.timetuple().tm_yday
        lines += [
            "",
            "Examples of file names that are found at once:",
            f"  COD0MGXFIN_{day.year}{doy:03d}0000_01D_05M_ORB.SP3",
            f"  BRDC00IGS_R_{day.year}{doy:03d}0000_01D_MN.rnx",
            f"  brdc{doy:03d}0.{yy:02d}n   brdm{doy:03d}0.{yy:02d}p",
            "Files with other names are also found, by their content.",
        ]
        return "\n".join(lines)


def search_folders(folder: str, day: date) -> List[str]:
    """
    The selected folder plus the usual archive sub-folders for the day
    (``YYYY``, ``YYYY/DDD``, ``DDD``), where they exist.
    """
    doy = f"{day.timetuple().tm_yday:03d}"
    candidates = [folder,
                  os.path.join(folder, str(day.year)),
                  os.path.join(folder, str(day.year), doy),
                  os.path.join(folder, doy)]
    return [p for p in candidates if os.path.isdir(p)]


def _list_files(folders: List[str]) -> List[str]:
    files = []
    for d in folders:
        try:
            for entry in os.scandir(d):
                if entry.is_file():
                    files.append(entry.path)
        except OSError:
            continue
    return files


def find_orbit_file(folder: str, when: datetime, systems: Iterable[str],
                    prefer: str = 'precise',
                    until: Optional[datetime] = None) -> SearchResult:
    """
    Find the best local orbit file for the period from ``when`` to ``until``.

    Args:
        folder: Folder selected by the user.
        when: First epoch the orbits are needed for (GPS time).
        systems: Systems the run needs, e.g. {'G'} or {'G', 'E'}.
        prefer: 'precise' tries SP3 files first, 'broadcast' tries navigation
            files first. The other kind is used when the preferred one is missing.
        until: Last epoch needed; default the end of the day of ``when``.
            Only the day of ``when`` is searched.

    Returns:
        SearchResult whose ``match`` is the chosen file, or None.
    """
    systems = set(systems)
    day = when.date()
    folders = search_folders(folder, day)
    result = SearchResult(match=None, folders=folders or [folder])
    if not folders:
        result.rejected.append((folder, "folder does not exist"))
        return result

    kind_order = ['sp3', 'nav'] if prefer == 'precise' else ['nav', 'sp3']

    def sort_key(info: OrbitFile):
        return (kind_order.index(info.kind), info.rank, info.name)

    def judge(info: OrbitFile) -> Optional[str]:
        if not info.covers(when, until):
            if info.kind == 'sp3' and info.start is not None:
                return (f"covers {info.start:%Y-%m-%d %H:%M} to {info.end:%Y-%m-%d %H:%M}, "
                        f"not the run on {day}")
            return f"holds data for {info.day}, not {day}"
        if not info.can_serve(systems):
            if info.kind == 'nav' and not systems <= BROADCAST_SYSTEMS:
                return "navigation file; the run needs systems other than GPS and Galileo"
            return f"contains {', '.join(sorted(info.systems)) or 'no known system'}, " \
                   f"the run needs {', '.join(sorted(systems))}"
        return None

    files = _list_files(folders)
    named: List[OrbitFile] = []
    others: List[str] = []
    for path in files:
        name = os.path.basename(path)
        if is_unsupported_compression(name):
            if classify_by_name(name[:name.rfind('.')], day) is not None:
                result.unsupported.append(path)
            continue
        hint = classify_by_name(name, day)
        if hint is not None:
            hint.path = path
            named.append(hint)
        else:
            others.append(path)

    # Step 1: standard names, best first; stop at the first usable file.
    for hint in sorted(named, key=sort_key):
        result.opened.append(hint.path)
        info = read_header(hint.path, found_by='name', hint=hint)
        if info is None:
            result.rejected.append((hint.path, "not a readable SP3 or RINEX navigation file"))
            continue
        reason = judge(info)
        if reason:
            result.rejected.append((hint.path, reason))
            continue
        result.match = info
        return result

    # Step 2: content scan of every other file (header only).
    result.content_scan = True
    usable: List[OrbitFile] = []
    for path in sorted(others):
        try:
            if os.path.getsize(path) > _MAX_SCAN_BYTES:
                continue
        except OSError:
            continue
        result.opened.append(path)
        info = read_header(path, found_by='content')
        if info is None:
            continue
        reason = judge(info)
        if reason:
            result.rejected.append((path, reason))
        else:
            usable.append(info)
    if usable:
        result.match = sorted(usable, key=sort_key)[0]
    return result


# ---------------------------------------------------------------------------
# Navigation file reader
# ---------------------------------------------------------------------------

_ORBIT_FIELDS = (
    ('IODE', 'Crs', 'Delta_n', 'M0'),
    ('Cuc', 'e', 'Cus', 'sqrt_A'),
    ('Toe', 'Cic', 'Omega0', 'Cis'),
    ('i0', 'Crc', 'omega', 'Omega_dot'),
    ('i_dot', 'Codes_on_L2', 'GPS_Week', 'L2_P_Data_flag'),
    ('SV_accuracy', 'SV_health', 'TGD', 'IODC'),
    ('Transmission_time',),
)

# Number of lines of one RINEX 3 navigation record, per system.
_RECORD_LINES_V3 = {'G': 8, 'E': 8, 'C': 8, 'J': 8, 'I': 8, 'R': 4, 'S': 4}


def _parse_record(lines: List[str], offset: int, epoch: datetime) -> dict:
    """Parse the 8 lines of a Keplerian navigation record into named parameters."""
    first = lines[0]
    params = {
        'epoch': epoch,
        'SV_clock_bias': _rinex_float(first[offset + 19:offset + 38]),
        'SV_clock_drift': _rinex_float(first[offset + 38:offset + 57]),
        'SV_clock_drift_rate': _rinex_float(first[offset + 57:offset + 76]),
    }
    for line, names in zip(lines[1:8], _ORBIT_FIELDS):
        for k, key in enumerate(names):
            params[key] = _rinex_float(line[offset + 19 * k:offset + 19 * (k + 1)])
    return params


def _epoch_v3(line: str) -> datetime:
    return datetime(int(line[4:8]), int(line[9:11]), int(line[12:14]),
                    int(line[15:17]), int(line[18:20]), int(line[21:23]))


def _epoch_v2(line: str) -> datetime:
    yy = int(line[3:5])
    second = float(line[17:22])
    return datetime(yy + (1900 if yy >= 80 else 2000), int(line[6:8]), int(line[9:11]),
                    int(line[12:14]), int(line[15:17]), int(second))


def _accept(system: str, prn: int) -> bool:
    if system == 'G':
        return 1 <= prn <= 32
    if system == 'E':
        return 1 <= prn <= 36
    return False


def read_nav_file(path: str, systems: Iterable[str] = BROADCAST_SYSTEMS) -> Dict[str, Dict[int, List[dict]]]:
    """
    Read GPS and Galileo broadcast ephemerides from a RINEX 2, 3 or 4
    navigation file (plain or ``.gz``).

    Args:
        path: Navigation file.
        systems: Systems to keep (subset of 'G', 'E').

    Returns:
        {system: {prn: [record, ...]}}; each record is a dict with 'epoch',
        'system' and the Keplerian parameters ('Toe', 'sqrt_A', 'e', ...).
        Empty dict if the file cannot be read.
    """
    keep = set(systems) & BROADCAST_SYSTEMS
    data: Dict[str, Dict[int, List[dict]]] = {}
    with open_text(path) as f:
        lines = f.readlines()

    header_end = next((i + 1 for i, line in enumerate(lines) if 'END OF HEADER' in line), 0)
    if header_end == 0 or 'RINEX VERSION / TYPE' not in lines[0]:
        return {}
    try:
        version = float(lines[0][:9])
    except ValueError:
        return {}

    def store(system: str, prn: int, record_lines: List[str], offset: int, epoch: datetime):
        try:
            rec = _parse_record(record_lines, offset, epoch)
        except (ValueError, IndexError):
            return
        rec['system'] = system
        data.setdefault(system, {}).setdefault(prn, []).append(rec)

    body = lines[header_end:]
    n = len(body)
    i = 0
    if version >= 4:
        # Records start with a '>' line naming the record type; collect the lines
        # up to the next '>' so records of any length are skipped correctly.
        while i < n:
            line = body[i]
            if not line.startswith('>'):
                i += 1
                continue
            parts = line[1:].split()
            j = i + 1
            while j < n and not body[j].startswith('>'):
                j += 1
            if len(parts) >= 3 and parts[0] == 'EPH':
                sat, msg = parts[1], parts[2].upper()
                system = sat[0]
                ok_msg = (system == 'G' and msg == 'LNAV') or \
                         (system == 'E' and msg in ('INAV', 'FNAV'))
                if system in keep and ok_msg and j - (i + 1) >= 8:
                    try:
                        prn = int(sat[1:3])
                        epoch = _epoch_v3(body[i + 1])
                    except (ValueError, IndexError):
                        prn, epoch = 0, None
                    if epoch is not None and _accept(system, prn):
                        store(system, prn, body[i + 1:i + 9], 4, epoch)
            i = j
    elif version >= 3:
        while i < n:
            line = body[i]
            m = _NAV_RECORD_V3.match(line)
            if not m:
                i += 1
                continue
            system, prn = m.group(1), int(m.group(2))
            length = _RECORD_LINES_V3.get(system, 8)
            if system in keep and _accept(system, prn) and i + 8 <= n:
                try:
                    epoch = _epoch_v3(line)
                except ValueError:
                    i += 1
                    continue
                store(system, prn, body[i:i + 8], 4, epoch)
            # Advance by the record length, but never past the next record
            # header: a record that is shorter than expected must not hide
            # the one after it.
            j = i + 1
            while j < min(i + length, n) and not _NAV_RECORD_V3.match(body[j]):
                j += 1
            i = j
    else:
        if 'G' not in keep:
            return {}
        while i < n:
            line = body[i]
            if not _NAV_RECORD_V2.match(line) or i + 8 > n:
                i += 1
                continue
            try:
                prn = int(line[0:2])
                epoch = _epoch_v2(line)
            except ValueError:
                i += 1
                continue
            if _accept('G', prn):
                store('G', prn, body[i:i + 8], 3, epoch)
            i += 8
    return data


# ---------------------------------------------------------------------------
# Broadcast orbit evaluation
# ---------------------------------------------------------------------------

def satellite_position(eph: dict, t_sow: float, system: Optional[str] = None) -> np.ndarray:
    """
    ECEF position [m] of a satellite from a Keplerian broadcast record
    (GPS LNAV or Galileo I/NAV, F/NAV) at seconds of week ``t_sow``.
    """
    system = system or eph.get('system', 'G')
    gm = GM.get(system, GM['G'])
    t_k = t_sow - eph['Toe']
    if t_k > 302400:
        t_k -= SECONDS_PER_WEEK
    if t_k < -302400:
        t_k += SECONDS_PER_WEEK

    a = eph['sqrt_A'] ** 2
    n = np.sqrt(gm / a ** 3) + eph['Delta_n']
    m_k = eph['M0'] + n * t_k
    e = eph['e']
    e_k = m_k
    for _ in range(20):
        e_new = m_k + e * np.sin(e_k)
        if abs(e_new - e_k) < 1e-13:
            e_k = e_new
            break
        e_k = e_new
    v_k = np.arctan2(np.sqrt(1 - e ** 2) * np.sin(e_k), np.cos(e_k) - e)
    phi = v_k + eph['omega']
    s2, c2 = np.sin(2 * phi), np.cos(2 * phi)
    u_k = phi + eph['Cus'] * s2 + eph['Cuc'] * c2
    r_k = a * (1 - e * np.cos(e_k)) + eph['Crs'] * s2 + eph['Crc'] * c2
    i_k = eph['i0'] + eph['Cis'] * s2 + eph['Cic'] * c2 + eph['i_dot'] * t_k
    x_p, y_p = r_k * np.cos(u_k), r_k * np.sin(u_k)
    omega_k = eph['Omega0'] + (eph['Omega_dot'] - OMEGA_E) * t_k - OMEGA_E * eph['Toe']
    return np.array([
        x_p * np.cos(omega_k) - y_p * np.cos(i_k) * np.sin(omega_k),
        x_p * np.sin(omega_k) + y_p * np.cos(i_k) * np.cos(omega_k),
        y_p * np.sin(i_k),
    ])


def _toe_time(eph: dict) -> datetime:
    """Absolute reference time of a record (week number + Toe, GPS time scale)."""
    return GPS_EPOCH + timedelta(seconds=eph['GPS_Week'] * SECONDS_PER_WEEK + eph['Toe'])


def select_ephemeris(records: List[dict], when: datetime) -> Optional[dict]:
    """
    The record to use at ``when``: the newest one not older than two hours,
    else the nearest one within two hours. None if no record qualifies.
    """
    best_before, best_before_dt = None, None
    best_near, best_near_dt = None, None
    for rec in records:
        dt = (when - _toe_time(rec)).total_seconds()
        if 0 <= dt <= EPHEMERIS_MAX_AGE_S and (best_before_dt is None or dt < best_before_dt):
            best_before, best_before_dt = rec, dt
        if abs(dt) <= EPHEMERIS_MAX_AGE_S and (best_near_dt is None or abs(dt) < best_near_dt):
            best_near, best_near_dt = rec, abs(dt)
    return best_before or best_near


def read_sp3_file(path: str) -> dict:
    """
    Satellite positions from an SP3 file (plain or ``.gz``), as
    {'orbits': {system: {prn: [(datetime, xyz_m), ...]}}}.
    """
    orbits: Dict[str, Dict[int, list]] = {}
    epoch = None
    with open_text(path) as f:
        for line in f:
            if line.startswith('*'):
                try:
                    sec = float(line[20:31])
                    epoch = datetime(int(line[3:7]), int(line[8:10]), int(line[11:13]),
                                     int(line[14:16]), int(line[17:19]), int(sec),
                                     int(round((sec - int(sec)) * 1e6)) % 1000000)
                except (ValueError, IndexError):
                    epoch = None
            elif line.startswith('P') and epoch is not None:
                try:
                    xyz = np.array([float(line[4:18]), float(line[18:32]),
                                    float(line[32:46])]) * 1000.0
                    prn = int(line[2:4])
                except (ValueError, IndexError):
                    continue
                # 0.000000 or 999999.999999 mark a missing position.
                if 0.0 < abs(xyz[0]) < 999999000.0:
                    orbits.setdefault(line[1], {}).setdefault(prn, []).append((epoch, xyz))
    return {'orbits': orbits}


def load_orbit_table(info: OrbitFile, day: date, step_s: float = 300.0) -> dict:
    """
    Orbit table for one day from a file found by ``find_orbit_file``, in the
    SP3 structure {'orbits': {system: {prn: [(datetime, xyz_m), ...]}}}.
    A navigation file is evaluated every ``step_s`` seconds, from 10 minutes
    before to 10 minutes after the day, like the nodes of a daily SP3 file.
    """
    if info.kind == 'sp3':
        return read_sp3_file(info.path)
    start = datetime(day.year, day.month, day.day) - timedelta(minutes=10)
    end = start + timedelta(days=1, minutes=20)
    return broadcast_to_orbit_table(read_nav_file(info.path), start, end, step_s)


def broadcast_to_orbit_table(nav: Dict[str, Dict[int, List[dict]]], start: datetime,
                             end: datetime, step_s: float = 300.0) -> dict:
    """
    Evaluate broadcast ephemerides on a regular time grid, in the structure an
    SP3 reader returns: {'orbits': {system: {prn: [(datetime, xyz), ...]}}}.

    This lets programs that interpolate SP3 positions use a navigation file
    without any other change.
    """
    orbits: Dict[str, Dict[int, list]] = {}
    n_steps = int((end - start).total_seconds() // step_s) + 1
    epochs = [start + timedelta(seconds=k * step_s) for k in range(n_steps)]
    for system, sats in nav.items():
        for prn, records in sats.items():
            track = []
            for t in epochs:
                rec = select_ephemeris(records, t)
                if rec is None:
                    continue
                _, sow = gps_week_and_sow(t)
                pos = satellite_position(rec, sow, system)
                if np.all(np.isfinite(pos)):
                    track.append((t, pos))
            if len(track) >= 2:
                orbits.setdefault(system, {})[prn] = track
    return {'orbits': orbits}
