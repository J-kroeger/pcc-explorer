# src/obstruction_mask.py
"""
Obstruction-mask support for PCC-Explorer.

Reads the horizon masks produced by RINEX-Masker and reports, for any azimuth,
the lowest elevation that is still unobstructed. Directions below that line are
dropped from the analysis, so a PCC impact can be evaluated for the sky a
station actually sees rather than an ideal open horizon.

Two input formats are accepted, both written by RINEX-Masker:

  * ``pcc-mask-v1`` JSON  - "Export Mask (JSON)"
  * the plain-text mask   - lines of ``azimuth elevation`` (horizon profile) or
    ``az_from az_to el_limit`` (sector), with an optional
    ``# Uniform cutoff: <deg>`` comment.

This module deliberately does **not** import RINEX-Masker. The mask is plain
data, and a user who has only the exported file must be able to use it without
RINEX-Masker installed.

Interpolation and blocking semantics mirror RINEX-Masker's
``elevation_mask.ElevationMask`` so both tools agree on which directions are
blocked.
"""

import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

MASK_FORMAT = 'pcc-mask-v1'


class ObstructionMask:
    """An azimuth-dependent horizon, optionally with sectors and a flat cutoff."""

    def __init__(self, uniform_cutoff: float = 0.0):
        self.uniform_cutoff: float = float(uniform_cutoff)
        self.sectors: List[Tuple[float, float, float]] = []
        self.horizon_profile: List[Tuple[float, float]] = []
        self.source_path: Optional[str] = None

    # ---- construction ----------------------------------------------------

    def set_horizon_profile(self, points: Sequence[Sequence[float]]) -> None:
        """Set the (azimuth, elevation) points; they are sorted by azimuth."""
        pts = [(float(az) % 360.0, float(el)) for az, el in points]
        if len(pts) < 2:
            raise ValueError("A horizon profile needs at least 2 points")
        self.horizon_profile = sorted(pts, key=lambda p: p[0])

    @classmethod
    def from_dict(cls, d: dict) -> "ObstructionMask":
        """Build from a ``pcc-mask-v1`` dictionary."""
        fmt = d.get('format')
        if fmt != MASK_FORMAT:
            raise ValueError(
                f"Unknown mask format {fmt!r} - expected {MASK_FORMAT!r}."
            )
        mask = cls(uniform_cutoff=d.get('uniform_cutoff_deg', 0.0) or 0.0)
        mask.sectors = [
            (float(s['az_from']), float(s['az_to']), float(s['el_limit']))
            for s in d.get('sectors', [])
        ]
        profile = d.get('horizon_profile', [])
        if profile:
            mask.set_horizon_profile(
                [(p['azimuth'], p['elevation']) for p in profile]
            )
        return mask

    @classmethod
    def from_file(cls, filepath: str) -> "ObstructionMask":
        """
        Load a mask from JSON or the plain-text format.

        The format is detected from the content, not the extension, because
        RINEX-Masker's text masks are handed around as .txt with varied names.
        """
        with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
            text = fh.read()

        stripped = text.lstrip()
        if stripped.startswith('{'):
            mask = cls.from_dict(json.loads(text))
        else:
            mask = cls._from_text(text)
        mask.source_path = os.path.abspath(filepath)
        return mask

    @classmethod
    def _from_text(cls, text: str) -> "ObstructionMask":
        """Parse RINEX-Masker's plain-text mask."""
        mask = cls()
        profile_points: List[Tuple[float, float]] = []

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                # The uniform cutoff travels as a comment in this format.
                if line.lower().startswith('# uniform cutoff:'):
                    try:
                        mask.uniform_cutoff = float(line.split(':', 1)[1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass
                continue

            parts = line.replace(',', ' ').split()
            try:
                values = [float(p) for p in parts]
            except ValueError:
                continue  # Ignore anything that is not numeric
            if len(values) == 3:
                mask.sectors.append((values[0], values[1], values[2]))
            elif len(values) == 2:
                profile_points.append((values[0], values[1]))

        if profile_points:
            mask.set_horizon_profile(profile_points)
        if not profile_points and not mask.sectors and mask.uniform_cutoff == 0.0:
            raise ValueError(
                "No mask data found in the file (expected 'azimuth elevation' "
                "points, 'az_from az_to el_limit' sectors, or JSON)."
            )
        return mask

    # ---- queries ---------------------------------------------------------

    @staticmethod
    def _azimuth_in_range(azimuth: float, az_from: float, az_to: float) -> bool:
        """Sector test that also handles ranges wrapping through north."""
        azimuth %= 360.0
        az_from %= 360.0
        az_to %= 360.0
        if az_from <= az_to:
            return az_from <= azimuth <= az_to
        return azimuth >= az_from or azimuth <= az_to

    def get_profile_elevation(self, azimuth: float) -> float:
        """Horizon elevation at `azimuth`, linearly interpolated with wrap-around."""
        if not self.horizon_profile:
            return 0.0
        azimuth %= 360.0
        azimuths = [p[0] for p in self.horizon_profile]
        elevations = [p[1] for p in self.horizon_profile]
        # Extend by one point either side so 0/360 interpolates across the seam.
        az_ext = [azimuths[-1] - 360.0] + azimuths + [azimuths[0] + 360.0]
        el_ext = [elevations[-1]] + elevations + [elevations[0]]
        return float(np.interp(azimuth, az_ext, el_ext))

    def get_min_elevation(self, azimuth: float) -> float:
        """Lowest unobstructed elevation at `azimuth`."""
        min_el = self.uniform_cutoff
        for az_from, az_to, el_limit in self.sectors:
            if self._azimuth_in_range(azimuth, az_from, az_to):
                min_el = max(min_el, el_limit)
        if self.horizon_profile:
            min_el = max(min_el, self.get_profile_elevation(azimuth))
        return min_el

    def is_obstructed(self, azimuth: float, elevation: float) -> bool:
        """True if a direction is blocked by the cutoff, a sector or the horizon."""
        return elevation < self.get_min_elevation(azimuth)

    # ---- reporting -------------------------------------------------------

    def summary(self) -> str:
        """One-line description for logs and the GUI."""
        bits = []
        if self.horizon_profile:
            els = [p[1] for p in self.horizon_profile]
            bits.append(f"{len(self.horizon_profile)} horizon points, "
                        f"elevation {min(els):.1f}-{max(els):.1f} deg")
        if self.sectors:
            bits.append(f"{len(self.sectors)} sector(s)")
        if self.uniform_cutoff:
            bits.append(f"uniform cutoff {self.uniform_cutoff:.1f} deg")
        return "; ".join(bits) if bits else "empty mask"


def load_obstruction_mask(filepath: str) -> Optional[ObstructionMask]:
    """
    Load a mask, returning None for a blank path.

    Raises ValueError / OSError with a readable message so callers can surface
    the reason instead of silently continuing without a mask.
    """
    if not filepath or not str(filepath).strip():
        return None
    filepath = str(filepath).strip()
    if not os.path.exists(filepath):
        raise OSError(f"Obstruction mask file not found: {filepath}")
    return ObstructionMask.from_file(filepath)
