# -*- coding: utf-8 -*-
"""
Version and release date, read from the program rather than from a file date.

A file's modification time says when it was written, not which release it is.

The shape used instead:

  * every program states its version and release date in its own Information
    box, from its `tool_version.py`;
  * the same pair sits in a fixed, machine-readable place, a tag at the top of
    the README;
  * the launcher reads that tag from each installed program.

The tag looks like this, and both forms parse the same way::

    Version: 1.1.0                  - Version: 1.1.0
    Release date: 2026-08-23        - Release date: 2026-08-23

The tag is written from `tool_version.py` when a package is built, and checked
against it, so the README cannot quietly drift from the code.

Every program of the suite uses this same module, unchanged.
"""

import os
import re
from datetime import datetime

# Tolerates a leading "- " or "* " so one regex serves plain text and markdown,
# and bold markers so a README can print the tag in bold if it wants to.
_VERSION_RE = re.compile(r"^[\s\-*]*\**\s*Version:\s*\**\s*([0-9][^\s*]*)",
                         re.IGNORECASE | re.MULTILINE)
_DATE_RE = re.compile(r"^[\s\-*]*\**\s*Release date:\s*\**\s*(\d{4}-\d{2}-\d{2})",
                      re.IGNORECASE | re.MULTILINE)

# Read only the head of a README: a version number mentioned in a change log
# further down must not be mistaken for the tag.
HEAD_LINES = 40

README_NAMES = ("README.md", "readme.txt", "README.txt", "readme.md",
                "Readme.txt")


def parse_text(text):
    """(version, release_date) from a README, either value None if absent."""
    head = "\n".join(text.split("\n")[:HEAD_LINES])
    v = _VERSION_RE.search(head)
    d = _DATE_RE.search(head)
    return (v.group(1) if v else None, d.group(1) if d else None)


def read_tag(folder):
    """
    Read the version tag from the first README in `folder` that carries one.

    Returns {'version', 'release_date', 'source'} or None. Never raises: a
    program that cannot read a readme must still start.
    """
    try:
        if not folder or not os.path.isdir(folder):
            return None
        for name in README_NAMES:
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    version, date = parse_text(fh.read())
            except Exception:
                continue
            if version:
                return {"version": version, "release_date": date,
                        "source": name}
    except Exception:
        pass
    return None


def format_date(iso_date):
    """2026-08-23 -> 23 August 2026, or the input unchanged if it is not one."""
    if not iso_date:
        return ""
    try:
        return datetime.strptime(iso_date, "%Y-%m-%d").strftime("%d %B %Y")
    except Exception:
        return iso_date


def describe(version, release_date=None):
    """The one line every program and the launcher print."""
    if not version:
        return ""
    if release_date:
        return "%s, released %s" % (version, format_date(release_date))
    return str(version)


def about_line(version_module, prefix="Version: "):
    """
    The line an Information box prints, from the tool's own tool_version.

    Returns a plain "Version: unknown" rather than raising, because an
    Information box that cannot open is worse than one missing a date.
    """
    try:
        return prefix + describe(version_module.VERSION,
                                 getattr(version_module, "RELEASE_DATE", None))
    except Exception:
        return prefix + "unknown"
