"""Compatibility shim for tests/examples.

The actual CRCA-SD implementation lives under `branches/crca_sd/` in this repo.
This package re-exports those modules under the import path `crca_sd.*`.
"""

from crca_sd.crca_sd_core import *  # noqa: F401,F403
from crca_sd.crca_sd_mpc import *  # noqa: F401,F403
from crca_sd.crca_sd_governance import *  # noqa: F401,F403
from crca_sd.crca_sd_realtime import *  # noqa: F401,F403
# NOTE: Do NOT import `crca_sd_tui` at package import time.
# It depends on Unix-only modules (e.g., `termios`) and breaks test collection on Windows.

__all__ = []  # populated by wildcard imports above

