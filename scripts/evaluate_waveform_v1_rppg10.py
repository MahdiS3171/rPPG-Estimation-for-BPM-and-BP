#!/usr/bin/env python
"""Read-only external rPPG-10 HR evaluation of official frozen Waveform v1."""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rppg_lab.rppg10_external import main


if __name__ == "__main__":
    main()
