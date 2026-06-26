#!/usr/bin/env python3
"""Fast Orbital Braille demo — completes in seconds (low grid/time resolution)."""

from __future__ import annotations

import sys

from run_demo import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--quick", *sys.argv[1:]]
    main()