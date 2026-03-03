#!/usr/bin/env python
"""
Run the LegacyLens scenario tests (6 queries from requirements.md).
Usage: py run_tests.py
"""
import sys
from pathlib import Path

# Load .env before imports
import os
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    from tests.scenario_tests import main
    sys.exit(main())
