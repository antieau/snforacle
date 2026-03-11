import sys
from pathlib import Path

# Make test-local helpers (e.g. _mathelpers) importable as plain modules.
sys.path.insert(0, str(Path(__file__).parent))
