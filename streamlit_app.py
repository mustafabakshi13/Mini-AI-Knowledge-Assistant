"""Entrypoint for Streamlit Community Cloud.

Ensures the repository root is properly added to sys.path and executes the
main Streamlit application from app.py.
"""

import sys
from pathlib import Path

# Guarantee that the repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import and execute the app directly
from app import main

if __name__ == "__main__":
    main()
