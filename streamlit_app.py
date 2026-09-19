"""Entrypoint alias for Streamlit Community Cloud.

Redirects execution to app.py so that deployments targeting either
'app.py' or 'streamlit_app.py' as the main file path work seamlessly.
"""

from pathlib import Path
import runpy

app_path = Path(__file__).parent / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
