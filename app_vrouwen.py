# app_vrouwen.py
from __future__ import annotations

import os
import runpy

os.environ["APP_DATA_ROOT"] = "data_vrouwen"

# Run the existing app.py as if it were executed directly
runpy.run_path("app.py", run_name="__main__")
