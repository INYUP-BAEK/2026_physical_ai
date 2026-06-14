#!/usr/bin/env python3
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().with_name("09_eval_v11_rollout_core.py")), run_name="__main__")
