#!/usr/bin/env python3
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

__all__ = ["load_crca_agent", "__version__"]
__version__ = "1.3.0"


def load_crca_agent():
    # In this repository layout the implementation file is `CRCA.py`
    module_path = Path(__file__).resolve().parent / "CRCA.py"
    spec = spec_from_file_location("crca.crca_impl", str(module_path))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod.CRCAAgent


