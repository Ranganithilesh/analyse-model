"""Make sure optional packages exist. If one is missing in the Python that is running the app,
install it into THAT interpreter (this is what fixed 'No module named cryptography' in SV-36/37)."""
import importlib
import subprocess
import sys


def ensure(module: str, pip_name: str = None) -> bool:
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        pass
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", pip_name or module],
                       check=True, capture_output=True, timeout=300)
        importlib.invalidate_caches()
        importlib.import_module(module)
        return True
    except Exception:
        return False
