from pathlib import Path
import sys

# Ensure root and server dirs are available in sys.path
_current_dir = Path(__file__).resolve().parent
_server_dir = _current_dir.parent
_root_dir = _server_dir.parent

for p in [str(_root_dir), str(_server_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)
