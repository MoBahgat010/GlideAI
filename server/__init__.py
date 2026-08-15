from pathlib import Path
import sys

_server_dir = Path(__file__).resolve().parent
_root_dir = _server_dir.parent
_src_dir = _server_dir / "src"

for p in [str(_root_dir), str(_server_dir), str(_src_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)
