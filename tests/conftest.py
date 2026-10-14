import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents import set_tracing_disabled  # noqa: E402

set_tracing_disabled(True)  # tests never upload traces
