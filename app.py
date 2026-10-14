"""ResearchPilot UI.

    python app.py                 # http://127.0.0.1:7860
    python app.py --port 7861     # another port
    python app.py --share         # temporary public link (Gradio)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from research_agent.config import ConfigError, load_settings  # noqa: E402
from research_agent.tracing import configure_tracing  # noqa: E402
from research_agent.ui.app import launch  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    try:
        settings = load_settings()
    except ConfigError as exc:
        raise SystemExit(f"[config] {exc}")
    configure_tracing(settings)
    print(f"ResearchPilot: http://{args.host}:{args.port}  (database: {settings.db_path})")
    launch(settings, server_name=args.host, server_port=args.port, share=args.share, inbrowser=False)
