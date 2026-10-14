import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Tests never call real embedding providers (Ollama/Gemini). Retrieval tests that need
# embeddings install a fake provider with research_agent.rag.set_embedder().
os.environ["EMBED_MODE"] = "off"

from agents import set_tracing_disabled  # noqa: E402

set_tracing_disabled(True)  # tests never upload traces
