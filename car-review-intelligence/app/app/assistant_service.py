"""In-process bridge between the Reflex UI and the routing agent.

Loaded once and shared across requests (see `get_service`). It calls the same
`src.agent.chat` the notebook and CLI use, so the app demonstrates the real
routing path -- Ollama when reachable, keyword fallback otherwise -- not a
separate UI-only mock.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import chat  # noqa: E402


class AssistantService:
    def respond(self, message, use_ollama=True):
        """Route one message and return a UI-friendly dict.

        Models download on first use of a given skill, then cache; the first
        message that triggers a new skill can take a few seconds.
        """
        out = chat(message, use_ollama=use_ollama)
        return {
            "reply": out["reply"],
            "skill": out["skill"] or "—",
            "via": out["via"],
        }


_service = None


def get_service():
    global _service
    if _service is None:
        _service = AssistantService()
    return _service
