"""
agentlib: internal library modules for agent.py

This package is intentionally small and dependency-free. It holds settings and
configuration parsing so agent.py can focus on the runtime agent behavior.
"""

from .settings import AgentSettings
from .session import AgentSession, SessionLineResult
from .embedding import build_embedded_session, fork_embedded_session

__all__ = [
    "AgentSettings",
    "AgentSession",
    "SessionLineResult",
    "build_embedded_session",
    "fork_embedded_session",
]

