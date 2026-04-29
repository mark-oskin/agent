"""
agentlib: internal library modules for agent.py

This package is intentionally small and dependency-free. It holds settings and
configuration parsing so agent.py can focus on the runtime agent behavior.
"""

from .settings import AgentSettings

__all__ = ["AgentSettings"]

