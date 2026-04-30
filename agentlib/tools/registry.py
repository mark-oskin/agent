from __future__ import annotations

"""
Tool registry for core + plugin tools.

This wraps the existing `agentlib.tools.routing` + `agentlib.tools.plugins` globals behind
an object so the composition root (`agentlib.app` / `agent.py`) doesn't need a pile of
module-level aliases and wrappers.
"""

from dataclasses import dataclass
from typing import AbstractSet, Callable, Optional

from agentlib.tools import plugins, routing


def _core_tool_ids() -> frozenset[str]:
    return frozenset({tid for tid, _label, _aliases in routing.CORE_TOOL_ENTRIES})


@dataclass
class ToolRegistry:
    """
    Mutable tool registry for a single process.

    Plugin toolsets are stored in module globals in `agentlib.tools.plugins`. This class provides
    a small interface that hides those globals from the rest of the app.
    """

    default_tools_dir: str

    def load_plugin_toolsets(self, tools_dir: Optional[str] = None) -> None:
        plugins.load_plugin_toolsets(
            tools_dir=tools_dir,
            default_tools_dir=self.default_tools_dir,
        )

    def register_aliases(self) -> None:
        routing.register_tool_aliases()

    @property
    def core_tools(self) -> frozenset[str]:
        return _core_tool_ids()

    @property
    def plugin_toolsets(self) -> dict:
        return plugins.PLUGIN_TOOLSETS

    @property
    def plugin_tool_handlers(self) -> dict[str, Callable[[dict], str]]:
        return plugins.PLUGIN_TOOL_HANDLERS

    def all_known_tools(self) -> frozenset[str]:
        return routing.all_known_tools()

    def coerce_enabled_tools(self, ets: Optional[AbstractSet[str]]) -> AbstractSet[str]:
        """`None` means all tools enabled (default)."""
        if ets is None:
            return self.all_known_tools()
        return frozenset(ets)

    def normalize_tool_name(self, token: str) -> Optional[str]:
        return routing.normalize_tool_name(token)

    def canonicalize_user_tool_phrase(self, phrase: str) -> str:
        return routing.canonicalize_user_tool_phrase(phrase)

    def effective_enabled_tools_for_turn(
        self,
        *,
        base_enabled_tools: AbstractSet[str],
        enabled_toolsets: AbstractSet[str],
        user_query: str,
    ) -> frozenset[str]:
        return routing.effective_enabled_tools_for_turn(
            base_enabled_tools=base_enabled_tools,
            enabled_toolsets=enabled_toolsets,
            user_query=user_query,
        )

    def plugin_tools_for_toolset(self, toolset: str) -> set[str]:
        return plugins.plugin_tools_for_toolset(toolset)

    def route_active_toolsets_for_request(
        self, user_query: str, enabled_toolsets: AbstractSet[str]
    ) -> set[str]:
        return routing.route_active_toolsets_for_request(user_query, enabled_toolsets)

    def format_unknown_tool_hint(self, phrase: str) -> str:
        return routing.format_unknown_tool_hint(phrase)

    def format_settings_tools_list(self, enabled_tools: AbstractSet[str]) -> str:
        return routing.format_settings_tools_list(enabled_tools)

    def describe_tool_call_contract(self, tool_id: str) -> str:
        return routing.describe_tool_call_contract(tool_id)

    def tool_policy_runner_text(self, enabled_tools: Optional[AbstractSet[str]]) -> str:
        return routing.tool_policy_runner_text(enabled_tools)

