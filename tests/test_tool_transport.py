"""Tool transport tagging (native API vs JSON-in-content)."""

from __future__ import annotations

import json

from agentlib import agent_json
from agentlib.agent_json import AgentJsonDeps, consume_last_tool_transport
from agentlib.llm.tool_schemas import tool_transport_label
from agentlib.tools import turn_support
from agentlib.tools.registry import ToolRegistry
import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REG = ToolRegistry(default_tools_dir=os.path.join(PROJECT_DIR, "tools"))
_REG.load_plugin_toolsets(_REG.default_tools_dir)
_REG.register_aliases()
_DEPS = AgentJsonDeps(
    all_known_tools=_REG.all_known_tools,
    coerce_enabled_tools=_REG.coerce_enabled_tools,
    merge_tool_param_aliases=turn_support.merge_tool_param_aliases,
)


def test_message_to_agent_json_text_records_native_transport():
    msg = {
        "content": '{"action":"answer","answer":"ignore"}',
        "tool_calls": [
            {
                "function": {
                    "name": "search_web",
                    "arguments": json.dumps({"query": "x"}),
                }
            }
        ],
    }
    agent_json.message_to_agent_json_text(msg, None, _DEPS)
    assert consume_last_tool_transport() == ("search_web", "native")


def test_message_to_agent_json_text_records_json_transport():
    msg = {
        "content": '{"action":"tool_call","tool":"grep","parameters":{"pattern":"foo"}}',
    }
    agent_json.message_to_agent_json_text(msg, None, _DEPS)
    assert consume_last_tool_transport() == ("grep", "json")


def test_tool_transport_label():
    assert tool_transport_label("search_web", "native") == "[native] search_web"