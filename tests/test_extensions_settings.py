"""Tests for ``extensions`` settings group and ``/set extensions`` backing store."""

from __future__ import annotations

from agentlib.settings import DEFAULT_SETTINGS, AgentSettings


def test_extensions_delta_persists_overrides():
    st = AgentSettings.defaults()
    st.extensions_set_kv("code_pipeline", "code_test_max", "9")
    d = st.as_groups_delta_dict()
    assert d.get("extensions", {}).get("code_pipeline", {}).get("code_test_max") == 9

    st2 = AgentSettings.defaults()
    st2.extensions_merge_from_prefs(d.get("extensions", {}))
    sub = st2.get(("extensions", "code_pipeline"))
    assert isinstance(sub, dict)
    assert sub.get("code_test_max") == 9


def test_extensions_unset_restores_default():
    st = AgentSettings.defaults()
    st.extensions_set_kv("code_pipeline", "inner_round_max", "1")
    st.extensions_unset_key("code_pipeline", "inner_round_max")
    sub = st.get(("extensions", "code_pipeline"))
    dflt = (DEFAULT_SETTINGS.get("extensions") or {}).get("code_pipeline") or {}
    assert isinstance(sub, dict)
    assert sub.get("inner_round_max") == dflt.get("inner_round_max")


def test_extensions_unknown_id_roundtrip():
    st = AgentSettings.defaults()
    st.extensions_set_kv("custom_ext", "alpha", "42")
    sub = st.get(("extensions", "custom_ext"))
    assert isinstance(sub, dict)
    assert sub.get("alpha") == 42
