from __future__ import annotations

import datetime
import html as html_module
import json
import re
from typing import Optional

import requests


def _settings_get_bool(settings, group: str, key: str, default: bool) -> bool:
    if settings is None:
        return default
    try:
        return bool(settings.get_bool((group, key), default=default))  # type: ignore[attr-defined]
    except Exception:
        return default


def _settings_get_int(settings, group: str, key: str, default: int) -> int:
    if settings is None:
        return default
    try:
        return int(settings.get_int((group, key), default=default))  # type: ignore[attr-defined]
    except Exception:
        return default


def _settings_get_str(settings, group: str, key: str, default: str) -> str:
    if settings is None:
        return default
    try:
        return str(settings.get_str((group, key), default=default))  # type: ignore[attr-defined]
    except Exception:
        return default


def default_search_web_max_results(settings=None) -> int:
    v = _settings_get_int(settings, "agent", "search_web_max_results", 5)
    return max(1, min(30, int(v)))


def search_web_max_results_clamped(n: object, *, fallback: int) -> int:
    if n is None or isinstance(n, bool):
        return fallback
    try:
        v = int(float(n))
    except (TypeError, ValueError):
        t = str(n).strip()
        if not t:
            return fallback
        try:
            v = int(float(t))
        except (TypeError, ValueError):
            return fallback
    return max(1, min(30, v))


def search_web_effective_max_results(params: object, *, settings=None) -> int:
    p = params if isinstance(params, dict) else {}
    d = default_search_web_max_results(settings)
    for k in ("max_results", "max", "num_results", "n", "limit"):
        if p.get(k) is not None:
            return search_web_max_results_clamped(p.get(k), fallback=d)
    return d


def enrich_search_query_for_present_day(query: str, *, settings=None) -> str:
    if not _settings_get_bool(settings, "ollama", "search_enrich", True):
        return query
    q = (query or "").strip()
    if not q:
        return q
    low = q.lower()
    if re.search(
        r"\b(who was|who were|first |second |third |fourth |fifth |"
        r"\d{1,2}(st|nd|rd|th) |original |founding )\b",
        low,
    ):
        return q
    if re.search(r"\b(in 19[0-9]{2}|in 20[01][0-9])\b", low):
        return q
    if re.search(r"\b(current|today|now|present|incumbent|latest)\b", low):
        if not re.search(r"\b20[0-9]{2}\b", low):
            y = datetime.date.today().year
            return f"{q.rstrip('.')} {y}"
        return q
    if re.search(r"\b20[2-9][0-9]\b", low):
        return q
    if re.search(r"\b(who is|who's|who are)\b", low) and re.search(
        r"\b(president|prime minister|governor|mayor|senator|representative|"
        r"ceo|chancellor|speaker|chief justice|king|queen)\b",
        low,
    ):
        y = datetime.date.today().year
        return f"{q.rstrip('.')} current {y}"
    return q


def search_web_backend(settings=None) -> str:
    v = _settings_get_str(settings, "agent", "search_web_backend", "ddg").strip().lower()
    if v in ("ddg", "duckduckgo"):
        return "ddg"
    if v in ("searx", "searxng"):
        return "searxng"
    return "ddg"


def searxng_base_url(settings=None) -> str:
    u = _settings_get_str(settings, "agent", "searxng_url", "https://searx.party").strip()
    u = u.rstrip("/")
    if not u.startswith(("http://", "https://")):
        return "https://searx.party"
    return u


def search_backend_banner_line(settings=None) -> str:
    bk = search_web_backend(settings)
    if bk == "searxng":
        return f"[Search backend] searxng ({searxng_base_url(settings)!r})"
    return "[Search backend] duckduckgo"


def _searxng_search(query: str, *, max_results: int, settings=None) -> str:
    q = (query or "").strip()
    if not q:
        return ""
    base = searxng_base_url(settings)
    url = base + "/search"
    try:
        r = requests.get(
            url,
            params={"q": q, "format": "json"},
            timeout=15,
            headers={"User-Agent": "agentlib/1.0"},
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return ""
    res = data.get("results") if isinstance(data, dict) else None
    if not isinstance(res, list) or not res:
        return ""
    lines = []
    for one in res[: max(1, min(30, int(max_results)))]:
        if not isinstance(one, dict):
            continue
        title = str(one.get("title") or "").strip()
        url0 = str(one.get("url") or "").strip()
        content = str(one.get("content") or "").strip()
        if not url0:
            continue
        if title:
            lines.append(f"- {title}\n  {content}\n  {url0}".rstrip())
        else:
            lines.append(f"- {content}\n  {url0}".rstrip())
    return "\n".join(lines)


def _ddg_instant_answer(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return ""
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": q, "format": "json", "no_redirect": "1", "no_html": "1"},
            timeout=12,
            headers={"User-Agent": "agentlib/1.0"},
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    parts = []
    for k in ("AbstractText", "Answer", "Definition"):
        v = str(data.get(k) or "").strip()
        if v:
            parts.append(v)
    if not parts:
        rel = data.get("RelatedTopics")
        if isinstance(rel, list) and rel:
            for one in rel:
                if isinstance(one, dict) and str(one.get("Text") or "").strip():
                    parts.append(str(one.get("Text") or "").strip())
                    break
    return "\n".join(parts).strip()


def _fetch_ddg_html(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return ""
    try:
        r = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": q},
            timeout=15,
            headers={"User-Agent": "agentlib/1.0"},
        )
        r.raise_for_status()
        return r.text or ""
    except Exception:
        return ""


def _parse_ddg_html_results(page: str, max_results: int = 5):
    if not page:
        return []
    out = []
    # Basic extraction from DDG "html" endpoint.
    for m in re.finditer(r'class="result__a"\s+href="([^"]+)"[^>]*>(.*?)</a>', page, re.I | re.S):
        url = html_module.unescape(m.group(1) or "").strip()
        title = re.sub(r"<[^>]*>", " ", m.group(2) or "")
        title = html_module.unescape(re.sub(r"\s+", " ", title)).strip()
        if not url:
            continue
        out.append(f"- {title}\n  {url}".rstrip())
        if len(out) >= max_results:
            break
    return out


def _wikipedia_opensearch(query: str, result_limit: int = 5) -> str:
    q = (query or "").strip()
    if not q:
        return ""
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "search": q,
                "limit": max(1, min(30, int(result_limit))),
                "namespace": 0,
                "format": "json",
            },
            timeout=12,
            headers={"User-Agent": "agentlib/1.0"},
        )
        r.raise_for_status()
        data = r.json()
        if len(data) < 4 or not data[1]:
            return ""
        lines = []
        for title, desc, url in zip(data[1], data[2], data[3]):
            d = (desc or "").strip()
            lines.append(f"- {title}\n  {d}\n  {url}")
        return "\n".join(lines)
    except Exception:
        return ""


def _first_url_in_text(s: str) -> str:
    if not s:
        return ""
    m = re.search(r"https?://\S+", s)
    return m.group(0).rstrip(").,]") if m else ""


def wikipedia_top_page_extract(query: str, *, fetch_page: callable) -> str:
    listing = _wikipedia_opensearch(query)
    url = _first_url_in_text(listing)
    if not url:
        return ""
    page = fetch_page(url)
    if not page or str(page).startswith("Fetch error:"):
        return ""
    return f"Top result URL: {url}\nExtract: {str(page)[:1200]}"


def search_web(query: str, params: Optional[dict] = None, *, settings=None, fetch_page: callable) -> str:
    q = enrich_search_query_for_present_day(str(query or ""), settings=settings)
    mr = search_web_effective_max_results(params or {}, settings=settings)
    backend = search_web_backend(settings)
    parts: list[str] = []

    parts.append(search_backend_banner_line(settings))
    rows_text = ""
    got_rows = False
    if backend == "searxng":
        rows_text = _searxng_search(q, max_results=mr, settings=settings)
        if rows_text:
            parts.append("[Web results]\n" + rows_text)
            got_rows = True
        else:
            parts.append(
                f"[Note] SearXNG returned no usable results (instance: {searxng_base_url(settings)!r}). "
                "Falling back to DuckDuckGo instant answers and Wikipedia."
            )
    if backend != "searxng" or not got_rows:
        ia = _ddg_instant_answer(q)
        if ia:
            parts.append("[DuckDuckGo instant answer]\n" + ia)
        page = _fetch_ddg_html(q)
        blocked = "anomaly-modal" in page or "bots use DuckDuckGo" in page
        rows = [] if blocked else _parse_ddg_html_results(page, max_results=mr)
        if rows:
            parts.append("[Web results]\n" + "\n".join(rows))
            got_rows = True
        elif blocked:
            parts.append(
                "[Note] DuckDuckGo returned a bot-check page instead of HTML results "
                "(common for datacenter IPs). Instant answer and Wikipedia fallback still apply."
            )
    if not got_rows:
        wiki = _wikipedia_opensearch(q, result_limit=mr)
        if wiki:
            parts.append("[Wikipedia search]\n" + wiki)
        wiki_extract = wikipedia_top_page_extract(q, fetch_page=fetch_page)
        if wiki_extract:
            parts.append("[Wikipedia top result extract]\n" + wiki_extract)
    if not parts:
        return (
            "No results found for this search. "
            "Try search_web again with a shorter or alternate query, a product/site/organization name, "
            "or a year (e.g. 2026) for time-sensitive topics. If the user provided a URL, use fetch_page on that URL."
        )
    return "\n\n".join(parts)

