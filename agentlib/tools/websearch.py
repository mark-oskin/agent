from __future__ import annotations

import datetime
import html as html_module
import json
import re
from typing import Callable, Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests

def _unique_in_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        s = (it or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def urls_from_search_output(search_output: str, *, max_urls: int = 10) -> list[str]:
    """
    Extract URLs from `search_web` formatted output, preserving order.

    This is intentionally simple: it looks for http(s):// links and strips trailing punctuation.
    """
    s = str(search_output or "")
    urls = []
    for m in re.finditer(r"https?://\\S+", s):
        u = (m.group(0) or "").rstrip(").,]>}\"'")
        if u:
            urls.append(u)
        if len(urls) >= max_urls * 3:
            break
    return _unique_in_order(urls)[: max(1, min(30, int(max_urls)))]


def _fetch_html(url: str) -> tuple[str, int, str]:
    """
    Fetch raw HTML for readability extraction.
    Returns (final_url, http_status, html_text_or_error).
    """
    u = (url or "").strip()
    if not u:
        return ("", 0, "Fetch error: empty url string.")
    if not re.match(r"^https?://", u, re.IGNORECASE):
        return ("", 0, f"Fetch error: URL must start with http:// or https:// (got {u!r}).")
    headers = {"User-Agent": "agentlib/1.0 (readability)"}
    try:
        r = requests.get(u, headers=headers, timeout=22, allow_redirects=True)
        st = int(r.status_code)
        if st >= 400:
            return (r.url or u, st, f"Fetch error: HTTP {st}")
        return (r.url or u, st, r.text or "")
    except Exception as ex:
        return ("", 0, f"Fetch error: {type(ex).__name__}: {ex}")


def _readability_excerpt_from_html(html_text: str, *, url: str, max_chars: int = 2600) -> tuple[str, str]:
    """
    Return (title, excerpt_text) using readability-lxml.
    Falls back to regex tag stripping if readability fails.
    """
    body = str(html_text or "")
    if not body.strip():
        return ("", "")
    title = ""
    text = ""
    try:
        from readability import Document  # type: ignore[import-not-found]

        doc = Document(body)
        title = (doc.short_title() or "").strip()
        summary_html = doc.summary(html_partial=True) or ""
        try:
            import lxml.html  # type: ignore[import-not-found]

            root = lxml.html.fromstring(summary_html)
            text = root.text_content()
        except Exception:
            text = re.sub(r"<[^>]*>", " ", summary_html)
    except Exception:
        text = re.sub(r"<[^>]*>", " ", body)

    text = re.sub(r"\\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return (title, text)


def search_web_fetch_top(
    query: str,
    params: Optional[dict] = None,
    *,
    settings=None,
    fetch_page: callable,
) -> str:
    """
    Combined tool: run `search_web`, then fetch and excerpt the top N URLs (default 10).

    Output is a single string containing the original search results plus per-URL excerpts.
    """
    p = params if isinstance(params, dict) else {}
    mr = search_web_effective_max_results(p, settings=settings)
    n = p.get("fetch_top_n")
    try:
        fetch_top_n = int(float(n)) if n is not None else 10
    except Exception:
        fetch_top_n = 10
    fetch_top_n = max(1, min(10, int(fetch_top_n)))

    listing = search_web(query, params={**p, "max_results": mr}, settings=settings, fetch_page=fetch_page)
    urls = urls_from_search_output(listing, max_urls=fetch_top_n)
    if not urls:
        return listing + "\\n\\n[Fetched excerpts]\\n(no urls extracted from search results)"

    # Fetch in parallel (IO-bound)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    recs = []
    with ThreadPoolExecutor(max_workers=min(10, max(4, len(urls)))) as ex:
        futs = {ex.submit(_fetch_html, u): u for u in urls}
        for fut in as_completed(futs):
            u0 = futs[fut]
            try:
                final_url, status, body = fut.result()
            except Exception as ex2:
                final_url, status, body = ("", 0, f"Fetch error: {type(ex2).__name__}: {ex2}")
            if isinstance(body, str) and body.startswith("Fetch error:"):
                recs.append((u0, final_url, status, "", body))
                continue
            title, excerpt = _readability_excerpt_from_html(body, url=u0)
            recs.append((u0, final_url, status, title, excerpt))

    # Preserve original URL order in output.
    by_url = {r[0]: r for r in recs}
    lines = ["[Fetched excerpts]"]
    for i, u in enumerate(urls, start=1):
        u0, final_url, status, title, excerpt = by_url.get(u, (u, "", 0, "", ""))
        t = title.strip() if isinstance(title, str) else ""
        if t:
            lines.append(f"{i}. {t}\n   {u0}")
        else:
            lines.append(f"{i}. {u0}")
        if final_url and final_url != u0:
            lines.append(f"   Final URL: {final_url}")
        if status:
            lines.append(f"   HTTP: {status}")
        if excerpt:
            lines.append(f"   Excerpt: {excerpt}")
        else:
            lines.append("   Excerpt: (empty)")
        lines.append("")
    return listing + "\n\n" + "\n".join(lines).rstrip()


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


def _debug_search_web_log(settings) -> Optional[Callable[[str], None]]:
    if not _settings_get_bool(settings, "agent", "debug_search_web", False):
        return None

    def log(msg: str) -> None:
        from agentlib.sink import sink_emit

        sink_emit({"type": "stderr", "text": f"[debug_search_web] {msg}"})

    return log


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


def resolved_http_url_from_href(href: str) -> str:
    """
    Normalize href values from DuckDuckGo HTML SERPs.

    DDG often emits scheme-relative redirects:
      //duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F...
    Those strings do not contain 'http(s)://', which broke URL-backed detection in
    `is_tool_result_weak_for_dedup` and made fetch_page harder for the model.

    Returns a canonical https URL when `uddg` decodes to one; otherwise applies
    minimal fixes (e.g. leading '//' -> 'https://').
    """
    raw = html_module.unescape((href or "").strip())
    if not raw:
        return ""
    u = raw
    if u.startswith("//"):
        u = "https:" + u
    try:
        parsed = urlparse(u)
        netloc = (parsed.netloc or "").lower()
        if "duckduckgo.com" in netloc and "/l/" in (parsed.path or ""):
            qs = parse_qs(parsed.query)
            uddg_vals = qs.get("uddg") or []
            if uddg_vals:
                inner = unquote(uddg_vals[0]).strip()
                if inner.startswith(("http://", "https://")):
                    return inner
    except Exception:
        pass
    if raw.startswith("//"):
        return "https:" + raw
    if raw.startswith(("http://", "https://")):
        return raw
    if u.startswith(("http://", "https://")):
        return u
    return raw


def search_backend_banner_line(settings=None) -> str:
    bk = search_web_backend(settings)
    if bk == "searxng":
        return f"[Search backend] searxng ({searxng_base_url(settings)!r})"
    return "[Search backend] duckduckgo"


def _searxng_search(
    query: str,
    *,
    max_results: int,
    settings=None,
    log: Optional[Callable[[str], None]] = None,
) -> str:
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
        if log:
            log(f"SearXNG GET {r.url!r} status={r.status_code} bytes={len(r.content)}")
        r.raise_for_status()
        data = r.json()
        if log and isinstance(data, dict):
            n = data.get("results")
            log(f"SearXNG JSON keys={sorted(data.keys())} results_len={len(n) if isinstance(n, list) else 'n/a'}")
    except Exception as ex:
        if log:
            log(f"SearXNG error: {type(ex).__name__}: {ex}")
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
    if log and not lines:
        log("SearXNG: results list non-empty but no lines after filtering (missing urls?)")
    return "\n".join(lines)


def _ddg_instant_answer(query: str, *, log: Optional[Callable[[str], None]] = None) -> str:
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
        if log:
            log(f"DDG instant GET {r.url!r} status={r.status_code} bytes={len(r.content)}")
        r.raise_for_status()
        data = r.json()
        if log and isinstance(data, dict):
            log(f"DDG instant JSON keys={sorted(data.keys())}")
    except Exception as ex:
        if log:
            log(f"DDG instant error: {type(ex).__name__}: {ex}")
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


def _fetch_ddg_html(query: str, *, log: Optional[Callable[[str], None]] = None) -> str:
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
        if log:
            log(f"DDG html GET {r.url!r} status={r.status_code} bytes={len(r.text or '')}")
        r.raise_for_status()
        page = r.text or ""
        if log:
            bot = "anomaly-modal" in page or "bots use DuckDuckGo" in page
            n_a = len(re.findall(r'class="result__a"', page, flags=re.I))
            log(f"DDG html bot_check_hint={bot} result__a_occurrences={n_a}")
        return page
    except Exception as ex:
        if log:
            log(f"DDG html error: {type(ex).__name__}: {ex}")
        return ""


def _parse_ddg_html_results(page: str, max_results: int = 5):
    if not page:
        return []
    out = []
    # Basic extraction from DDG "html" endpoint.
    for m in re.finditer(r'class="result__a"\s+href="([^"]+)"[^>]*>(.*?)</a>', page, re.I | re.S):
        href_raw = html_module.unescape(m.group(1) or "").strip()
        url = resolved_http_url_from_href(href_raw)
        title = re.sub(r"<[^>]*>", " ", m.group(2) or "")
        title = html_module.unescape(re.sub(r"\s+", " ", title)).strip()
        if not url:
            continue
        out.append(f"- {title}\n  {url}".rstrip())
        if len(out) >= max_results:
            break
    return out


def _wikipedia_opensearch(
    query: str, result_limit: int = 5, *, log: Optional[Callable[[str], None]] = None
) -> str:
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
        if log:
            log(f"Wikipedia opensearch GET {r.url!r} status={r.status_code} bytes={len(r.content)}")
        r.raise_for_status()
        data = r.json()
        if len(data) < 4 or not data[1]:
            return ""
        lines = []
        for title, desc, url in zip(data[1], data[2], data[3]):
            d = (desc or "").strip()
            lines.append(f"- {title}\n  {d}\n  {url}")
        if log:
            log(f"Wikipedia opensearch titles={len(data[1]) if len(data) >= 2 else 0}")
        return "\n".join(lines)
    except Exception as ex:
        if log:
            log(f"Wikipedia opensearch error: {type(ex).__name__}: {ex}")
        return ""


def first_url_in_text(s: str) -> str:
    if not s:
        return ""
    m = re.search(r"https?://\S+", s)
    return m.group(0).rstrip(").,]") if m else ""


def wikipedia_top_page_extract(query: str, *, fetch_page: callable) -> str:
    listing = _wikipedia_opensearch(query)
    url = first_url_in_text(listing)
    if not url:
        return ""
    page = fetch_page(url)
    if not page or str(page).startswith("Fetch error:"):
        return ""
    return f"Top result URL: {url}\nExtract: {str(page)[:1200]}"


def search_web(query: str, params: Optional[dict] = None, *, settings=None, fetch_page: callable) -> str:
    dbg = _debug_search_web_log(settings)
    raw_q = str(query or "")
    q = enrich_search_query_for_present_day(raw_q, settings=settings)
    mr = search_web_effective_max_results(params or {}, settings=settings)
    backend = search_web_backend(settings)
    parts: list[str] = []

    if dbg:
        p = params if isinstance(params, dict) else {}
        dbg(
            f"start backend={backend!r} max_results={mr} "
            f"query_raw={raw_q!r} query_enriched={q!r} param_keys={sorted(p.keys())!r}"
        )

    parts.append(search_backend_banner_line(settings))
    rows_text = ""
    got_rows = False
    if backend == "searxng":
        rows_text = _searxng_search(q, max_results=mr, settings=settings, log=dbg)
        if rows_text:
            parts.append("[Web results]\n" + rows_text)
            got_rows = True
        else:
            parts.append(
                f"[Note] SearXNG returned no usable results (instance: {searxng_base_url(settings)!r}). "
                "Falling back to DuckDuckGo instant answers and Wikipedia."
            )
    if backend != "searxng" or not got_rows:
        ia = _ddg_instant_answer(q, log=dbg)
        if dbg:
            dbg(f"DDG instant answer text_chars={len(ia)} non_empty={bool((ia or '').strip())}")
        if ia:
            parts.append("[DuckDuckGo instant answer]\n" + ia)
        page = _fetch_ddg_html(q, log=dbg)
        blocked = "anomaly-modal" in page or "bots use DuckDuckGo" in page
        rows = [] if blocked else _parse_ddg_html_results(page, max_results=mr)
        if dbg:
            dbg(f"DDG parsed_html_rows={len(rows)} blocked_page={blocked}")
        if rows:
            parts.append("[Web results]\n" + "\n".join(rows))
            got_rows = True
        elif blocked:
            parts.append(
                "[Note] DuckDuckGo returned a bot-check page instead of HTML results "
                "(common for datacenter IPs). Instant answer and Wikipedia fallback still apply."
            )
    if not got_rows:
        wiki = _wikipedia_opensearch(q, result_limit=mr, log=dbg)
        if dbg:
            dbg(f"Wikipedia listing_chars={len(wiki)} non_empty={bool((wiki or '').strip())}")
        if wiki:
            parts.append("[Wikipedia search]\n" + wiki)
        wiki_extract = wikipedia_top_page_extract(q, fetch_page=fetch_page)
        if dbg:
            dbg(f"Wikipedia top_extract_chars={len(wiki_extract)} non_empty={bool((wiki_extract or '').strip())}")
        if wiki_extract:
            parts.append("[Wikipedia top result extract]\n" + wiki_extract)
    if not parts:
        return (
            "No results found for this search. "
            "Try search_web again with a shorter or alternate query, a product/site/organization name, "
            "or a year (e.g. 2026) for time-sensitive topics. If the user provided a URL, use fetch_page on that URL."
        )
    out = "\n\n".join(parts)
    if dbg:
        has_url = bool(re.search(r"https?://", out))
        dbg(
            f"done merged_parts={len(parts)} output_chars={len(out)} "
            f"got_rows={got_rows} output_contains_http_url={has_url}"
        )
    return out

