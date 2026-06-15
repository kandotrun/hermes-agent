"""Perplexity Search API provider — plugin form.

Routes Hermes ``web_search`` tool calls through Perplexity's first-party
Search API (``POST /search``), not through Sonar/chat completions. This returns
raw ranked web results as structured JSON and does not ask a Perplexity LLM to
write an answer.

Config keys this provider responds to::

    web:
      search_backend: "perplexity"     # explicit per-capability
      backend: "perplexity"            # shared fallback for search
      perplexity:
        timeout: 60                     # optional seconds
        base_url: "https://api.perplexity.ai"  # optional
        search_context_size: "high"     # optional: low|medium|high
        country: "JP"                  # optional ISO-3166 alpha-2
        search_language_filter: ["ja", "en"]  # optional ISO-639-1 list
        search_domain_filter: ["example.com"]  # optional domain list

Env vars::

    PERPLEXITY_API_KEY=...              # required
    PERPLEXITY_BASE_URL=...             # optional base URL override
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.perplexity.ai"
DEFAULT_TIMEOUT = 60.0
DEFAULT_SEARCH_CONTEXT_SIZE = "high"
VALID_CONTEXT_SIZES = {"low", "medium", "high"}


def _env_value(name: str) -> str:
    """Resolve an env var via Hermes config-aware env, then process env."""
    try:
        from hermes_cli.config import get_env_value

        value = get_env_value(name)
    except Exception:  # noqa: BLE001
        value = None
    if value is None:
        value = os.getenv(name, "")
    return (value or "").strip().strip('"').strip("'")


def _load_perplexity_config() -> Dict[str, Any]:
    """Read ``web.perplexity`` from config.yaml (returns {} on miss)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        web_section = cfg.get("web") if isinstance(cfg, dict) else None
        section = web_section.get("perplexity") if isinstance(web_section, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load web.perplexity config: %s", exc)
        return {}


def _perplexity_api_key() -> str:
    return _env_value("PERPLEXITY_API_KEY")


def _perplexity_base_url(config: Optional[Dict[str, Any]] = None) -> str:
    cfg = config if config is not None else _load_perplexity_config()
    configured = cfg.get("base_url") if isinstance(cfg.get("base_url"), str) else ""
    return (configured.strip() or _env_value("PERPLEXITY_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def _perplexity_timeout(config: Optional[Dict[str, Any]] = None) -> float:
    cfg = config if config is not None else _load_perplexity_config()
    try:
        return float(cfg.get("timeout", DEFAULT_TIMEOUT))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT


def _search_context_size(config: Optional[Dict[str, Any]] = None) -> str:
    cfg = config if config is not None else _load_perplexity_config()
    value = cfg.get("search_context_size") if isinstance(cfg.get("search_context_size"), str) else ""
    value = (value or DEFAULT_SEARCH_CONTEXT_SIZE).strip().lower()
    return value if value in VALID_CONTEXT_SIZES else DEFAULT_SEARCH_CONTEXT_SIZE


def _list_config_value(config: Dict[str, Any], key: str, *, max_items: int = 20) -> List[str]:
    raw = config.get(key)
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = raw
    else:
        return []
    cleaned: List[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if value:
            cleaned.append(value)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _coerce_limit(limit: int) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = 5
    # Perplexity Search API accepts max_results 1..20.
    return max(1, min(value, 20))


def _normalize_search_results(response: Dict[str, Any], *, limit: int) -> List[Dict[str, Any]]:
    """Normalize Perplexity Search API ``results`` rows to Hermes web rows."""
    web_results: List[Dict[str, Any]] = []
    seen: set[str] = set()

    raw_results = response.get("results")
    if not isinstance(raw_results, list):
        return web_results

    for raw in raw_results:
        if not isinstance(raw, dict):
            continue
        url = str(raw.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        title = str(raw.get("title") or url).strip()
        description = str(raw.get("snippet") or "").strip()
        result: Dict[str, Any] = {
            "title": title,
            "url": url,
            "description": description,
            "position": len(web_results) + 1,
        }
        if raw.get("date"):
            result["date"] = raw.get("date")
        if raw.get("last_updated"):
            result["last_updated"] = raw.get("last_updated")
        web_results.append(result)
        if len(web_results) >= limit:
            break

    return web_results


class PerplexityWebSearchProvider(WebSearchProvider):
    """Search-only provider backed by Perplexity's raw Search API."""

    @property
    def name(self) -> str:
        return "perplexity"

    @property
    def display_name(self) -> str:
        return "Perplexity Search API"

    def is_available(self) -> bool:
        """Return True when ``PERPLEXITY_API_KEY`` is configured."""
        return bool(_perplexity_api_key())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        # Search API returns ranked search result snippets, not arbitrary URL fetches.
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a raw Perplexity Search API query and normalize results."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}
        except Exception:  # noqa: BLE001 — interrupt module is best-effort
            pass

        api_key = _perplexity_api_key()
        if not api_key:
            return {
                "success": False,
                "error": (
                    "PERPLEXITY_API_KEY environment variable not set. "
                    "Get your API key at https://docs.perplexity.ai/"
                ),
            }

        limit = _coerce_limit(limit)
        cfg = _load_perplexity_config()
        base_url = _perplexity_base_url(cfg)
        timeout = _perplexity_timeout(cfg)

        payload: Dict[str, Any] = {
            "query": query,
            "max_results": limit,
            "search_context_size": _search_context_size(cfg),
        }

        country = cfg.get("country")
        if isinstance(country, str) and len(country.strip()) == 2:
            payload["country"] = country.strip().upper()

        language_filter = _list_config_value(cfg, "search_language_filter")
        if language_filter:
            payload["search_language_filter"] = language_filter

        domain_filter = _list_config_value(cfg, "search_domain_filter")
        if domain_filter:
            payload["search_domain_filter"] = domain_filter

        try:
            import httpx

            logger.info("Perplexity raw search: '%s' (limit=%d)", query, limit)
            response = httpx.post(
                f"{base_url}/search",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                return {"success": False, "error": "Perplexity search returned a non-object response"}
        except Exception as exc:  # noqa: BLE001 — includes httpx + JSON errors
            logger.warning("Perplexity raw search error: %s", exc)
            return {"success": False, "error": f"Perplexity search failed: {exc}"}

        web_results = _normalize_search_results(data, limit=limit)

        return {
            "success": True,
            "provider": "perplexity",
            "api": "search",
            "id": data.get("id"),
            "server_time": data.get("server_time"),
            "results": data.get("results") or [],
            "data": {"web": web_results},
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Perplexity Search API",
            "badge": "paid",
            "tag": "Raw ranked web results via Perplexity Search API. No Sonar/chat-completions LLM step.",
            "env_vars": [
                {
                    "key": "PERPLEXITY_API_KEY",
                    "prompt": "Perplexity API key",
                    "url": "https://docs.perplexity.ai/",
                },
            ],
        }
