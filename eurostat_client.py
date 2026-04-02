"""
eurostat_client.py — Reusable Eurostat API wrapper

Use this as the foundation for any additional dataset scripts
(e.g. mortality, migration, population pyramids).
"""

import json
import logging
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)

EUROSTAT_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

DEFAULT_TIMEOUT = 30  # seconds


class EurostatClient:
    """
    Thin, cache-aware wrapper around the Eurostat JSON API.

    Usage:
        client = EurostatClient(cache_dir=Path("data/raw"))
        raw = client.fetch("demo_frate", geo="CZ")
    """

    def __init__(self, cache_dir: Path, timeout: int = DEFAULT_TIMEOUT):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _cache_path(self, dataset: str, **filters) -> Path:
        suffix = "_".join(f"{k}{v}" for k, v in sorted(filters.items()) if v)
        name = f"{dataset}_{suffix}.json" if suffix else f"{dataset}.json"
        return self.cache_dir / name

    def fetch(
        self,
        dataset: str,
        geo: Optional[str] = None,
        indic_de: Optional[str] = None,
        age: Optional[str] = None,
        unit: Optional[str] = None,
        lang: str = "EN",
        force: bool = False,
    ) -> dict:
        """
        Fetch a dataset from Eurostat, using a local JSON cache.

        Args:
            dataset:   Eurostat dataset code, e.g. "demo_frate".
            geo:       Country/region filter, e.g. "CZ".
            indic_de:  Demographic indicator filter.
            age:       Age group filter.
            unit:      Unit filter.
            lang:      Language for labels ("EN", "DE", "FR").
            force:     Bypass cache and re-download.

        Returns:
            Parsed JSON dict (Eurostat SDMX-JSON format).

        Raises:
            requests.HTTPError on non-2xx responses.
            requests.Timeout if server is unresponsive.
            ValueError if response is not valid JSON.
        """
        filters = {k: v for k, v in {"geo": geo, "indic_de": indic_de,
                                       "age": age, "unit": unit}.items() if v}
        cache_file = self._cache_path(dataset, **filters)

        if cache_file.exists() and not force:
            log.info("[cache] %s", cache_file.name)
            return json.loads(cache_file.read_text(encoding="utf-8"))

        url = f"{EUROSTAT_BASE}/{dataset}"
        params = {"format": "JSON", "lang": lang, **filters}

        log.info("[API] GET %s  params=%s", url, params)

        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            log.error("Timeout after %ds — %s", self.timeout, url)
            raise
        except requests.exceptions.HTTPError as exc:
            log.error("HTTP %s — %s", exc.response.status_code, exc.response.text[:300])
            raise
        except requests.exceptions.ConnectionError as exc:
            log.error("Connection error: %s", exc)
            raise

        try:
            data = resp.json()
        except ValueError:
            log.error("Invalid JSON response (first 500 chars):\n%s", resp.text[:500])
            raise

        cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        log.info("[saved] %s", cache_file)
        return data
