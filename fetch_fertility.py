"""
Eurostat Fertility Rate Fetcher — Czech Republic
Dataset: demo_frate (Fertility rates by age)
Docs:    https://ec.europa.eu/eurostat/web/json-and-unicode-web-services

Saves raw JSON + clean CSV to data/ so the API is only hit when needed.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
LOG_DIR = ROOT / "logs"

for d in (RAW_DIR, PROCESSED_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

RAW_FILE = RAW_DIR / "demo_frate_CZ_raw.json"
CSV_FILE = PROCESSED_DIR / "fertility_rate_CZ.csv"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "fetch_fertility.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Eurostat API config ────────────────────────────────────────────────────────
EUROSTAT_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

PARAMS = {
    "dataset": "demo_frate",
    "geo":     "CZ",          # Czech Republic
    # "indic_de": leave blank → fetch all indicators (TFR, ASFR, etc.)
    "format":  "JSON",
    "lang":    "EN",
}


# ── Core functions ─────────────────────────────────────────────────────────────

def fetch_from_api(force: bool = False) -> dict:
    """
    Download raw JSON from Eurostat.  Caches to RAW_FILE.

    Args:
        force: Re-download even if cache exists.

    Returns:
        Parsed JSON dict.

    Raises:
        requests.HTTPError:  Non-2xx response.
        requests.Timeout:    Server did not respond in time.
        ValueError:          Response is not valid JSON.
    """
    if RAW_FILE.exists() and not force:
        log.info("Cache hit — loading raw JSON from %s", RAW_FILE)
        return json.loads(RAW_FILE.read_text(encoding="utf-8"))

    url = f"{EUROSTAT_BASE}/{PARAMS['dataset']}"
    query_params = {k: v for k, v in PARAMS.items() if k != "dataset"}

    log.info("Fetching from Eurostat API: %s  params=%s", url, query_params)

    try:
        resp = requests.get(url, params=query_params, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        log.error("Request timed out after 30 s.")
        raise
    except requests.exceptions.HTTPError as exc:
        log.error("HTTP %s — %s", exc.response.status_code, exc.response.text[:300])
        raise
    except requests.exceptions.ConnectionError as exc:
        log.error("Network error: %s", exc)
        raise

    try:
        data = resp.json()
    except ValueError:
        log.error("Response is not valid JSON:\n%s", resp.text[:500])
        raise

    RAW_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Raw JSON saved → %s", RAW_FILE)
    return data


def parse_eurostat_json(data: dict) -> pd.DataFrame:
    """
    Convert Eurostat SDMX-JSON structure into a tidy DataFrame.

    Eurostat JSON layout:
        data["value"]       — flat dict  { "linear_index": value }
        data["dimension"]   — ordered dimension metadata
        data["id"]          — ordered list of dimension names

    Returns:
        DataFrame with columns: year, indicator, age, geo, value
    """
    dims = data["dimension"]
    dim_ids = data["id"]          # e.g. ["freq","indic_de","age","unit","geo","time"]
    dim_sizes = data["size"]      # e.g. [1, 3, 20, 1, 1, 40]

    # Build position→code and position→label maps for each dimension.
    # Eurostat "index" is  { code_string: int_position }, so we invert it.
    index_maps: dict[str, dict] = {}
    for dim_id in dim_ids:
        cat = dims[dim_id]["category"]
        # pos → code  (e.g. {0: "CZ", 1: "DE"})
        pos_to_code = {int(pos): code for code, pos in cat["index"].items()}
        # pos → human label
        pos_to_label = {pos: cat["label"].get(code, code)
                        for pos, code in pos_to_code.items()}
        index_maps[dim_id] = pos_to_code
        index_maps[dim_id + "_label"] = pos_to_label

    # Walk the flat value dict and reconstruct dimension coordinates
    values_raw = data.get("value", {})
    rows = []

    # Precompute stride for each dimension (row-major order)
    strides = []
    stride = 1
    for size in reversed(dim_sizes):
        strides.insert(0, stride)
        stride *= size

    for linear_idx_str, value in values_raw.items():
        linear_idx = int(linear_idx_str)
        coords = {}
        remaining = linear_idx
        for dim_id, dim_stride in zip(dim_ids, strides):
            pos = remaining // dim_stride
            remaining %= dim_stride
            coords[dim_id] = pos

        rows.append({
            "year":      index_maps["time"].get(coords.get("time", -1), "?"),
            "indicator": index_maps["indic_de"].get(coords.get("indic_de", -1), "?")
                         if "indic_de" in index_maps else "?",
            "age":       index_maps["age"].get(coords.get("age", -1), "?")
                         if "age" in index_maps else "TOTAL",
            "geo":       index_maps["geo"].get(coords.get("geo", -1), "CZ"),
            "value":     value,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        log.warning("Parser produced an empty DataFrame — check API response.")
        return df

    # Clean up
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.sort_values(["indicator", "age", "year"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    log.info(
        "Parsed %d rows | years %s–%s | indicators: %s",
        len(df),
        df["year"].min(),
        df["year"].max(),
        df["indicator"].unique().tolist(),
    )
    return df


def save_csv(df: pd.DataFrame, path: Path = CSV_FILE) -> None:
    """Write DataFrame to CSV with a metadata header comment."""
    meta = (
        f"# Czech Republic Fertility Rates — Eurostat dataset: demo_frate\n"
        f"# Retrieved: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"# Source: https://ec.europa.eu/eurostat\n"
    )
    path.write_text(meta, encoding="utf-8")
    df.to_csv(path, mode="a", index=False, encoding="utf-8")
    log.info("CSV saved → %s  (%d rows)", path, len(df))


# ── Entry point ────────────────────────────────────────────────────────────────

def main(force_refresh: bool = False) -> pd.DataFrame:
    """
    Orchestrates fetch → parse → cache → return.

    Args:
        force_refresh: If True, bypass local cache and re-download.

    Returns:
        Tidy DataFrame ready for analysis.
    """
    # Short-circuit: if clean CSV already exists, load it directly
    if CSV_FILE.exists() and not force_refresh:
        log.info("CSV cache found — loading from %s (use --refresh to re-fetch)", CSV_FILE)
        df = pd.read_csv(CSV_FILE, comment="#")
        log.info("Loaded %d rows from CSV cache.", len(df))
        return df

    raw = fetch_from_api(force=force_refresh)
    df = parse_eurostat_json(raw)

    if df.empty:
        log.error("No data parsed — CSV not written.")
        sys.exit(1)

    save_csv(df)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch CZ fertility rates from Eurostat.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore local cache and re-download from the API.",
    )
    args = parser.parse_args()

    df = main(force_refresh=args.refresh)

    print("\n── Preview ──────────────────────────────────────")
    print(df.head(20).to_string(index=False))
    print(f"\nShape: {df.shape}  |  Columns: {list(df.columns)}")
