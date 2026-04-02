# Czech Republic Demographic Data — Eurostat

A clean, cache-aware pipeline for pulling Czech demographic data from the
[Eurostat JSON API](https://ec.europa.eu/eurostat/web/json-and-unicode-web-services).

## Project Structure

```
cz_demography/
├── data/
│   ├── raw/          # Cached Eurostat JSON responses (auto-created)
│   └── processed/    # Clean CSVs ready for analysis (auto-created)
├── logs/             # Fetch logs (auto-created)
├── src/
│   ├── eurostat_client.py   # Reusable API wrapper — use for any new dataset
│   └── fetch_fertility.py   # Fertility rates (demo_frate) for Czech Republic
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### First run (downloads from API, caches locally)
```bash
python src/fetch_fertility.py
```

### Subsequent runs (loads from CSV cache — no API call)
```bash
python src/fetch_fertility.py
```

### Force refresh from API
```bash
python src/fetch_fertility.py --refresh
```

## Caching Strategy

| Layer | File | Description |
|-------|------|-------------|
| Raw   | `data/raw/demo_frate_CZraw.json` | Full Eurostat JSON — skip API next run |
| Clean | `data/processed/fertility_rate_CZ.csv` | Parsed, tidy CSV — loaded directly |

Delete either file to trigger a re-fetch/re-parse.

## Adding More Datasets

Use `EurostatClient` from `src/eurostat_client.py`:

```python
from pathlib import Path
from src.eurostat_client import EurostatClient

client = EurostatClient(cache_dir=Path("data/raw"))

# Example: mortality rates
raw = client.fetch("demo_magec", geo="CZ")
```

Common Czech demographic datasets:
| Code | Description |
|------|-------------|
| `demo_frate` | Fertility rates by age |
| `demo_magec` | Deaths by age and sex |
| `demo_pjan`  | Population on 1 January |
| `demo_gind`  | General population indicators |
| `demo_minfind` | Migration and integration |

## Output CSV columns

| Column | Description |
|--------|-------------|
| `year` | Reference year (Int64) |
| `indicator` | Eurostat indicator code (e.g. TOTFERRT, ASFR) |
| `age` | Age group (TOTAL or specific age) |
| `geo` | Country code (CZ) |
| `value` | Rate value |
