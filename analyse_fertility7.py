"""
analyse_fertility.py — Basic stats, trend projection, and chart
for Czech Republic fertility rates from demo_frate.

Run from the demography/ folder:
    python analyse_fertility.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
CSV_FILE  = ROOT / "data" / "processed" / "fertility_rate_CZ.csv"
OUT_DIR   = ROOT / "data" / "processed"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── Load ───────────────────────────────────────────────────────────────────────

def load_tfr(path: Path = CSV_FILE) -> pd.Series:
    df = pd.read_csv(path, comment="#")
    age_specific = df[~df["age"].isin(["TOTAL", "?"])].copy()
    tfr = (
        age_specific.groupby("year")["value"].sum()
        .sort_index()
        .dropna()
    )
    if tfr.empty:
        log.error("No age-specific rows found. Available ages: %s", df["age"].unique().tolist())
        sys.exit(1)
    log.info("Loaded TFR: %d years (%d – %d)", len(tfr), tfr.index.min(), tfr.index.max())
    return tfr


# ── Stats ──────────────────────────────────────────────────────────────────────

def compute_stats(tfr: pd.Series) -> dict:
    yoy_abs = tfr.diff().dropna()
    yoy_pct = tfr.pct_change().dropna() * 100
    n_years = len(tfr) - 1
    cagr    = (tfr.iloc[-1] / tfr.iloc[0]) ** (1 / n_years) - 1
    return {
        "first_year":   int(tfr.index.min()),
        "last_year":    int(tfr.index.max()),
        "n_years":      len(tfr),
        "tfr_start":    round(tfr.iloc[0], 3),
        "tfr_end":      round(tfr.iloc[-1], 3),
        "tfr_min":      round(tfr.min(), 3),
        "tfr_min_year": int(tfr.idxmin()),
        "tfr_max":      round(tfr.max(), 3),
        "tfr_max_year": int(tfr.idxmax()),
        "mean_yoy_abs": round(yoy_abs.mean(), 4),
        "mean_yoy_pct": round(yoy_pct.mean(), 3),
        "cagr_pct":     round(cagr * 100, 3),
    }


def print_stats(s: dict) -> None:
    print("\n" + "═" * 52)
    print("  Czech Republic — Total Fertility Rate")
    print(f"  {s['first_year']} – {s['last_year']}  ({s['n_years']} observations)")
    print("═" * 52)
    print(f"  Start value          {s['tfr_start']:.3f}")
    print(f"  End value            {s['tfr_end']:.3f}")
    print(f"  All-time high        {s['tfr_max']:.3f}  ({s['tfr_max_year']})")
    print(f"  All-time low         {s['tfr_min']:.3f}  ({s['tfr_min_year']})")
    print("  ──────────────────────────────────────")
    print(f"  Avg annual change    {s['mean_yoy_abs']:+.4f}  children/woman")
    print(f"  Avg annual change    {s['mean_yoy_pct']:+.3f} %")
    print(f"  CAGR                 {s['cagr_pct']:+.3f} %")
    print("═" * 52 + "\n")


# ── Projection ─────────────────────────────────────────────────────────────────

def project(tfr: pd.Series, horizon: int = 80, base_margin: float = 0.10) -> pd.DataFrame:
    yoy       = tfr.diff().dropna()
    base_rate = yoy.mean()
    last_year  = int(tfr.index[-1])
    last_value = tfr.iloc[-1]
    rows = []
    for i, yr in enumerate(range(last_year + 1, last_year + horizon + 1), start=1):
        base   = max(0, last_value + base_rate * i)
        margin = base_margin * (1 + (i - 1) / (horizon - 1)) if horizon > 1 else base_margin
        rows.append({
            "year":   yr,
            "base":   base,
            "upper":  base * (1 + margin),
            "lower":  max(0, base * (1 - margin)),
            "margin": margin,
        })
    proj = pd.DataFrame(rows).set_index("year")
    log.info(
        "Projection to %d — base ends at %.3f  (margin %.0f%%→%.0f%%: band %.3f–%.3f)",
        last_year + horizon, proj["base"].iloc[-1],
        base_margin * 100, proj["margin"].iloc[-1] * 100,
        proj["lower"].iloc[-1], proj["upper"].iloc[-1],
    )
    return proj


# ── Era background bands ───────────────────────────────────────────────────────
# (start_year, end_year, hex_color, short_label)
ERAS = [
    (1960, 1967, "#4a90d9", "Early communism\n& baby boom"),
    (1968, 1973, "#c0392b", "Prague Spring\n& Soviet occupation"),
    (1974, 1989, "#8e44ad", "Husák\nnormalisation"),
    (1990, 2003, "#e67e22", "Post-communist\ntransition"),
    (2004, 2012, "#27ae60", "EU accession\n& prosperity"),
    (2013, 2019, "#f39c12", "Austerity\n& stagnation"),
    (2020, 2023, "#1abc9c", "COVID-19\n& rebound"),
]

# ── Point annotations ─────────────────────────────────────────────────────────
# (year, label, x_offset_pts, y_offset_pts)
# All offsets push INTO the chart — no labels escape the axes.
# Negative y_off = label below the dot; positive = above.
HISTORICAL_EVENTS = [
    (1963, "Baby boom peak",                                  -30,  +55),
    (1966, "Stat redefinition:\nrecorded births inflated,\nthen baseline correction",
                                                              -55, -110),
    (1967, "Stat anomaly resolves\n+ brief stability",        +65, -140),
    (1968, "Soviet invasion:\nPrague Spring crushed",         +50,  +55),
    (1974, "Husák pro-natalist\npolicy",                      -45,  +55),
    (1980, "Policy effect\ndecay begins",                     -50,  -85),
    (1990, "Velvet Revolution:\neconomic shock onset",        +55,  +55),
    (1999, "Transition nadir",                                +40,  -65),
    (2008, "EU boom echo +\nHusák cohort peak",               -50,  +55),
    (2021, "Post-COVID\nrebound",                             -50,  +55),
]

# ── Measurement-adjusted segments ─────────────────────────────────────────────
# Linear interpolation between anchor years — fills known distortion windows.
ADJUSTMENTS = [
    (1963, 1969),   # Stat redefinition artefact
    (1990, 1999),   # Transition registration chaos
    (2019, 2022),   # COVID birth postponement
]

REPLACEMENT_LEVEL = 2.1
COHORT_START, COHORT_END = 1999, 2015


# ── Chart ──────────────────────────────────────────────────────────────────────

def plot(tfr: pd.Series, proj: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(24, 9))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    last_year  = int(tfr.index[-1])
    first_year = int(tfr.index.min())
    proj_end   = int(proj.index.max())
    y_max      = tfr.max() * 1.12

    ax.set_ylim(0, y_max)
    ax.set_xlim(first_year - 1, proj_end + 1)

    # ── Era background bands — staggered labels ────────────────────────────────
    # Two alternating y positions so adjacent labels never collide.
    era_patches = []
    ERA_Y_POSITIONS = [y_max * 0.97, y_max * 0.88]   # high / low row

    for i, (start, end, color, label) in enumerate(ERAS):
        band_start = max(start, first_year)
        band_end   = min(end, last_year)
        if band_start >= band_end:
            continue
        ax.axvspan(band_start, band_end, color=color, alpha=0.10, zorder=0)
        mid   = (band_start + band_end) / 2
        row_y = ERA_Y_POSITIONS[i % 2]
        ax.text(
            mid, row_y, label,
            ha="center", va="top", fontsize=6.5, color=color,
            alpha=0.88, style="italic",
            bbox=dict(boxstyle="round,pad=0.18", fc="#0f1117", ec="none", alpha=0.65),
        )
        era_patches.append(Patch(facecolor=color, alpha=0.4, label=label.replace("\n", " ")))

    # ── Husák cohort echo band ─────────────────────────────────────────────────
    COHORT_COLOR = "#f0c060"
    ax.axvspan(COHORT_START, COHORT_END, color=COHORT_COLOR, alpha=0.07, zorder=0)
    for x in (COHORT_START, COHORT_END):
        ax.axvline(x, color=COHORT_COLOR, linewidth=0.7, linestyle=":", alpha=0.5, zorder=1)
    mid_cohort = (COHORT_START + COHORT_END) / 2
    cohort_label_y = 1.02
    ax.annotate(
        "",
        xy=(COHORT_END - 0.5,       cohort_label_y),
        xytext=(COHORT_START + 0.5, cohort_label_y),
        arrowprops=dict(arrowstyle="<->", color=COHORT_COLOR, lw=1.0),
    )
    ax.text(
        mid_cohort, cohort_label_y + y_max * 0.025,
        "Husák cohort (b.1974–80)\nin peak fertility age (25–35)",
        ha="center", va="bottom", fontsize=7, color=COHORT_COLOR,
        bbox=dict(boxstyle="round,pad=0.25", fc="#0f1117", ec="#6a5000", alpha=0.85),
    )
    era_patches.append(Patch(facecolor=COHORT_COLOR, alpha=0.35,
                             label="Husák cohort in peak fertility (1999–2015)"))

    # ── Replacement level ──────────────────────────────────────────────────────
    ax.axhline(REPLACEMENT_LEVEL, color="#666", linewidth=0.8, linestyle="--", zorder=1)
    ax.text(first_year + 1, REPLACEMENT_LEVEL + (y_max -1) * 0.015,
            "Replacement level (2.1)", color="#888", fontsize=8)

    # ── Error band & projection ────────────────────────────────────────────────
    ax.fill_between(proj.index, proj["lower"], proj["upper"],
                    color="#3a6fa8", alpha=0.22, label="±10–20 % error margin", zorder=1)
    ax.plot(tfr.index, tfr.values,
            color="#4fa3e0", linewidth=2.2, zorder=3, label="Historical TFR")
    ax.plot(proj.index, proj["base"],
            color="#4fa3e0", linewidth=1.6, linestyle="--", zorder=3, label="Base projection")

    # ── History / projection divider ──────────────────────────────────────────
    ax.axvline(last_year, color="#555", linewidth=0.8, linestyle="-", zorder=2)
    ax.text(last_year + 0.5, y_max * 0.03,
            "← Historical  |  Projected →", color="#777", fontsize=7.5)

    # ── Measurement-adjusted underlying trend lines ────────────────────────────
    adj_plotted = False
    for cs, ce in ADJUSTMENTS:
        if cs not in tfr.index or ce not in tfr.index:
            continue
        yrs  = list(range(cs, ce + 1))
        v_s, v_e = tfr[cs], tfr[ce]
        vals = [v_s + (v_e - v_s) * (i / (len(yrs) - 1)) for i in range(len(yrs))]
        lbl  = "underlying trend (measurement-adjusted)" if not adj_plotted else "_nolegend_"
        ax.plot(yrs, vals, color="#aaaaaa", linewidth=1.3,
                linestyle=(0, (4, 3)), zorder=3, alpha=0.75, label=lbl)
        adj_plotted = True

    # ── Historical event annotations ──────────────────────────────────────────
    # connectionstyle rad alternates sign to reduce leader-line crossings
    for idx, (yr, label, x_off, y_off) in enumerate(HISTORICAL_EVENTS):
        if yr not in tfr.index:
            yr = min(tfr.index, key=lambda y: abs(y - yr))
        val = tfr[yr]
        rad = 0.15 * (1 if idx % 2 == 0 else -1)
        ax.scatter(yr, val, color="#f0c060", s=28, zorder=5)
        ax.annotate(
            label,
            xy=(yr, val),
            xytext=(x_off, y_off), textcoords="offset points",
            ha="center", va="center", fontsize=6.5, color="#f0c060",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1e1a0e", ec="#6a5000", alpha=0.88),
            arrowprops=dict(arrowstyle="-", color="#6a5000", lw=0.9,
                            connectionstyle="arc3, rad=0.3"
                            # f"arc3,rad={rad}"
                            ),
            annotation_clip=False,
        )

    # ── Trend annotations ─────────────────────────────────────────────────────
    ax.annotate(
        "Slow policy decay\n(incentives adapt, intentions don't)",
        xy=(1986, tfr.iloc[tfr.index.get_indexer([1986], method='nearest')[0]]),
        xytext=(1981, tfr.min() * 0.78),
        fontsize=6.5, color="#cc88ff", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="#110022", ec="#553388", alpha=0.85),
        arrowprops=dict(arrowstyle="->", color="#553388", lw=0.9,
                        connectionstyle="arc3,rad=0.2"),
    )

    yr_start, yr_end = 1990, 1999
    if yr_start in tfr.index and yr_end in tfr.index:
        ax.annotate(
            "",
            xy=(yr_end, tfr[yr_end]), xytext=(yr_start, tfr[yr_start]),
            arrowprops=dict(arrowstyle="->", color="#e05c5c", lw=1.5,
                            connectionstyle="arc3,rad=-0.15"),
            zorder=4,
        )
        ax.text(
            (yr_start + yr_end) / 2 - 1,
            (tfr[yr_start] + tfr[yr_end]) / 2 + 0.08,
            "Steepest peacetime\nfertility collapse\nin European history",
            ha="center", fontsize=6.5, color="#e05c5c",
            bbox=dict(boxstyle="round,pad=0.25", fc="#1a0505", ec="#7a2222", alpha=0.85),
        )

    # ── Projection intersect markers every 5 years ────────────────────────────
    proj_5yr = proj["base"].loc[proj.index % 5 == 0]
    for i, (yr, val) in enumerate(proj_5yr.items()):
        if val <= 0 or val > y_max:
            continue
        offset_y = 14 if i % 2 == 0 else -20
        ax.plot([yr, yr], [0, val], color="#4fa3e0", linewidth=0.4,
                linestyle=":", alpha=0.35, zorder=2)
        ax.plot([first_year, yr], [val, val], color="#4fa3e0", linewidth=0.4,
                linestyle=":", alpha=0.35, zorder=2)
        ax.scatter(yr, val, color="#4fa3e0", s=22, zorder=4)
        ax.annotate(
            f"{yr}\n{val:.2f}",
            xy=(yr, val), xytext=(0, offset_y), textcoords="offset points",
            ha="center", fontsize=7, color="#c8dff0",
            bbox=dict(boxstyle="round,pad=0.2", fc="#1a2a3a", ec="none", alpha=0.7),
        )

    # ── X-axis every 5 years ──────────────────────────────────────────────────
    x_start = int(np.floor(first_year / 5) * 5)
    x_end   = int(np.ceil(proj_end    / 5) * 5)
    ax.set_xticks(range(x_start, x_end + 1, 5))
    ax.tick_params(axis="x", rotation=45)

    # ── Styling ───────────────────────────────────────────────────────────────
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("Year", color="#aaa", fontsize=10)
    ax.set_ylabel("Children per woman", color="#aaa", fontsize=10)
    ax.set_title(
        "Czech Republic — Total Fertility Rate\nHistorical trend & projection to 2100",
        color="white", fontsize=13, pad=14,
    )

    # Two-part legend
    line_handles, line_labels = ax.get_legend_handles_labels()
    era_legend = ax.legend(
        handles=era_patches,
        title="Historical eras", title_fontsize=7,
        loc="upper right",
        framealpha=0.3, facecolor="#1a1a2e", edgecolor="#333",
        labelcolor="white", fontsize=7,
    )
    ax.add_artist(era_legend)
    ax.legend(
        handles=line_handles, labels=line_labels,
        loc="lower right",
        framealpha=0.3, facecolor="#1a1a2e", edgecolor="#333",
        labelcolor="white", fontsize=8.5,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    log.info("Chart saved → %s", out_path)
    plt.show()


# ── Save projection CSV ────────────────────────────────────────────────────────

def save_projection(proj: pd.DataFrame, out_path: Path) -> None:
    proj.round(4).to_csv(out_path)
    log.info("Projection CSV saved → %s", out_path)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tfr   = load_tfr()
    stats = compute_stats(tfr)
    print_stats(stats)
    proj  = project(tfr)

    chart_path = OUT_DIR / "fertility_rate_CZ_chart.png"
    proj_path  = OUT_DIR / "fertility_rate_CZ_projection.csv"

    plot(tfr, proj, out_path=chart_path)
    save_projection(proj, out_path=proj_path)
