"""
Generate Task 3 KB plots. Run from repo root: python scripts/generate_kb_plots.py

Outputs in models/task3/plots/:
  kb_chunks_per_station.png   chunks per station (bar)
  kb_region_distribution.png  chunks by region (pie)
  kb_score_distribution.png   cosine similarity scores for sample queries (hist)
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from db import get_conn, put_conn

PLOT_DIR = REPO_ROOT / "models" / "task3" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_QUERIES = [
    "fire evacuation procedure",
    "train overrun at platform",
    "passenger taken ill on train",
    "signal failure contingency",
    "level crossing incident",
    "trespass on the line",
    "fatality on track",
    "severe weather disruption",
    "power failure at station",
    "flooding on the line",
]

PALETTE = "#2c7bb6"
ACCENT  = "#d7191c"


# helpers

def _query(sql: str, params=()) -> list[tuple]:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
        return rows
    finally:
        put_conn(conn)


# Plot 1: chunks per station

def plot_chunks_per_station():
    rows = _query(
        "SELECT station, COUNT(*) FROM documents GROUP BY station ORDER BY COUNT(*) DESC"
    )
    if not rows:
        print("  No documents found, skipping.")
        return

    stations = [r[0] for r in rows]
    counts   = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(max(10, len(stations) * 0.55), 5))
    bars = ax.bar(stations, counts, color=PALETTE, edgecolor="white", width=0.7)

    ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
    ax.set_title("Knowledge Base: Document Chunks per Station", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Station", fontsize=10)
    ax.set_ylabel("Number of Chunks", fontsize=10)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.xticks(rotation=45, ha="right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = PLOT_DIR / "kb_chunks_per_station.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


# Plot 2: region distribution

def plot_region_distribution():
    rows = _query(
        "SELECT COALESCE(region, 'Unknown'), COUNT(*) FROM documents GROUP BY region ORDER BY COUNT(*) DESC"
    )
    if not rows:
        print("  No documents found, skipping.")
        return

    labels = [r[0] for r in rows]
    sizes  = [r[1] for r in rows]

    cmap   = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        pctdistance=0.82,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(9)

    ax.legend(
        wedges, labels,
        title="Region",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=9,
    )
    ax.set_title("Knowledge Base: Chunks by Region", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    out = PLOT_DIR / "kb_region_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# Plot 3: retrieval score distribution

def plot_score_distribution():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  sentence-transformers not available, skipping score plot.")
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")
    all_scores = []

    for q in SAMPLE_QUERIES:
        emb = model.encode(q).tolist()
        rows = _query(
            "SELECT 1 - (embedding <=> %s::vector) AS score FROM documents ORDER BY embedding <=> %s::vector LIMIT 5",
            (emb, emb),
        )
        all_scores.extend([float(r[0]) for r in rows])

    if not all_scores:
        print("  No scores returned, skipping.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_scores, bins=20, color=PALETTE, edgecolor="white", alpha=0.9)
    ax.axvline(np.mean(all_scores), color=ACCENT, linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(all_scores):.3f}")
    ax.set_title("RAG Retrieval: Cosine Similarity Score Distribution\n(top-5 results, 10 sample queries)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Cosine Similarity Score", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = PLOT_DIR / "kb_score_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


# main

if __name__ == "__main__":
    print("Generating Task 3 KB plots …\n")

    print("[1/3] Chunks per station …")
    plot_chunks_per_station()

    print("[2/3] Region distribution …")
    plot_region_distribution()

    print("[3/3] Retrieval score distribution …")
    plot_score_distribution()

    print(f"\nDone. Plots saved to {PLOT_DIR}/")
