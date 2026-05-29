"""
RAG evaluation with real Q&A pairs from SWR Station Disruption Plan documents.

Questions and expected answers are taken word-for-word from:
  - Portsmouth Harbour Issue 2 (January 2021)
  - Woking Issue 1 (April 2018)
  - Milford Issue 2 (2020)
  - Surbiton Issue 1 (April 2018)
  - Ryde Pier Head Issue 2 (January 2021)

Run:
    python tests/eval_rag.py              # retrieval metrics only
    python tests/eval_rag.py --judge      # also score answers with LLM
    python tests/eval_rag.py --plot       # save slide-ready plots to models/task3/plots/
"""

import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tasks.task3.retriever import retrieve
from tasks.task3.llm import answer_contingency_query

# ---------------------------------------------------------------------------
# Test set - 15 Q&A pairs extracted directly from the documents
# keywords are distinctive phrases from the expected answer used for
# retrieval relevance checking (Hit@5 / MRR).
# ---------------------------------------------------------------------------

TEST_SET = [
    {
        "query": "How is additional staff support requested during disruption?",
        "expected": (
            "Additional support will be requested via the Level 2 On Call Manager "
            "through control."
        ),
        "keywords": ["level 2 on call manager", "additional support"],
    },
    {
        "query": "How should volunteers be briefed when they arrive at the station?",
        "expected": (
            "Volunteers are briefed on their arrival using the Site Specific Brief "
            "for the location along with any other local information, the nature of "
            "the disruption and any alternative travel arrangements that have been "
            "put in place."
        ),
        "keywords": ["site specific brief", "volunteer", "briefed"],
    },
    {
        "query": "What should be done for colleagues and customers during hot weather disruption?",
        "expected": (
            "If prolonged disruption occurs during hot weather, it will be necessary "
            "for customer and colleague welfare to provide water at key locations. "
            "This should be arranged via station colleagues and through the on call structure."
        ),
        "keywords": ["hot weather", "water", "key locations"],
    },
    {
        "query": "Who is responsible for making sure colleagues get breaks during prolonged disruption?",
        "expected": (
            "It is the responsibility of the Level 1 On Call Manager to liaise with "
            "the station manager to ensure that this is being achieved."
        ),
        "keywords": ["level 1 on call manager", "breaks", "station manager"],
    },
    {
        "query": "What should a colleague do if they feel fatigued during a shift?",
        "expected": (
            "If you feel the effects of fatigue you should highlight this immediately "
            "so cover for the role you are performing can be arranged."
        ),
        "keywords": ["fatigue", "highlight", "cover"],
    },
    {
        "query": "How should staff handle a frustrated or aggressive customer?",
        "expected": (
            "Remain calm, polite and respectful at all times. If required you should "
            "politely excuse yourself and make your way to a position of safety and "
            "request assistance, which may be from a colleague in the first instance "
            "or BTP if appropriate."
        ),
        "keywords": ["calm", "polite", "position of safety", "btp"],
    },
    {
        "query": "How quickly should the initial disruption message be sent after an incident?",
        "expected": (
            "SWR aims to send an initial message within 10 minutes of any incident "
            "that takes place."
        ),
        "keywords": ["10 minutes", "initial message"],
    },
    {
        "query": "What 3 pieces of information are included in every SWR disruption message?",
        "expected": (
            "Every message shows: (1) the cause of disruption - what the problem is "
            "and where it is; (2) the impact - what this means for the journey; "
            "(3) the advice - how to continue the journey."
        ),
        "keywords": ["cause of disruption", "impact", "advice", "3 key"],
    },
    {
        "query": "What system does SWR use to update the JourneyCheck website during disruption?",
        "expected": (
            "The Tyrell system updates SWR's own JourneyCheck site, ensuring customers "
            "can get the latest information about an incident."
        ),
        "keywords": ["tyrell", "journeycheck"],
    },
    {
        "query": "What should staff tell customers when there is no information available yet?",
        "expected": (
            "If there is no information to provide at the time, this should be relayed "
            "to customers so they are kept up to date."
        ),
        "keywords": ["no information", "relayed", "kept up to date"],
    },
    {
        "query": "Should gatelines be kept open or closed during disruption?",
        "expected": (
            "Gatelines should be kept in operation at all times unless there is an "
            "emergency, as they provide an effective way of managing the flow of "
            "people in and out of stations."
        ),
        "keywords": ["gateline", "kept in operation", "unless there is an emergency"],
    },
    {
        "query": "What must be done if a gateline needs to be opened during disruption?",
        "expected": (
            "If the gate does need to be opened it must be recorded in the gateline log."
        ),
        "keywords": ["gateline log", "recorded", "opened"],
    },
    {
        "query": "What should I do if passenger information screens are showing wrong information?",
        "expected": (
            "Make your local SCP aware - they will be able to help by updating the "
            "screens. If you do not have an SCP, contact the local information "
            "controllers in the WICC."
        ),
        "keywords": ["scp", "wicc", "screens", "updating"],
    },
    {
        "query": "What is the role of the Level 1 On Call Manager?",
        "expected": (
            "The Level 1 On Call Manager is responsible for the area the station is "
            "located in. They keep an overview of the whole area, pass on information "
            "as required, and have direct contact with the Level 2 On Call Manager "
            "for escalation purposes."
        ),
        "keywords": ["level 1 on call manager", "overview", "escalation"],
    },
    {
        "query": "What is the Duty Resource Manager responsible for?",
        "expected": (
            "The Duty Resource Manager is responsible for managing train crews through "
            "normal service and during disruption, including allocating any reserve "
            "crew members to services during disruption."
        ),
        "keywords": ["duty resource manager", "train crews", "reserve crew"],
    },
]

THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45]
TOP_K = 5


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def is_relevant(chunk: dict, keywords: list[str]) -> bool:
    text = (chunk.get("chunk_text", "") + " " + chunk.get("section", "")).lower()
    return any(kw.lower() in text for kw in keywords)


def eval_retrieval(threshold: float, collect_per_query: bool = False) -> dict:
    hits, reciprocal_ranks, no_results = 0, 0.0, 0
    per_query = []

    for item in TEST_SET:
        chunks = retrieve(item["query"], top_k=TOP_K, min_score=threshold)

        if not chunks:
            no_results += 1
            if collect_per_query:
                per_query.append({"query": item["query"], "rank": None, "score": 0.0})
            continue

        hit = False
        for rank, chunk in enumerate(chunks, start=1):
            if is_relevant(chunk, item["keywords"]):
                if not hit:
                    reciprocal_ranks += 1 / rank
                    hit = True
                    if collect_per_query:
                        per_query.append({
                            "query": item["query"],
                            "rank":  rank,
                            "score": chunk["score"],
                        })
        if hit:
            hits += 1
        elif collect_per_query:
            per_query.append({"query": item["query"], "rank": None, "score": 0.0})

    n = len(TEST_SET)
    result = {
        "threshold":   threshold,
        "hit@5":       hits / n * 100,
        "mrr":         reciprocal_ranks / n,
        "no_result_%": no_results / n * 100,
    }
    if collect_per_query:
        result["per_query"] = per_query
    return result


# ---------------------------------------------------------------------------
# Slide plots
# ---------------------------------------------------------------------------

PLOT_DIR = _ROOT / "models" / "task3" / "plots"

def save_plots(per_query: list[dict], summary: dict) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    SWR_RED   = "#C8102E"
    SWR_DARK  = "#1A1A2E"
    GREY      = "#E8E8E8"
    HIT_COLOR = SWR_RED
    MISS_COLOR = "#AAAAAA"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    # --- Plot 1: Rank distribution bar chart ---
    ranks = [p["rank"] for p in per_query]
    rank_counts = {r: ranks.count(r) for r in [1, 2, 3, 4, 5]}
    miss_count  = ranks.count(None)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    labels = ["Rank 1", "Rank 2", "Rank 3", "Rank 4", "Rank 5", "Miss"]
    values = [rank_counts.get(i, 0) for i in range(1, 6)] + [miss_count]
    colors = [HIT_COLOR if v > 0 and i < 5 else (MISS_COLOR if i == 5 else GREY)
              for i, v in enumerate(values)]
    colors = [HIT_COLOR] * 5 + [MISS_COLOR]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5, width=0.6)

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    str(val), ha="center", va="bottom", fontsize=13, fontweight="bold",
                    color=SWR_DARK)

    ax.set_ylim(0, max(values) + 2)
    ax.set_ylabel("Number of queries", fontsize=11, color=SWR_DARK)
    ax.set_title("Where does the right answer appear? (Top-5 retrieval, 15 queries)",
                 fontsize=13, fontweight="bold", color=SWR_DARK, pad=14)
    ax.tick_params(colors=SWR_DARK)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    hit_patch  = mpatches.Patch(color=HIT_COLOR,  label=f"Hit  ({sum(v for v in values[:5])} queries)")
    miss_patch = mpatches.Patch(color=MISS_COLOR, label=f"Miss ({miss_count} queries)")
    ax.legend(handles=[hit_patch, miss_patch], fontsize=10, framealpha=0)

    plt.tight_layout()
    out1 = PLOT_DIR / "eval_rank_distribution.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out1}")

    # --- Plot 2: Per-query similarity scores, coloured by rank ---
    short_labels = [p["query"][:48] + "…" if len(p["query"]) > 48 else p["query"]
                    for p in per_query]
    scores = [p["score"] for p in per_query]
    bar_colors = [HIT_COLOR if p["rank"] == 1 else
                  "#E07070"  if p["rank"] in (2, 3) else
                  "#F0A0A0"  if p["rank"] in (4, 5) else
                  MISS_COLOR
                  for p in per_query]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("white")

    y_pos = np.arange(len(per_query))
    ax.barh(y_pos, scores, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_labels, fontsize=8.5, color=SWR_DARK)
    ax.invert_yaxis()
    ax.set_xlabel("Cosine similarity score of first relevant chunk", fontsize=10, color=SWR_DARK)
    ax.set_title("Per-query retrieval scores", fontsize=13, fontweight="bold",
                 color=SWR_DARK, pad=14)
    ax.axvline(0.35, color=SWR_DARK, linestyle="--", linewidth=1, alpha=0.4,
               label="Min-score threshold (0.35)")
    ax.set_xlim(0, 1.05)
    ax.tick_params(colors=SWR_DARK)

    r1 = mpatches.Patch(color=HIT_COLOR,  label="Rank 1")
    r23= mpatches.Patch(color="#E07070",  label="Rank 2–3")
    r45= mpatches.Patch(color="#F0A0A0",  label="Rank 4–5")
    ms = mpatches.Patch(color=MISS_COLOR, label="Miss")
    ax.legend(handles=[r1, r23, r45, ms], fontsize=9, framealpha=0, loc="lower right")

    plt.tight_layout()
    out2 = PLOT_DIR / "eval_per_query_scores.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out2}")

    # --- Plot 3: Summary metrics card ---
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.patch.set_facecolor("white")

    metrics = [
        ("Hit@5", f"{summary['hit@5']:.0f}%",   "Queries with a\nrelevant result in top 5"),
        ("MRR",   f"{summary['mrr']:.3f}",       "Mean Reciprocal Rank\n(1.0 = always rank 1)"),
        ("Rank 1\nRate", f"{rank_counts.get(1,0)}/{len(per_query)}",
                                                 "Queries where the best\nresult is ranked 1st"),
    ]

    for ax, (title, value, subtitle) in zip(axes, metrics):
        ax.set_facecolor(SWR_RED)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.5, 0.72, value,    ha="center", va="center", fontsize=34,
                fontweight="bold", color="white", transform=ax.transAxes)
        ax.text(0.5, 0.38, title,    ha="center", va="center", fontsize=13,
                fontweight="bold", color="white", alpha=0.9, transform=ax.transAxes)
        ax.text(0.5, 0.13, subtitle, ha="center", va="center", fontsize=8,
                color="white", alpha=0.75, transform=ax.transAxes)

    fig.suptitle("Task 3 - RAG Retrieval Performance  (15 real staff queries)",
                 fontsize=12, fontweight="bold", color=SWR_DARK, y=1.02)
    plt.tight_layout()
    out3 = PLOT_DIR / "eval_summary_card.png"
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out3}")


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def judge_answer(query: str, expected: str, actual: str) -> int:
    """Ask the LLM to score the actual answer vs the expected answer (1-5)."""
    from llm.client import chat_text
    prompt = (
        "You are evaluating a RAG system answer.\n\n"
        f"Question: {query}\n\n"
        f"Expected answer (from the document): {expected}\n\n"
        f"System answer: {actual}\n\n"
        "Score the system answer from 1 to 5:\n"
        "5 = covers the expected answer fully and accurately\n"
        "4 = mostly correct, minor omission\n"
        "3 = partially correct\n"
        "2 = barely relevant\n"
        "1 = wrong or missing\n\n"
        "Reply with a single digit only."
    )
    try:
        result = chat_text([{"role": "user", "content": prompt}]).strip()
        return int(result[0]) if result and result[0].isdigit() else 0
    except Exception:
        return 0


def eval_answers() -> list[dict]:
    results = []
    for i, item in enumerate(TEST_SET, 1):
        print(f"  [{i}/{len(TEST_SET)}] {item['query'][:60]}...")
        answer = answer_contingency_query(item["query"])
        if isinstance(answer, tuple):
            answer = answer[0]
        score = judge_answer(item["query"], item["expected"], answer)
        results.append({
            "query":    item["query"],
            "expected": item["expected"],
            "answer":   answer[:300],
            "score":    score,
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true",
                        help="Also generate and score answers with the LLM")
    parser.add_argument("--plot", action="store_true",
                        help="Save slide-ready plots to models/task3/plots/")
    args = parser.parse_args()

    # --- Retrieval metrics ---
    print(f"Evaluating retrieval for {len(TEST_SET)} queries...\n")
    retrieval_results = [eval_retrieval(t, collect_per_query=(t == 0.35))
                         for t in THRESHOLDS]

    col = f"{'Threshold':>10} {'Hit@5 (%)':>10} {'MRR':>8} {'No-result (%)':>14}"
    print(col)
    print("-" * len(col))
    for r in retrieval_results:
        marker = "  <-- current" if r["threshold"] == 0.35 else ""
        print(f"{r['threshold']:>10.2f} {r['hit@5']:>10.1f} "
              f"{r['mrr']:>8.3f} {r['no_result_%']:>14.1f}{marker}")

    best = max(retrieval_results, key=lambda r: r["mrr"])
    print(f"\nBest MRR at threshold={best['threshold']:.2f}  "
          f"(Hit@5={best['hit@5']:.1f}%  No-result={best['no_result_%']:.1f}%)")

    # --- Plots ---
    if args.plot:
        print("\nGenerating slide plots...")
        current = next(r for r in retrieval_results if r["threshold"] == 0.35)
        save_plots(current["per_query"], current)

    # --- LLM answer judge ---
    if args.judge:
        print("\n" + "=" * 60)
        print("Scoring generated answers with LLM judge...")
        print("=" * 60 + "\n")
        answer_results = eval_answers()

        print(f"\n{'#':<4} {'Score':>5}  Query")
        print("-" * 60)
        for i, r in enumerate(answer_results, 1):
            print(f"{i:<4} {r['score']:>5}/5  {r['query'][:55]}")

        scored = [r for r in answer_results if r["score"] > 0]
        if scored:
            avg = sum(r["score"] for r in scored) / len(scored)
            print(f"\nAverage score: {avg:.2f}/5  ({len(scored)}/{len(TEST_SET)} answered)")
