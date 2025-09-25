"""Plot summary graphs for question_type metrics.

Generates:
 1. Donut chart: overall answerable vs impossible counts.
 2. Donut chart: per-category share (by total_records).

Optional (added) 3. Horizontal bar chart comparing F1 per question_type.

Removes dependency on notebook-only display tools and writes files to an
output directory (default: ./plots). Use: python plot_graph.py [output_dir]
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------
# 1) Input data (from the user message)
# ---------------------------
data = [
    # question_type, hit@1, exact_match, precision, recall, f1, impossible_accuracy, agg_metrics, total_answerable, total_impossible, total_records
    ("back_court", 87.10, 84.95, 89.77, 91.58, 90.49, 80.56, 90.49, 93, 108, 201),
    ("bottom_player_counter_attack", 87.32, 85.92, 86.38, 86.38, 86.38, 92.96, 86.38, 71, 71, 142),
    ("flat_shot_sequence", 88.89, 55.56, 86.11, 85.19, 84.63, 100.00, 84.63, 18, 18, 36),
    ("four_corner", 90.00, 90.00, 98.00, 100.00, 98.89, 100.00, 98.89, 10, 12, 22),
    ("hit_area_only", 88.03, 59.08, 87.32, 83.11, 83.77, 79.08, 83.77, 1195, 502, 1697),
    ("net_shot", 79.43, 75.00, 80.49, 80.26, 80.12, 93.68, 80.12, 316, 190, 506),
    ("player_stroke", 92.49, 86.78, 92.65, 91.33, 91.49, 96.21, 91.49, 1331, 501, 1832),
    ("player_stroke_area", 85.33, 81.90, 85.55, 85.11, 84.96, 97.33, 84.96, 1254, 487, 1741),
    ("stroke_only", 92.76, 82.24, 92.64, 91.80, 91.51, 93.29, 91.51, 1216, 507, 1723),
    ("upper_player_counter_attack", 84.51, 84.51, 85.45, 85.45, 85.45, 92.19, 85.45, 71, 64, 135),
]

df = pd.DataFrame(
    data,
    columns=[
        "question_type",
        "hit@1",
        "exact_match",
        "precision",
        "recall",
        "f1",
        "impossible_accuracy",
        "agg_metrics",
        "total_answerable",
        "total_impossible",
        "total_records",
    ],
)

# Totals from the user's global metrics
total_answerable_overall = 5575
total_impossible_overall = 2460
total_records_overall = 8035

# ---------------------------
# 2) Donut chart: total_answerable vs total_impossible
# ---------------------------
def make_overall_donut(out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    sizes = [total_answerable_overall, total_impossible_overall]
    labels = ["Answerable", "Impossible"]

    def autopct_factory(values):
        total = sum(values)
        def _fmt(pct):
            # Raw count (rounded) instead of percentage
            val = int(round(pct * total / 100.0))
            return f"{val}"
        return _fmt

    ax.pie(sizes, labels=labels, startangle=90, autopct=autopct_factory(sizes))
    centre_circle = plt.Circle((0, 0), 0.60, fc="white")
    ax.add_artist(centre_circle)
    ax.axis("equal")
    ax.set_title("Answerable vs Impossible (Overall)")
    path = os.path.join(out_dir, "donut_overall_answerable_impossible.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path

# ---------------------------
# 3) Donut chart: per-category share of all queries
# ---------------------------
# Compute share by total_records
def make_category_share_donut(out_dir: str) -> str:
    """Create a donut chart with a side legend (color mapping) to keep the plot clean."""
    category_shares = df[["question_type", "total_records"]].copy()
    category_shares["share"] = category_shares["total_records"] / total_records_overall

    # Define a consistent color mapping (extend / reuse if more categories appear)
    base_colors = plt.get_cmap("tab20").colors
    colors = [base_colors[i % len(base_colors)] for i in range(len(category_shares))]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, _ = ax.pie(
        category_shares["total_records"],
        labels=None,  # no labels directly on wedges
        startangle=90,
        counterclock=False,
        colors=colors,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    # Donut hole
    centre_circle = plt.Circle((0, 0), 0.55, fc="white")
    ax.add_artist(centre_circle)
    ax.axis("equal")
    ax.set_title("Per-Category Share of All Queries")

    # Build legend labels with percentage
    legend_labels = [
        f"{row.question_type}  {row.total_records}" for row in category_shares.itertuples()
    ]
    ax.legend(
        wedges,
        legend_labels,
        title="question_type",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )

    path = os.path.join(out_dir, "donut_category_share.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path


def make_f1_bar(out_dir: str) -> str:
    sort_df = df.sort_values("f1", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(sort_df["question_type"], sort_df["f1"], color="#4C72B0")
    ax.invert_yaxis()
    ax.set_xlabel("F1 (%)")
    ax.set_title("F1 by Question Type")
    for i, (val) in enumerate(sort_df["f1" ]):
        ax.text(val + 0.3, i, f"{val:.2f}", va="center", fontsize=8)
    path = os.path.join(out_dir, "bar_f1_by_question_type.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path

# Prepare a small table for reference (optional to display)
def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "plots"
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "overall_donut": make_overall_donut(out_dir),
        "category_share_donut": make_category_share_donut(out_dir),
        "f1_bar": make_f1_bar(out_dir),
    }
    # Also write a CSV for reference
    csv_path = os.path.join(out_dir, "question_type_metrics.csv")
    df.to_csv(csv_path, index=False)
    paths["metrics_csv"] = csv_path
    print("Saved files:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
