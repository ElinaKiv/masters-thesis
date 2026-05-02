"""
Error Analysis and Visualization for Whisper--Gold Comparisons.

This module analyzes word-level alignment results from the Whisper--gold
matching pipeline. It aggregates error counts and error rates across
multiple Whisper model sizes and outputs both CSV summaries and
publication-ready visualizations.

The analysis:
- Processes aligned JSON files containing gold and Whisper annotations
- Computes word counts and error frequencies per model
- Calculates per-category and total error rates
- Exports results as CSV files
- Generates comparative plots (bar charts, stacked bars, pie/donut charts)

The module assumes input data follows the alignment pipeline schema
(sections → turns → words with per-model metadata) and that mismatches
may carry multiple error categories.
"""

import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import csv


# ---------------------------------------------------
# Configuration
# ---------------------------------------------------

# Directory containing processed Whisper–gold JSON files
input_folder = "final_whisper_and_gold_transcriptions"

# Directory where CSVs and figures will be saved
output_folder = "analysis_output_test"

# Whisper model sizes to include in all analyses
MODELS = ["tiny", "medium", "large-v3"]


# ---------------------------------------------------
# Matplotlib global styling
# ---------------------------------------------------

# Global font scaling parameter for figures (1.0 = default)
FONT_SCALE = 1.4   # Adjust once to scale all text uniformly

base_size = 10  # Matplotlib default font size

# Apply consistent font scaling to all plots
plt.rcParams.update({
    "font.size": base_size * FONT_SCALE,
    "axes.titlesize": base_size * 1.2 * FONT_SCALE,
    "axes.labelsize": base_size * 1.1 * FONT_SCALE,
    "xtick.labelsize": base_size * 1.0 * FONT_SCALE,
    "ytick.labelsize": base_size * 1.0 * FONT_SCALE,
    "legend.fontsize": base_size * 1.0 * FONT_SCALE
})


# ---------------------------------------------------
# Error counting logic
# ---------------------------------------------------

def count_error_categories_single(data):
    """
    Count error categories and total word counts for a single JSON file.

    Parameters:
        data (dict): Parsed JSON data containing sections, turns, and words.

    Returns:
        tuple:
            - error_counts (dict): {model -> {error_category -> count}}
            - word_counts (dict): {model -> total words}

    Notes:
        Word counting uses a corrected rule:
        a word is skipped only if gold, segment_id, and word_start
        are ALL None. This ensures fair normalization across models.
    """
    error_counts = {model: defaultdict(int) for model in MODELS}
    word_counts = {model: 0 for model in MODELS}

    for section in data.get("sections", []):
        for turn in section.get("turns", []):
            for w in turn.get("words", []):

                gold = w.get("gold")

                # ---------------------------------
                # WORD COUNTING
                # ---------------------------------
                for model in MODELS:
                    model_info = w.get(model, {})
                    segment_id = model_info.get("segment_id")
                    word_start = model_info.get("word_start")

                    # Skip only if all alignment evidence is missing
                    should_skip = (
                        gold is None and
                        segment_id is None and
                        word_start is None
                    )

                    if not should_skip:
                        word_counts[model] += 1

                # ---------------------------------
                # ERROR COUNTING
                # ---------------------------------
                for model in MODELS:
                    info = w.get(model, {})
                    if info.get("status") == "mismatch":
                        categories = info.get("error_category")

                        # Error categories may be a list or a single string
                        if isinstance(categories, list):
                            for c in categories:
                                error_counts[model][c] += 1
                        elif isinstance(categories, str):
                            error_counts[model][categories] += 1

    return error_counts, word_counts


def merge_counts(total_errors, total_words, file_errors, file_words):
    """
    Merge per-file error and word counts into corpus-level totals.

    Parameters:
        total_errors (dict): Accumulated error counts.
        total_words (dict): Accumulated word counts.
        file_errors (dict): Error counts from one file.
        file_words (dict): Word counts from one file.
    """
    for model in MODELS:
        total_words[model] += file_words[model]
        for cat, n in file_errors[model].items():
            total_errors[model][cat] += n


def analyze_folder(folder_path):
    """
    Analyze all processed JSON files in a folder.

    Parameters:
        folder_path (str): Path to directory containing JSON files.

    Returns:
        tuple:
            - total_errors (dict): Aggregated error counts per model.
            - total_words (dict): Aggregated word counts per model.
    """
    total_errors = {model: defaultdict(int) for model in MODELS}
    total_words = {model: 0 for model in MODELS}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            file_errors, file_words = count_error_categories_single(data)
            merge_counts(total_errors, total_words, file_errors, file_words)

    return total_errors, total_words


# ---------------------------------------------------
# CSV export functions
# ---------------------------------------------------

def export_csv_error_counts(error_counts, output_file):
    """
    Export raw error counts per category and model to CSV.
    """
    all_categories = sorted({c for model in MODELS for c in error_counts[model].keys()})

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Error Category"] + MODELS)

        for category in all_categories:
            row = [category] + [error_counts[model].get(category, 0) for model in MODELS]
            writer.writerow(row)


def export_csv_error_rates(error_counts, word_counts, output_file):
    """
    Export error rates (errors / word) per category and model to CSV.
    """
    all_categories = sorted({c for model in MODELS for c in error_counts[model].keys()})

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Error Category"] + MODELS)

        for category in all_categories:
            row = [category]
            for model in MODELS:
                total_words = word_counts[model] if word_counts[model] > 0 else 1
                rate = error_counts[model].get(category, 0) / total_words
                row.append(rate)
            writer.writerow(row)


def export_csv_total_error_rates(error_counts, word_counts, output_file):
    """
    Export total error rate per model to CSV.
    """
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Total Error Rate"])

        for model in MODELS:
            total_err = sum(error_counts[model].values())
            total_words = word_counts[model] or 1
            total_rate = total_err / total_words
            writer.writerow([model, total_rate])


# ---------------------------------------------------
# Plotting functions
# ---------------------------------------------------

def plot_error_counts(error_counts, save_path):
    """
    Plot absolute error counts per category and model (grouped bars).
    """
    all_categories = sorted({c for model in MODELS for c in error_counts[model].keys()})
    x = np.arange(len(all_categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(MODELS):
        counts = [error_counts[model].get(cat, 0) for cat in all_categories]
        ax.bar(x + i * width, counts, width, label=model)

    ax.set_xlabel("Error category")
    ax.set_ylabel("Count")
    ax.set_xticks(x + width)
    ax.set_xticklabels(all_categories, rotation=45)
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_error_rates(error_counts, word_counts, save_path):
    """
    Plot error rates (percentage) per category and model.
    """
    all_categories = sorted({c for model in MODELS for c in error_counts[model].keys()})
    x = np.arange(len(all_categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(MODELS):
        rates = [
            error_counts[model].get(cat, 0) / (word_counts[model] or 1) * 100
            for cat in all_categories
        ]
        ax.bar(x + i * width, rates, width, label=model)

    ax.set_xlabel("Error category")
    ax.set_ylabel("Error rate (errors / word), %")
    ax.set_xticks(x + width)
    ax.set_xticklabels(all_categories, rotation=45)
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_total_error_rates(error_counts, word_counts, save_path):
    """
    Plot total error rate per model.
    """
    rates = {}
    for model in MODELS:
        total_err = sum(error_counts[model].values())
        total_words = word_counts[model] or 1
        rates[model] = total_err / total_words * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rates.keys(), rates.values(), color=["#1f77b4", "#ff7f0e", "#2ca02c"])

    ax.set_xlabel("Model")
    ax.set_ylabel("Total error rate (errors / word), %")
    ax.set_ylim(0, max(rates.values()) * 1.2)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_single_model_error_rates(model, error_counts, word_counts, save_path):
    """
    Plot per-category error rates for a single model.
    """
    categories = sorted(error_counts[model].keys())
    rates = [
        error_counts[model].get(cat, 0) / (word_counts[model] or 1) * 100
        for cat in categories
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(categories, rates, color="#1f77b4")

    ax.set_xlabel("Error category")
    ax.set_ylabel("Error rate (errors / word), %")
    ax.set_xticklabels(categories, rotation=45)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_all_single_models(error_counts, word_counts, output_folder):
    """
    Generate separate error-rate bar charts for each model.
    """
    for model in MODELS:
        save_path = os.path.join(output_folder, f"{model}_error_rates.png")
        plot_single_model_error_rates(model, error_counts, word_counts, save_path)


def plot_error_rate_pies(error_counts, word_counts, output_folder):
    """
    Create one pie chart per model showing proportional error distribution
    across categories.
    """
    for model in MODELS:
        categories = sorted(error_counts[model].keys())
        rates = [
            error_counts[model].get(cat, 0) / (word_counts[model] or 1)
            for cat in categories
        ]

        if sum(rates) == 0:
            print(f"Skipping {model} (no errors).")
            continue

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            rates,
            labels=categories,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.8
        )

        plt.tight_layout()
        fig.savefig(os.path.join(output_folder, f"{model}_error_rate_pie.png"), dpi=300)
        plt.close(fig)


def plot_combined_error_rate_pies(error_counts, word_counts, output_path):
    """
    Create a combined donut-style pie chart figure with one subplot per model
    and a shared legend across all error categories.
    """
    all_categories = sorted({c for m in MODELS for c in error_counts[m].keys()})
    cmap = plt.get_cmap("tab20")
    category_colors = {cat: cmap(i % 20) for i, cat in enumerate(all_categories)}

    n_models = len(MODELS)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 7))
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, MODELS):
        categories = sorted(error_counts[model].keys())
        rates = [
            error_counts[model].get(cat, 0) / (word_counts[model] or 1)
            for cat in categories
        ]

        if sum(rates) == 0:
            ax.text(0.5, 0.5, "No errors", ha="center", va="center")
            ax.set_title(model)
            ax.axis("off")
            continue

        model_colors = [category_colors[cat] for cat in categories]

        ax.pie(
            rates,
            labels=None,
            colors=model_colors,
            autopct=lambda p: f"{p:.1f}%" if p >= 0.01 else "",
            startangle=140,
            pctdistance=1.1,
            wedgeprops=dict(width=0.4),
        )
        ax.set_title(model)

    fig.legend(
        handles=[
            plt.matplotlib.patches.Patch(color=category_colors[cat], label=cat)
            for cat in all_categories
        ],
        title="Error Categories",
        loc="center right",
        bbox_to_anchor=(1.05, 0.5),
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_error_groups_bar_chart(error_counts, word_counts, save_path):
    """
    Plot grouped error rates:
    - Orthographic
    - Linguistic (lexical, morphological, phonological, pragmatic)
    - Structural (omission, addition)
    """
    groups = {
        "Orthographic": ["orthographic"],
        "Linguistic": ["lexical", "morphological", "phonological", "pragmatic"],
        "Structural": ["omission", "addition"],
    }

    group_rates = {group: [] for group in groups}

    for group, cats in groups.items():
        for model in MODELS:
            total_err = sum(error_counts[model].get(cat, 0) for cat in cats)
            total_words = word_counts[model] or 1
            group_rates[group].append((total_err / total_words) * 100)

    x = np.arange(len(groups))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(MODELS):
        values = [group_rates[group][i] for group in groups]
        ax.bar(x + i * width, values, width, label=model)

    ax.set_xticks(x + width)
    ax.set_xticklabels(groups.keys())
    ax.set_ylabel("Error rate (errors / word), %")
    ax.set_title("Grouped Error Rates per Model")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_structural_errors_stacked(error_counts, word_counts, save_path):
    """
    Create a stacked bar chart showing omission vs. addition error rates
    per model, with percentage labels inside each segment.
    """
    omission_rates = []
    addition_rates = []

    for model in MODELS:
        total_words = word_counts[model] or 1
        omission_rates.append(error_counts[model].get("omission", 0) / total_words * 100)
        addition_rates.append(error_counts[model].get("addition", 0) / total_words * 100)

    x = np.arange(len(MODELS))
    width = 0.6

    fig, ax = plt.subplots(figsize=(8, 6))
    bars_omission = ax.bar(x, omission_rates, width, label="Omission")
    bars_addition = ax.bar(x, addition_rates, width, bottom=omission_rates, label="Addition")

    # Annotate bar segments with percentage labels
    for i, bar in enumerate(bars_omission):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h / 2, f"{h:.2f}%", ha="center", va="center")

    for i, bar in enumerate(bars_addition):
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                omission_rates[i] + h / 2,
                f"{h:.2f}%",
                ha="center",
                va="center"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Error rate (errors / word), %")
    ax.set_title("Structural Error Rates per Model (Stacked)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------
# Main execution
# ---------------------------------------------------

if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)

    error_counts, word_counts = analyze_folder(input_folder)

    print("\nTotal words per model:", word_counts)
    print("Error counts per model:", error_counts)

    export_csv_error_counts(error_counts, os.path.join(output_folder, "error_counts.csv"))
    export_csv_error_rates(error_counts, word_counts, os.path.join(output_folder, "error_rates.csv"))
    export_csv_total_error_rates(error_counts, word_counts, os.path.join(output_folder, "total_error_rates.csv"))

    plot_error_counts(error_counts, os.path.join(output_folder, "error_counts.png"))
    plot_error_rates(error_counts, word_counts, os.path.join(output_folder, "error_rates.png"))
    plot_total_error_rates(error_counts, word_counts, os.path.join(output_folder, "total_error_rates.png"))

    plot_all_single_models(error_counts, word_counts, output_folder)
    plot_error_rate_pies(error_counts, word_counts, output_folder)
    plot_combined_error_rate_pies(
        error_counts,
        word_counts,
        os.path.join(output_folder, "combined_error_rate_pies.png")
    )
    plot_error_groups_bar_chart(
        error_counts,
        word_counts,
        os.path.join(output_folder, "error_groups_bar_chart.png")
    )
    plot_structural_errors_stacked(
        error_counts,
        word_counts,
        os.path.join(output_folder, "structural_errors_stacked.png")
    )

    print(f"\nAll outputs saved to: {output_folder}")