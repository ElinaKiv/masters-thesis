import json
import os
import csv
from collections import defaultdict
from statsmodels.stats.contingency_tables import mcnemar

# -------- SETTINGS --------
DATA_DIR = "json_gold_processed"

MODELS = ["tiny", "medium", "large-v3"]

ERROR_TYPES = [
    "phonological",
    "morphological",
    "lexical",
    "pragmatic",
    "omission",
    "addition",
    "orthographic"
]

# -------- HELPERS --------

def is_correct(word_obj):
    return word_obj.get("status") == "match"

def has_error_type(word_obj, error_type):
    if word_obj.get("status") != "mismatch":
        return False
    return error_type in word_obj.get("error_category", [])

def iterate_words(data):
    for section in data.get("sections", []):
        for turn in section.get("turns", []):
            for word in turn.get("words", []):
                yield word


# -------- AGGREGATION STRUCTURE --------
results = defaultdict(lambda: defaultdict(lambda: {"b": 0, "c": 0}))

# Include overall results
ALL_TYPES = [None] + ERROR_TYPES

# -------- PROCESS ALL FILES --------

for filename in os.listdir(DATA_DIR):
    if not filename.endswith(".json"):
        continue

    path = os.path.join(DATA_DIR, filename)

    with open(path, "r") as f:
        data = json.load(f)

    # Loop over model pairs
    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            model_a = MODELS[i]
            model_b = MODELS[j]

            for error_type in ALL_TYPES:
                b, c = 0, 0

                for word in iterate_words(data):
                    A = word.get(model_a)
                    B = word.get(model_b)

                    if not A or not B:
                        continue

                    # ---- define binary values ----
                    if error_type is None:
                        A_val = not is_correct(A)
                        B_val = not is_correct(B)
                    else:
                        A_val = has_error_type(A, error_type)
                        B_val = has_error_type(B, error_type)

                    # ---- update counts ----
                    if A_val and not B_val:
                        b += 1
                    elif B_val and not A_val:
                        c += 1

                # ---- accumulate across files ----
                key = (model_a, model_b)
                results[key][error_type]["b"] += b
                results[key][error_type]["c"] += c


# -------- MCNEMAR TEST --------

def run_mcnemar(b, c):
    table = [[0, b],
             [c, 0]]

    result = mcnemar(
        table,
        exact=False,
        correction=True
    )

    return result.statistic, result.pvalue


# -------- OUTPUT RESULTS --------

csv_rows = []

for (model_a, model_b), res_dict in results.items():
    print(f"\n=== {model_a} vs {model_b} (aggregated) ===")

    for error_type in ALL_TYPES:
        label = "overall" if error_type is None else error_type

        b = res_dict[error_type]["b"]
        c = res_dict[error_type]["c"]

        stat, p = run_mcnemar(b, c)

        print(f"\n{label}")
        print(f"b={b}, c={c}")
        print(f"chi2={stat:.3f}, p={p:.6f}")

        # store for CSV
        csv_rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "error_type": label,
            "b": b,
            "c": c,
            "chi2": stat,
            "p_value": p,
            "significant": p < 0.05
        })


# -------- WRITE CSV --------

output_file = "mcnemar_results.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model_a",
            "model_b",
            "error_type",
            "b",
            "c",
            "chi2",
            "p_value",
            "significant"
        ]
    )
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"\nResults saved to {output_file}")