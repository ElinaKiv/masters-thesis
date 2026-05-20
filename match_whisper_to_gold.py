"""
Whisper--Gold Alignment and Error Classification Pipeline.

This script aligns word-level Whisper ASR outputs with gold standard
transcriptions and annotates each word with match status and
linguistically motivated error categories.

For each gold JSON file, the pipeline:
- Matches the corresponding Whisper output per selected model using
  filename prefix and sample ordinal.
- Aligns words via Whisper segment IDs and word start timestamps.
- Compares Whisper and gold words under multiple normalization levels
  (strict, case-insensitive, punctuation-stripped).
- Classifies mismatches into error types such as word mismatch,
  orthographic, pragmatic, omission, and addition.
- The category "word_mismatch" is inteded for further manual division
  into linguistic error categories.

The output is one enriched JSON file per gold transcription, preserving
the original structure while adding per-model comparison results.
"""

import json
import re
import os
import logging
from tqdm import tqdm

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

gold_folder = r"raw_data/gold_standard_json_files"  # directory of gold standard transcriptions
whisper_folder = "raw_data"                         # base directory for whisper transcriptions
output_folder = "matched_whisper_and_gold_2"          # directory for processed output files

# Whisper models to include in the comparison
selected_models = [
    "tiny",
    "medium",
    "large-v3"
]

# Mapping from model names to their respective transcription directories
model_folders = {
    "tiny": "whisper_transcriptions_tiny",
    "medium": "whisper_transcriptions_medium",
    "large-v3": "whisper_transcriptions_large-v3"
}

diagnostic_mode = True                              # prints/logs detailed processing information
strict_mode = True                                  # if True, case and punctuation differences count as errors

# Logging configuration: console + file output for diagnostics
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("diagnostic.log", mode="w", encoding="utf-8")
    ]
)

# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def normalize(word):
    """
    Normalize a word for comparison depending on strictness settings.

    Parameters:
        word (str or None): A word token to normalize.

    Returns:
        str or None: The normalized word, or None if the input is None.

    Notes:
        - In strict mode, only leading/trailing whitespace is removed.
        - In non-strict mode, punctuation is stripped and the word is lowercased.
    """
    if word is None:
        return None
    if strict_mode:
        return word.strip()
    return re.sub(r"[^\w]", "", word).lower().strip()


def extract_ordinal_from_title(title):
    """
    Extract an ordinal sample identifier (e.g. '1st', '2nd') from a section title.

    Parameters:
        title (str): Section title containing an ordinal word (e.g. 'First interview').

    Returns:
        str or None: Ordinal string if detected, otherwise None.

    Notes:
        This function assumes English ordinal words and a fixed mapping.
    """
    mapping = {
        "first": "1st",
        "second": "2nd",
        "third": "3rd",
        "fourth": "4th",
        "fifth": "5th",
        "sixth": "6th",
        "seventh": "7th",
        "eight": "8th"      # the typo 'eight' instead of 'eighth' is intentional
    }
    title_lower = title.lower()
    for word, ordinal in mapping.items():
        if word in title_lower:
            return ordinal
    return None


def match_whisper_file(model_folder, gold_prefix, ordinal):
    """
    Match exactly one Whisper JSON file to a gold section based on naming conventions.

    Parameters:
        model_folder (str): Directory containing Whisper JSON files for one model.
        gold_prefix (str): Speaker or interview prefix derived from the gold filename.
        ordinal (str): Ordinal sample identifier (e.g. '1st').

    Returns:
        str or None: Filename of the matched Whisper file, or None if no match is found.

    Notes:
        Matching assumes filenames follow the pattern:
        <prefix>_<ordinal>_...json
    """
    ordinal = ordinal.lower()
    gold_prefix = gold_prefix.lower()

    candidates = []
    for filename in os.listdir(model_folder):
        if not filename.lower().endswith(".json"):
            continue

        parts = filename.split("_")
        if len(parts) < 2:
            continue

        prefix = parts[0].lower()
        whisper_ordinal = parts[1].lower()

        if prefix == gold_prefix and whisper_ordinal == ordinal:
            candidates.append(filename)

    if not candidates:
        return None

    # Deterministic choice if multiple candidates exist
    return sorted(candidates)[0]


def load_whisper_json(model_folder, filename):
    """
    Load a Whisper transcription JSON file.

    Parameters:
        model_folder (str): Directory of the Whisper model outputs.
        filename (str or None): Name of the Whisper JSON file.

    Returns:
        dict or None: Parsed JSON data if loaded successfully, otherwise None.
    """
    if filename is None:
        return None
    path = os.path.join(model_folder, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    return None


def build_whisper_index(whisper_json):
    """
    Build an index mapping Whisper segment IDs to their word lists.

    Parameters:
        whisper_json (dict or None): Parsed Whisper JSON transcription.

    Returns:
        dict or None: Mapping from segment ID to list of word dictionaries.
    """
    if whisper_json is None:
        return None
    index = {}
    for seg in whisper_json.get("segments", []):
        index[seg["id"]] = seg.get("words", [])
    return index


def find_whisper_word(whisper_words, start):
    """
    Locate a Whisper word by matching its start timestamp.

    Parameters:
        whisper_words (list or None): List of Whisper word dictionaries.
        start (float): Start time of the target word (from alignment data).

    Returns:
        str or None: The matched Whisper word string, or None if not found.

    Notes:
        A tolerance of 0.001 seconds is used to account for floating-point
        representation differences in timestamps.
    """
    if whisper_words is None:
        return None
    for w in whisper_words:
        if abs(w["start"] - start) <= 0.001:
            return w["word"]
    return None


def is_pragmatic_error(word1, word2):
    """
    Determine whether two words differ in pragmatic force.

    Parameters:
        word1 (str): First word token.
        word2 (str): Second word token.

    Returns:
        bool: True if interrogative/exclamatory punctuation differs.

    Notes:
        Pragmatic errors are identified via '?' and '!' punctuation.
    """
    punct1 = '!' if '!' in word1 else '?' if '?' in word1 else None
    punct2 = '!' if '!' in word2 else '?' if '?' in word2 else None
    return punct1 != punct2


def has_different_punctuation(word1, word2):
    """
    Check whether two words differ in punctuation.

    Parameters:
        word1 (str): First word token.
        word2 (str): Second word token.

    Returns:
        bool: True if punctuation sets differ.
    """
    punct1 = set(re.findall(r"[^\w\s]", word1))
    punct2 = set(re.findall(r"[^\w\s]", word2))
    return punct1 != punct2


def has_different_case(word1, word2):
    """
    Check whether two words differ in letter casing.

    Parameters:
        word1 (str): First word token.
        word2 (str): Second word token.

    Returns:
        bool: True if casing differs in any form (upper/lower/title).
    """
    same_case = (
        word1.isupper() == word2.isupper() and
        word1.islower() == word2.islower() and
        word1.istitle() == word2.istitle()
    )
    return not same_case


def reorder_section_keys(section):
    """
    Reorder section dictionary keys so that 'sample' appears first.

    Parameters:
        section (dict): A section dictionary.

    Returns:
        dict: Reordered section dictionary.

    Notes:
        This is purely cosmetic and intended to improve readability of output JSON.
    """
    return {"sample": section["sample"], **{k: v for k, v in section.items() if k != "sample"}}


# ---------------------------------------------------------
# Core processing functions
# ---------------------------------------------------------

def process_gold_file(gold_path, whisper_folder, selected_models, output_path):
    """
    Process a single gold transcription file and align it with Whisper outputs.

    Parameters:
        gold_path (str): Path to the gold standard JSON file.
        whisper_folder (str): Base directory containing Whisper model folders.
        selected_models (list[str]): Whisper models to include.
        output_path (str): Path where the processed JSON will be saved.

    Behavior:
        - Extracts sample numbers from section titles.
        - Matches Whisper files per model.
        - Aligns words via segment ID and timestamps.
        - Assigns match status and error categories.
        - Writes the enriched JSON to disk.
    """
    logging.info(f"Processing gold file: {gold_path}")

    with open(gold_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    gold_filename = os.path.basename(gold_path)
    gold_prefix = gold_filename.split("_")[0]

    # Placeholder field retained for compatibility with downstream schemas
    gold_data["interview"] = None

    for section in gold_data["sections"]:
        title = section["title"]

        # Determine ordinal/sample number from section title
        ordinal = extract_ordinal_from_title(title)
        if ordinal is None:
            raise ValueError(f"Could not extract ordinal from section title: {title}")

        sample_number = int(ordinal[:-2])
        section["sample"] = sample_number
        del section["title"]

        # Normalize structure: entries → turns
        section["turns"] = section.pop("entries")

        whisper_indices = {}

        # Load and index Whisper data for each selected model
        for model_key in selected_models:
            model_folder = os.path.join(whisper_folder, model_folders[model_key])
            match = match_whisper_file(model_folder, gold_prefix, ordinal)

            if match is None:
                logging.warning(
                    f"No Whisper file for model {model_key} prefix={gold_prefix} ordinal={ordinal}"
                )
                whisper_indices[model_key] = None
                continue

            whisper_json = load_whisper_json(model_folder, match)
            whisper_indices[model_key] = build_whisper_index(whisper_json)

        # Compare gold and Whisper words
        for turn in section["turns"]:
            for word_entry in turn["words"]:
                gold_word = word_entry["gold"]

                # Remove any non-relevant model keys
                keys_to_keep = {"gold"} | set(selected_models)
                for k in list(word_entry.keys()):
                    if k not in keys_to_keep:
                        del word_entry[k]

                for model_key in selected_models:
                    model_info = word_entry.get(model_key)
                    if not isinstance(model_info, dict):
                        logging.warning(
                            f"No model info for model {model_key} for word {word_entry} in turn {turn} in section {section} in file {gold_filename}."
                        )
                        continue

                    seg_id = model_info.get("segment_id")
                    start = model_info.get("word_start")

                    index = whisper_indices.get(model_key)

                    # Whisper file missing entirely
                    if index is None:
                        model_info["status"] = None
                        model_info["error_category"] = ["missing_whisper_file"]
                        continue

                    # Alignment metadata missing
                    if (seg_id is None or start is None) and gold_word is not None:
                        model_info["status"] = "mismatch"
                        model_info["error_category"] = ["omission"]
                        continue
                    
                    # Both gold and whisper are null
                    if (seg_id is None or start is None) and gold_word is None:
                        model_info["status"] = "match"
                        continue

                    whisper_words = index.get(seg_id, [])
                    whisper_word = find_whisper_word(whisper_words, start)
                    model_info["word"] = whisper_word

                    # Gold word missing
                    if gold_word is None and seg_id is not None and start is not None:
                        model_info["status"] = "mismatch"
                        model_info["error_category"] = ["addition"]
                        continue

                    # Whisper lookup failed
                    if whisper_word is None:
                        model_info["status"] = None
                        model_info["error_category"] = ["whisper_word_lookup_failed"]
                        continue

                    # Prepare normalized comparison variants
                    strict_gold = gold_word.strip()
                    strict_whisper = whisper_word.strip()

                    ci_gold = strict_gold.lower()
                    ci_whisper = strict_whisper.lower()

                    punct_gold = re.sub(r"[^\w]", "", gold_word).strip()
                    punct_whisper = re.sub(r"[^\w]", "", whisper_word).strip()

                    fully_gold = punct_gold.lower()
                    fully_whisper = punct_whisper.lower()

                    # Hierarchical comparison logic from strict to fully normalized
                    if strict_gold == strict_whisper:
                        model_info["status"] = "match"
                    elif ci_gold == ci_whisper:
                        model_info["status"] = "mismatch"
                        model_info["error_category"] = ["orthographic"]
                    elif punct_gold == punct_whisper:
                        model_info["status"] = "mismatch"
                        model_info["error_category"] = ["orthographic"]
                        if is_pragmatic_error(gold_word, whisper_word):
                            model_info["error_category"] = ["pragmatic"]
                    elif fully_gold == fully_whisper:
                        model_info["status"] = "mismatch"
                        model_info["error_category"] = ["orthographic"]
                        if is_pragmatic_error(gold_word, whisper_word):
                            model_info["error_category"].append("pragmatic")
                    else:
                        model_info["status"] = "mismatch"
                        model_info["error_category"] = ["word_mismatch"]
                        if has_different_case(strict_gold, strict_whisper) or has_different_punctuation(strict_gold, strict_whisper):
                            model_info["error_category"] = ["word_mismatch", "orthographic"]
                        if has_different_punctuation(strict_gold, strict_whisper) and is_pragmatic_error(strict_gold, strict_whisper):
                            model_info["error_category"] = ["word_mismatch", "pragmatic"]
                        if has_different_case(strict_gold, strict_whisper) and is_pragmatic_error(strict_gold, strict_whisper):
                            model_info["error_category"] = ["word_mismatch", "orthographic", "pragmatic"]

    # Cosmetic reordering of keys before saving
    gold_data["sections"] = [reorder_section_keys(sec) for sec in gold_data["sections"]]

    # Save processed output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gold_data, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved processed file to {output_path}")


def process_all_gold_files(gold_folder, whisper_folder, selected_models, output_folder):
    """
    Process all gold transcription files in a directory.

    Parameters:
        gold_folder (str): Directory containing gold JSON files.
        whisper_folder (str): Base directory for Whisper outputs.
        selected_models (list[str]): Whisper models to include.
        output_folder (str): Directory to save processed outputs.
    """
    os.makedirs(output_folder, exist_ok=True)
    gold_files = [f for f in os.listdir(gold_folder) if f.endswith(".json")]

    for filename in tqdm(gold_files, desc="Processing gold files"):
        gold_path = os.path.join(gold_folder, filename)
        output_path = os.path.join(output_folder, filename)
        process_gold_file(gold_path, whisper_folder, selected_models, output_path)


# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------

if __name__ == "__main__":
    logging.info("Starting Whisper-to-gold comparison pipeline")

    process_all_gold_files(
        gold_folder=gold_folder,
        whisper_folder=whisper_folder,
        selected_models=selected_models,
        output_folder=output_folder
    )

    logging.info("Done.")