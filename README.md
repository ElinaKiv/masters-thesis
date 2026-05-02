This repository contains material for comparing Whisper transcriptions to human-made (gold standard) transcriptions. The material is used in the Master's thesis "Careless Whisper?: A lingusitic evaluation of the AI-based automatic speech recognition tool" by Elina Kivari (2026) at the Univeristy of Helsinki in collaboration with CSC - IT Center for Science.

## Running Whisper transcriptions

The folder "running_whisper" contains a script for running Whisper transcriptions with different model sizes on audio files. The script runs Whisper for sizes tiny, medium, and large-v3 for .mp3 and .mp4 files. The audio files used in this work are not public.

## Raw data

Raw transcription data for both Whisper and gold standard transcriptions is stored in the "raw_data" folder. 

The gold standard JSON-files contain word level references (which were made manually) to the corresponding words transcribed with Whisper. Each gold standard JSON-file contains transcriptions for one of the eight interviews, i.e., each JSON-file contains all eight samples from one interview.

The Whisper transcriptions are grouped to folders according to model size. Each folder contains the transcriptions for all eight samples from all eight interviews.

## Matching Whisper and gold standard transcriptions

The Python script "match_whisper_to_gold.py" compares the two transcriptions word by word and flags all mismatches. Further, the script categorizes errors based on case, punctuation or word mismatch. The outputs are located in the folder "matched_whisper_and_gold".

## Final Whisper and gold standard trancriptions

The word mismatches are further categorized manually to produce the JSON-files in the folder "final_whisper_and_gold_trancriptions".

## Analyzing Whisper transcription errors

The Python script "analyze_whisper_errors.py" counts and groups the errors listed in the final Whisper and gold standard transcription JSON-files. Further, the script produces various charts and tables into the folder "analysis_output".