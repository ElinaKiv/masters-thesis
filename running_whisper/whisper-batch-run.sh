#!/bin/bash
#SBATCH --account=XXX
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:v100:1


module load whisper

for file in *.mp4 *.mp3; do

	[ -e "$file" ] || continue

	srun whisper "$file" --model medium --language en --threads 2 --diarize pyannote_v3.0 --diarize_threads 2 --num_speakers 2 -o whisper_transcriptions_medium -f json

	srun whisper "$file" --model large-v3 --language en --threads 2 --diarize pyannote_v3.0 --diarize_threads 2 --num_speakers 2 -o whisper_transcriptions_large-v3 -f json

	srun whisper "$file" --model tiny --language en --threads 2 --diarize pyannote_v3.0 --diarize_threads 2 --num_speakers 2 -o whisper_transcriptions_tiny -f json

done
