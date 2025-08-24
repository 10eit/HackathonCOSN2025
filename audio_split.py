import os
import re
import json
import glob
from faster_whisper import WhisperModel

# --- Controls ---
# Set these booleans to True or False to control which transcription formats are generated.
GENERATE_WORD_LEVEL = True
GENERATE_CHAR_LEVEL = True

# --- Model Configuration ---
# Choose your model size: "tiny", "base", "small", "medium", "large"
# Set your device: "cpu" or "cuda"
# Set compute type for CPU: "int8", "int16", "float32"
# Set compute type for CUDA: "int8_float16", "float16"
model = WhisperModel("tiny", device="cpu", compute_type="int8")

target_folder = "xiaowangzi_male_sbject_m1"

all_transcriptions = {}
search_pattern = os.path.join(target_folder, '*.wav')
audio_files_to_process = glob.glob(search_pattern)
audio_files_to_process.sort(key=lambda f: int(re.search(r'(\d+)', os.path.basename(f)).group(1)))

for audio_file_path in audio_files_to_process:
    audio_filename = os.path.basename(audio_file_path)
    
    word_level_data = []
    char_level_data = []
    
    segments, info = model.transcribe(audio_file_path, word_timestamps=True)
    
    for segment in segments:
        for word in segment.words:

            # --- Mode 1: Word-level Transcription ---
            if GENERATE_WORD_LEVEL:
                word_info = {
                    "word": word.word.strip(),
                    "start": round(word.start, 4),
                    "end": round(word.end, 4)
                }
                word_level_data.append(word_info)

            # --- Mode 2: Character-level Transcription ---
            if GENERATE_CHAR_LEVEL:
                text = word.word.strip()
                clean_text = re.sub(r'[^\u4e00-\u9fffa-zA-Z0-9]', '', text)
                word_duration = word.end - word.start
                num_chars = len(clean_text)
                char_duration = word_duration / num_chars if num_chars > 0 else 0

                for i, char in enumerate(clean_text):
                    char_start_time = word.start + (i * char_duration)
                    char_end_time = char_start_time + char_duration
                    
                    char_info = {
                        "char": char,
                        "start": round(char_start_time, 4),
                        "end": round(char_end_time, 4),
                        "original_word": text # Add original word for context
                    }
                    char_level_data.append(char_info)
    
    output_data = {}
    if GENERATE_WORD_LEVEL:
        output_data["word_level_transcription"] = word_level_data
    if GENERATE_CHAR_LEVEL:
        output_data["character_level_transcription"] = char_level_data

    if output_data:
        all_transcriptions[audio_filename] = output_data
    print(f"Finished processing: {audio_filename}. Found {len(word_level_data)} words, found {len(char_level_data)} chars.")

if all_transcriptions:
    output_filename = f"transcription_results_{target_folder}.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_transcriptions, f, ensure_ascii=False, indent=4)

    print(f"\n✅ All done! Results saved to '{output_filename}'")