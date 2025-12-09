cat > data_prep/2_transcribe.py <<EOF
import whisper
import glob
import json
import os
import argparse

def transcribe_folder(input_folder, output_json):
    print(f"Loading Whisper for {input_folder}...")
    model = whisper.load_model("medium") # Use 'medium' or 'large-v3' for real data
    
    audio_files = glob.glob(os.path.join(input_folder, "*.wav"))
    dataset_meta = []

    print(f"Transcribing {len(audio_files)} files...")

    for audio_path in audio_files:
        try:
            result = model.transcribe(audio_path)
            text = result["text"].strip()
            
            meta_entry = {
                "audio_filepath": audio_path,
                "text": text,
                "duration": result["segments"][-1]["end"]
            }
            dataset_meta.append(meta_entry)
            print(f"  Processed: {os.path.basename(audio_path)}")
        except Exception as e:
            print(f"  Failed: {audio_path} ({e})")

    with open(output_json, "w") as f:
        json.dump(dataset_meta, f, indent=4)
    print(f"Saved metadata to {output_json}")

if __name__ == "__main__":
    transcribe_folder("raw_data/vocals", "dataset_metadata.json")
EOF
