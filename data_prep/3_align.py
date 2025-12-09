cat > data_prep/3_align.py <<EOF
import torch
import librosa
import json
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def align_dataset(metadata_file, output_file):
    print("Loading Alignment Model (Wav2Vec2)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)

    with open(metadata_file, "r") as f:
        dataset = json.load(f)

    print(f"Aligning {len(dataset)} files...")
    aligned_dataset = []

    for entry in dataset:
        audio_path = entry["audio_filepath"]
        text = entry["text"].upper()
        
        try:
            # Resample to 16k for Wav2Vec2
            audio, sr = librosa.load(audio_path, sr=16000)
            
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)
            
            with torch.no_grad():
                logits = model(input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            
            # Save alignment info
            entry["alignment"] = {
                "transcription": transcription,
                "duration_path": audio_path.replace(".wav", ".dur.npy")
            }
            aligned_dataset.append(entry)
            print(f"  Aligned: {os.path.basename(audio_path)}")
            
        except Exception as e:
            print(f"  Failed: {audio_path} ({e})")

    with open(output_file, "w") as f:
        json.dump(aligned_dataset, f, indent=4)
    print(f"Alignment complete. Saved to {output_file}")

if __name__ == "__main__":
    align_dataset("dataset_metadata.json", "dataset_metadata_aligned.json")
EOF
