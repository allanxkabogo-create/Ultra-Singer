cat > scripts/dataset.py <<EOF
import torch
from torch.utils.data import Dataset
import json
import torchaudio
import os
import numpy as np

class UltraDataset(Dataset):
    def __init__(self, metadata_path, tokenizer):
        with open(metadata_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.sr = 44100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item['audio_filepath']
        
        # 1. Load Audio
        wav, sr = torchaudio.load(audio_path)
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            wav = resampler(wav)
            
        # 2. Pad/Trim to fixed length (e.g., 5 seconds) for batching
        target_samples = self.sr * 5
        if wav.shape[-1] < target_samples:
            wav = torch.nn.functional.pad(wav, (0, target_samples - wav.shape[-1]))
        else:
            wav = wav[:, :target_samples]
            
        # 3. Add Batch Dim for Tokenizer
        wav = wav.unsqueeze(0) 
        
        # Note: In real training, we pre-compute tokens to save GPU.
        # Here we return raw audio and let the loop handle it.
        return wav.squeeze(0), item['text']
EOF
