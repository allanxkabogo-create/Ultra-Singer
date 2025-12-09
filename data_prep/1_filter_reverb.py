import os
import torch
import torchaudio
import numpy as np
from scipy.signal import correlate

def estimate_rt60(audio_path, threshold_db=-60):
    """
    Estimates Reverb Time (RT60) of an audio file.
    If RT60 > 0.3s, the file is likely too 'wet' for training.
    """
    try:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True) # Convert to Mono
        
        # Calculate Energy Decay
        impulse_response = waveform.numpy()[0]
        decay = 10 * np.log10(np.cumsum(impulse_response[::-1]**2)[::-1] + 1e-10)
        
        # Find time to drop by threshold_db
        peak_energy = np.max(decay)
        target_energy = peak_energy + threshold_db
        
        # Find where it crosses the threshold
        decay_slice = np.where(decay <= target_energy)[0]
        if len(decay_slice) == 0:
            return 99.0 # Infinite reverb (bad file)
            
        t_60 = decay_slice[0] / sr
        return t_60
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return 99.0

if __name__ == "__main__":
    # Example usage
    # You will point this to your downloaded MOGG/Atmos folder
    print("RT60 Estimator Ready.")
