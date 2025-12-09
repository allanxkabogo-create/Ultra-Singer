import torch
import torch.nn as nn
from dac.model.dac import DAC
from dac.utils import load_model

class UltraTokenizer(nn.Module):
    def __init__(self, model_type='44khz'):
        super().__init__()
        print(f"Loading DAC Model: {model_type}...")
        # Downloads/Loads the pre-trained Descript Audio Codec
        model_path = load_model(tag=model_type)
        self.dac = DAC.load(model_path)
        
        # Freeze parameters (we don't train the ears, we train the brain)
        self.dac.eval()
        for param in self.dac.parameters():
            param.requires_grad = False
            
    def encode(self, audio_tensor):
        # Input: [Batch, 1, Time] -> Output: [Batch, N_Codebooks, Time]
        with torch.no_grad():
            x = self.dac.preprocess(audio_tensor, 44100)
            _, codes, _, _, _ = self.dac.encode(x)
        return codes

    def decode(self, codes):
        # Input: [Batch, N_Codebooks, Time] -> Output: Audio
        with torch.no_grad():
            z_q, _, _ = self.dac.quantizer.from_codes(codes)
            audio = self.dac.decode(z_q)
        return audio
