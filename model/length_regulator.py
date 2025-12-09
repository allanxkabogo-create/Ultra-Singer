cat > model/length_regulator.py <<EOF
import torch
import torch.nn as nn
import torch.nn.functional as F

class LengthRegulator(nn.Module):
    """
    The 'Control' module.
    Input: Phoneme Embeddings + Durations (in frames)
    Output: Frame-level Embeddings (Exact Length)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, durations, target_len=None):
        # x: [Batch, Phoneme_Len, Dim]
        # durations: [Batch, Phoneme_Len] (Int number of frames)
        
        output = []
        for batch_i in range(x.size(0)):
            expanded = []
            for i in range(x.size(1)):
                # Repeat the phoneme 'N' times based on duration
                repeat_count = max(1, int(durations[batch_i, i].item()))
                expanded.append(x[batch_i, i].unsqueeze(0).repeat(repeat_count, 1))
            
            output.append(torch.cat(expanded, dim=0))
        
        # Pad to longest in batch
        max_len = max([o.size(0) for o in output])
        if target_len is not None:
            max_len = target_len
            
        padded_output = []
        for o in output:
            pad_amount = max_len - o.size(0)
            if pad_amount > 0:
                o = F.pad(o, (0, 0, 0, pad_amount))
            elif pad_amount < 0:
                o = o[:max_len]
            padded_output.append(o)
            
        return torch.stack(padded_output)
EOF
