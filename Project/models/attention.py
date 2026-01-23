import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hid_dim] (ultimo stato del decoder)
        # encoder_outputs: [batch_size, src_len, hid_dim]
        src_len = encoder_outputs.shape[1]
        
        # Ripeti hidden per ogni step temporale del sorgente
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        attention = self.v(energy).squeeze(2)
        # Softmax per ottenere una distribuzione di probabilità (somma = 1)
        return F.softmax(attention, dim=1)
