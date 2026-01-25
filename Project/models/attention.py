import torch
import torch.nn as nn
import torch.nn.functional as F

# --- [ATTENTION MECHANISM CLASS] ---
class Attention(nn.Module):

    def __init__(self, hid_dim):
        """ Attention Mechanism initialization. """

        super().__init__()

        # Linear layers for attention computation
        self.attn = nn.Linear(hid_dim * 2, hid_dim)

        # Final linear layer to produce attention scores
        self.v = nn.Linear(hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        """ Forward pass for the attention mechanism. """

        # hidden: [batch_size, hid_dim] (last state of the decoder)
        # encoder_outputs: [batch_size, src_len, hid_dim]
        src_len = encoder_outputs.shape[1]
        
        # Repeat hidden for each source time step
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        attention = self.v(energy).squeeze(2)
        
        # Softmax to obtain a probability distribution (sum = 1)
        return F.softmax(attention, dim=1)
