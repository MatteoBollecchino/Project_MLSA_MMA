import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        # input: [batch_size] -> un singolo token (il precedente)
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        
        # Calcolo pesi di attenzione basati sull'ultimo hidden state
        a = self.attention(hidden[-1], encoder_outputs).unsqueeze(1)
        # Prodotto scalare per ottenere il context vector: [batch_size, 1, hid_dim]
        weighted = torch.bmm(a, encoder_outputs)
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        # Uniamo tutto per la proiezione finale sul vocabolario
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2))
        return prediction.squeeze(1), hidden, cell