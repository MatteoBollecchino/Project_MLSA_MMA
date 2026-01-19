import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # LSTM riceve (embedding + context_vector)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        # input: [batch_size] -> token precedente
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input)) # [batch_size, 1, emb_dim]
        
        # Calcolo Attention tra l'ultimo hidden state e tutti gli encoder_outputs
        a = self.attention(hidden[-1], encoder_outputs) # [batch_size, src_len]
        a = a.unsqueeze(1) # [batch_size, 1, src_len]
        
        # Context vector: somma pesata degli encoder outputs
        weighted = torch.bmm(a, encoder_outputs) # [batch_size, 1, hid_dim]
        
        # Concatenazione embedding + context per la RNN
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        # Predizione finale concatenando tutto per massimizzare il segnale
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2))
        
        return prediction.squeeze(1), hidden, cell