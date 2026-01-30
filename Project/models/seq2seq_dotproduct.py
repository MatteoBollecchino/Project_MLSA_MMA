import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class DotProductAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        # Proiezioni lineari come nelle slide del Prof
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.scale = np.sqrt(hid_dim)

    def forward(self, query, keys):
        # query: [batch, hid_dim] | keys: [batch, src_len, hid_dim]
        q = self.w_q(query).unsqueeze(1) # [batch, 1, hid_dim]
        k = self.w_k(keys).transpose(1, 2) # [batch, hid_dim, src_len]
        
        # Scaled Dot Product: (Q * K^T) / sqrt(d)
        scores = torch.bmm(q, k) / self.scale # [batch, 1, src_len]
        alphas = F.softmax(scores, dim=-1)
        
        # Context vector
        context = torch.bmm(alphas, keys) # [batch, 1, hid_dim]
        return alphas.squeeze(1), context

class DecoderDot(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.attention = DotProductAttention(hid_dim)
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        
        # Usiamo l'ultimo stato nascosto come Query
        _, context = self.attention(hidden[-1], encoder_outputs)
        
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        prediction = self.fc_out(output)
        return prediction.squeeze(1), hidden, cell

class Seq2SeqDotProduct(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout, device):
        super().__init__()
        from models.seq2seq_bahdanau import Encoder # Riutilizziamo l'encoder standard
        self.encoder = Encoder(vocab_size, emb_dim, hid_dim, n_layers, dropout)
        self.decoder = DecoderDot(vocab_size, emb_dim, hid_dim, n_layers, dropout)
        self.device = device
        self.output_dim = vocab_size

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape[0], trg.shape[1]
        outputs = torch.zeros(batch_size, trg_len, self.output_dim).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else output.argmax(1)
        return outputs