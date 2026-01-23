import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def train_model(model, train_loader, valid_loader, config, device):
    # Loss & Optimizer
    # Ignoriamo il token <PAD> (ID 0) nel calcolo dell'errore (Sparsity mask)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_valid_loss = float('inf')

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        
        # Tqdm per il feedback visivo (UX per il developer)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for src, trg in pbar:
            src, trg = src.to(device), trg.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: l'output ha forma [batch_size, trg_len, output_dim]
            output = model(src, trg)
            
            # Reshape per la CrossEntropy (appiattiamo la sequenza)
            # output: [(batch_size * trg_len), output_dim]
            # trg: [(batch_size * trg_len)]
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            loss.backward()
            
            # Gradient Clipping per evitare l'esplosione dei gradienti nelle RNN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # Qui andrebbe la logica di validazione (omessa per brevità, ma essenziale)
        logger.info(f"Epoch {epoch+1} completata. Average Loss: {epoch_loss/len(train_loader):.4f}")