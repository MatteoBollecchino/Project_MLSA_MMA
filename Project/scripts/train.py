import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def train_model(model, train_loader, valid_loader, config, device):
    # --- [FASE 0] AUDIT & CONFIGURAZIONE ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"🚀 TRAINING START | Params: {total_params:,} | Device: {device}")

    # 1. Label Smoothing: Previene l'overconfident del modello (Migliora Generalizzazione)
    # 
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    # 2. AdamW: Gestisce meglio il decadimento dei pesi rispetto ad Adam standard
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    # 3. OneCycleLR: Il "Proiettile d'Argento" per i Transformer. 
    # Implementa un Warmup iniziale (LR sale) e un Decay finale (LR scende).
    # 
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.0005, 
        steps_per_epoch=len(train_loader), 
        epochs=config.epochs,
        pct_start=0.1 # 10% del tempo in warmup
    )
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    best_valid_loss = float('inf')

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for src, trg in pbar:
            src, trg = src.to(device, non_blocking=True), trg.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                # Il teacher forcing è ignorato dal Transformer grazie ai **kwargs nel forward
                tf_ratio = max(0.2, 0.5 - (epoch * 0.1)) 
                output = model(src, trg, teacher_forcing_ratio=tf_ratio)
                
                output_dim = output.shape[-1]
                # Escludiamo il primo token (<SOS>) dal calcolo della loss
                loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # CRITICO: Per OneCycleLR, lo scheduler deve avanzare a ogni BATCH, non a ogni epoca
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

        # Validazione
        valid_loss = evaluate_validation(model, valid_loader, criterion, device)
        
        status_msg = f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {valid_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            status_msg += " 🔥 [NEW BEST]"
        
        logger.info(status_msg)

def evaluate_validation(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                output = model(src, trg, teacher_forcing_ratio=0)
                output_dim = output.shape[-1]
                loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
                epoch_loss += loss.item()
    return epoch_loss / len(loader)