import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def train_model(model, train_loader, valid_loader, config, device):
    # --- FASE 0: AUDIT HARDWARE & MODELLO (PRINCIPLE OF OBSERVABILITY) ---
    logger.info("="*50)
    logger.info("AUDIT ESECUZIONE TRAINING")
    logger.info(f"Target Device: {device.type.upper()}")
    
    if device.type == 'cuda':
        logger.info(f"GPU Model: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM Totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info("Mixed Precision: ABILITATA (torch.amp.autocast)")
    else:
        logger.warning("ATTENZIONE: Esecuzione su CPU. Il processo sarà lento.")
        logger.info("Mixed Precision: DISABILITATA (Non supportata su CPU standard)")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parametri Addestrabili: {total_params:,}")
    logger.info("="*50)

    # Inizializzazione standard
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Scaler per AMP
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_valid_loss = float('inf')

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for src, trg in pbar:
            src, trg = src.to(device, non_blocking=True), trg.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # Esecuzione con Autocast (Automatic Mixed Precision)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                output = model(src, trg)
                output_dim = output.shape[-1]
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

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", gpu_mem=f"{torch.cuda.memory_allocated(0)/1e9:.1f}GB" if device.type == 'cuda' else "N/A")

        # Validazione
        valid_loss = evaluate_validation(model, valid_loader, criterion, device)
        
        status_msg = f"Epoch {epoch+1} FINISHED | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {valid_loss:.4f}"
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            status_msg += " [NEW BEST]"
        
        logger.info(status_msg)

def evaluate_validation(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device, non_blocking=True), trg.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                output = model(src, trg, teacher_forcing_ratio=0)
                output_dim = output.shape[-1]
                loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
                epoch_loss += loss.item()
    return epoch_loss / len(loader)