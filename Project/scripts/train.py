import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time # Indispensabile per il profiling
from tqdm import tqdm

logger = logging.getLogger(__name__)

def train_model(model, train_loader, valid_loader, config, device, telemetry=None):
    # --- [FASE 0] AUDIT HARDWARE ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"🚀 TRAINING START | Params: {total_params:,} | Device: {device}")

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("❌ Il train_loader è vuoto.")

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    # Configurazione OneCycleLR con protezione per piccoli subset
    total_steps = config.epochs * steps_per_epoch
    if total_steps < 2:
        scheduler = None
    else:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.0005, steps_per_epoch=steps_per_epoch, 
            epochs=config.epochs, pct_start=0.3
        )
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    best_valid_loss = float('inf')

    for epoch in range(config.epochs):
        epoch_start_time = time.time() # ⏱️ START TIMER EPOCA
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for src, trg in pbar:
            src, trg = src.to(device, non_blocking=True), trg.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                tf_ratio = max(0.2, 0.5 - (epoch * 0.1)) 
                output = model(src, trg, teacher_forcing_ratio=tf_ratio)
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

            if scheduler:
                scheduler.step()

            train_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}")

        # --- FINE EPOCA: CALCOLO METRICHE E TEMPI ---
        epoch_duration = time.time() - epoch_start_time # ⏱️ STOP TIMER EPOCA
        avg_train_loss = train_loss / steps_per_epoch
        valid_loss = evaluate_validation(model, valid_loader, criterion, device)
        
        # --- [TELEMETRIA] Chiamata aggiornata con epoch_duration ---
        if telemetry:
            telemetry.log_epoch(
                epoch=epoch + 1, 
                train_loss=avg_train_loss, 
                val_loss=valid_loss, 
                lr=current_lr, 
                epoch_duration=epoch_duration # Il parametro mancante
            )
        
        status_msg = f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val: {valid_loss:.4f} | Time: {epoch_duration:.2f}s"
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            status_msg += " 🔥 [NEW BEST]"
        
        logger.info(status_msg)

def evaluate_validation(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    num_batches = len(loader)
    if num_batches == 0: return 0
    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                output = model(src, trg, teacher_forcing_ratio=0)
                output_dim = output.shape[-1]
                loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
                epoch_loss += loss.item()
    return epoch_loss / num_batches