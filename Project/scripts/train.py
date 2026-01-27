import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time # Indispensabile per il profiling
from tqdm import tqdm

# Logger Configuration
logger = logging.getLogger(__name__)

# --- [TRAINING FUNCTION] ---
def train_model(model, train_loader, valid_loader, config, device, telemetry=None):
    """ Trains the model for a specified number of epochs using the provided data loaders and configuration. """

    # --- [PHASE 0] AUDIT HARDWARE ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"🚀 TRAINING START | Params: {total_params:,} | Device: {device}")

    # DataLoader length check
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("❌ The train_loader is empty.")

    # --- [PHASE 1] OPTIMIZER AND CRITERION CONFIGURATION ---

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    # Optimizer configuration
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    # OneCycleLR configuration with protection for small subsets
    total_steps = config.epochs * steps_per_epoch
    if total_steps < 2:
        scheduler = None
    else:
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.0005, steps_per_epoch=steps_per_epoch, 
            epochs=config.epochs, pct_start=0.3
        )
    
    # --- [PHASE 2] TRAINING LOOP ---

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # Best validation loss tracker
    best_valid_loss = float('inf')
 
    # Epoch loop
    for epoch in range(config.epochs):
        epoch_start_time = time.time() # START TIMER EPOCH
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for src, trg in pbar:
            src, trg = src.to(device, non_blocking=True), trg.to(device, non_blocking=True)
            
            # --- FORWARD AND BACKWARD PASS ---

            # Clear gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass with mixed precision
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                tf_ratio = max(0.2, 0.5 - (epoch * 0.1)) 
                output = model(src, trg, teacher_forcing_ratio=tf_ratio)
                output_dim = output.shape[-1]
                loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))

            # Backward pass and optimization step
            if scaler:
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Scheduler step
            if scheduler:
                scheduler.step()

            # --- END BATCH: UPDATE METRICS AND PROGRESS BAR ---
            train_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}")

        # --- END EPOCH: CALCULATE METRICS AND TIMING ---
        epoch_duration = time.time() - epoch_start_time # ⏱️ STOP TIMER EPOCH
        avg_train_loss = train_loss / steps_per_epoch
        valid_loss = evaluate_validation(model, valid_loader, criterion, device)
        
        # --- [TELEMETRY] Updated call with epoch_duration ---
        if telemetry:
            telemetry.log_epoch(
                epoch=epoch + 1, 
                train_loss=avg_train_loss, 
                val_loss=valid_loss, 
                lr=current_lr, 
                epoch_duration=epoch_duration # The missing parameter
            )
        
        # --- [LOGGING] EPOCH SUMMARY ---
        status_msg = f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val: {valid_loss:.4f} | Time: {epoch_duration:.2f}s"
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            status_msg += " 🔥 [NEW BEST]"
        
        # Log epoch summary
        logger.info(status_msg)

# --- [VALIDATION FUNCTION] ---
def evaluate_validation(model, loader, criterion, device):
    """ Evaluates the model on the validation set and returns the average loss. """

    # Set model to evaluation mode
    model.eval()

    # Initialize loss accumulator
    epoch_loss = 0

    # Calculate number of batches
    num_batches = len(loader)

    # Guard against empty loader to avoid division by zero
    if num_batches == 0: return 0   

    # Evaluate without gradient calculation
    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):

                # Forward pass
                output = model(src, trg, teacher_forcing_ratio=0)
                
                # Calculate loss
                output_dim = output.shape[-1]
                loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
                epoch_loss += loss.item()

    # Return average loss over all batches
    return epoch_loss / num_batches