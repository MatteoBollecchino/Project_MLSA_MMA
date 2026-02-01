"""
================================================================================
TRAINING ENGINE
================================================================================
ROLE: Core Optimization Loop and Gradient Descent Orchestrator.

DESIGN PRINCIPLES:
- Numerical Stability: Implements Gradient Clipping and Label Smoothing.
- Hardware Acceleration: Fully integrated with Mixed Precision (FP16/BF16).
- Adaptive Regularization: Hyperparameters scale based on the model phenotype 
  (Transformer vs. LSTM) to combat Overfitting vs. Underfitting.
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
from tqdm import tqdm

# Standard logger hook for training progress and diagnostic telemetry.
logger = logging.getLogger(__name__)

def train_model(model, train_loader, valid_loader, config, device, telemetry=None):
    """
    Main Training Loop Orchestrator.
    
    Args:
        model: PyTorch Module (LSTM or Transformer).
        train_loader: Iterator for training data batches.
        valid_loader: Iterator for validation data batches.
        config: Namespace containing epochs, model_type, and batch_size.
        device: Target compute device (cuda:0, cuda:1, or cpu).
        telemetry: Optional logger class for persisting training history.
    """

    # --- [PHASE 0] ARCHITECTURAL AUDIT ---
    # Log total trainable parameters to estimate model capacity and memory footprint.
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"🚀 START | Model: {config.model} | Params: {total_params:,} | Device: {device}")

    # Integrity Check: Prevents runtime division errors if the dataset refinery failed.
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("❌ Error: train_loader is empty. Check data refinery output.")

    # --- [PHASE 1] DIFFERENTIATED HYPERPARAMETER INJECTION ---
    # Mental Model: 'The Transformer' (Titan) needs high pressure regularization 
    # to avoid memorization, while 'The LSTM' (Tailor) needs guidance.
    
    if config.model == "transformer":
        # Strategy for Transformers: Prevent 'Dirac Delta' distributions in Softmax.
        weight_decay = 0.1       # Strong L2 penalty to keep weights from exploding.
        label_smoothing = 0.2    # Soften targets (e.g. 0.9 instead of 1.0) to encourage exploration.
        max_lr = 0.0001          # Transformers are sensitive to high LR during the early phase.
        pct_start = 0.3          # 30% of total steps used for warm-up.
    else:
        # Strategy for LSTM: Standard Sequence-to-Sequence approach.
        weight_decay = 0.01      # Conservative weight decay.
        label_smoothing = 0.1    # Standard smoothing.
        max_lr = 0.0005          # Recurrent nets can often handle faster initial learning.
        pct_start = 0.3

    # CRITERION: CrossEntropy with ignore_index=0 ensures <PAD> tokens do not compute gradients.
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)

    # OPTIMIZER: AdamW decouples Weight Decay from the gradient update (crucial for Attention).
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=weight_decay)

    # SCHEDULER: OneCycleLR implements 'Super-Convergence' by cycling LR and Momentum.
    total_steps = config.epochs * steps_per_epoch
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr, 
        steps_per_epoch=steps_per_epoch, 
        epochs=config.epochs, 
        pct_start=pct_start
    ) if total_steps >= 2 else None

    # --- [PHASE 2] CORE TRAINING LOOP ---
    # SCALER: Manages precision scaling for FP16 training to prevent underflow.
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    best_valid_loss = float('inf')

    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        model.train() # Enable Dropout and BatchNorm.
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for src, trg in pbar:
            # Move data to VRAM. non_blocking=True allows overlapping compute/transfer.
            src, trg = src.to(device, non_blocking=True), trg.to(device, non_blocking=True)
            
            # Efficiently reset gradients for the new batch.
            optimizer.zero_grad(set_to_none=True)

            # AUTO-MIXED PRECISION (AMP): Automatically casts tensors to half-precision.
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                # Decay Teacher Forcing: Start by guiding the model, end by letting it guess.
                tf_ratio = max(0.2, 0.5 - (epoch * 0.1)) 
                output = model(src, trg, teacher_forcing_ratio=tf_ratio)
                
                # Reshape logic: Flatten [Batch, Seq, Vocab] for standard CrossEntropy format.
                output_dim = output.shape[-1]
                # Slice [:, 1:] to skip <SOS> token, matching target alignment.
                loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))

            # BACKWARD PASS: Scale loss, compute gradients, and step optimizer.
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # Gradient Clipping: Prevents the 'Exploding Gradient' problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if scheduler:
                scheduler.step()

            # Batch Diagnostics: Update progress bar postfix with real-time loss/LR.
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

        # --- [PHASE 3] EPOCH EVALUATION ---
        # Mental Model: Validating generalizability on unseen data.
        epoch_duration = time.time() - epoch_start_time
        avg_train_loss = train_loss / steps_per_epoch
        valid_loss = evaluate_validation(model, valid_loader, criterion, device)
        
        # Persist epoch results to log files for future plotting/analysis.
        if telemetry:
            telemetry.log_epoch(epoch+1, avg_train_loss, valid_loss, optimizer.param_groups[0]['lr'], epoch_duration)
        
        # LOGGING SUMMARY: Report performance and check for 'New Best' checkpoint.
        status_msg = f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val: {valid_loss:.4f} | Time: {epoch_duration:.2f}s"
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            status_msg += " 🔥 [NEW BEST]"
        
        logger.info(status_msg)

def evaluate_validation(model, loader, criterion, device):
    """ 
    Calculates average loss on the validation set.
    Teacher Forcing is strictly disabled (ratio=0) for honest assessment.
    """
    model.eval() # Disable Dropout and BatchNorm.
    epoch_loss = 0
    num_batches = len(loader)
    if num_batches == 0: return 0   

    with torch.no_grad(): # Disable gradient graph construction to save memory.
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                # Model predicts autoregressively based on context.
                output = model(src, trg, teacher_forcing_ratio=0)
                output_dim = output.shape[-1]
                # Compare predicted tokens with actual ground truth.
                loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
                epoch_loss += loss.item()

    return epoch_loss / num_batches