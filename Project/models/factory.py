"""
================================================================================
ARCHITECTURAL FACTORY UNIT
================================================================================
ROLE: Centralized Model Synthesis and Hyperparameter Configuration.

DESIGN RATIONALE:
- Abstraction: Allows the C2Orchestrator to request an architecture by string 
  identifier without knowing internal implementation details.
- Phenotypic Calibration: Specifically tuned hyperparameters based on 
  architectural behavior (Sequential vs. Parallel).
- Hardware Mapping: Ensures immediate VRAM occupancy upon instantiation.
================================================================================
"""

import logging
from models.transformer import Seq2SeqTransformer
from models.seq2seq_bahdanau import Seq2SeqBahdanau
from models.seq2seq_dotproduct import Seq2SeqDotProduct

# Module-level logger for auditing architectural synthesis.
logger = logging.getLogger(__name__)

def get_model_architecture(model_type, device, vocab_size=20000):
    """
    FACTORY ACTUATOR v3.5 - Tailored Architectures
    ---------------------------------------------
    Calibrated parameters based on architectural phenotype:
    
    - LSTM: Maintains statistical conservatism (Dropout 0.1). 
      Recurrent nets are naturally regularized by their sequential nature.
      
    - TRANSFORMER: Employs aggressive regularization (Dropout 0.3) 
      to disrupt rote learning (memorization) of fixed strings, forcing the 
      attention heads to learn generalized syntactic patterns.
    """

    # --- [PHASE 1] LSTM CONFIGURATION (Sequential Backbones) ---
    # Common dimensions for Bahdanau (Additive) and DotProduct (Multiplicative).
    # Focus: Balance between latent capacity and training stability.
    lstm_params = {
        "emb_dim": 256,   # Dimension of the dense vector space for tokens.
        "hid_dim": 512,   # Latent space capacity (Hidden State size).
        "n_layers": 2,    # Depth of the recurrent stack.
        "dropout": 0.1    # Standard regularization protocol for RNNs.
    }

    # --- [PHASE 2] TRANSFORMER CONFIGURATION (Parallel Backbone) ---
    # Focus: Fighting Overfitting. Since Transformers have massive capacity, 
    # they tend to "shortcut" the learning process by memorizing the dataset.
    trans_params = {
        "emb_dim": 256,
        "hid_dim": 512,            # Maps to d_model in Transformer nomenclature.
        "nhead": 8,                # Number of parallel attention heads.
        "layers": 6,               # Total depth of Encoder and Decoder stacks.
        "dropout": 0.3,            # HIGH ENTROPY FIX: Forces model to find robust pathways.
        "dim_feedforward": 2048    # Expansion factor for the point-wise MLP.
    }

    # --- [PHASE 3] LOGICAL BRANCHING & INSTANTIATION ---

    # ARCHITECTURE A: LSTM + BAHDANAU ATTENTION
    # Uses a neural network layer to learn alignments (Additive Attention).
    if model_type == "lstm_bahdanau":
        logger.info(f"Factory: Generating LSTM + Bahdanau (Conservative Dropout: {lstm_params['dropout']})")
        return Seq2SeqBahdanau(
            vocab_size=vocab_size, 
            emb_dim=lstm_params["emb_dim"], 
            hid_dim=lstm_params["hid_dim"], 
            n_layers=lstm_params["n_layers"], 
            dropout=lstm_params["dropout"], 
            device=device
        ).to(device) # Immediate hardware mapping.

    # ARCHITECTURE B: LSTM + DOTPRODUCT ATTENTION
    # Scaled Dot-Product alignment mechanism (Geometric/Multiplicative Attention).
    elif model_type == "lstm_dotproduct":
        logger.info(f"Factory: Generating LSTM + DotProduct (Conservative Dropout: {lstm_params['dropout']})")
        return Seq2SeqDotProduct(
            vocab_size=vocab_size, 
            emb_dim=lstm_params["emb_dim"], 
            hid_dim=lstm_params["hid_dim"], 
            n_layers=lstm_params["n_layers"], 
            dropout=lstm_params["dropout"], 
            device=device
        ).to(device)

    # ARCHITECTURE C: TRANSFORMER (Attention Is All You Need)
    # The parallel titan. Requires strict regularization to converge on code logic.
    elif model_type == "transformer":
        logger.info(f"Factory: Generating Transformer (Aggressive Dropout: {trans_params['dropout']})")
        return Seq2SeqTransformer(
            vocab_size=vocab_size, 
            d_model=trans_params["hid_dim"],
            nhead=trans_params["nhead"],
            num_layers=trans_params["layers"], 
            dim_feedforward=trans_params["dim_feedforward"],
            dropout=trans_params["dropout"]
        ).to(device)

    # ERROR HANDLING: Prevents the pipeline from entering an undefined state.
    else:
        logger.error(f"Critical Error: Architecture '{model_type}' not supported by current Factory build.")
        raise ValueError(f"Unknown model type requested: {model_type}")