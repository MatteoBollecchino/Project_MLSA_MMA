import logging
from models.transformer import Seq2SeqTransformer
from models.seq2seq_bahdanau import Seq2SeqBahdanau
from models.seq2seq_dotproduct import Seq2SeqDotProduct

logger = logging.getLogger(__name__)

def get_model_architecture(model_type, device, vocab_size=50000):
    params = {
        "emb_dim": 256, "hid_dim": 512, "n_layers": 2, 
        "dropout": 0.1, "nhead": 8, "transformer_layers": 6
    }

    if model_type == "lstm_bahdanau":
        logger.info("🏗️ Factory: Generating LSTM + Bahdanau (Additive)")
        return Seq2SeqBahdanau(vocab_size, params["emb_dim"], params["hid_dim"], 
                               params["n_layers"], params["dropout"], device).to(device)

    elif model_type == "lstm_dotproduct":
        logger.info("🏗️ Factory: Generating LSTM + DotProduct (Professor's Version)")
        return Seq2SeqDotProduct(vocab_size, params["emb_dim"], params["hid_dim"], 
                                 params["n_layers"], params["dropout"], device).to(device)

    elif model_type == "transformer":
        logger.info(f"⚡ Factory: Generating Transformer (Heads: {params['nhead']})")
        return Seq2SeqTransformer(vocab_size, d_model=params["hid_dim"], nhead=params["nhead"],
                                  num_layers=params["transformer_layers"], dropout=0.1).to(device)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")