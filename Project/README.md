# Project Description

The C2 Master Orchestrator is a modular neural pipeline designed for Code Summarization (automatic analysis and summarization of source code). The system aims to transform raw Python code into concise natural language descriptions, optimizing the signal-to-noise ratio through a series of data refinement stages and model training.

The system follows the "Factory" design pattern to decouple architecture selection from the training loop and implements a strict separation between data ingestion, cleaning, and training.

## Authors

Marco Pietri, Alessandro Nesti, Matteo Bollecchino

---

## Project Structure
The code is organized into the following main directories and files:

- `C2Orchestrator.py`: The "Central Management Unit" (CMU). It is the main script that orchestrates the entire machine learning lifecycle, managing both training and evaluation.

- `data/`: Contains scripts for dataset management:
    - `dataset_cleaner.py`: Module for code sanitization (comment removal, MD5 deduplication).
    - `dataset.py`: Handles data loading (DataLoaders).
    - `download_dataset.py`: Downloads the dataset (CodeSearchNet in our case).
    - `inspect_dataset.py`: Generates human-readable snapshots of compressed for a qualitative evaluation. 
    - `tokenizer.py`: Construction and management of the BPE vocabulary.
- `Datasets/`: Default directory for storing raw and processed dataset.
- `logs/`: Directory where execution logs and training metrics are saved.
- `models/`: Contains neural architecture definitions (e.g., Transformer, Encoder-Decoder with LSTM).
- `scripts/`: Utility scripts and training logic (`train.py`, `log_manager.py`).

- `colab_remote_exec.txt`: Script to be executed in Colab environment to train and evaluate a model
- `evaluate.py`: Script dedicated to checkpoint evaluation, calculating metrics such as BLEU and ROUGE, and managing "Audit Mode".
- `requirements.txt`: List of requirements needed to run the code
- `test_audit.py`: Model assessment via Beam Search and Suite Testing.
- `tokenizer.json`: Contains the BPE Tokenizer (general info, vocabulary and merges)

---

## Usage Guide

### Prerequisites

1.  Ensure you are in the root directory of the project (`Project`).
2.  Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Training Commands
To start training a new model, use the `C2Orchestrator.py` script with the `--mode train` flag (default). It is mandatory to specify the model architecture using `--model`.

**Example 1: Transformer Training**
Trains a Transformer model on a subset of 50,000 samples for 10 epochs:
```bash
python C2Orchestrator.py --mode train --model transformer --subset 50000 --epochs 10 --batch_size 128 --force_train

```

**Example 2: LSTM Training (Bahdanau Attention)**
Trains an LSTM model with additive attention:

```bash
python C2Orchestrator.py --mode train --model lstm_bahdanau --subset 50000 --epochs 10 --batch_size 64

```

### Evaluation

To evaluate saved models (checkpoints), use the `--mode eval` mode. The system will look for `.pt` files in the checkpoints folder.

**Example: Audit of the last saved model**

```bash
python C2Orchestrator.py --mode eval --neval 1 --evaluation fast

```

**Example: Audit of all available checkpoints**

```bash
python C2Orchestrator.py --mode eval --neval all

```

---

## Command-line Arguments

The `C2Orchestrator.py` script accepts several arguments to control its behavior:

### Main Modes
- `--mode`: Choose the operational mode.
    - `train` (default): Train a new model.
    - `eval`: Audit and evaluate existing checkpoints.
- `--model`: (Required for `train` mode) Specifies the model architecture.
    - `transformer`: Use the Transformer model.
    - `lstm_bahdanau`: Use the Encoder-Decoder LSTM with Bahdanau Attention model.
    - `lstm_dotproduct`: Use the Encoder-Decoder LSTM with Dot-Product Attention model.

### Training Parameters
- `--subset`: Specifies the number of samples from the dataset to use for training. (e.g., `50000`). If not provided, the entire dataset is used.
- `--epochs`: The number of complete passes through the training dataset. (default: `10`).
- `--batch_size`: The number of samples per batch to load. (default: `128`).

### Evaluation Parameters
- `--neval`: (For `eval` mode) Specifies which saved models to evaluate.
    - `1` (default): Evaluates the most recent model.
    - `all`: Evaluates all models found in the checkpoints directory.
    - `1,2,4`: Evaluates specific models by their index in a reverse-chronological list.
- `--evaluation`: Determines the number of samples to use for the evaluation process.
    - `instant`: 100 samples.
    - `fast` (default): 200 samples.
    - `deep`: 1000 samples.

### Infrastructure Flags
- `--force_download`: A flag that, when present, forces the re-download of the dataset even if it already exists.
- `--force_preprocess`: A flag that forces the re-creation of the tokenizer vocabulary.
- `--force_train`: Overwrites existing sessions.


---