# Project Core

This directory contains the core source code for the Code Summarization project.

## Project Structure

- `data/`: Scripts for data downloading, preprocessing, and loading (`dataset.py`, `download_dataset.py`, `preprocess.py`).
- `models/`: Contains the implementation of the neural network models, such as `transformer.py` and `seq2seq.py`.
- `scripts/`: Main scripts to execute training (`train.py`) and other utilities.
- `C2Orchestrator.py`: The main orchestrator script to run training and evaluation tasks.
- `evaluate.py`: Script to evaluate a trained model.
- `requirements.txt`: A list of the Python dependencies for the project.
- `logs/`: Contains log files from training runs.
- `Datasets/`: Default directory where the datasets are stored.

## How to Run

### Prerequisites

1.  Make sure you are in this (`Project`) directory.
2.  Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Training Commands

**Transformer Model:**
To train the Transformer model with a subset of 50,000 samples for 10 epochs and a batch size of 128, run:
```sh
python C2Orchestrator.py --model transformer --subset 50000 --epochs 10 --batch_size 128 --force_train
```

**LSTM with Attention Model:**
To train the LSTM with Attention model with a subset of 50,000 samples for 5 epochs and a batch size of 64, run:
```sh
python C2Orchestrator.py --model lstm_attention --subset 50000 --epochs 5 --batch_size 64
```

## Command-line Arguments

The `C2Orchestrator.py` script accepts several arguments to control its behavior:

### Main Modes
-   `--mode`: Choose the operational mode.
    -   `train` (default): Train a new model.
    -   `eval`: Audit and evaluate existing checkpoints.
-   `--model`: (Required for `train` mode) Specifies the model architecture.
    -   `transformer`: Use the Transformer model.
    -   `lstm_attention`: Use the LSTM with Attention model.

### Training Parameters
-   `--subset`: Specifies the number of samples from the dataset to use for training. (e.g., `50000`). If not provided, the entire dataset is used.
-   `--epochs`: The number of complete passes through the training dataset. (default: `10`).
-   `--batch_size`: The number of samples per batch to load. (default: `128`).

### Evaluation Parameters
-   `--neval`: (For `eval` mode) Specifies which saved models to evaluate.
    -   `1` (default): Evaluates the most recent model.
    -   `all`: Evaluates all models found in the checkpoints directory.
    -   `1,2,4`: Evaluates specific models by their index in a reverse-chronological list.
-   `--evaluation`: Determines the number of samples to use for the evaluation process.
    -   `instant`: 100 samples.
    -   `fast` (default): 200 samples.
    -   `deep`: 1000 samples.

### Infrastructure Flags
-   `--force_download`: A flag that, when present, forces the re-download of the dataset even if it already exists.
-   `--force_preprocess`: A flag that forces the re-creation of the tokenizer vocabulary.
