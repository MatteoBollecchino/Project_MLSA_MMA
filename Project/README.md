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
