# Project_MLSA_MMA

## Authors

Matteo Bollecchino, Alessandro Nesti, Marco Pietri

## Project Structure

The project is organized into the following main directories:

- `Project/`: Contains the core source code for the project.
  - `data/`: Scripts for data downloading, preprocessing, and loading (`dataset.py`, `download_dataset.py`, `preprocess.py`).
  - `models/`: Contains the implementation of the neural network models, such as `transformer.py` and `seq2seq.py`.
  - `scripts/`: Main scripts to execute training (`train.py`) and other utilities.
  - `C2Orchestrator.py`: The main orchestrator script to run training and evaluation tasks.
  - `evaluate.py`: Script to evaluate a trained model.
  - `requirements.txt`: A list of the Python dependencies for the project.
- `Datasets/`: Directory where the datasets are stored, once downloaded.
- `Examples/`: Contains various examples and tests, like scripts to test HuggingFace models.
- `Lectures/`: Contains PDF files and notes from lectures.
- `TOOLS/`: Contains utility scripts for project maintenance, such as code snapshot generation and dependency reporting.

## How to Run

To run the training for the different models, use the `C2Orchestrator.py` script from the `Project/` directory.

### Prerequisites

1.  Navigate to the `Project` directory:
    ```sh
    cd Project
    ```
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
