import torch
import torch.nn.functional as F
import json
import gzip
import os
import glob
import random
import sys
from tokenizers import Tokenizer
from models.factory import get_model_architecture

# --- [PHASE 1] DETERMINISTIC NORMALIZATION ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

def clean_output(text):
    """Removes BPE artifacts and normalizes spacing."""
    return text.replace('Ġ', ' ').replace('  ', ' ').strip()

# --- UNIVERSAL DECODING LOGIC ---
def autoregressive_decode(model, src_tensor, tokenizer, model_tag, max_len=40, device="cpu"):
    model.eval()
    sos_id = tokenizer.token_to_id("<SOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    
    with torch.no_grad():
        if model_tag == "lstm_attention":
            encoder_outputs, hidden, cell = model.encoder(src_tensor)
            input_token = torch.LongTensor([sos_id]).to(device)
            predicted_indices = []
            for _ in range(max_len):
                output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
                top1 = output.argmax(1)
                if top1.item() == eos_id: break
                predicted_indices.append(top1.item())
                input_token = top1
            return predicted_indices
        else:
            # Transformer logic
            ys = torch.ones(1, 1).fill_(sos_id).type(torch.long).to(device)
            predicted_indices = []
            for _ in range(max_len):
                out = model(src_tensor, ys)
                next_word = out[:, -1].argmax(1).item()
                if next_word == eos_id: break
                predicted_indices.append(next_word)
                ys = torch.cat([ys, torch.ones(1, 1).type(torch.long).fill_(next_word).to(device)], dim=1)
            return predicted_indices

# --- DATA LOADING ---
def load_samples(data_dir, split, num_samples=10):
    search_pattern = os.path.join(data_dir, split, "*.jsonl.gz")
    files = glob.glob(search_pattern)
    if not files: return []
    
    samples = []
    random.shuffle(files)
    for file_path in files:
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                lines = f.readlines()
                chosen = random.sample(lines, min(num_samples - len(samples), len(lines)))
                for c in chosen:
                    item = json.loads(c)
                    samples.append({
                        'code': item.get('code', item.get('func_code_string', '')),
                        'doc': item.get('docstring', item.get('func_documentation_string', ''))
                    })
            if len(samples) >= num_samples: break
        except: continue
    return samples[:num_samples]

# --- CORE AUDIT ENGINE ---
def run_deep_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_path = os.path.join(project_root, "tokenizer.json")
    checkpoint_dir = os.path.join(project_root, "models", "checkpoints")
    data_root = os.path.join(project_root, "Datasets", "python", "final", "jsonl")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    # --- SUITE 10-10-10 ALIGNMENT ---
    suites = {
        "TRAIN (Memorization)": load_samples(data_root, "train", 10),
        "TEST (Generalization)": load_samples(data_root, "test", 10),
        "CUSTOM (High Complexity Alignment)": [
            {"code": "def load_config(path):\n    with open(path, 'r') as f:\n        return json.load(f)", "doc": "Loads a configuration file from a JSON path."},
            {"code": "def get_db_connection(url):\n    try:\n        return create_engine(url).connect()\n    except Exception as e:\n        raise ConnectionError(e)", "doc": "Establishes a connection to the database or raises error."},
            {"code": "def process_image_batch(images, size):\n    return [cv2.resize(img, size) for img in images]", "doc": "Resizes a batch of images to a specific dimension."},
            {"code": "def calculate_rdd_mean(rdd):\n    return rdd.map(lambda x: (x, 1)).reduce(lambda a, b: (a[0]+b[0], a[1]+b[1]))", "doc": "Computes the mean of an RDD using map and reduce."},
            {"code": "def detect_faces_cnn(img):\n    return cnn_face_detector(img, upsample_num=1)", "doc": "Uses a CNN model to detect human faces in an image."},
            {"code": "def walk_dir_tree(root):\n    for root, dirs, files in os.walk(root):\n        yield files", "doc": "Recursively yields files from a directory tree structure."},
            {"code": "def apply_mask(tensor, mask_val=0):\n    return tensor.masked_fill(tensor == mask_val, float('-inf'))", "doc": "Applies a padding mask to a given input tensor."},
            {"code": "def validate_token_auth(token):\n    if not token.startswith('Bearer '):\n        return False\n    return verify_jwt(token)", "doc": "Verifies if the provided token is a valid JWT Bearer."},
            {"code": "def serialize_row_as_dict(row):\n    return {field: getattr(row, field) for field in row.__fields__}", "doc": "Converts a dataset row object into a standard dictionary."},
            {"code": "def compute_euclidean_dist(a, b):\n    return np.linalg.norm(a - b, axis=1)", "doc": "Calculates the euclidean distance between two numpy arrays."}
        ]
    }

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    print(f"\n{'='*90}\n🚀 AUDIT START | DEVICE: {device} | SAMPLES: 10-10-10\n{'='*90}")

    for ckpt in sorted(checkpoints):
        print(f"\n🔍 CHECKPOINT: {ckpt}")
        model_tag = "transformer" if "transformer" in ckpt.lower() else "lstm_attention"
        
        try:
            model = get_model_architecture(model_tag, device, vocab_size=vocab_size)
            model.load_state_dict(torch.load(os.path.join(checkpoint_dir, ckpt), map_location=device))
            model.eval()

            for suite_name, samples in suites.items():
                print(f"\n   >>> SUITE: {suite_name}")
                for i, s in enumerate(samples):
                    ids_input = tokenizer.encode(s['code']).ids
                    src_tensor = torch.LongTensor([1] + ids_input + [2]).unsqueeze(0).to(device)
                    
                    ids_pred = autoregressive_decode(model, src_tensor, tokenizer, model_tag, device=device)
                    prediction = clean_output(tokenizer.decode(ids_pred, skip_special_tokens=True))
                    
                    print(f"    S#{i+1} | CODE: {s['code'].strip().replace('\\n', ' ')[:45]}...")
                    print(f"        REAL: {s['doc'][:60].strip()}")
                    print(f"        PRED: {prediction}")
                    print(f"        {'-'*20}")
        except Exception as e:
            print(f"    ❌ Error: {e}")

if __name__ == "__main__":
    run_deep_audit()