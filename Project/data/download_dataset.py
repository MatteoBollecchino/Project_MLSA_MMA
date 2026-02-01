"""
================================================================================
DATA ACQUISITION UNIT
================================================================================
ROLE: Automating the Ingestion of the CodeSearchNet Corpus.

DESIGN RATIONALE:
- Non-Persistent Intermediate Storage: Downloads are handled in RAM-buffers 
  (BytesIO) to reduce SSD wear and tear during the decompression phase.
- Network Robustness: Uses specific User-Agent headers to bypass CDN anti-bot 
  heuristics that frequently block vanilla Python-requests signatures.
- High-Availability Source: Targets the Hugging Face CDN as the most stable 
  and high-bandwidth 'Source of Truth' for this dataset.
================================================================================
"""

import os
import requests
import zipfile
import io

# DOWNLOAD AND EXTRACTION OF CODESEARCHNET ROBUST DATASET
def download_codesearchnet_robust(output_dir="Project/Datasets"):
    """ 
    Orchestrates the automated retrieval and extraction of the Python dataset.
    
    Args:
        output_dir (str): Final destination for the unzipped JSONL files.
    """

    # SOURCE SELECTION: Mirrors on Hugging Face Assets are preferred for 
    # their consistent uptime and integrity-checking capabilities.
    url = "https://huggingface.co/datasets/code_search_net/resolve/main/data/python.zip"
    
    # Filesystem readiness check.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"--- Attempting acquisition via Hugging Face CDN ---")
    print(f"URL: {url}")
    print("Download in progress (about 900MB). Grab a coffee...")

    try:
        # BOT BYPASS PROTOCOL: Vanilla requests often trigger 403 Forbidden errors 
        # from CDNs (Cloudflare/Akamai). A human-like User-Agent ensures connectivity.
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        # STREAMING INITIATION: stream=True allows us to process the incoming 
        # binary stream without loading the entire 1GB into active memory at once.
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status() # Network-level sanity check (Exception on 4xx/5xx).

        # IN-MEMORY EXTRACTION: 
        # Mental Model: 'io.BytesIO' creates a virtual file in RAM.
        # This allows 'zipfile.ZipFile' to read the content as if it were on disk,
        # but with zero I/O latency.
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            print("Download completed. Extracting...")
            # Extractall flattens the hierarchy into the target Dataset directory.
            zip_ref.extractall(output_dir)
        
        # Verification phase: Audit the resulting directory structure.
        print(f"SUCCESS: Dataset extracted to {output_dir}")
        print("Detected structure: " + str(os.listdir(output_dir)))

    except Exception as e:
        # ERROR TELEMETRY: Provides the user with a manual remediation path.
        print(f"CRITICAL FAILURE: {e}")
        print("\n--- MANUAL SOLUTION ---")
        print(f"If the script fails, manually download from here: {url}")
        print(f"And extract the contents to: {os.path.abspath(output_dir)}")

# STANDALONE EXECUTION LOGIC
if __name__ == "__main__":
    download_codesearchnet_robust()