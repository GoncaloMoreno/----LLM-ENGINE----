import torch
from tokenizers import Tokenizer
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
import json


# Honestly this code gave me WinError 1450 a bunch of times but i didnt feel like debugging it lol
# Total tokens turned out to be = 21,180,260,593
total_tokens = 0

# --- Config ---
TOKENIZER_PATH = "chess_tokenizer.json"
INPUT_FILES = [
    r'C:\Users\moren\Desktop\Masters\--- LLM ENGINE ---\1 DATA + CLEANUP\game_files\lichess_1.txt',
    r'C:\Users\moren\Desktop\Masters\--- LLM ENGINE ---\1 DATA + CLEANUP\game_files\lichess_2.txt',
    r'C:\Users\moren\Desktop\Masters\--- LLM ENGINE ---\1 DATA + CLEANUP\game_files\lichess_3.txt',
    r'C:\Users\moren\Desktop\Masters\--- LLM ENGINE ---\1 DATA + CLEANUP\game_files\lichess_4.txt'
]

OUTPUT_DIR = r'C:\Users\moren\Desktop\Masters\--- LLM ENGINE ---\1 DATA + CLEANUP\game_files'

LINES_PER_CHUNK = 2_500_000
NUM_WORKERS = 2

# --- Tokenization function to run in subprocesses ---
def tokenize_lines(lines):
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    return [tokenizer.encode(line).ids for line in lines]

# --- Main tokenization + save ---
def process_files():
    chunk_idx = 0                                                    
    for file_path in INPUT_FILES:
        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            batch_lines = []
            for line in tqdm(f, desc=f"Reading {filename}"):
                line = line.strip()
                if not line:
                    continue
                batch_lines.append(line)

                if len(batch_lines) >= LINES_PER_CHUNK:
                    save_chunk(batch_lines, chunk_idx)
                    chunk_idx += 1
                    batch_lines = []

            if batch_lines:
                save_chunk(batch_lines, chunk_idx)
                chunk_idx += 1

# --- Save a single chunk ---
def save_chunk(lines, chunk_idx):
    global total_tokens
    print(f"Tokenizing chunk {chunk_idx} with {len(lines)} lines...")

    # Split into mini-batches to prevent one giant list
    BATCH_SIZE = 50_000
    MAX_JOBS = 2
    all_ids = []
    index_map = []  # To store the start and end positions of each game
    current_position = 0  # Tracks the current position in the flat tensor

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for i in range(0, len(lines), BATCH_SIZE):
            mini_batch = lines[i:i + BATCH_SIZE]
            futures.append(executor.submit(tokenize_lines, mini_batch))

        # If the number of jobs exceeds MAX_JOBS, wait for some to complete
        if len(futures) >= MAX_JOBS:
                for fut in tqdm(futures, desc=f"Processing chunk {chunk_idx} (waiting for jobs)"):
                    ids = fut.result()
                    for game_ids in ids:
                        all_ids.extend(game_ids)  # Flatten the tokenized IDs
                        index_map.append((current_position, current_position + len(game_ids)))
                        current_position += len(game_ids)
                futures = []  # Clear the completed futures

        for fut in tqdm(futures, desc=f"Processing chunk {chunk_idx}"):
            ids = fut.result()
            for game_ids in ids:
                all_ids.extend(game_ids)  # Flatten the tokenized IDs
                index_map.append((current_position, current_position + len(game_ids)))
                current_position += len(game_ids)
    
    # save the tokenized tensor
    token_tensor = torch.tensor(all_ids, dtype=torch.int32)
    output_path = os.path.join(OUTPUT_DIR, f"tokenized_chunk_{chunk_idx}.pt")
    torch.save(token_tensor, output_path)
    print(f"Saved chunk {chunk_idx} to {output_path} ({len(token_tensor)} tokens)")

    # Save the index map
    index_map_path = os.path.join(OUTPUT_DIR, f"index_map_{chunk_idx}.json")
    with open(index_map_path, "w") as f:
        json.dump(index_map, f)
    print(f"Saved index map for chunk {chunk_idx} to {index_map_path}")

    # just a counter
    total_tokens += len(token_tensor)
    print(f"Total tokens so far: {total_tokens}")

if __name__ == "__main__":
    process_files()