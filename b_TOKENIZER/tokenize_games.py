# Total games: 10,000,000
#Total tokens: 1,453,788,622

import torch
from tokenizers import Tokenizer
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
import json

# --- Config ---
# Use absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(SCRIPT_DIR, "chess_tokenizer_CLEAN.json")
INPUT_FILES = [
    r'a_DATA_CLEANUP\lichess_1_CLEAN.txt'
]

OUTPUT_DIR = r'a_DATA_CLEANUP\games_for_CLEAN'
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output directory if it doesn't exist

LINES_PER_CHUNK = 1_000_000  # 2.5M games per chunk
MAX_CHUNKS = 10  # Stop after 4 chunks (10M games total)
NUM_WORKERS = 2

# --- Tokenization function to run in subprocesses ---
def tokenize_lines(lines):
    try:
        # Use absolute path for tokenizer
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        # Add <S> token (id: 0) at the start of each line
        return [[0] + tokenizer.encode(line).ids for line in lines]
    except Exception as e:
        print(f"Error in tokenize_lines: {e}")
        print(f"Tokenizer path: {TOKENIZER_PATH}")
        return []

# --- Main tokenization + save ---
def process_files():
    chunk_idx = 0
    total_tokens = 0
    total_games = 0
    
    for file_path in INPUT_FILES:
        if chunk_idx >= MAX_CHUNKS:
            print(f"\nReached maximum number of chunks ({MAX_CHUNKS}). Stopping.")
            break
            
        filename = os.path.basename(file_path)
        print(f"\nProcessing {filename}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                batch_lines = []
                for line in tqdm(f, desc=f"Reading {filename}"):
                    if chunk_idx >= MAX_CHUNKS:
                        break
                        
                    line = line.strip()
                    if not line:
                        continue
                    batch_lines.append(line)
                    total_games += 1

                    if len(batch_lines) >= LINES_PER_CHUNK:
                        tokens_in_chunk = save_chunk(batch_lines, chunk_idx)
                        total_tokens += tokens_in_chunk
                        chunk_idx += 1
                        batch_lines = []
                        
                        print(f"\nProgress: {chunk_idx}/{MAX_CHUNKS} chunks")
                        print(f"Total games processed: {total_games:,}")
                        print(f"Total tokens: {total_tokens:,}")

                if batch_lines and chunk_idx < MAX_CHUNKS:
                    tokens_in_chunk = save_chunk(batch_lines, chunk_idx)
                    total_tokens += tokens_in_chunk
                    chunk_idx += 1
                    
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue

    print(f"\nProcessing complete!")
    print(f"Total chunks processed: {chunk_idx}")
    print(f"Total games: {total_games:,}")
    print(f"Total tokens: {total_tokens:,}")

# --- Save a single chunk ---
def save_chunk(lines, chunk_idx):
    print(f"\nTokenizing chunk {chunk_idx} with {len(lines):,} games...")

    # Split into mini-batches to prevent one giant list
    BATCH_SIZE = 50_000
    MAX_JOBS = 2
    all_ids = []
    index_map = []  # To store the start and end positions of each game
    current_position = 0  # Tracks the current position in the flat tensor

    try:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for i in range(0, len(lines), BATCH_SIZE):
                mini_batch = lines[i:i + BATCH_SIZE]
                futures.append(executor.submit(tokenize_lines, mini_batch))

            # Process completed futures
            for fut in tqdm(futures, desc=f"Processing chunk {chunk_idx}"):
                try:
                    ids = fut.result()
                    for game_ids in ids:
                        all_ids.extend(game_ids)  # Flatten the tokenized IDs
                        index_map.append((current_position, current_position + len(game_ids)))
                        current_position += len(game_ids)
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue

        # Save the tokenized tensor
        token_tensor = torch.tensor(all_ids, dtype=torch.int32)
        output_path = os.path.join(OUTPUT_DIR, f"tokenized_chunk_{chunk_idx}.pt")
        torch.save(token_tensor, output_path)
        print(f"Saved chunk {chunk_idx} to {output_path} ({len(token_tensor):,} tokens)")

        # Save the index map
        index_map_path = os.path.join(OUTPUT_DIR, f"index_map_{chunk_idx}.json")
        with open(index_map_path, "w") as f:
            json.dump(index_map, f)
        print(f"Saved index map for chunk {chunk_idx} to {index_map_path}")

        return len(token_tensor)
        
    except Exception as e:
        print(f"Error saving chunk {chunk_idx}: {e}")
        return 0

if __name__ == "__main__":
    process_files()