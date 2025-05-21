################################################################################################
################################################################################################
######### This script processes a large PGN file (lichess_db_standard_rated_2025-01.pgn)   #####
#########                extracting only the moves from each game                          #####
#########    It splits the file into smaller chunks, processes each chunk in parallel,     #####
#########             and then merges the results into a final output file                 #####
#########             games from lichess_db_standard_rated_2025-01.pgn
################################################################################################
################################################################################################
#################      Reading PGN: 2008410160it [41:10, 813095.89it/s]      ###################
#################       Processing 4017 chunks using 12 cores [10:53]        ###################
################################################################################################
################################################################################################

import os
from glob import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# === CONFIG ===
GAMES_PER_CHUNK = 50000  # Good for ~16GB RAM
CHUNKS_DIR = "chunks"
OUTPUT_FILE = "moves_only.pgn"


def split_pgn_by_game(input_path, output_dir, games_per_file=GAMES_PER_CHUNK):
    os.makedirs(output_dir, exist_ok=True)
    game_buffer = []
    file_count = 0
    game_count = 0

    print("🔪 Splitting PGN into chunks...")

    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, desc="Reading PGN"):
            game_buffer.append(line)
            if line.strip() == "":
                game_count += 1
            if game_count >= games_per_file:
                chunk_path = os.path.join(output_dir, f"chunk_{file_count}.pgn")
                with open(chunk_path, 'w', encoding='utf-8') as out:
                    out.writelines(game_buffer)
                game_buffer = []
                game_count = 0
                file_count += 1

        if game_buffer:
            chunk_path = os.path.join(output_dir, f"chunk_{file_count}.pgn")
            with open(chunk_path, 'w', encoding='utf-8') as out:
                out.writelines(game_buffer)


def process_chunk(chunk_file):
    out_file = chunk_file.replace(".pgn", "_moves_only.pgn")

    with open(chunk_file, 'r', encoding='utf-8') as infile, open(out_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()
            if line and not line.startswith('[') and line[0].isdigit() and '.' in line:
                outfile.write(line + '\n')

    return out_file


def parallel_process_chunks(chunks_dir):
    chunk_files = glob(os.path.join(chunks_dir, "chunk_*.pgn"))
    print(f"⚙️ Processing {len(chunk_files)} chunks using {cpu_count()} cores...")

    results = []
    with Pool() as pool:
        for result in tqdm(pool.imap_unordered(process_chunk, chunk_files), total=len(chunk_files), desc="Processing chunks"):
            results.append(result)

    return results


#######################################################################
# def merge_processed_chunks(processed_files, output_path):
#     print("🧩 Merging all processed chunks into final output...")
#     with open(output_path, 'w', encoding='utf-8') as outfile:
#         for fpath in tqdm(processed_files, desc="Merging"):
#             with open(fpath, 'r', encoding='utf-8') as infile:
#                 outfile.write(infile.read())

#######################################################################
# Merge all processed chunks into one big PGN file without loading into RAM

# CHUNKS_DIR = "chunks"  # Change if your files are elsewhere
# OUTPUT_FILE = "moves_only.pgn"

# def merge_processed_chunks(output_path):
#     """Merges all *_moves_only.pgn files into one big PGN file without loading into RAM."""
#     chunk_files = sorted(glob(os.path.join(CHUNKS_DIR, "*_moves_only.pgn")))  # Sort to maintain order

#     print(f"🧩 Merging {len(chunk_files)} files without using RAM...")
    
#     with open(output_path, 'w', encoding='utf-8') as outfile:
#         for chunk in tqdm(chunk_files, desc="Merging", unit="file"):
#             with open(chunk, 'r', encoding='utf-8') as infile:
#                 for line in infile:
#                     outfile.write(line)

#     print(f"✅ Done! Final merged file: {output_path}")

# # Run the merge function
# merge_processed_chunks(OUTPUT_FILE)
#######################################################################

def main():
    input_file = r'C:\Users\moren\Desktop\Masters\--- LLM ENGINE ---\1 DATA + CLEANUP\game_files\lichess_db_standard_rated_2025-01.pgn'  # change this if needed
    split_pgn_by_game(input_file, CHUNKS_DIR)
    processed = parallel_process_chunks(CHUNKS_DIR)
    #merge_processed_chunks(processed, OUTPUT_FILE)
    print(f"✅ Done! Final file: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

# delete original chunk files
CHUNKS_DIR = "chunks"
# Find all original chunk files (without "_moves_only")
chunk_files = glob(os.path.join(CHUNKS_DIR, "chunk_*.pgn"))
chunk_files = [f for f in chunk_files if "_moves_only" not in f]
# chunk_files = [f for f in chunk_files if "_moves_only_cleaned" not in f]

print(f"🗑️ Deleting {len(chunk_files)} original chunk files...")

for file in chunk_files:
    os.remove(file)

print("✅ Done! Only the 'moves_only' files remain.")

# Run clean_moves_only.py to clean the moves_only files
# os.system("python clean_moves_only.py chunks")

# keep only cleaned moves files
CHUNKS_DIR = "chunks"
# Find all original chunk files (without "_moves_only")
chunk_files = glob(os.path.join(CHUNKS_DIR, "chunk_*.pgn"))
chunk_files = [f for f in chunk_files if "_moves_only_cleaned" not in f]

print(f"🗑️ Deleting {len(chunk_files)} original chunk files...")

for file in chunk_files:
    os.remove(file)
