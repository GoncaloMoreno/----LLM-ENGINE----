from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tqdm import tqdm
import os
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


# normal BPE took ~9-10 mins per game file (10GB); 4 files total
# ByteLevel took ~25mins per game file

# --- Setup ---
tokenizer = Tokenizer(models.BPE())
#tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
#tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
tokenizer.decoder = ByteLevelDecoder()

trainer = trainers.BpeTrainer(
    vocab_size=2048,
    special_tokens=["<PAD>", "<MASK>", "<S>", "<UNK>"]
)

# --- Generator that yields lines with progress ---
def line_generator_with_progress(file_paths):
    for file_path in file_paths:
        with open(file_path, "r") as f:
            for line in tqdm(f, desc=f"Processing {os.path.basename(file_path)}"):
                yield line.strip()

#--- Files ---
files = [
    r"C:\Users\moren\Desktop\Masters\--- LLM ENGINE ---\1 DATA + CLEANUP\game_files\lichess_1.txt",
    r"C:\Users\moren\Desktop\Masters\--- LLM ENGINE ---\1 DATA + CLEANUP\game_files\lichess_2.txt",
    r"C:\Users\moren\Desktop\Masters\--- LLM ENGINE ---\1 DATA + CLEANUP\game_files\lichess_3.txt",
    r"C:\Users\moren\Desktop\Masters\--- LLM ENGINE ---\1 DATA + CLEANUP\game_files\lichess_4.txt"
]

# for testing easily
#files = [r"C:\Users\moren\Desktop\Masters\--- LLM ENGINE ---\1 DATA + CLEANUP\game_files\chunk_1.txt"]

# --- Get total line count (optional but cleaner progress) ---
total_lines = 100152321 # should be 100412379(?) oops

# --- Train tokenizer ---
tokenizer.train_from_iterator(line_generator_with_progress(files), trainer=trainer, length=total_lines)

# --- Save ---
tokenizer.save("chess_tokenizer.json")