"""
Chess Transformer Generation Script

Edit the variables below to:
1. Set the CHECKPOINT_PATH (or leave as None for random initialization)
2. Set the START_SEQUENCE (the input text for the model)
3. Adjust generation parameters (NUM_TOKENS, TEMPERATURE, TOP_K)

Then run:
$ python generate_chess.py
"""

import os
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
from tokenizers import Tokenizer
import pickle

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .nanomodel_chess import GPTConfig, GPT

# --- User Configuration ---
CHECKPOINT_PATH = r"E:\LLM_ENGINE\checkpoints\CLEANModels\L1_1506.pt"  # Set to None for random init
START_SEQUENCE = "<S>"  # Your input sequence in plain text
NUM_TOKENS = 50  # Number of new tokens to generate
TEMPERATURE = 1.0  # Sampling temperature (higher = more random)
TOP_K = 50  # Top-k sampling (None = no limit, 0 = no limit)

# --- Model and Tokenizer Configuration (should generally match training) ---
BLOCK_SIZE = 550
VOCAB_SIZE = 530 # From your tokenizer
N_LAYER = 8
N_HEAD = 8
N_EMBD = 512
DROPOUT = 0.1
BIAS = True
TOKENIZER_FILE = "chess_tokenizer_CLEAN.json" # Relative to b_TOKENIZER directory

def load_model(checkpoint_path_str=None):
    config = GPTConfig(
        block_size=BLOCK_SIZE,
        vocab_size=VOCAB_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
        bias=BIAS
    )
    model = GPT(config)

    if checkpoint_path_str:
        checkpoint_path = Path(checkpoint_path_str).resolve()
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            except RuntimeError as e:
                print(f"Initial torch.load failed: {e}. Trying with pickle_module...")
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', pickle_module=pickle)
                except Exception as e_pickle:
                    print(f"torch.load with pickle_module also failed: {e_pickle}.")
                    print("Model loading failed. Please check the checkpoint file and PyTorch versions.")
                    return None # Or raise an error
            
            if 'model_args' in checkpoint:
                print("Checkpoint model args:")
                for k, v in checkpoint['model_args'].items():
                    print(f"  {k}: {v}")
            else:
                print("Warning: 'model_args' not found in checkpoint.")

            try:
                model.load_state_dict(checkpoint['model'])
                print(f"Successfully loaded model state_dict. Iteration: {checkpoint.get('iter_num', 'N/A')}")
            except Exception as e_state_dict:
                print(f"Error loading state_dict: {e_state_dict}")
                print("Trying with strict=False...")
                try:
                    model.load_state_dict(checkpoint['model'], strict=False)
                    print(f"Successfully loaded model state_dict with strict=False. Iteration: {checkpoint.get('iter_num', 'N/A')}")
                except Exception as e_strict_false:
                    print(f"Error loading state_dict with strict=False: {e_strict_false}")
                    print("Model loading failed. State_dict mismatch.")
                    return None # Or raise an error
        else:
            print(f"Checkpoint not found at {checkpoint_path}. Using randomly initialized model.")
    else:
        print("No checkpoint path provided. Using randomly initialized model.")
    
    return model

def load_tokenizer():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer_full_path = os.path.join(base_dir, 'b_TOKENIZER', TOKENIZER_FILE)
    if not os.path.exists(tokenizer_full_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_full_path}. Ensure TOKENIZER_FILE is correct.")
    return Tokenizer.from_file(tokenizer_full_path)

def generate_text(model, tokenizer, start_sequence, num_tokens, temperature, top_k):
    model.eval()
    device = next(model.parameters()).device # Infer device from model
    
    encoded_input = tokenizer.encode(start_sequence)
    input_ids = torch.tensor([encoded_input.ids], dtype=torch.long, device=device) # Add batch dimension
    
    print(f"\nInput sequence: '{start_sequence}' (Tokens: {encoded_input.ids})")
    print(f"Generating {num_tokens} new tokens...")
    
    with torch.no_grad():
        output_ids_tensor = model.generate(input_ids, num_tokens, temperature=temperature, top_k=top_k if top_k > 0 else None)
    
    # The model.generate returns the full sequence (input + generated)
    # We need to slice off the input part if we only want the generated part
    # However, the tokenizer.decode handles the full sequence naturally.
    generated_tokens = output_ids_tensor[0].tolist()
    decoded_text = tokenizer.decode(generated_tokens)
    
    return decoded_text, generated_tokens

def play(sequence="<S>", num_tokens=10, temperature=1.0, top_k=50, checkpoint_path=None):
    """
    Generate a sequence of tokens using the chess transformer model.
    
    Args:
        sequence: Input text sequence (default: "<S>")
        num_tokens: Number of tokens to generate (default: 10)
        temperature: Sampling temperature (default: 1.0)
        top_k: Top-k sampling parameter (default: 50)
        checkpoint_path: Path to model checkpoint (default: uses L1_1506.pt)
    
    Returns:
        Generated sequence as a string
    """
    # Set default checkpoint if none provided
    if checkpoint_path is None:
        checkpoint_path = r"E:\LLM_ENGINE\checkpoints\CLEANModels\L1_1506.pt"
    
    # Load model and tokenizer
    model = load_model(checkpoint_path)
    if model is None:
        raise RuntimeError("Failed to load model")
    
    tokenizer = load_tokenizer()
    
    # Move model to appropriate device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # Encode the input sequence
    encoded_input = tokenizer.encode(sequence)
    input_ids = torch.tensor([encoded_input.ids], dtype=torch.long, device=device)
    
    print(f"\nInput sequence: '{sequence}' (Tokens: {encoded_input.ids})")
    print(f"Generating {num_tokens} new tokens...")
    
    with torch.no_grad():
        output_ids_tensor = model.generate(input_ids, num_tokens, temperature=temperature, top_k=top_k if top_k > 0 else None)
    
    # Decode the generated sequence
    generated_tokens = output_ids_tensor[0].tolist()
    decoded_text = tokenizer.decode(generated_tokens)
    
    return decoded_text

def main():
    # Example usage
    test_sequence = "<S>"
    print(f"\nGenerating sequence starting with: '{test_sequence}'")
    generated_text = play(test_sequence, num_tokens=20)
    print(f"\nGenerated sequence: {generated_text}")

if __name__ == '__main__':
    main() 