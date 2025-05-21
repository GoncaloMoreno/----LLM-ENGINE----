import os
import sys
import torch

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from e_TRANSFORMER.generate import load_model, generate_continuation
from tokenizers import Tokenizer
from e_TRANSFORMER.transformer_model import ChessTransformerDecoder, ChessTransformerConfig

# Set up device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define model configuration
model_config = ChessTransformerConfig(
    vocab_size=852,
    d_model=256,
    nhead=8,
    num_decoder_layers=6,
    dim_feedforward=1024,
    dropout=0.1,
    max_seq_length=500
)

# Initialize model with configuration
model = ChessTransformerDecoder(model_config).to(device)

# Try to load checkpoint if it exists
checkpoint_path = r'E:\LLM_ENGINE\checkpoints\model_step_2500.pt'
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print("No checkpoint found, using randomly initialized model")

model.eval()

# Load tokenizer
tokenizer = Tokenizer.from_file(r'E:\LLM_ENGINE\b_TOKENIZER\chess_tokenizer.json')

# Generate continuation
start_sequence = "1. "
print(f"\nGenerating continuation for: '{start_sequence}'")
print(f"Model configuration:")
print(f"- Temperature: 1.2 (increased for more randomness)")
print(f"- Top-k: 50 (increased for more variety)")
print(f"- Number of tokens: 20 (increased for longer sequence)")

continuation = generate_continuation(
    model,
    tokenizer,
    start_sequence,
    num_tokens=20,  # increased number of tokens
    temperature=1.2,  # increased temperature for more randomness
    top_k=50,  # increased top_k for more variety
    device=device
)

print(f"\nFull sequence: {start_sequence}{continuation}")
