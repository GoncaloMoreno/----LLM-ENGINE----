import os
import sys
import torch

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from e_TRANSFORMER.generate import load_model, generate_continuation
from tokenizers import Tokenizer

# Set up device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model and tokenizer
model, config = load_model(r'E:\LLM_ENGINE\checkpoints\supersmallmodel_epoch_last.pt', device)
tokenizer = Tokenizer.from_file(r'E:\LLM_ENGINE\b_TOKENIZER\chess_tokenizer.json')

# Generate continuation
start_sequence = "1"
continuation = generate_continuation(
    model,
    tokenizer,
    start_sequence,
    num_tokens=30,  # number of tokens to generate
    temperature=0.6,  # control randomness
    top_k=10,  # limit vocabulary to top k tokens
    top_p=0.98,  # nucleus sampling parameter
    device=device
)

print(f"Continuation: {continuation}")


# supposed to be: 38, 8, 46, 14, 46, 15, 42, 8,52, 13, 55,