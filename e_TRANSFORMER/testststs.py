import sys
import os
import torch
# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from e_TRANSFORMER.transformer_model import ChessTransformerDecoder
from tokenizers import Tokenizer
from pathlib import Path
import argparse

def load_model(checkpoint_path, device):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # If config is not in checkpoint, use default values
    if 'config' not in checkpoint:
        config = {
            'vocab_size': 852,  # Default vocab size
            'd_model': 128,
            'nhead': 8,
            'num_decoder_layers': 4,
            'dim_feedforward': 1024,
            'dropout': 0.1,
            'max_seq_length': 600
        }
    else:
        config = checkpoint['config']
    
    # Initialize model with config
    model = ChessTransformerDecoder(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_seq_length=config['max_seq_length']
    ).to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config

config = {
            'vocab_size': 852,  # Default vocab size
            'd_model': 256,
            'nhead': 8,
            'num_decoder_layers': 6,
            'dim_feedforward': 1024,
            'dropout': 0.1,
            'max_seq_length': 500
        }

model = ChessTransformerDecoder(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_seq_length=config['max_seq_length']
    )

print(model)

print(f'total parameters: {sum(p.numel() for p in model.parameters())}')
print(f'total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
