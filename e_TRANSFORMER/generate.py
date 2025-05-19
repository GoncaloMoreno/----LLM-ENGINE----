import torch
import os
import sys

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
            'num_decoder_layers': 2,
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

def generate_games(
    model,
    tokenizer,
    num_games=5,
    max_length=200,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    device='cuda'
):
    # Start token is <S> (id=2)
    start_tokens = torch.tensor([[2]], device=device)
    start_tokens = start_tokens.repeat(num_games, 1)
    
    # Generate sequences
    with torch.no_grad():
        generated = model.generate(
            start_tokens=start_tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
    # Decode generated sequences
    games = []
    for seq in generated:
        # Convert to list and remove padding tokens
        token_ids = seq.tolist()
        token_ids = [t for t in token_ids if t != 0]  # Remove padding
        # Decode tokens to text
        game = tokenizer.decode(token_ids)
        games.append(game)
    
    return games

def generate_continuation(
    model,
    tokenizer,
    start_sequence,
    num_tokens=10,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    device='cuda'
):
    # Tokenize the start sequence
    start_tokens = tokenizer.encode(start_sequence).ids
    start_tokens = torch.tensor([start_tokens], device=device)
    
    print(f"Start tokens: {start_tokens[0].tolist()}")
    print(f"Start sequence decoded: {tokenizer.decode(start_tokens[0].tolist())}")
    
    # Generate continuation
    with torch.no_grad():
        generated = model.generate(
            start_tokens=start_tokens,
            max_length=len(start_tokens[0]) + num_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
    # Get only the new tokens
    new_tokens = generated[0, len(start_tokens[0]):]
    new_tokens = [t.item() for t in new_tokens if t != 0]  # Remove padding
    
    print(f"Generated token IDs: {new_tokens}")
    
    # Decode the new tokens
    continuation = tokenizer.decode(new_tokens)
    
    return continuation

# Example of how to use the functions directly
def example_usage():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model and tokenizer
    model, config = load_model('path/to/checkpoint.pt', device)
    tokenizer = Tokenizer.from_file('path/to/tokenizer.json')
    
    # Generate continuation
    start_sequence = "your starting sequence here"
    continuation = generate_continuation(
        model,
        tokenizer,
        start_sequence,
        num_tokens=10,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        device=device
    )
    print(f"Continuation: {continuation}")

def main():
    parser = argparse.ArgumentParser(description='Generate chess games using trained transformer')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer file')
    parser.add_argument('--mode', type=str, choices=['game', 'continuation'], default='game',
                        help='Generation mode: game or continuation')
    parser.add_argument('--start_sequence', type=str, help='Starting sequence for continuation mode')
    parser.add_argument('--num_tokens', type=int, default=10, help='Number of tokens to generate in continuation mode')
    parser.add_argument('--num_games', type=int, default=5, help='Number of games to generate in game mode')
    parser.add_argument('--max_length', type=int, default=200, help='Maximum sequence length')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.95, help='Nucleus sampling parameter')
    parser.add_argument('--output', type=str, default='generated_games.txt', help='Output file path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run generation on')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, config = load_model(args.checkpoint, args.device)
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    
    if args.mode == 'continuation':
        if not args.start_sequence:
            raise ValueError("start_sequence is required for continuation mode")
        
        print(f"Generating {args.num_tokens} tokens from sequence: {args.start_sequence}")
        continuation = generate_continuation(
            model,
            tokenizer,
            args.start_sequence,
            num_tokens=args.num_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device
        )
        
        print("\nGenerated continuation:")
        print(f"Start: {args.start_sequence}")
        print(f"Continuation: {continuation}")
        
        # Save to file
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            f.write(f"Start sequence: {args.start_sequence}\n")
            f.write(f"Continuation: {continuation}\n")
    
    else:  # game mode
        # Generate games
        print(f"Generating {args.num_games} games...")
        games = generate_games(
            model,
            tokenizer,
            num_games=args.num_games,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device
        )
        
        # Save generated games
        output_path = Path(args.output)
        print(f"Saving generated games to {output_path}")
        with open(output_path, 'w') as f:
            for i, game in enumerate(games, 1):
                f.write(f"Game {i}:\n{game}\n\n")
        
        # Also print the games
        print("\nGenerated games:")
        for i, game in enumerate(games, 1):
            print(f"\nGame {i}:")
            print(game)

if __name__ == '__main__':
    main() 