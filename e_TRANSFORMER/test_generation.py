import torch
from transformer_model import ChessTransformerDecoder
from tokenizers import Tokenizer
from pathlib import Path
import os

def main():
    # Get the absolute path to the tokenizer file
    current_dir = Path(__file__).parent
    tokenizer_path = current_dir.parent / "b_TOKENIZER" / "chess_tokenizer.json"
    
    # Configuration
    config = {
        'vocab_size': 852,  # This should match your tokenizer's vocab size
        'd_model': 512,
        'nhead': 8,
        'num_decoder_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'max_seq_length': 5000,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Initialize model
    print("Initializing model...")
    model = ChessTransformerDecoder(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_seq_length=config['max_seq_length']
    ).to(config['device'])
    model.eval()
    
    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    # Generate some test sequences
    print("\nGenerating test sequences...")
    
    # Start token is <S> (id=2)
    start_tokens = torch.tensor([[2]], device=config['device'])
    start_tokens = start_tokens.repeat(3, 1)  # Generate 3 sequences
    
    # Try different temperatures
    temperatures = [0.5, 1.0, 2.0]
    
    for temp in temperatures:
        print(f"\nGenerating with temperature {temp}:")
        with torch.no_grad():
            generated = model.generate(
                start_tokens=start_tokens,
                max_length=50,  # Short sequences for testing
                temperature=temp,
                top_k=50,
                top_p=0.95
            )
        
        # Decode and print sequences
        for i, seq in enumerate(generated):
            # Convert to list and remove padding tokens
            token_ids = seq.tolist()
            token_ids = [t for t in token_ids if t != 0]  # Remove padding
            
            # Print raw tokens and decoded text
            print(f"\nSequence {i + 1}:")
            print(f"Token IDs: {token_ids}")
            print(f"Decoded text: {tokenizer.decode(token_ids)}")
            
            # Also print token meanings for debugging
            print("Token meanings:")
            for token_id in token_ids:
                token = tokenizer.decode([token_id])
                print(f"  {token_id}: '{token}'")

if __name__ == '__main__':
    main() 