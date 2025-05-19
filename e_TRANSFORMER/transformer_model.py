import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        if x.size(0) > self.max_len:
            print(f"Warning: Input sequence length {x.size(0)} exceeds max_len {self.max_len}. Truncating.")
            x = x[:self.max_len]
        return x + self.pe[:x.size(0)]

class ChessTransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 2048,
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 5000,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Create decoder layer and stack them
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Final linear layer to project back to vocabulary size
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize the parameters using standard transformer initialization."""
        # Initialize all parameters with xavier uniform
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Initialize embedding layer with standard xavier uniform
        nn.init.xavier_uniform_(self.embedding.weight)
        # Keep padding token as zeros
        with torch.no_grad():
            self.embedding.weight[self.pad_token_id].fill_(0)
        
        # Initialize output layer normally (remove the small gain)
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: Target sequence [batch_size, seq_len]
            memory: Memory from encoder (None for decoder-only) [batch_size, memory_len, d_model]
            tgt_mask: Target sequence mask
            memory_mask: Memory mask
            tgt_key_padding_mask: Target key padding mask
            memory_key_padding_mask: Memory key padding mask
        """
        device = tgt.device
        
        # Truncate sequence if it's too long
        if tgt.size(1) > self.max_seq_length:
            #print(f"Warning: Input sequence length {tgt.size(1)} exceeds max_seq_length {self.max_seq_length}. Truncating.")
            tgt = tgt[:, :self.max_seq_length]
        
        # Create target mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), device)
        
        # Create padding mask if not provided
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = (tgt == self.pad_token_id)
        
        # Convert boolean mask to float mask for consistency
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.float().masked_fill(
                tgt_key_padding_mask == True, float('-inf')
            ).masked_fill(tgt_key_padding_mask == False, float(0.0))
        
        # Embed tokens and add positional encoding
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt.transpose(0, 1)).transpose(0, 1)
        
        # If no memory is provided (decoder-only), use target as memory
        if memory is None:
            memory = tgt
            memory_key_padding_mask = tgt_key_padding_mask
        
        # Convert memory padding mask to float mask for consistency
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.float().masked_fill(
                memory_key_padding_mask == True, float('-inf')
            ).masked_fill(memory_key_padding_mask == False, float(0.0))
        
        # Pass through transformer decoder
        output = self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Project to vocabulary size
        output = self.output_layer(output)
        
        return output

    def generate(
        self,
        start_tokens: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None
    ) -> torch.Tensor:
        """
        Generate a sequence given starting tokens.
        
        Args:
            start_tokens: Starting tokens [batch_size, seq_len]
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Cumulative probability for nucleus sampling
        """
        self.eval()
        device = start_tokens.device
        
        with torch.no_grad():
            batch_size = start_tokens.size(0)
            current_seq = start_tokens
            
            for _ in range(max_length - start_tokens.size(1)):
                # Get model predictions
                logits = self(current_seq)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Get probabilities
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # Print top 5 tokens and their probabilities
                top_probs, top_indices = torch.topk(probs[0], 5)
                print("\nTop 5 tokens and probabilities:")
                for prob, idx in zip(top_probs, top_indices):
                    print(f"Token {idx.item()}: {prob.item():.4f}")
                
                # Apply top-k sampling if specified
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) sampling if specified
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                print(f"Selected token: {next_token[0].item()}")
                
                # Concatenate with current sequence
                current_seq = torch.cat([current_seq, next_token], dim=1)
                
                # Stop if all sequences have reached the end token (you may want to modify this based on your tokenizer)
                if (next_token == 2).all():  # Assuming 2 is your end token (<S>)
                    break
            
            return current_seq 