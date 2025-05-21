import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from torch.nn import functional as F

@dataclass
class ChessTransformerConfig:
    vocab_size: int = 2048
    d_model: int = 512
    nhead: int = 8
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 5000
    pad_token_id: int = 0
    bias: bool = True

class ChessTransformerDecoder(nn.Module):
    def __init__(self, config: ChessTransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embedding layer
        self.wte = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        # Positional embedding layer (learned)
        self.wpe = nn.Embedding(config.max_seq_length, config.d_model)
        
        self.drop = nn.Dropout(config.dropout)
        
        # Create decoder layer and stack them
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            bias=config.bias
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers
        )
        
        # Final linear layer to project back to vocabulary size
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight

        # Initialize parameters
        self.apply(self._init_weights)

        # Apply special scaled init to the residual projections (c_proj in NanoGPT's Block)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                pass

    def _init_weights(self, module):
        """Initialize the parameters, similar to NanoGPT."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                 with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = idx.device
        b, t = idx.size()

        if t > self.config.max_seq_length:
            idx = idx[:, :self.config.max_seq_length]
            if targets is not None:
                targets = targets[:, :self.config.max_seq_length]
            t = self.config.max_seq_length

        tgt_mask = self.generate_square_subsequent_mask(t, device)
        
        tgt_key_padding_mask = (idx == self.config.pad_token_id)
        
        tok_emb = self.wte(idx)
        
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.wpe(pos)
        
        x = self.drop(tok_emb + pos_emb) 
        
        output = self.transformer_decoder(
            tgt=x,
            memory=x,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=tgt_key_padding_mask
        )
        
        loss = None
        if targets is not None:
            logits = self.lm_head(output)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.config.pad_token_id)
        else:
            logits = self.lm_head(output[:, [-1], :])

        return logits, loss

    def generate(
        self,
        start_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None
    ) -> torch.Tensor:
        self.eval()
        current_seq = start_tokens
            
        for _ in range(max_new_tokens):
            seq_cropped = current_seq if current_seq.size(1) <= self.config.max_seq_length else current_seq[:, -self.config.max_seq_length:]
            
            logits, _ = self(seq_cropped)
            
            next_token_logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1)
            
            current_seq = torch.cat([current_seq, next_token], dim=1)
            
        return current_seq