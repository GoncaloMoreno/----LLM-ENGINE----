import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from pathlib import Path
from tqdm import tqdm
import wandb
import os
import sys
import random  # Added for shuffling
from natsort import natsorted # Added for consistent file ordering before shuffle

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer_model import ChessTransformerDecoder, ChessTransformerConfig
from d_DATALOADER.my_dataloader import my_Dataset, my_collate_fn

def train_epoch(model, train_loader, optimizer, criterion, device, scheduler=None, global_step=0, save_dir=None):
    model.train()
    total_loss = 0
    total_tokens = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, batch in enumerate(progress_bar):
        # Convert batch to correct type and device
        batch = batch.long().to(device)
        
        # Prepare input and target sequences (shift by 1)
        input_seq = batch[:, :-1]
        target_seq = batch[:, 1:]
        
        # Forward pass (model will generate mask internally after potential truncation)
        outputs, _ = model(input_seq)
        
        # Debug: Print unique tokens in first batch to verify diversity
        if batch_idx == 0:
            unique_tokens = torch.unique(outputs.argmax(dim=-1))
            print(f"\nUnique tokens in first batch: {unique_tokens.size(0)}")
            print(f"Sample of predicted tokens: {outputs.argmax(dim=-1)[0, :10]}")
        
        # Ensure target sequence matches output sequence length
        if target_seq.size(1) > outputs.size(1):
            target_seq = target_seq[:, :outputs.size(1)]
        
        # Reshape outputs and targets for loss calculation
        outputs = outputs.contiguous().view(-1, outputs.size(-1))
        target_seq = target_seq.contiguous().view(-1)
        
        # Calculate loss (ignore padding tokens)
        loss = criterion(outputs, target_seq)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Update metrics
        total_loss += loss.item() * target_seq.ne(0).sum().item()
        total_tokens += target_seq.ne(0).sum().item()
        
        # Update progress bar
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'ppl': f'{perplexity:.2f}',
            'step': global_step
        })
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                'batch_loss': loss.item(),
                'batch_perplexity': math.exp(loss.item()) if loss.item() < 100 else float('inf'),
                'avg_loss': avg_loss,
                'avg_perplexity': perplexity,
                'step': global_step
            })
        
        # Save checkpoint every X steps
        if save_dir is not None and global_step % 2500 == 0:
            checkpoint_path = save_dir / f'model_step_{global_step}.pt'
            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': avg_loss,
                'perplexity': perplexity
            }, checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")
        
        global_step += 1
    
    return avg_loss, perplexity, global_step

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        for batch in progress_bar:
            # Convert batch to correct type and device
            batch = batch.long().to(device)
            
            # Prepare input and target sequences
            input_seq = batch[:, :-1]
            target_seq = batch[:, 1:]
            
            # Forward pass
            outputs = model(input_seq)
            
            # Ensure target sequence matches output sequence length
            if target_seq.size(1) > outputs.size(1):
                target_seq = target_seq[:, :outputs.size(1)]
            
            # Reshape outputs and targets
            outputs = outputs.contiguous().view(-1, outputs.size(-1))
            target_seq = target_seq.contiguous().view(-1)
            
            # Calculate loss
            loss = criterion(outputs, target_seq)
            
            # Update metrics
            total_loss += loss.item() * target_seq.ne(0).sum().item()
            total_tokens += target_seq.ne(0).sum().item()
            
            # Update progress bar
            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
            perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
            progress_bar.set_postfix({
                'val_loss': f'{avg_loss:.4f}',
                'val_ppl': f'{perplexity:.2f}'
            })
    
    return avg_loss, perplexity

def main():
    # Training configuration
    config = {
        'vocab_size': 852,  # Updated to match your tokenizer's vocab size
        'd_model': 256,
        'nhead': 8,
        'num_decoder_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'max_seq_length': 500,
        'batch_size': 2,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'warmup_steps': 4000,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 0,  # Set to 0 for debugging
        'save_dir': 'checkpoints',
        'use_wandb': True,
        'val_split_ratio': 0.1 # 10% for validation
    }
    
    # Initialize wandb
    if config['use_wandb']:
        wandb.init(project='chess-transformer', config=config)
    
    # Create save directory
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(exist_ok=True)
    
    # Initialize model
    model_config = ChessTransformerConfig(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_seq_length=config['max_seq_length']
    )
    model = ChessTransformerDecoder(model_config).to(config['device'])
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / config['warmup_steps'], 
                        math.sqrt(config['warmup_steps'] / (step + 1)))
    )
    
    # Load checkpoint if specified
    checkpoint_path = ''  # Update this to your checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        global_step = checkpoint.get('step', 0)
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
        print(f"Previous best validation loss: {best_val_loss}")
    else:
        print("No checkpoint found, starting from scratch")
        global_step = 0
        best_val_loss = float('inf')
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding token (0)
    
    # --- Prepare Train/Validation File Split ---
    data_in_folder = Path('a_DATA_CLEANUP/practice_tokenized')
    data_idx_folder = Path('a_DATA_CLEANUP/practice_maps')

    print(f"\nChecking data directories:")
    print(f"Input folder exists: {data_in_folder.exists()}")
    print(f"Index folder exists: {data_idx_folder.exists()}")
    
    all_in_chunks = natsorted([f for f in data_in_folder.iterdir() if f.name.endswith('.pt')])
    all_maps = natsorted([f for f in data_idx_folder.iterdir() if f.name.endswith('.json')])
    
    print(f"\nFound files:")
    print(f"Input chunks: {[f.name for f in all_in_chunks]}")
    print(f"Map files: {[f.name for f in all_maps]}")

    if len(all_in_chunks) != len(all_maps):
        raise ValueError("Number of input chunks and index files do not match in a_DATA_CLEANUP.")

    all_chunk_map_pairs = list(zip(all_in_chunks, all_maps))
    print(f"\nCreated {len(all_chunk_map_pairs)} chunk-map pairs")
    
    random.shuffle(all_chunk_map_pairs) # Shuffle all pairs

    # For single file case, put it in training set
    if len(all_chunk_map_pairs) == 1:
        train_files = all_chunk_map_pairs
        val_files = []
    else:
        split_idx = int(len(all_chunk_map_pairs) * (1 - config['val_split_ratio']))
        # Ensure at least one file goes to training
        split_idx = max(1, split_idx)
        train_files = all_chunk_map_pairs[:split_idx]
        val_files = all_chunk_map_pairs[split_idx:]

    print(f"\nSplit results:")
    print(f"Split index: {split_idx if len(all_chunk_map_pairs) > 1 else 'N/A (single file)'}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")

    if not train_files:
        raise ValueError("Training file list is empty. Check data paths and split ratio.")
    if not val_files and len(all_chunk_map_pairs) > 1:
        raise ValueError("Validation file list is empty. Check data paths and split ratio. You might need more data or a smaller val_split_ratio.")
    # --- End File Split ---

    # Setup data loaders
    train_dataset = my_Dataset(
        chunk_map_pairs_list=train_files,
        shuffle=True
    )
    
    val_dataset = my_Dataset(
        chunk_map_pairs_list=val_files,
        shuffle=False  # No need to shuffle validation data
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        collate_fn=my_collate_fn,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        collate_fn=my_collate_fn,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    # Training loop
    best_val_loss = float('inf')
    global_step = 0
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        train_loss, train_ppl, global_step = train_epoch(
            model, train_loader, optimizer, criterion, 
            config['device'], scheduler, global_step, save_dir
        )
        
        # Validate
        val_loss, val_ppl = validate(
            model, val_loader, criterion, config['device']
        )
        
        # Log metrics
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_perplexity': train_ppl,
            'val_loss': val_loss,
            'val_perplexity': val_ppl,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'step': global_step
        }
        
        if config['use_wandb']:
            wandb.log(metrics)
        
        print(f"\nMetrics: {metrics}")
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = save_dir / f'model_epoch_{epoch + 1}_val_loss_{val_loss:.4f}.pt'
            torch.save({
                'epoch': epoch + 1,
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")

if __name__ == '__main__':
    main() 