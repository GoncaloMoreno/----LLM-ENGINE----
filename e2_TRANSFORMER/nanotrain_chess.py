"""
Chess Transformer Training Script (adapted from NanoGPT)

To run on a single GPU, example:
$ python nanotrain_chess.py --batch_size=32 --compile=False
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import random
from pathlib import Path

import torch
# import numpy as np # Not strictly needed with custom dataloader
from torch.utils.data import DataLoader
from natsort import natsorted
import wandb # Added for wandb logging


# Assuming nanomodel_chess.py is in the same directory or accessible via PYTHONPATH
from nanomodel_chess import GPTConfig, GPT
# Adjust path to your custom dataloader
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add parent for d_DATALOADER
from d_DATALOADER.my_dataloader import my_Dataset, my_collate_fn


# -----------------------------------------------------------------------------
# Configuration for Chess Transformer
# I/O
out_dir = './checkpoints/CLEANXLModels'  # Relative path to checkpoints directory
eval_interval = 100  # Run validation less frequently
save_interval = 50  # Save checkpoints more frequently
log_interval = 1
eval_iters = 50 # batches for validation
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
checkpoint_path = r'checkpoints\CLEANXLModels\ckpt_iter650_loss1.5125.pt'  # Path to checkpoint to resume from

# Wandb logging
wandb_log = True # Enable wandb logging
wandb_project = 'chess-transformer-CLEANXL' # Your wandb project name
wandb_run_name = 'chess_run_' + str(int(time.time())) # Unique run name

# Data
# data_in_folder_path = 'a_DATA_CLEANUP/practice_tokenized' # Path to your tokenized .pt files
# data_idx_folder_path = 'a_DATA_CLEANUP/practice_maps'    # Path to your .json map files
# Relative paths from the workspace root e.g., /e:/LLM_ENGINE/
data_in_folder_path = r'a_DATA_CLEANUP\games_for_CLEAN\games'
data_idx_folder_path = r'a_DATA_CLEANUP\games_for_CLEAN\maps'
val_split_ratio = 0.1 # 10% for validation, adjust as needed

gradient_accumulation_steps = 4  # Accumulate gradients over 4 steps before updating
batch_size = 8
block_size = 550 # Increased from 500 to handle longer sequences
num_workers = 0 # For DataLoader. Set to 0 for debugging, >0 for parallel loading.

# Model
vocab_size = 530 # Your specific vocab size
n_layer = 14 #8
n_head = 12 #8
n_embd = 768 #512 # d_model
dropout = 0.1 
bias = True # NanoGPT default is False, but your original ChessTransformer used True for nn.TransformerDecoderLayer

# AdamW Optimizer
learning_rate = 3e-5 #3e-4  # Constant, very low learning rate
max_iters = 5000 # Total number of training iterations (batches)
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95 # Increased beta2 for more smoothed gradient variance estimation
grad_clip = 1 # Reduced grad_clip further

# Learning rate decay settings
decay_lr = True # Using a constant learning rate
warmup_iters = 250    # Not used with constant LR
lr_decay_iters = 1000 # Not strictly used, but kept for consistency
min_lr = 3e-6 #3e-5        # Same as learning_rate for constant LR

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'float32' # Forcing float32 for stability testing
compile_model = False # PyTorch 2.0 compilation

# Loss tracking
loss_window_size = 100  # Number of iterations to average over

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # For logging and saving
# -----------------------------------------------------------------------------

# Simplified init: no DDP
master_process = True
seed_offset = 0 # Not used as no DDP

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Wandb init
if wandb_log and master_process:
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# --- Custom Data Loading ---
print("Initializing custom DataLoaders...")
data_in_folder = Path(data_in_folder_path)
data_idx_folder = Path(data_idx_folder_path)

if not data_in_folder.exists() or not data_idx_folder.exists():
    print(f"Data folders not found. Searched at: {data_in_folder.resolve()} and {data_idx_folder.resolve()}")
    print("Please ensure 'data_in_folder_path' and 'data_idx_folder_path' are correct relative to your workspace root or absolute.")
    exit() # Or raise an error

all_in_chunks = natsorted([f for f in data_in_folder.iterdir() if f.name.endswith('.pt')])
all_maps = natsorted([f for f in data_idx_folder.iterdir() if f.name.endswith('.json')])

if not all_in_chunks or not all_maps:
    print("No .pt or .json files found in the specified data directories.")
    exit()

if len(all_in_chunks) != len(all_maps):
    raise ValueError("Number of input chunks and index files do not match.")

all_chunk_map_pairs = list(zip(all_in_chunks, all_maps))
random.shuffle(all_chunk_map_pairs)

if len(all_chunk_map_pairs) == 1 and val_split_ratio > 0:
    print("Warning: Only one data file found. Using it for training and disabling validation.")
    train_files = all_chunk_map_pairs
    val_files = []
elif len(all_chunk_map_pairs) == 0:
    print("Error: No data files found after pairing. Check data paths and file naming.")
    exit()
else:
    split_idx = int(len(all_chunk_map_pairs) * (1 - val_split_ratio))
    if split_idx == 0 and len(all_chunk_map_pairs) > 0 : # Ensure at least one training file if there's data
        print("Warning: val_split_ratio resulted in 0 training files. Adjusting to use at least one for training.")
        split_idx = 1
    train_files = all_chunk_map_pairs[:split_idx]
    val_files = all_chunk_map_pairs[split_idx:]
    if not val_files and len(all_chunk_map_pairs) > 1 :
        print("Warning: No validation files after split. Consider adjusting val_split_ratio or adding more data.")


print(f"Total data file pairs: {len(all_chunk_map_pairs)}")
print(f"Training file pairs: {len(train_files)}")
print(f"Validation file pairs: {len(val_files)}")

train_dataset = my_Dataset(chunk_map_pairs_list=train_files, shuffle=True)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    collate_fn=my_collate_fn,
    pin_memory=True if device_type == 'cuda' else False
)

if val_files:
    val_dataset = my_Dataset(chunk_map_pairs_list=val_files, shuffle=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=my_collate_fn,
        pin_memory=True if device_type == 'cuda' else False
    )
else:
    val_loader = None
    print("Validation loader is not created as there are no validation files.")
# --- End Custom Data Loading ---


iter_num = 0
best_val_loss = 1e9
epoch_num = 0


# Model Initialization
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)

# Initialize loss tracking variables that might be updated if resuming
loss_history = []
running_avg_loss = None

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    # Declare that we are modifying the global variables inside this block
    # No, this is not how global works. We modify them directly.
    print(f"Resuming training from {checkpoint_path}")
    
    loaded_checkpoint_loss = None
    try:
        loaded_checkpoint_loss = float(Path(checkpoint_path).name.split('loss')[-1].split('.pt')[0])
        print(f"Extracted loss from checkpoint filename: {loaded_checkpoint_loss:.4f}")
    except (ValueError, IndexError, AttributeError) as e:
        print(f"Could not reliably extract loss from filename '{Path(checkpoint_path).name}': {e}.")

    checkpoint = torch.load(checkpoint_path, map_location=device) # Load checkpoint once

    if loaded_checkpoint_loss is None: # If filename parsing failed, try from checkpoint content
        print("Attempting to use 'best_val_loss' from checkpoint content for running_avg_loss initialization.")
        # Note: A checkpoint might not have best_val_loss if saved before first eval or if key is different.
        # It also might not have a direct equivalent of running_avg_loss saved.
        # Using best_val_loss is a reasonable proxy if a direct running_avg_loss isn't stored.
        loaded_checkpoint_loss = checkpoint.get('best_val_loss', 1.6232) # Default if not found
        print(f"Using loss value from checkpoint content (or default): {loaded_checkpoint_loss:.4f}")

    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    iter_num = checkpoint['iter_num']
    epoch_num = checkpoint.get('epoch_num', 0)
    best_val_loss = checkpoint['best_val_loss']
    print(f"Resumed from iteration {iter_num}, epoch {epoch_num}, best_val_loss {best_val_loss:.4f}")

    # Initialize global loss tracking variables using the determined loss
    loss_history = [loaded_checkpoint_loss] * loss_window_size
    running_avg_loss = loaded_checkpoint_loss 
    print(f"Initialized loss tracking with running_avg_loss: {running_avg_loss:.4f}")
model.to(device)

scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
    if 'optimizer' in checkpoint: 
        optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None 

if compile_model:
    print("Compiling the model... (takes a ~minute)")
    model = torch.compile(model)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    
    if not hasattr(estimate_loss, 'val_iter'):
        estimate_loss.val_iter = iter(val_loader) if val_loader is not None else None
    if not hasattr(estimate_loss, 'train_iter'):
        estimate_loss.train_iter = iter(train_loader)
    
    if val_loader is not None:
        val_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            try:
                batch = next(estimate_loss.val_iter)
            except StopIteration:
                estimate_loss.val_iter = iter(val_loader)
                batch = next(estimate_loss.val_iter)
            
            X = batch.long().to(device)
            Y = X[:, 1:].contiguous()
            X = X[:, :-1].contiguous()
            
            with ctx:
                logits, loss = model(X, Y)
            val_losses[k] = loss.item()
        out['val'] = val_losses.mean()
        out['val_perplexity'] = torch.exp(val_losses.mean()).item()
    else:
        out['val'] = float('inf')
        out['val_perplexity'] = float('inf')

    train_losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        try:
            batch = next(estimate_loss.train_iter)
        except StopIteration:
            estimate_loss.train_iter = iter(train_loader)
            batch = next(estimate_loss.train_iter)
        
        X = batch.long().to(device)
        Y = X[:, 1:].contiguous()
        X = X[:, :-1].contiguous()
        with ctx:
            logits, loss = model(X, Y)
        train_losses[k] = loss.item()
    out['train'] = train_losses.mean()
    out['train_perplexity'] = torch.exp(train_losses.mean()).item()
    
    model.train()
    return out

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    # Modified cosine decay to be more pronounced
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    # Apply additional decay factor to make the curve steeper
    coeff = coeff ** 2  # Square the coefficient to make decay more aggressive
    return min_lr + coeff * (learning_rate - min_lr)

# Training loop
t0 = time.time()
# current_batch_X, current_batch_Y = None, None # Not used

print(f"Starting training for {max_iters} iterations...")
train_loader_iter = iter(train_loader)

# loss_history and running_avg_loss are now correctly initialized globally before this loop.

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        val_loss_to_print = losses['val'] if val_loader is not None else 'N/A'
        # Ensure running_avg_loss has a value for printing if it's the first eval after resuming
        current_running_avg_loss_for_print = running_avg_loss if running_avg_loss is not None else losses['train']
        print(f"step {iter_num} (epoch {epoch_num}): train loss {losses['train']:.4f}, val loss {val_loss_to_print}, avg_loss {current_running_avg_loss_for_print:.4f}")
        
        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "epoch": epoch_num,
                "train/loss": losses['train'],
                "train/perplexity": losses['train_perplexity'],
                "train/running_avg_loss": current_running_avg_loss_for_print,
                "lr": lr,
            }
            if val_loader is not None and losses['val'] != float('inf'):
                log_dict["val/loss"] = losses['val']
                log_dict["val/perplexity"] = losses['val_perplexity']
            wandb.log(log_dict)

        if val_loader is not None and (losses['val'] < best_val_loss or always_save_checkpoint):
            if losses['val'] < best_val_loss:
                 best_val_loss = losses['val']
            if iter_num > 0:
                # Ensure running_avg_loss is not None before using in filename
                loss_for_filename = running_avg_loss if running_avg_loss is not None else losses['train'] 
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'epoch_num': epoch_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    # Consider saving running_avg_loss in checkpoint explicitly if needed
                }
                checkpoint_name = f'ckpt_iter{iter_num}_loss{loss_for_filename:.4f}.pt'
                checkpoint_s_path = os.path.join(out_dir, checkpoint_name) # Renamed to avoid conflict
                print(f"saving checkpoint to {checkpoint_s_path}")
                torch.save(checkpoint, checkpoint_s_path)
                
                if losses['val'] == best_val_loss:
                    best_checkpoint_name = f'best_ckpt_iter{iter_num}_loss{losses["val"]:.4f}.pt'
                    best_checkpoint_path = os.path.join(out_dir, best_checkpoint_name)
                    torch.save(checkpoint, best_checkpoint_path)
                    print(f"saved best checkpoint to {best_checkpoint_path}")

    if iter_num % save_interval == 0 and master_process and iter_num > 0:
        # Ensure running_avg_loss is not None before using in filename
        loss_for_filename_save = running_avg_loss if running_avg_loss is not None else (sum(loss_history) / len(loss_history) if loss_history else losses['train'])
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'epoch_num': epoch_num,
            'best_val_loss': best_val_loss,
            'config': config,
        }
        checkpoint_name = f'ckpt_iter{iter_num}_loss{loss_for_filename_save:.4f}.pt'
        checkpoint_s_path = os.path.join(out_dir, checkpoint_name) # Renamed to avoid conflict
        print(f"saving checkpoint to {checkpoint_s_path}")
        torch.save(checkpoint, checkpoint_s_path)

    if iter_num == 0 and eval_only:
        break
    
    try:
        batch_data = next(train_loader_iter)
    except StopIteration: 
        epoch_num +=1 
        print(f"Completed data pass (Epoch {epoch_num-1}). Starting new pass (Epoch {epoch_num}).")
        train_loader_iter = iter(train_loader)
        batch_data = next(train_loader_iter)

    current_batch = batch_data.long().to(device)
    X = current_batch[:, :-1].contiguous()
    Y = current_batch[:, 1:].contiguous()

    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps 
        scaler.scale(loss).backward()
    
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        batch_loss = loss.item() * gradient_accumulation_steps
        loss_history.append(batch_loss)
        if len(loss_history) > loss_window_size:
            loss_history.pop(0)
        if loss_history: # Ensure loss_history is not empty
             running_avg_loss = sum(loss_history) / len(loss_history)
        else: # Should not happen if iter_num >=0 and log_interval is met, but as a safeguard
             running_avg_loss = batch_loss 
        
        print(f"iter {iter_num} (epoch {epoch_num}): batch_loss {batch_loss:.4f}, avg_loss {running_avg_loss:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")
        if wandb_log: 
            wandb.log({
                'iter': iter_num, 
                'batch_loss': batch_loss, 
                'running_avg_loss': running_avg_loss,
                'lr': lr, 
                'epoch': epoch_num
            })
    
    iter_num += 1

    if iter_num > max_iters:
        print(f"Reached max_iters ({max_iters}). Training finished.")
        break

print("Training complete.")

if master_process:
    # Ensure losses['train'] is available or use running_avg_loss for the final checkpoint name
    final_loss_for_filename = losses.get('train') if 'losses' in locals() and losses.get('train') is not None else (running_avg_loss if running_avg_loss is not None else 0.0)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'epoch_num': epoch_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    final_checkpoint_name = f'final_ckpt_iter{iter_num}_loss{final_loss_for_filename:.4f}.pt'
    final_checkpoint_path = os.path.join(out_dir, final_checkpoint_name)
    print(f"saving final checkpoint to {final_checkpoint_path}")
    torch.save(checkpoint, final_checkpoint_path)
    if wandb_log:
        wandb.finish()

