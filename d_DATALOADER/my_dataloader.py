import torch
import math
from natsort import natsorted
import random
import json

from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path

class my_Dataset(IterableDataset):

    def __init__(self, in_folder, idx_folder, shuffle=True):
        self.shuffle = shuffle
        # chunk and map paths
        self.in_folder = Path(in_folder) if not isinstance(in_folder, Path) else in_folder
        self.idx_folder = Path(idx_folder) if not isinstance(idx_folder, Path) else idx_folder
        # chunk and map variables
        self.chunks_maps = None
        # functions to run on init
        self.cm_tuple()

    def cm_tuple(self):   # chunk-maps tuples
        in_chunks = natsorted([f for f in self.in_folder.iterdir() if '.pt' in str(f)])   # list of chunk paths (without game_counts)
        self.n_chunks = len(in_chunks)
        maps = natsorted([f for f in self.idx_folder.iterdir()])   # list of map paths
        if len(in_chunks) != len(maps):   # shouldnt need to be ran but just in case
            raise ValueError("Number of input chunks(files) and index files do not match.")
        else:
            self.chunks_maps = list(zip(in_chunks, maps)) # (chunk, map)

    def __iter__(self):
        # --- Worker information and workload splitting ---
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        # --- Determine the chunks for this worker ---
        all_chunk_map_pairs = list(self.chunks_maps) # Get the original list (make copy if needed)
        num_chunks = len(all_chunk_map_pairs)

        if num_chunks == 0:
            return # Stop iteration if dataset is empty

        # Calculate the range of chunks this worker should process (based on original order)
        per_worker = int(math.ceil(num_chunks / float(num_workers)))
        iter_start = worker_id * per_worker
        iter_end = min(iter_start + per_worker, num_chunks)

        # Get the subset of chunk paths assigned to this worker
        assigned_chunk_map_pairs = all_chunk_map_pairs[iter_start:iter_end] # Slice FIRST

        # Shuffle the subset assigned to this worker
        if self.shuffle:
            random.shuffle(assigned_chunk_map_pairs) # Shuffle the worker's ASSIGNED portion SECOND

        # --- Process assigned chunks ---
        print(f"[Worker {worker_id}] Processing {len(assigned_chunk_map_pairs)} chunks (indices from original list: {iter_start} to {iter_end-1}).") # Optional debug print

        for chunk_path, map_path in assigned_chunk_map_pairs:
            try:
                # Load data for the current chunk
                # Consider adding more robust file existence checks if needed
                if not chunk_path.exists() or not map_path.exists():
                     print(f"[Worker {worker_id}] Warning: File missing for chunk {chunk_path.name} or map {map_path.name}. Skipping.")
                     continue
                     
                # print(f"[Worker {worker_id}] Loading chunk: {chunk_path.name}") # Optional debug print
                chunk_data = torch.load(chunk_path)
                with open(map_path, 'r') as f:
                    map_data = json.load(f) # Should be a list of [start, end] pairs

                num_games_in_chunk = len(map_data)
                if num_games_in_chunk == 0:
                    # print(f"[Worker {worker_id}] Warning: Chunk {chunk_path.name} has 0 games in map. Skipping.") # Optional debug print
                    continue # Skip empty chunks

                # Generate local game indices (0 to num_games-1)
                local_game_indices = list(range(num_games_in_chunk))

                # Shuffle local game indices within the chunk (optional)
                if self.shuffle:
                    random.shuffle(local_game_indices)

                # Yield games from this chunk based on shuffled local indices
                for game_idx in local_game_indices:
                    try:
                        start, end = map_data[game_idx]
                        yield chunk_data[start:end]
                    except IndexError:
                         print(f"[Worker {worker_id}] Error: Game index {game_idx} out of bounds for map {map_path.name} (len={num_games_in_chunk}). Skipping game.")
                    except Exception as game_e:
                         print(f"[Worker {worker_id}] Error yielding game {game_idx} from chunk {chunk_path.name}: {game_e}")
                         
                # Optional: Explicitly free memory for the large chunk data
                del chunk_data
                del map_data
                del local_game_indices

            except FileNotFoundError:
                 print(f"[Worker {worker_id}] Error: File not found for chunk {chunk_path} or map {map_path}. Skipping chunk.")
            except Exception as e:
                # Log error or decide how to handle other corrupted files/chunks
                print(f"[Worker {worker_id}] Error processing chunk {chunk_path.name}: {e}")
                # Optionally 'del chunk_data' etc. here too in case they exist
                continue # Skip to the next chunk

##########################

from torch.nn.utils.rnn import pad_sequence

def my_collate_fn(batch):
    games = [tensor for tensor in batch]  # list of game tensors
    games = pad_sequence(games, batch_first=True, padding_value=0)  # pad to max length in batch
    return games