import re
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def clean_pgn_content(pgn_text):
    # 1. Remove comments { ... }
    # Pattern matches braces and their content only
    comment_pattern = r"\{.*?\}"
    cleaned_text = re.sub(comment_pattern, "", pgn_text)

    # 2. Remove redundant Black move numbers N...
    move_num_pattern = r"\d+\.\.\. "
    cleaned_text = re.sub(move_num_pattern, "", cleaned_text)

    # 3. Remove move annotations ?, !, ??, !!, ?!, !?
    # Pattern matches one or more ? or ! characters
    annotation_pattern = r"[?!]+"
    cleaned_text = re.sub(annotation_pattern, "", cleaned_text)

    # 4. Remove move numbers followed by a dot (e.g., "2.", "7.", "23.", "54.")
    move_number_pattern = r"\d+\. "
    cleaned_text = re.sub(move_number_pattern, "", cleaned_text)

    # 5. Clean up multiple spaces resulting from previous removals
    cleaned_text = re.sub(r" +", " ", cleaned_text)

    # 6. Final line formatting and result spacing
    cleaned_lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
    # Add a space at the start of each line
    cleaned_lines = [" " + line for line in cleaned_lines]
    final_text = "\n".join(cleaned_lines)
    final_text = re.sub(r"(\S)(1-0|0-1|1/2-1/2|\*)$", r"\1 \2", final_text)
    return final_text

def process_large_file(input_filepath, chunk_size=1024*1024*100):  # 1GB chunks
    if not os.path.isfile(input_filepath):
        print(f"Skipping: '{input_filepath}' is not a valid file.")
        return

    try:
        # Fix file extension handling
        base, ext = os.path.splitext(input_filepath)
        output_filepath = f"{base}_CLEAN{ext}" if ext else f"{input_filepath}_CLEAN"
        
        # Get total file size for progress tracking
        total_size = os.path.getsize(input_filepath)
        processed_size = 0
        start_time = time.time()
        
        # Try different encodings
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
        encoding = None
        
        # First, try to determine the encoding
        for enc in encodings_to_try:
            try:
                with open(input_filepath, 'r', encoding=enc) as f:
                    f.read(1024)  # Read a small sample
                encoding = enc
                break
            except UnicodeDecodeError:
                continue
        
        if encoding is None:
            print(f"Error: Could not determine encoding for file '{input_filepath}'")
            return

        # Process the file in chunks
        with open(input_filepath, 'r', encoding=encoding) as f_in, \
             open(output_filepath, 'w', encoding='utf-8') as f_out:
            
            buffer = ""
            while True:
                chunk = f_in.read(chunk_size)
                if not chunk:
                    break
                
                # Update progress
                processed_size += len(chunk.encode(encoding))
                progress = (processed_size / total_size) * 100
                elapsed_time = time.time() - start_time
                speed = processed_size / (1024 * 1024 * elapsed_time)  # MB/s
                
                # Print progress
                print(f"\rProgress: {progress:.1f}% ({processed_size/1024/1024:.1f}MB/{total_size/1024/1024:.1f}MB) "
                      f"Speed: {speed:.1f}MB/s", end='', flush=True)
                
                # Add the chunk to our buffer
                buffer += chunk
                
                # Find the last complete game in the buffer
                last_game_end = buffer.rfind('\n\n')
                if last_game_end != -1:
                    # Process complete games
                    games_to_process = buffer[:last_game_end]
                    buffer = buffer[last_game_end:]
                    
                    # Clean and write the complete games
                    cleaned_games = clean_pgn_content(games_to_process)
                    f_out.write(cleaned_games)
                    if not cleaned_games.endswith('\n'):
                        f_out.write('\n')
            
            # Process any remaining content in the buffer
            if buffer:
                cleaned_remaining = clean_pgn_content(buffer)
                f_out.write(cleaned_remaining)
                if not cleaned_remaining.endswith('\n'):
                    f_out.write('\n')

        # Print final progress
        total_time = time.time() - start_time
        print(f"\nCompleted in {total_time:.1f} seconds. Average speed: {total_size/(1024*1024*total_time):.1f}MB/s")
        return f"✓ Cleaned: {input_filepath}"

    except Exception as e:
        return f"✗ Failed: {input_filepath} — {e}"

def get_pgn_files(folder_path):
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith('.pgn') and os.path.isfile(os.path.join(folder_path, f))
    ]

def main():
    if len(sys.argv) != 2:
        print("Usage: python clean_moves_only.py <folder_path_or_file>")
        sys.exit(1)

    target_path = os.path.normpath(sys.argv[1])

    files_to_process = []

    if os.path.isdir(target_path):
        files_to_process = get_pgn_files(target_path)
        if not files_to_process:
            print("No .pgn files found in the folder.")
            return
    elif os.path.isfile(target_path):
        files_to_process = [target_path]
    else:
        print(f"Error: '{target_path}' is not a valid directory or file.")
        return

    print(f"Processing {len(files_to_process)} file(s)...\n")

    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_large_file, f): f for f in files_to_process}
        for future in as_completed(future_to_file):
            result = future.result()
            if result:
                print(result)

    print("\nDone.")

if __name__ == "__main__":
    main()