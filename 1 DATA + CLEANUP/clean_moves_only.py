import re
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    cleaned_text = re.sub(annotation_pattern, "", cleaned_text) # <-- ADDED THIS STEP

    # 4. Clean up multiple spaces resulting from previous removals
    cleaned_text = re.sub(r" +", " ", cleaned_text)

    # 5. Final line formatting and result spacing
    cleaned_lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
    final_text = "\n".join(cleaned_lines)
    final_text = re.sub(r"(\S)(1-0|0-1|1/2-1/2|\*)$", r"\1 \2", final_text)
    return final_text

def process_file(input_filepath):
    if not os.path.isfile(input_filepath):
        print(f"Skipping: '{input_filepath}' is not a valid file.")
        return

    try:
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
        content = None
        for enc in encodings_to_try:
            try:
                with open(input_filepath, 'r', encoding=enc) as f_in:
                    content = f_in.read()
                break
            except UnicodeDecodeError:
                continue
            except Exception:
                continue

        if content is None:
            print(f"Error: Could not decode file '{input_filepath}'")
            return

        cleaned_content = clean_pgn_content(content)
        base, ext = os.path.splitext(input_filepath)
        output_filepath = f"{base}_cleanedv2{ext}" if ext.lower() == '.pgn' else f"{input_filepath}_cleanedv2"

        with open(output_filepath, 'w', encoding='utf-8') as f_out:
            f_out.write(cleaned_content)
            if not cleaned_content.endswith('\n'):
                f_out.write('\n')

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
        future_to_file = {executor.submit(process_file, f): f for f in files_to_process}
        for future in as_completed(future_to_file):
            result = future.result()
            if result:
                print(result)

    print("\nDone.")

if __name__ == "__main__":
    main()