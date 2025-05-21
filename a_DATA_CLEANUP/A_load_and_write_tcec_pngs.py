import requests
import os
import re
import fileinput
from concurrent.futures import ThreadPoolExecutor

repo_api_url = "https://api.github.com/repos/TCEC-Chess/tcecgames/contents/master-archive"
raw_base_url = "https://raw.githubusercontent.com/TCEC-Chess/tcecgames/main/master-archive/"
output_file = r"C:\Users\moren\Desktop\Masters\2nd Semester\Deep Learning\personal engine\game_files\tcec_games_cleaned.txt"
n = 500

if os.path.exists(output_file):
    os.remove(output_file)

# Precompile regular expressions
event_pattern = re.compile(r"(\[Event .+?\])")
bracket_pattern = re.compile(r"\[.*?\]")
curly_pattern = re.compile(r"\{.*?\}")
newline_pattern = re.compile(r"(\d+\.)\s*\n\s*")
space_newline_pattern = re.compile(r"(\w[^\.\n]*)\n(\w)")
multiple_spaces_pattern = re.compile(r" +")
result_pattern = re.compile(r"\s*(1-0|0-1|1/2-1/2)\s*$")
comment_pattern = re.compile(r";.*")
move_number_space_pattern = re.compile(r"(\d+)\.([a-zA-Z])")

def clean_pgn(pgn):
    games = event_pattern.split(pgn)
    clean_games = []
    for i in range(1, len(games), 2):
        game_header = games[i]
        game_content = games[i + 1]
        game_content = bracket_pattern.sub("", game_content)
        game_content = curly_pattern.sub("", game_content)
        game_content = comment_pattern.sub("", game_content)
        game_content = "\n".join(line.strip() for line in game_content.split("\n") if line.strip())
        game_content = move_number_space_pattern.sub(r"\1. \2", game_content)
        game_content = newline_pattern.sub(r"\1 ", game_content)
        game_content = space_newline_pattern.sub(r"\1 \2", game_content)
        game_content = multiple_spaces_pattern.sub(" ", game_content)
        game_content = result_pattern.sub(r"", game_content)
        clean_lines = []
        for line in game_content.split("\n"):
            if re.match(r"^\d+\.", line) and "{" not in line:
                clean_lines.append(line)
        clean_games.append("\n".join(clean_lines))

    return "\n".join(clean_games)

def download_and_clean(filename):
    file_url = raw_base_url + filename
    file_response = requests.get(file_url)
    if file_response.status_code == 200:
        pgn_content = file_response.text
        return clean_pgn(pgn_content)
    else:
        print(f"Failed to download {filename}, Status code: {file_response.status_code}")
        return None

response = requests.get(repo_api_url)
if response.status_code != 200:
    print(f"Failed to fetch file list. Status code: {response.status_code}")
    print("Response:", response.text)
    exit()

files = response.json()
file_names = [file["name"] for file in files if file["type"] == "file"][:n]

cleaned_contents = []
processed_count = 0

with ThreadPoolExecutor(max_workers=10) as executor:
    results = executor.map(download_and_clean, file_names)
    for result in results:
        if result:
            cleaned_contents.append(result)
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} files.")

with open(output_file, "w", encoding="utf-8") as outfile:
    outfile.write("\n".join(cleaned_contents))

for line in fileinput.input(output_file, inplace=True):
    if len(line) > 50:
        print(line, end='')

print("All files processed.")
