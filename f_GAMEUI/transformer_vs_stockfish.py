import sys
import os
import chess
import chess.pgn
import pandas as pd
import random
from stockfish import Stockfish
import time

# --- Setup Python Path to find your custom modules ---
try:
    # Assumes this script is in f_GAMEUI, and project root is one level up
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
except NameError: # Fallback for environments where __file__ is not defined (e.g. some notebooks)
    current_file_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_file_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from e2_TRANSFORMER.generate_chess import play as play_transformer_move
except ModuleNotFoundError as e:
    print(f"Error importing 'play_transformer_move' function from e2_TRANSFORMER.generate_chess: {e}")
    print("Ensure that the project root is correctly added to sys.path and the module exists.")
    raise

# --- Configuration ---
NUM_GAMES = 50  # Total number of games to play
# !!! IMPORTANT: SET THE PATH TO YOUR STOCKFISH EXECUTABLE HERE !!!
# Examples:
# STOCKFISH_PATH = "/usr/games/stockfish"  # Linux
# STOCKFISH_PATH = "/opt/homebrew/bin/stockfish" # macOS (Homebrew)
STOCKFISH_PATH = r"E:\Stockfish\stockfish\stockfish-windows-x86-64-avx2.exe"  # Windows (REMEMBER TO UPDATE THIS)

# Transformer model parameters (can be adjusted)
TRANSFORMER_CHECKPOINT_PATH = None # Uses default in generate_chess.py, or specify your XL model
TRANSFORMER_TEMP = 0.8
TRANSFORMER_TOP_K = 40
TRANSFORMER_TOKENS_TO_GENERATE = 5 # How many tokens the transformer should generate to find a move
MAX_TRANSFORMER_RETRIES = 10 # How many times to retry if transformer provides an invalid move

# Stockfish parameters
STOCKFISH_ELO = 1350 # Approximate ELO for Stockfish (adjust for difficulty)
STOCKFISH_THINK_TIME_MS = 500 # Milliseconds for Stockfish to think

def get_transformer_move(board, game_history_for_ai, transformer_checkpoint, temp, top_k, tokens_to_gen):
    """
    Gets a move from the transformer model.
    Returns the move in SAN (Standard Algebraic Notation) or None if a valid move isn't found.
    """
    current_game_string_for_ai = " ".join(game_history_for_ai)
    print(f"Transformer thinking... Input: '{current_game_string_for_ai}'")

    for attempt in range(MAX_TRANSFORMER_RETRIES):
        try:
            ai_full_response_sequence = play_transformer_move(
                sequence=current_game_string_for_ai,
                num_tokens=tokens_to_gen,
                temperature=temp,
                top_k=top_k,
                checkpoint_path=transformer_checkpoint
            )

            if ai_full_response_sequence.startswith(current_game_string_for_ai):
                generated_part = ai_full_response_sequence[len(current_game_string_for_ai):].strip()
            else:
                print(f"Warning: AI response didn't start with input. Full response: '{ai_full_response_sequence}'")
                # Take the last few tokens instead of just the last one
                generated_part = " ".join(ai_full_response_sequence.split()[-3:])

            if not generated_part:
                print(f"Transformer generated empty response (Attempt {attempt + 1}/{MAX_TRANSFORMER_RETRIES}).")
                game_history_for_ai.append("[tf_empty_response]") # Log placeholder
                continue

            # Try to extract the first token as a move
            potential_move_san = generated_part.split(' ')[0]
            if not potential_move_san:
                print(f"Transformer generated empty move token (Attempt {attempt + 1}/{MAX_TRANSFORMER_RETRIES}).")
                game_history_for_ai.append("[tf_empty_token]") # Log placeholder
                continue
            
            # Validate the move with the chess library
            try:
                board.parse_san(potential_move_san) # Check if legal without making the move
                print(f"Transformer proposes move: {potential_move_san}")
                return potential_move_san
            except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError) as e:
                print(f"Transformer proposed illegal move '{potential_move_san}': {e} (Attempt {attempt + 1}/{MAX_TRANSFORMER_RETRIES}).")
                game_history_for_ai.append(f"[tf_illegal:{potential_move_san}]") # Log placeholder

        except RuntimeError as e: # Catch model loading errors from play_transformer_move
            print(f"RuntimeError during transformer move generation: {e}")
            return None # Critical error, stop trying for this game
        except Exception as e:
            print(f"Unexpected error during transformer move generation: {e} (Attempt {attempt + 1}/{MAX_TRANSFORMER_RETRIES}).")
            if 'potential_move_san' not in locals(): potential_move_san = "[tf_parsing_failed]"
            game_history_for_ai.append(f"[tf_error:{potential_move_san}]") # Log placeholder
    
    print("Transformer failed to make a legal move after retries.")
    return None


def play_one_game(stockfish, transformer_plays_white, game_num):
    """Plays a single game between Stockfish and the Transformer."""
    board = chess.Board()
    moves = []  # Store just the moves
    game_history_for_transformer = ["<S>"] # Transformer's view of the game
    current_move_number = 1  # Track move numbers

    while not board.is_game_over():
        is_transformer_turn = (board.turn == chess.WHITE and transformer_plays_white) or \
                              (board.turn == chess.BLACK and not transformer_plays_white)

        move_san = None
        move_uci = None

        if is_transformer_turn:
            print(f"\n--- Game {game_num}: Transformer's turn ({'White' if board.turn == chess.WHITE else 'Black'}) ---")
            move_san = get_transformer_move(
                board.copy(), # Pass a copy of the board for validation
                game_history_for_transformer,
                TRANSFORMER_CHECKPOINT_PATH,
                TRANSFORMER_TEMP,
                TRANSFORMER_TOP_K,
                TRANSFORMER_TOKENS_TO_GENERATE
            )
            if move_san is None:
                print("Transformer forfeits or critical error.")
                break # End game if transformer can't make a move

            try:
                move_obj = board.push_san(move_san)
                move_uci = move_obj.uci()
                game_history_for_transformer.append(move_san)
                # Add move with number if it's White's move
                if board.turn == chess.BLACK:  # If it's now Black's turn, we just made a White move
                    moves.append(f"{current_move_number}.{move_san}")
                else:  # If it's still White's turn, we just made a Black move
                    moves.append(move_san)
                    current_move_number += 1
            except Exception as e: # Should be caught by get_transformer_move, but as a safeguard
                print(f"Error pushing transformer move '{move_san}' to board: {e}")
                break
        else:
            print(f"\n--- Game {game_num}: Stockfish's turn ({'White' if board.turn == chess.WHITE else 'Black'}) ---")
            stockfish.set_fen_position(board.fen())
            if transformer_plays_white is False and board.fullmove_number == 1 and board.turn == chess.WHITE : # Stockfish is White, first move
                legal_moves = list(board.legal_moves)
                if not legal_moves: # Should not happen in a normal game start
                     break
                chosen_move = random.choice(legal_moves)
                print(f"Stockfish (White, 1st move) plays randomly: {board.san(chosen_move)}")
                move_uci = chosen_move.uci()
            else:
                move_uci = stockfish.get_best_move_time(STOCKFISH_THINK_TIME_MS)
                if move_uci is None: # Can happen if Stockfish is in a checkmate/stalemate position already
                    print("Stockfish returned no move (likely game over).")
                    if not board.is_game_over(): # If game isn't actually over, means SF failed
                        print("Stockfish failed to provide a move but game not over. SF Forfeits.")
                        break
                    continue # Let the loop re-evaluate board.is_game_over()

            move_obj = chess.Move.from_uci(move_uci)
            move_san = board.san(move_obj) # Get SAN for transformer history
            board.push(move_obj)
            game_history_for_transformer.append(move_san)
            # Add move with number if it's White's move
            if board.turn == chess.BLACK:  # If it's now Black's turn, we just made a White move
                moves.append(f"{current_move_number}.{move_san}")
            else:  # If it's still White's turn, we just made a Black move
                moves.append(move_san)
                current_move_number += 1
            print(f"Stockfish plays: {move_san}")

        clear_output(wait=True) # Optional: for cleaner output in compatible terminals/notebooks
        print(f"Game {game_num} - Move {board.fullmove_number}: {'Transformer' if transformer_plays_white else 'Stockfish'} vs {'Stockfish' if transformer_plays_white else 'Transformer'}")
        print(board)
        print(f"Last move: {move_san}")
        print(f"Transformer history: {' '.join(game_history_for_transformer)}")

    # Game finished
    outcome = board.outcome()
    result = outcome.result() if outcome else "*"
    
    # Create simple PGN with just moves
    pgn_string = " ".join(moves)
    num_total_moves = board.fullmove_number # Number of full moves
    return pgn_string, num_total_moves, result


def main():
    if not os.path.exists(STOCKFISH_PATH):
        print(f"Stockfish executable not found at: {STOCKFISH_PATH}")
        print("Please download Stockfish and update the STOCKFISH_PATH variable in this script.")
        return

    try:
        stockfish = Stockfish(path=STOCKFISH_PATH, parameters={"UCI_Elo": STOCKFISH_ELO, "Threads": 2})
        stockfish.set_elo_rating(STOCKFISH_ELO) # Ensure ELO is set
        print(f"Stockfish initialized: {stockfish.get_parameters()}")
    except Exception as e:
        print(f"Failed to initialize Stockfish: {e}")
        print("Check the STOCKFISH_PATH and ensure Stockfish can run.")
        return

    game_results = []

    for i in range(NUM_GAMES):
        game_num = i + 1
        transformer_plays_white = (i % 2 == 0) # Transformer plays White on even games (0, 2, ...), Black on odd
        
        print(f"\nStarting Game {game_num}/{NUM_GAMES}")
        print(f"Transformer plays {'White' if transformer_plays_white else 'Black'}")
        print(f"Stockfish plays {'Black' if transformer_plays_white else 'White'}")

        pgn, num_moves, result = play_one_game(stockfish, transformer_plays_white, game_num)
        game_results.append({
            "GameID": game_num,
            "Transformer_Color": "White" if transformer_plays_white else "Black",
            "PGN": pgn,
            "NumMoves": num_moves,
            "Result": result
        })
        
        # Optional: Save incrementally or less frequently if games are very long
        if game_num % 10 == 0: # Save every 10 games
            df = pd.DataFrame(game_results)
            df.to_csv("transformer_vs_stockfish_results.csv", index=False)
            print(f"Results for {game_num} games saved to transformer_vs_stockfish_results.csv")


    df_final = pd.DataFrame(game_results)
    df_final.to_csv("transformer_vs_stockfish_results.csv", index=False)
    print("\nAll games finished. Final results saved to transformer_vs_stockfish_results.csv")
    print(df_final.head())
    
    # Basic summary
    if not df_final.empty:
        transformer_wins = len(df_final[((df_final['Transformer_Color'] == 'White') & (df_final['Result'] == '1-0')) |
                                        ((df_final['Transformer_Color'] == 'Black') & (df_final['Result'] == '0-1'))])
        stockfish_wins = len(df_final[((df_final['Transformer_Color'] == 'White') & (df_final['Result'] == '0-1')) |
                                      ((df_final['Transformer_Color'] == 'Black') & (df_final['Result'] == '1-0'))])
        draws = len(df_final[df_final['Result'] == '1/2-1/2'])
        print("\n--- Summary ---")
        print(f"Total Games: {len(df_final)}")
        print(f"Transformer Wins: {transformer_wins}")
        print(f"Stockfish Wins: {stockfish_wins}")
        print(f"Draws: {draws}")


# For clear_output in non-notebook environments
def clear_output(wait=False):
    # For Windows
    if os.name == 'nt':
        os.system('cls')
    # For macOS and Linux
    else:
        os.system('clear')
    if wait: # Not a perfect equivalent of notebook's wait, but a small pause
        time.sleep(0.1)


if __name__ == "__main__":
    main() 