import chess
import chess.svg
import os
import time
from chessboard import display

valid_fen = 'rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2'

# Initialization
game_board = display.start()

# Position change/update
display.update(valid_fen, game_board)

# Checking GUI window for QUIT event. (Esc or GUI CANCEL)
display.check_for_quit()
# Close window
display.terminate()