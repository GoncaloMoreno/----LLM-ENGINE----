very subpar vibe-coded llm, completely adapted from karpathy's nanoGPT (https://github.com/karpathy/nanoGPT) that plays chess

there's some useless files that i'm keeping for memory, and there's also some ipynbs that were just for tests

process:
- Download games from one month of lichess
- run the a_DATA_CLEANUP scripts to clean and split the files
- run the b_TOKENIZER scripts to train a tokenizer and encode/save the game files and index maps
- c_DATASPLIT isn't actually needed anymore because I ended up doing val split on training
- d_DATALOADER takes care of all the intricacies of custom iterabledatasets
- run nanotrain_chess.py on e2_TRANSFORMER
- play on chessuitests on f_GAMEUI

