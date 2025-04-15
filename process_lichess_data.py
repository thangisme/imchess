import chess
import numpy as np
import json
import zstandard as zstd
import io
import os
import time

INPUT_ZST_FILE = "lichess_db_eval.jsonl.zst"
OUTPUT_NPZ_FILE = "lichess_data_processed.npz"

MAX_POSITIONS_TO_COLLET = 100000000
MIN_DEPTH_FILTER = 18
CP_SCALING_FACTOR = 800

piece_map = {
    'P' : 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6, # WHite pieces
    'p' : 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6, # Black pieces
    '.': 0 # Empty square
}

def board_to_vector(board):
    # 64 squares + 4 for castling + 1 for turn
    vector = np.zeros(64 + 4 + 1, dtype=np.float32)

    for i in range (64):
        piece = board.piece_at(i)
        if piece:
            vector[i] = piece_map[piece.symbol()]

    vector[64] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    vector[65] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    vector[66] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
    vector[67] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0

    vector[68] = 1 if board.turn == chess.WHITE else -1

    return vector


def normalize_score(score_cp=None, mate_in=None):
    if mate_in is not None:
        return 1 if mate_in > 9 else -1
    elif score_cp is not None:
        return np.tanh(score_cp / CP_SCALING_FACTOR)
    else:
        return 0

if not os.path.exists(INPUT_ZST_FILE):
    print(f"ERROR: Input file not found")
    exit()

data_x = []
data_y = []

position_processed_total = 0
position_collected_valid = 0
lines_read_count = 5
start_time = time.time()

print(f"Processing up to {MAX_POSITIONS_TO_COLLET} position with minimum depth {MIN_DEPTH_FILTER}")

try:
    with open(INPUT_ZST_FILE, 'rb') as compressed_file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')

            for line in text_stream:
                lines_read_count += 1
                if not line.strip(): continue # SKip blank lines

                position_processed_total += 1

                try:
                    pos_data = json.loads(line)
                    fen = pos_data.get("fen")
                    evals = pos_data.get("evals")

                    if not fen or not evals:
                        continue

                    best_eval_entry = None
                    highest_depth_found = -1

                    for eval_entry in evals:
                        depth = eval_entry.get("depth", 0)
                        if depth >= MIN_DEPTH_FILTER and depth > highest_depth_found:
                            if eval_entry.get("pvs"):
                                highest_depth_found = depth
                                best_eval_entry = eval_entry

                    if best_eval_entry:
                        first_pv = best_eval_entry["pvs"][0]
                        cp = first_pv.get("cp")
                        mate = first_pv.get("mate")

                        if cp is not None or mate is not None:
                            normalized_eval = normalize_score(score_cp=cp, mate_in=mate)

                            try:
                                board = chess.Board(fen)
                            except ValueError:
                                continue

                            board_vec = board_to_vector(board)

                            data_x.append(board_vec)
                            data_y.append(normalized_eval)
                            position_collected_valid += 1

                except json.JSONDecodeError:
                    print(f"Malformed JSON on line {lines_read_count}")
                    continue
                except Exception as e:
                    print(f"Error processing line {lines_read_count}: {e}")
                    continue

                if position_processed_total % 100000 == 0:
                    elapsed_time = time.time() - start_time
                    pps = position_processed_total /elapsed_time if elapsed_time > 0 else 0
                    print(f'''
                    Lines read: {lines_read_count}, Processed: {position_processed_total}
                    Collected: {position_collected_valid}/{MAX_POSITIONS_TO_COLLET} ({pps:.1f} pos/sec)
                    ''')

                if position_collected_valid >= MAX_POSITIONS_TO_COLLET:
                    break

except Exception as e:
    print(f"Error processing: {e}")

if position_collected_valid > 0:
    x_train = np.array(data_x, dtype=np.float32)
    y_train = np.array(data_y, dtype=np.float32)

    print(f"Shape of x_train (board vectors): {x_train.shape}")
    print(f"Shape of y_train (board vectors): {y_train.shape}")

    np.savez_compressed(OUTPUT_NPZ_FILE, X=x_train, y=y_train)
    print("Data saved successfully.")
else:
    print(f"No valid position collected")

print(f"Total time taken: {time.time() - start_time:.2f}")
