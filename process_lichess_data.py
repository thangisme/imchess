import chess
import numpy as np
import json
import zstandard as zstd
import io
import os
import time

INPUT_ZST_FILE = "lichess_db_eval.jsonl.zst"
OUTPUT_NPZ_FILE = "lichess_data_processed.npz"

MAX_POSITIONS_TO_COLLET = 2000000
MIN_DEPTH_FILTER = 18
CP_SCALING_FACTOR = 800

piece_to_plane = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def board_to_planes(board):
    """
    Converts a chess.Board to an 8x8x12 tensor.
    Each of the 12 planes corresponds to one piece type for one color.
    """
    planes = np.zeros((8, 8, 12), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            channel = piece_to_plane[piece.symbol()]
            planes[rank, file, channel] = 1.0
    return planes

def normalize_score(score_cp=None, mate_in=None):
    if mate_in is not None:
        return 1 if mate_in > 9 else -1
    elif score_cp is not None:
        return np.tanh(score_cp / CP_SCALING_FACTOR)
    else:
        return 0

if not os.path.exists(INPUT_ZST_FILE):
    print("ERROR: Input file not found")
    exit()

data_x = []
data_y = []

position_processed_total = 0
position_collected_valid = 0
lines_read_count = 5
start_time = time.time()

print(f"Processing up to {MAX_POSITIONS_TO_COLLET} positions with minimum depth {MIN_DEPTH_FILTER}")
print("Press Ctrl+C to abort and save progress.")

try:
    with open(INPUT_ZST_FILE, 'rb') as compressed_file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            for line in text_stream:
                lines_read_count += 1
                if not line.strip():
                    continue  # Skip blank lines

                position_processed_total += 1

                try:
                    pos_data = json.loads(line)
                    fen = pos_data.get("fen")
                    eval_entries = pos_data.get("evals")
                    if not fen or not eval_entries:
                        continue

                    best_eval_entry = None
                    highest_depth_found = -1

                    for eval_entry in eval_entries:
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

                            board_planes = board_to_planes(board)
                            data_x.append(board_planes)
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
                    pps = position_processed_total / elapsed_time if elapsed_time > 0 else 0
                    print(
                        f"Lines read: {lines_read_count}, Processed: {position_processed_total}, "
                        f"Collected: {position_collected_valid}/{MAX_POSITIONS_TO_COLLET} ({pps:.1f} pos/sec)"
                    )

                if position_collected_valid >= MAX_POSITIONS_TO_COLLET:
                    break

except KeyboardInterrupt:
    print("Keyboard interruption detected. Saving progress...")
except Exception as e:
    print("An unexpected error occurred:", e)
finally:
    if position_collected_valid > 0:
        x_train = np.array(data_x, dtype=np.float32) 
        y_train = np.array(data_y, dtype=np.float32) 
        print(f"Shape of x_train (board planes): {x_train.shape}")
        print(f"Shape of y_train (evaluations): {y_train.shape}")

        np.savez_compressed(OUTPUT_NPZ_FILE, X=x_train, y=y_train)
        print("Data saved successfully to", OUTPUT_NPZ_FILE)
    else:
        print("No valid positions collected to save.")

    total_elapsed = time.time() - start_time
    print(f"Total time taken: {total_elapsed:.2f} seconds")
