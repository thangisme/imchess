import chess
import chess.pgn
import numpy as np
import os
import io
import time
import sys
import re
import multiprocessing
import gc
from tqdm import tqdm
import zstandard as zstd

PGN_FILE = "lichess_db_standard_rated_2025-02.pgn.zst"
OUTPUT_DIR = "policy_training_data_mp"
MAX_GAMES = 200000
MIN_ELO = 2400
VALIDATION_SPLIT = 0.1
MIN_FULLMOVE_NUMBER = 1
BATCH_SIZE = 10000 
SAVE_INTERMEDIATE = True 
RESERVE_CORES = 2 

MAX_MOVE_ID = 63 * 320 + 63 * 5 + 4
NUM_POSSIBLE_MOVES = MAX_MOVE_ID + 1

WHITE_ELO_RE = re.compile(r'\[WhiteElo\s+"(\d+)"\]')
BLACK_ELO_RE = re.compile(r'\[BlackElo\s+"(\d+)"\]')

def encode_move(move: chess.Move) -> int:
    from_square = move.from_square
    to_square = move.to_square
    promotion_type = 0
    if move.promotion is not None:
        promotion_map = {chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}
        promotion_type = promotion_map.get(move.promotion, 0)
    move_index = from_square * (64 * 5) + to_square * 5 + promotion_type
    return move_index

def board_to_planes(board: chess.Board) -> np.ndarray:
    planes = np.zeros((8, 8, 13), dtype=np.float32)
    piece_to_plane_idx = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank, file = chess.square_rank(square), chess.square_file(square)
            plane_base_idx = piece_to_plane_idx[piece.piece_type]
            color_offset = 0 if piece.color == chess.WHITE else 6
            planes[7 - rank, file, plane_base_idx + color_offset] = 1.0
    if board.turn == chess.BLACK:
        planes[:, :, 12] = 1.0
    return planes

def process_pgn_string_worker(pgn_string_with_config):
    pgn_string, min_elo_worker, min_fullmove_worker = pgn_string_with_config
    extracted_data = []
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_string))
        if game is None:
            return extracted_data

        white_elo = int(game.headers.get("WhiteElo", "0"))
        black_elo = int(game.headers.get("BlackElo", "0"))
        if min(white_elo, black_elo) < min_elo_worker:
            return extracted_data

        result = game.headers.get("Result", "*")
        result_value = 0.0
        if result == "1-0": result_value = 1.0
        elif result == "0-1": result_value = -1.0

        board = game.board()
        for move in game.mainline_moves():
            is_valid_state = (
                board.fullmove_number >= min_fullmove_worker
                and not board.is_game_over(claim_draw=True)
            )
            if is_valid_state:
                board_state_planes = board_to_planes(board)
                encoded_move = encode_move(move)
                position_value = result_value if board.turn == chess.WHITE else -result_value
                
                extracted_data.append(
                    (board_state_planes.tolist(), encoded_move, position_value)
                )
            try:
                board.push(move)
            except (ValueError, AssertionError) as e:
                break

    except Exception as e:
        pass
    return extracted_data

def save_batch_data(batch_idx, positions, moves, values):
    batch_dir = os.path.join(OUTPUT_DIR, "batches")
    os.makedirs(batch_dir, exist_ok=True)
    
    if len(positions) > 0:
        X = np.array(positions, dtype=np.float32)
        y_policy = np.array(moves, dtype=np.int32)
        y_value = np.array(values, dtype=np.float32)
        
        np.save(os.path.join(batch_dir, f"X_batch_{batch_idx}.npy"), X)
        np.save(os.path.join(batch_dir, f"y_policy_batch_{batch_idx}.npy"), y_policy)
        np.save(os.path.join(batch_dir, f"y_value_batch_{batch_idx}.npy"), y_value)
        
        return X.shape[0]  
    return 0

def run_data_preparation():
    if not os.path.exists(PGN_FILE):
        print(f"ERROR: Input PGN file not found: {PGN_FILE}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    print(f"Starting data preparation from {PGN_FILE}...")
    start_time = time.time()

    filtered_pgn_strings = []
    
    dctx = zstd.ZstdDecompressor()
    with open(PGN_FILE, "rb") as compressed_file:
        with dctx.stream_reader(compressed_file) as binary_reader:
            text_stream = io.TextIOWrapper(binary_reader, encoding="utf-8", errors="ignore")
            
            current_game_text = ""
            in_headers = False
            white_elo = black_elo = 0
            game_count = 0
            filtered_count = 0
            
            pbar = tqdm(desc="Finding games", unit=" games")
            
            for line in text_stream:
                if line.strip().startswith('[Event '):
                    if current_game_text:
                        if white_elo >= MIN_ELO and black_elo >= MIN_ELO:
                            filtered_pgn_strings.append(current_game_text)
                            filtered_count += 1
                            
                            if filtered_count >= MAX_GAMES:
                                break
                        
                        game_count += 1
                        pbar.update(1)
                        pbar.set_description(f"Games found: {game_count}, kept: {filtered_count}")
                    
                    current_game_text = line
                    in_headers = True
                    white_elo = black_elo = 0
                    continue
                
                current_game_text += line
                
                if in_headers:
                    if line.strip() == "":
                        in_headers = False
                    
                    white_elo_match = WHITE_ELO_RE.search(line)
                    if white_elo_match:
                        try:
                            white_elo = int(white_elo_match.group(1))
                        except ValueError:
                            white_elo = 0
                    
                    black_elo_match = BLACK_ELO_RE.search(line)
                    if black_elo_match:
                        try:
                            black_elo = int(black_elo_match.group(1))
                        except ValueError:
                            black_elo = 0
            
            if current_game_text and filtered_count < MAX_GAMES:
                if white_elo >= MIN_ELO and black_elo >= MIN_ELO:
                    filtered_pgn_strings.append(current_game_text)
                    filtered_count += 1
                game_count += 1
                pbar.update(1)
            
            pbar.close()

    print(f"Step 1 Done: Examined {game_count} games, {len(filtered_pgn_strings)} passed pre-filter.")
    print(f"Time elapsed: {time.time() - start_time:.2f}s")

    if not filtered_pgn_strings:
        print("No games passed the pre-filter. Exiting.")
        sys.exit(0)

    available_cores = multiprocessing.cpu_count()
    num_processes = max(1, available_cores - RESERVE_CORES)
    print(f"Using {num_processes} worker processes (reserving {RESERVE_CORES} cores)")

    total_datapoints = 0
    batch_files = []
    
    for batch_idx in range(0, len(filtered_pgn_strings), BATCH_SIZE):
        batch_end = min(batch_idx + BATCH_SIZE, len(filtered_pgn_strings))
        print(f"\nProcessing batch {batch_idx//BATCH_SIZE + 1} (games {batch_idx+1}-{batch_end} of {len(filtered_pgn_strings)})")
        
        batch_positions = []
        batch_moves = []
        batch_values = []
        processed_game_count = 0
        
        batch_args = [(pgn_str, MIN_ELO, MIN_FULLMOVE_NUMBER) 
                      for pgn_str in filtered_pgn_strings[batch_idx:batch_end]]
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            with tqdm(total=len(batch_args), desc="Processing games", unit="game") as pbar:
                for result_list in pool.imap_unordered(process_pgn_string_worker, batch_args):
                    if result_list:
                        processed_game_count += 1
                        for planes_list, move_idx, val in result_list:
                            batch_positions.append(planes_list)
                            batch_moves.append(move_idx)
                            batch_values.append(val)
                    pbar.update(1)
                    pbar.set_postfix({"Data points": len(batch_moves), "Games OK": processed_game_count})
        
        if SAVE_INTERMEDIATE and batch_positions:
            points_saved = save_batch_data(batch_idx//BATCH_SIZE, batch_positions, batch_moves, batch_values)
            batch_files.append(batch_idx//BATCH_SIZE)
            total_datapoints += points_saved
            print(f"Saved batch {batch_idx//BATCH_SIZE} with {points_saved} data points. Total so far: {total_datapoints}")
            
            # Clear memory
            del batch_positions
            del batch_moves
            del batch_values
            gc.collect()
    
    print(f"\nProcessed all batches. Total data points: {total_datapoints}")
    
    print("\nCombining all batches, shuffling, and splitting...")
    
    if SAVE_INTERMEDIATE and batch_files:
        batch_dir = os.path.join(OUTPUT_DIR, "batches")
        all_X = []
        all_policy = []
        all_value = []
        
        for batch_idx in batch_files:
            X = np.load(os.path.join(batch_dir, f"X_batch_{batch_idx}.npy"))
            policy = np.load(os.path.join(batch_dir, f"y_policy_batch_{batch_idx}.npy"))
            value = np.load(os.path.join(batch_dir, f"y_value_batch_{batch_idx}.npy"))
            
            all_X.append(X)
            all_policy.append(policy)
            all_value.append(value)
        
        X = np.concatenate(all_X, axis=0)
        y_policy = np.concatenate(all_policy, axis=0)
        y_value = np.concatenate(all_value, axis=0)
        
        del all_X
        del all_policy
        del all_value
        gc.collect()
    else:
        print("No intermediate batch files found. This should not happen.")
        sys.exit(1)
    
    total_datapoints = X.shape[0]
    print(f"Final X shape: {X.shape}, y_policy shape: {y_policy.shape}, y_value shape: {y_value.shape}")

    print("Shuffling data...")
    indices = np.random.permutation(total_datapoints)
    X = X[indices]
    y_policy = y_policy[indices]
    y_value = y_value[indices]

    val_size = int(VALIDATION_SPLIT * total_datapoints)
    train_size = total_datapoints - val_size

    print(f"Splitting into {train_size} training and {val_size} validation samples.")
    X_train, X_val = X[:train_size], X[train_size:]
    y_policy_train, y_policy_val = y_policy[:train_size], y_policy[train_size:]
    y_value_train, y_value_val = y_value[:train_size], y_value[train_size:]

    print("Saving processed data to .npy files...")
    try:
        np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
        np.save(os.path.join(OUTPUT_DIR, "y_policy_train.npy"), y_policy_train)
        np.save(os.path.join(OUTPUT_DIR, "y_value_train.npy"), y_value_train)
        if val_size > 0:
            np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
            np.save(os.path.join(OUTPUT_DIR, "y_policy_val.npy"), y_policy_val)
            np.save(os.path.join(OUTPUT_DIR, "y_value_val.npy"), y_value_val)
        print(f"Data successfully saved to {OUTPUT_DIR}")
    except Exception as e:
        print(f"ERROR: Failed to save NumPy arrays: {e}", file=sys.stderr)
        sys.exit(1)

    end_time = time.time()
    print(f"\nTotal data preparation completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_data_preparation()

