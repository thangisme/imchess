import chess
import chess.pgn
import numpy as np
import os
import io
import time
import sys
from tqdm import tqdm

PGN_FILE = "lichess_db_standard_rated_2025-02.pgn.zst"
OUTPUT_DIR = "policy_training_data"
MAX_GAMES = 100000
MIN_ELO = 2300
VALIDATION_SPLIT = 0.1
MIN_FULLMOVE_NUMBER = 5

# Calculate the maximum possible move ID based on our encoding scheme
# Scheme: from_square * 320 + to_square * 5 + promotion_type
# Max from/to = 63, Max promotion_type = 4 (Knight)
MAX_MOVE_ID = 63 * 320 + 63 * 5 + 4  # = 20479
NUM_POSSIBLE_MOVES = MAX_MOVE_ID + 1  # = 20480 (Output layer size for NN)


def encode_move(move: chess.Move) -> int:
    from_square = move.from_square  # 0-63
    to_square = move.to_square  # 0-63

    promotion_type = 0
    if move.promotion is not None:
        promotion_map = {
            chess.QUEEN: 1,
            chess.ROOK: 2,
            chess.BISHOP: 3,
            chess.KNIGHT: 4,
        }
        promotion_type = promotion_map.get(move.promotion, 0)  # Default to 0 if unknown

    # Calculate unique index
    # Multiply 'from' by possibilities for 'to' and 'promo'
    # Multiply 'to' by possibilities for 'promo'
    move_index = from_square * (64 * 5) + to_square * 5 + promotion_type

    if move_index > MAX_MOVE_ID:
        print(
            f"Warning: Calculated move index {move_index} exceeds MAX_MOVE_ID {MAX_MOVE_ID} for move {move.uci()}"
        )

    return move_index


def decode_move(index: int) -> chess.Move | None:
    if not (0 <= index <= MAX_MOVE_ID):
        return None

    promotion_type = index % 5
    to_square = (index // 5) % 64
    from_square = index // (64 * 5)

    promotion = None
    if promotion_type > 0:
        promotion_map_rev = {
            1: chess.QUEEN,
            2: chess.ROOK,
            3: chess.BISHOP,
            4: chess.KNIGHT,
        }
        promotion = promotion_map_rev.get(promotion_type)

    try:
        move = chess.Move(from_square, to_square, promotion)
        return move
    except ValueError:
        return None


def board_to_planes(board: chess.Board) -> np.ndarray:
    planes = np.zeros((8, 8, 13), dtype=np.float32)

    piece_to_plane_idx = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = chess.square_rank(square)
            file = chess.square_file(square)

            plane_base_idx = piece_to_plane_idx[piece.piece_type]
            color_offset = 0 if piece.color == chess.WHITE else 6
            plane_idx = plane_base_idx + color_offset

            planes[7 - rank, file, plane_idx] = 1.0

    if board.turn == chess.BLACK:
        planes[:, :, 12] = 1.0

    return planes


def process_pgn_stream(pgn_stream, positions, target_moves, target_values):
    games_processed_in_stream = 0
    global game_count

    pbar = tqdm(total=MAX_GAMES, initial=game_count, desc="Processing Games")

    while game_count < MAX_GAMES:
        try:
            game = chess.pgn.read_game(pgn_stream)
        except Exception as e:
            print(f"\nWarning: Skipping game due to PGN parsing error: {e}")
            for _ in range(10):
                if pgn_stream.readline() == "":
                    break
            continue

        if game is None:
            print("\nEnd of PGN stream reached.")
            break

        try:
            white_elo = int(game.headers.get("WhiteElo", "0"))
            black_elo = int(game.headers.get("BlackElo", "0"))
            if min(white_elo, black_elo) < MIN_ELO:
                continue
        except ValueError:
            continue

        # Extract game result
        result = game.headers.get("Result", "*")
        result_value = 0.0  # Default for draw or unknown
        if result == "1-0":
            result_value = 1.0
        elif result == "0-1":
            result_value = -1.0

        board = game.board()
        try:
            for move in game.mainline_moves():
                is_valid_state = (
                    board.fullmove_number >= MIN_FULLMOVE_NUMBER
                    and not board.is_game_over()
                    and not board.is_check()
                )

                if is_valid_state:
                    board_state_planes = board_to_planes(board)
                    encoded_move = encode_move(move)

                    # Store position value from current player's perspective
                    position_value = (
                        result_value if board.turn == chess.WHITE else -result_value
                    )

                    positions.append(board_state_planes)
                    target_moves.append(encoded_move)
                    target_values.append(position_value)

                board.push(move)

        except Exception as e:
            print(f"\nWarning: Error processing moves in game {game_count + 1}: {e}")

        game_count += 1
        games_processed_in_stream += 1
        pbar.update(1)

    pbar.close()
    return games_processed_in_stream


def run_data_preparation():
    if not os.path.exists(PGN_FILE):
        print(f"ERROR: Input PGN file not found: {PGN_FILE}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    all_positions = []
    all_target_moves = []
    all_target_values = []
    global game_count
    game_count = 0

    print(f"Starting data preparation from {PGN_FILE}...")
    start_time = time.time()

    if PGN_FILE.lower().endswith(".zst"):
        try:
            import zstandard as zstd

            print("Detected .zst file, using Zstandard stream decompression...")
            with open(PGN_FILE, "rb") as compressed_file:
                decompressor = zstd.ZstdDecompressor()
                with decompressor.stream_reader(compressed_file) as binary_reader:
                    text_stream = io.TextIOWrapper(
                        binary_reader, encoding="utf-8", errors="ignore"
                    )
                    process_pgn_stream(
                        text_stream, all_positions, all_target_moves, all_target_values
                    )

        except ImportError:
            print(
                "ERROR: 'zstandard' library not installed, but input file is .zst.",
                file=sys.stderr,
            )
            print("Please install it: pip install zstandard", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(
                f"\nERROR during Zstandard decompression or processing: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

    else:
        print("Detected .pgn file, processing directly...")
        try:
            with open(
                PGN_FILE, "r", encoding="utf-8", errors="ignore"
            ) as pgn_file_handle:
                process_pgn_stream(
                    pgn_file_handle, all_positions, all_target_moves, all_target_values
                )
        except Exception as e:
            print(f"\nERROR processing PGN file: {e}", file=sys.stderr)
            sys.exit(1)

    total_datapoints = len(all_positions)
    print(f"\nFinished reading PGNs. Processed {game_count} games.")
    print(f"Extracted {total_datapoints} (position, move, value) triplets.")

    if total_datapoints == 0:
        print(
            "ERROR: No data points were extracted. Check PGN file, ELO filter, or other settings."
        )
        sys.exit(1)

    print("Converting data to NumPy arrays...")
    X = np.array(all_positions, dtype=np.float32)
    y_policy = np.array(all_target_moves, dtype=np.int32)
    y_value = np.array(all_target_values, dtype=np.float32)
    print(
        f"X shape: {X.shape}, y_policy shape: {y_policy.shape}, y_value shape: {y_value.shape}"
    )

    print("Shuffling data...")
    indices = np.random.permutation(total_datapoints)
    X = X[indices]
    y_policy = y_policy[indices]
    y_value = y_value[indices]

    val_size = int(VALIDATION_SPLIT * total_datapoints)
    train_size = total_datapoints - val_size

    if train_size <= 0 or val_size <= 0:
        print("ERROR: Not enough data points to create train/validation split.")
        print(
            f"Total points: {total_datapoints}, Validation size requested: {val_size}"
        )
        sys.exit(1)

    print(f"Splitting into {train_size} training and {val_size} validation samples.")
    X_train, X_val = X[:train_size], X[train_size:]
    y_policy_train, y_policy_val = y_policy[:train_size], y_policy[train_size:]
    y_value_train, y_value_val = y_value[:train_size], y_value[train_size:]

    print("Saving processed data to .npy files...")
    try:
        np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
        np.save(os.path.join(OUTPUT_DIR, "y_policy_train.npy"), y_policy_train)
        np.save(os.path.join(OUTPUT_DIR, "y_value_train.npy"), y_value_train)
        np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
        np.save(os.path.join(OUTPUT_DIR, "y_policy_val.npy"), y_policy_val)
        np.save(os.path.join(OUTPUT_DIR, "y_value_val.npy"), y_value_val)
        print(f"Data successfully saved to {OUTPUT_DIR}")
    except Exception as e:
        print(f"ERROR: Failed to save NumPy arrays: {e}", file=sys.stderr)
        sys.exit(1)

    end_time = time.time()
    print(f"\nData preparation completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    run_data_preparation()
