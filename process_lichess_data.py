import os
import io
import time

import orjson
import numpy as np
import zstandard as zstd

INPUT_ZST_FILE = "lichess_db_eval.jsonl.zst"
MAX_POSITIONS_TO_COLLECT = 3_000_000
TARGET_PER_PHASE = MAX_POSITIONS_TO_COLLECT // 3
MIN_DEPTH_FILTER = 18
CP_SCALING_FACTOR = 800

BOARD_MEMMAP_FILENAME = "board_states.dat"
SCORE_MEMMAP_FILENAME = "eval_scores.dat"
COUNT_FILENAME = "valid_positions_count.txt"

N_PLANES = 13  # 12 pieces + 1 side-to-move plane

PIECE_TO_PLANE = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}

count_open, count_mid, count_end = 0, 0, 0


def classify_phase(planes):
    pieces_count = int(planes.sum())
    if pieces_count >= 26:
        return "open"
    elif pieces_count <= 10:
        return "end"
    else:
        return "mid"


def normalize_score(cp, mate):
    if mate is not None:
        return 1.0 if mate > 0 else -1.0
    return np.tanh(cp / CP_SCALING_FACTOR)


def fen_to_uint8_planes(fen):
    parts = fen.split()
    board, stm = parts[0], parts[1]
    planes = np.zeros((8, 8, N_PLANES), dtype=np.uint8)

    rows = board.split("/")
    for r, row in enumerate(rows):
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                planes[r, f, PIECE_TO_PLANE[ch]] = 1
                f += 1

    if stm == "b":
        planes[:, :, 12] = 1

    return planes


if not os.path.exists(INPUT_ZST_FILE):
    print("ERROR: Input file not found")
    exit(1)

# Pre‑allocate memmaps on disk
board_memmap = np.memmap(
    BOARD_MEMMAP_FILENAME,
    mode="w+",
    dtype=np.uint8,
    shape=(MAX_POSITIONS_TO_COLLECT, 8, 8, N_PLANES),
)
score_memmap = np.memmap(
    SCORE_MEMMAP_FILENAME,
    mode="w+",
    dtype=np.float16,
    shape=(MAX_POSITIONS_TO_COLLECT,),
)

total_lines_processed = 0
total_positions_collected = 0
start_time = time.time()

with open(INPUT_ZST_FILE, "rb") as compressed_file:
    decompressor = zstd.ZstdDecompressor()
    with decompressor.stream_reader(compressed_file) as stream_reader:
        text_reader = io.TextIOWrapper(stream_reader, encoding="utf-8")
        for line in text_reader:
            if (count_open + count_mid + count_end) >= TARGET_PER_PHASE * 3:
                break

            total_lines_processed += 1
            if not line.strip():
                continue
            try:
                record = orjson.loads(line)
            except:
                continue

            fen_string = record.get("fen")
            eval_list = record.get("evals")
            if not fen_string or not eval_list:
                continue

            best_eval = None
            best_eval_depth = -1
            for evaluation in eval_list:
                depth = evaluation.get("depth", 0)
                if depth >= MIN_DEPTH_FILTER and depth > best_eval_depth:
                    principal_variations = evaluation.get("pvs")
                    if principal_variations:
                        best_eval_depth = depth
                        best_eval = evaluation

            if best_eval is None:
                continue

            first_pv = best_eval["pvs"][0]
            cp_score = first_pv.get("cp")
            mate_score = first_pv.get("mate")
            if cp_score is None and mate_score is None:
                continue

            normalized_value = normalize_score(cp_score, mate_score)

            try:
                board_planes = fen_to_uint8_planes(fen_string)
            except:
                continue

            phase = classify_phase(board_planes)
            if phase == "open" and count_open >= TARGET_PER_PHASE:
                continue
            if phase == "mid" and count_mid >= TARGET_PER_PHASE:
                continue
            if phase == "end" and count_end >= TARGET_PER_PHASE:
                continue

            board_memmap[total_positions_collected] = board_planes
            score_memmap[total_positions_collected] = np.float16(normalized_value)
            total_positions_collected += 1

            if phase == "open":
                count_open += 1
            if phase == "mid":
                count_mid += 1
            if phase == "end":
                count_end += 1

            if total_positions_collected >= MAX_POSITIONS_TO_COLLECT:
                break

            if total_lines_processed % 100_000 == 0:
                elapsed = time.time() - start_time
                rate = total_lines_processed / elapsed
                print(
                    f"Lines read: {total_lines_processed:,}; "
                    f"Collected: {total_positions_collected:,}; Open={count_open}; Mid={count_mid}; End={count_end}; "
                    f"{rate:.1f} lines/sec"
                )

board_memmap.flush()
score_memmap.flush()

with open(COUNT_FILENAME, "w") as f:
    f.write(str(total_positions_collected))

total_time = time.time() - start_time
print(
    f"Finished. Processed {total_lines_processed:,} lines, "
    f"collected {total_positions_collected:,} positions in {total_time:.1f}s"
)
