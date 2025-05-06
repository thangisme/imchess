import chess
import chess.polyglot
import numpy as np
import onnxruntime as ort
import time
import random
import sys
import os


class ChessAI:
    MAX_MOVE_ID = 63 * (64 * 5) + 63 * 5 + 4
    N_PLANES = 13
    
    def __init__(self, policy_model_path="policy_models/policy_model.onnx", 
                opening_book_path="baron30.bin", temperature=0.3):
        self.board = chess.Board()
        self.policy_model_path = policy_model_path
        self.opening_book_path = opening_book_path
        self.temperature = temperature
        self.nodes_searched = 0
        self.current_best_move = None
        self.policy_cache = {}
        
        try:
            if not os.path.exists(self.policy_model_path):
                raise FileNotFoundError(f"Policy model not found at '{self.policy_model_path}'")
                
            self.policy_session = ort.InferenceSession(
                self.policy_model_path, 
                providers=['CPUExecutionProvider']
            )
            self.policy_input_name = self.policy_session.get_inputs()[0].name
        except Exception as e:
            print(f"Failed to load ONNX Policy Network: {e}", file=sys.stderr)
            raise

    def reset_board(self):
        self.board.reset()
        self.policy_cache.clear()

    def set_board_from_fen(self, fen):
        try:
            self.board.set_fen(fen)
            self.policy_cache.clear()
        except ValueError:
            print(f"Invalid FEN string: {fen}", file=sys.stderr)
            self.board.reset()

    def make_move(self, move_uci_or_object):
        try:
            if isinstance(move_uci_or_object, str):
                move = self.board.parse_uci(move_uci_or_object)
            elif isinstance(move_uci_or_object, chess.Move):
                move = move_uci_or_object
            else:
                return False

            if move in self.board.legal_moves:
                self.board.push(move)
                self.policy_cache.clear()
                return True
            return False
        except Exception:
            return False

    def get_random_move(self):
        legal_moves = list(self.board.legal_moves)
        return random.choice(legal_moves) if legal_moves else None

    def _board_to_planes(self, board):
        planes = np.zeros((8, 8, self.N_PLANES), dtype=np.float32)
        piece_to_plane_idx = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
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
            planes[:, :, self.N_PLANES - 1] = 1.0
        return planes

    def _encode_move(self, move):
        from_square = move.from_square
        to_square = move.to_square
        promotion_type = 0
        if move.promotion is not None:
            promotion_map = {chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}
            promotion_type = promotion_map.get(move.promotion, 0)
        move_index = from_square * (64 * 5) + to_square * 5 + promotion_type
        return min(move_index, self.MAX_MOVE_ID)

    def _decode_move(self, index):
        if not (0 <= index <= self.MAX_MOVE_ID): 
            return None
        promotion_type = index % 5
        to_square = (index // 5) % 64
        from_square = index // (64 * 5)
        promotion = None
        if promotion_type > 0:
            promotion_map_rev = {1: chess.QUEEN, 2: chess.ROOK, 3: chess.BISHOP, 4: chess.KNIGHT}
            promotion = promotion_map_rev.get(promotion_type)
        try: 
            return chess.Move(from_square, to_square, promotion)
        except ValueError: 
            return None

    def get_policy_probabilities(self, board):
        board_fen = board.fen()
        if board_fen in self.policy_cache:
            return self.policy_cache[board_fen]

        board_planes = self._board_to_planes(board)
        batch = np.expand_dims(board_planes, axis=0).astype(np.float32)

        try:
            policy_logits_all_moves = self.policy_session.run(None, {self.policy_input_name: batch})[0][0]
        except Exception as e:
            print(f"Error during policy network inference: {e}", file=sys.stderr)
            return {}

        move_probabilities = {}
        legal_moves = list(board.legal_moves)
        sum_legal_probs = 0.0
        
        for move in legal_moves:
            encoded_move_idx = self._encode_move(move)
            if 0 <= encoded_move_idx < len(policy_logits_all_moves):
                prob = float(policy_logits_all_moves[encoded_move_idx])
                move_probabilities[move] = prob
                sum_legal_probs += prob
            else:
                move_probabilities[move] = 1e-9

        if sum_legal_probs > 1e-8:
            for move in move_probabilities:
                move_probabilities[move] /= sum_legal_probs
        elif legal_moves:
            prob_per_move = 1.0 / len(legal_moves)
            for move in legal_moves:
                move_probabilities[move] = prob_per_move

        self.policy_cache[board_fen] = move_probabilities
        return move_probabilities

    def get_book_move(self):
        if not self.opening_book_path or not os.path.exists(self.opening_book_path):
            return None
        try:
            with chess.polyglot.open_reader(self.opening_book_path) as reader:
                entries = list(reader.find_all(self.board))
                if entries:
                    return random.choices([e.move for e in entries], weights=[e.weight for e in entries], k=1)[0]
            return None
        except Exception:
            return None

    def get_best_move_iterative_deepening(self, max_depth=1, time_limit=1.0, external_stop_callback=None):
        self.nodes_searched = 0
        start_time = time.time()

        if self.board.is_game_over():
            return None

        move_probs = self.get_policy_probabilities(self.board)
        self.nodes_searched = len(move_probs)

        if not move_probs:
            self.current_best_move = self.get_random_move()
            return self.current_best_move

        moves = list(move_probs.keys())
        raw_probabilities = np.array([move_probs[m] for m in moves], dtype=np.float64)

        if self.temperature == 0:
            best_move_idx = np.argmax(raw_probabilities)
            selected_move = moves[best_move_idx]
        else:
            try:
                probs_temp_scaled = np.power(raw_probabilities + 1e-9, 1.0 / self.temperature)
                probs_temp_scaled /= np.sum(probs_temp_scaled)
                selected_move = np.random.choice(moves, p=probs_temp_scaled)
            except ValueError:
                best_move_idx = np.argmax(raw_probabilities)
                selected_move = moves[best_move_idx]

        self.current_best_move = selected_move

        elapsed = time.time() - start_time
        nps = int(self.nodes_searched / elapsed) if elapsed > 0 else 0
        score_for_uci = int(move_probs.get(selected_move, 0) * 1000)
        pv_str = selected_move.uci()
        
        print(f"info depth 1 score cp {score_for_uci} nodes {self.nodes_searched} nps {nps} time {int(elapsed * 1000)} pv {pv_str}")

        return selected_move

    def get_current_best_move(self):
        return self.current_best_move
