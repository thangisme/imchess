import chess
import chess.polyglot
import numpy as np
import onnxruntime as ort
import time
import random
import sys
import os
import math


class MCTSNode:
    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children = {}

        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior

        self.nn_value = None

        self._is_terminal = None
        self._terminal_value = None

    def expand(self, policy_value_fn):
        probs, self.nn_value = policy_value_fn(self.board)

        for move, prob in probs.items():
            if move not in self.children:
                new_board = self.board.copy()
                new_board.push(move)
                self.children[move] = MCTSNode(
                    new_board, parent=self, move=move, prior=prob
                )

    def select_child(self, c_puct=1.5):
        best_score = float("-inf")
        best_move = None
        best_child = None

        for move, child in self.children.items():
            exploitation = 0
            if child.visits > 0:
                exploitation = child.value_sum / child.visits

            exploration = (
                c_puct * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            )

            ucb_score = exploitation + exploration

            if ucb_score > best_score:
                best_score = ucb_score
                best_move = move
                best_child = child

        return best_child, best_move

    def is_terminal(self):
        if self._is_terminal is None:
            self._is_terminal = self.board.is_game_over()
        return self._is_terminal

    def terminal_value(self, perspective):
        if self._terminal_value is None:
            if self.board.is_checkmate():
                self._terminal_value = -1.0
            else:
                self._terminal_value = 0.0

        to_move = self.board.turn
        if perspective != to_move:
            return -self._terminal_value
        return self._terminal_value

    def backpropagate(self, value):
        node = self
        while node is not None:
            node.visits += 1
            node.value_sum += value
            value = -value
            node = node.parent


class ChessAI:
    MAX_MOVE_ID = 63 * (64 * 5) + 63 * 5 + 4
    N_PLANES = 13

    def __init__(
        self,
        policy_model_path="policy_models/policy_value_model.onnx",
        opening_book_path="baron30.bin",
        temperature=0.3,
    ):
        self.board = chess.Board()
        self.policy_model_path = policy_model_path
        self.opening_book_path = opening_book_path
        self.temperature = temperature
        self.nodes_searched = 0
        self.current_best_move = None
        self.policy_value_cache = {}
        self.mcts_cache = {}

        try:
            if not os.path.exists(policy_model_path):
                raise FileNotFoundError(
                    f"Policy model not found at '{policy_model_path}'"
                )

            self.policy_value_session = ort.InferenceSession(
                policy_model_path, providers=["CPUExecutionProvider"]
            )
            self.policy_input_name = self.policy_value_session.get_inputs()[0].name

            print(f"ONNX model loaded. Input name: {self.policy_input_name}")
            print(
                f"Model outputs: {[o.name for o in self.policy_value_session.get_outputs()]}"
            )

        except Exception as e:
            print(f"Failed to load ONNX Policy-Value Network: {e}", file=sys.stderr)
            raise

    def reset_board(self):
        self.board.reset()
        self.policy_value_cache.clear()
        self.mcts_cache.clear()

    def set_board_from_fen(self, fen):
        try:
            self.board.set_fen(fen)
            self.policy_value_cache.clear()
            self.mcts_cache.clear()
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
                self.policy_value_cache.clear()
                self.mcts_cache.clear()
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
            planes[:, :, self.N_PLANES - 1] = 1.0
        return planes

    def _encode_move(self, move):
        from_square = move.from_square
        to_square = move.to_square
        promotion_type = 0
        if move.promotion is not None:
            promotion_map = {
                chess.QUEEN: 1,
                chess.ROOK: 2,
                chess.BISHOP: 3,
                chess.KNIGHT: 4,
            }
            promotion_type = promotion_map.get(move.promotion, 0)
        move_index = from_square * (64 * 5) + to_square * 5 + promotion_type
        return min(move_index, self.MAX_MOVE_ID)

    def get_policy_and_value(self, board):
        board_fen = board.fen()
        if board_fen in self.policy_value_cache:
            return self.policy_value_cache[board_fen]

        board_planes = self._board_to_planes(board)
        batch = np.expand_dims(board_planes, axis=0).astype(np.float32)

        try:
            # Explicitly request outputs by name instead of using None
            outputs = self.policy_value_session.run(
                ["policy_head", "value_head"], {self.policy_input_name: batch}
            )

            # Now we get both outputs properly
            policy_logits = outputs[0][0]  # Policy head
            value = float(outputs[1][0])  # Value head

        except Exception as e:
            print(f"Error during policy-value inference: {e}", file=sys.stderr)
            return {}, 0.0

        # Rest of function unchanged
        move_probabilities = {}
        legal_moves = list(board.legal_moves)
        sum_legal_probs = 0.0

        for move in legal_moves:
            encoded_move_idx = self._encode_move(move)
            if 0 <= encoded_move_idx < len(policy_logits):
                prob = float(policy_logits[encoded_move_idx])
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

        self.policy_value_cache[board_fen] = (move_probabilities, value)
        return move_probabilities, value

    def mcts_search(
        self, root_board, num_simulations=800, exploration=1.5, time_limit=3.0
    ):
        root = MCTSNode(root_board)
        self.nodes_searched = 0

        def policy_value_fn(board):
            return self.get_policy_and_value(board)

        root.expand(policy_value_fn)

        start_time = time.time()
        simulation = 0

        while time.time() - start_time < time_limit and simulation < num_simulations:
            node = root
            search_path = [node]

            while node.children and not node.is_terminal():
                child, move = node.select_child(exploration)
                if child is None:
                    break

                node = child
                search_path.append(node)

            if not node.is_terminal() and not node.children:
                node.expand(policy_value_fn)

            value = 0
            if node.is_terminal():
                value = node.terminal_value(root_board.turn)
            else:
                value = node.nn_value
                if node.board.turn != root_board.turn:
                    value = -value

            for bnode in reversed(search_path):
                bnode.backpropagate(value)
                value = -value

            simulation += 1
            self.nodes_searched += 1

        best_move = None
        best_visits = -1

        for move, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = move

        move_stats = sorted(
            [
                (move, child.visits, child.value_sum / max(1, child.visits))
                for move, child in root.children.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        for i, (move, visits, value) in enumerate(move_stats[:5]):
            cp_score = int(value * 100)
            print(
                f"info string {i + 1}. {move.uci()} visits: {visits} score: {cp_score}"
            )

        elapsed = time.time() - start_time
        print(f"info string MCTS completed {simulation} simulations in {elapsed:.2f}s")

        return best_move, root

    def get_best_move_iterative_deepening(
        self, max_depth=1, time_limit=3.0, external_stop_callback=None
    ):
        start_time = time.time()

        # book_move = self.get_book_move()
        # if book_move:
        #     self.current_best_move = book_move
        #     print(f"info string book move {book_move.uci()}")
        #     return book_move

        if self.board.is_game_over():
            return None

        simulation_count = min(800, int(time_limit * 300))

        best_move, root = self.mcts_search(
            self.board, num_simulations=simulation_count, time_limit=time_limit * 0.95
        )

        if not best_move:
            move_probs, value = self.get_policy_and_value(self.board)
            moves = list(move_probs.keys())
            if moves:
                best_move = max(moves, key=lambda m: move_probs[m])
            else:
                best_move = self.get_random_move()

        elapsed = time.time() - start_time
        best_child = root.children.get(best_move)
        if best_child:
            score = int(100 * best_child.value_sum / max(1, best_child.visits))
            visits = best_child.visits
        else:
            score = 0
            visits = 0

        effective_depth = 1 + int(math.log2(max(1, self.nodes_searched)) / 2)

        print(
            f"info depth {effective_depth} score cp {score} nodes {self.nodes_searched} time {int(elapsed * 1000)} pv {best_move.uci()}"
        )

        self.current_best_move = best_move
        return best_move

    def get_book_move(self):
        if not self.opening_book_path or not os.path.exists(self.opening_book_path):
            return None
        try:
            with chess.polyglot.open_reader(self.opening_book_path) as reader:
                entries = list(reader.find_all(self.board))
                if entries:
                    return random.choices(
                        [e.move for e in entries],
                        weights=[e.weight for e in entries],
                        k=1,
                    )[0]
            return None
        except Exception:
            return None

    def get_current_best_move(self):
        return self.current_best_move
