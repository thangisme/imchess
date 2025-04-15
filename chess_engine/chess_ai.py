import chess
import chess.polyglot
import random
import time
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

NN_MODEL_PATH = "chess_eval.keras"
NN_SCORE_SCALING_FACTOR = 800.0

class ChessAI:
    def __init__(self, evaluation_mode="nn", nn_model_path=NN_MODEL_PATH):
        self.board = chess.Board()
        self.transposition_table = {}
        self.nodes_searched = 0
        self.opening_book = "baron30.bin"
        self.killer_moves = [[None,None] for _ in range(100)]
        self.current_ply = 0
        self.current_best_move = None
        self.evaluation_mode = evaluation_mode
        self.nn_model = None
        if self.evaluation_mode == "nn":
            if os.path.exists(nn_model_path):
                try:
                    self.nn_model = tf.keras.models.load_model(nn_model_path)
                    print("NN Model loaded successfully", file=sys.stderr)
                except Exception as e:
                    print(f"ERROR: failed to load NN model from {nn_model_path}: {e}", file=sys.stderr)
                    print("Falling back to 'algo' evaluation mode", file=sys.stderr)
                    self.evaluation_mode = "algo"
            else:
                print("ERROR: NN Model file not found", file=sys.stderr)
                print("Falling back to 'algo' evaluation mode", file=sys.stderr)
                self.evaluation_mode = "algo"

    def reset_board(self):
        self.board = chess.Board()

    def set_board_from_fen(self, fen):
        self.board = chess.Board(fen)

    def make_move(self, move):
        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        return False

    def get_random_move(self):
        legal_moves = list(self.board.legal_moves)
        if legal_moves:
            return random.choice(legal_moves)
        return None

    def evaluate_board(self):
        if self.evaluation_mode == "nn":
            return self._evaluate_board_nn()
        elif self.evaluation_mode == "algo":
            return self._evaluate_board_algo()
        else:
            print("Unknown evaluation mode, falling back to 'algo'", file=sys.stderr)
            self.evaluation_mode = "algo"
            return self._evaluate_board_algo()
        return self._evaluate_board_nn()
        
    def _evaluate_board_algo(self):
        if self.board.is_checkmate():
            return -10000 if self.board.turn else 10000

        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0 # Draw

        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }

        pawn_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]

        knight_table = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]

        bishop_table = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5,  5,  5,  5,  5,-10,
            -10,  0,  5,  0,  0,  5,  0,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ]

        rook_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
        ]

        queen_table = [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ]

        king_middle_game_table = [
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            20, 20,  0,  0,  0,  0, 20, 20,
            20, 30, 10,  0,  0, 10, 30, 20
        ]

        material_score = 0
        position_score = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)

                pos_value = 0
                if piece.piece_type == chess.PAWN:
                    if piece.color == chess.WHITE:
                        pos_value = pawn_table[square]
                    else: 
                        pos_value = pawn_table[chess.square_mirror(square)]
                elif piece.piece_type == chess.KNIGHT:
                    if piece.color == chess.WHITE:
                        pos_value = knight_table[square]
                    else:
                        pos_value = knight_table[chess.square_mirror(square)]
                elif piece.piece_type == chess.BISHOP:
                    if piece.color == chess.WHITE:
                        pos_value = bishop_table[square]
                    else:
                        pos_value = bishop_table[chess.square_mirror(square)]
                elif piece.piece_type == chess.ROOK:
                    if piece.color == chess.WHITE:
                        pos_value = rook_table[square]
                    else:
                        pos_value = rook_table[chess.square_mirror(square)]
                elif piece.piece_type == chess.QUEEN:
                    if piece.color == chess.WHITE:
                        pos_value = queen_table[square]
                    else:
                        pos_value = queen_table[chess.square_mirror(square)]
                elif piece.piece_type == chess.KING:
                    if piece.color == chess.WHITE:
                        pos_value = king_middle_game_table[square]
                    else:
                        pos_value = king_middle_game_table[chess.square_mirror(square)]

                if piece.color == chess.WHITE:
                    material_score += value
                    position_score += pos_value
                else:
                    material_score -= value
                    position_score -= pos_value

        mobility_score = self.evaluate_mobility()

        pawn_structure_score = self.evaluate_pawn_structure()
        king_safety_score = self.evaluate_king_safety()
        
        score = material_score + position_score * 0.3 + mobility_score + pawn_structure_score + king_safety_score

        return score

    def get_board_hash(self):
        return chess.polyglot.zobrist_hash(self.board)
    
    def minimax(self, depth, alpha, beta, maximizing_player, stop_callback=None):
        if stop_callback and stop_callback():
            return 0
        self.nodes_searched += 1

        current_ply = self.current_ply

        if self.board.is_repetition(3) or self.board.is_fifty_moves():
            return 0

        board_hash = self.get_board_hash()
        if board_hash in self.transposition_table:
            stored_depth, stored_value, stored_flag = self.transposition_table[board_hash]
            if stored_depth >= depth:
                if stored_flag == "EXACT":
                    return stored_value
                elif stored_flag == "LOWERBOUND" and stored_value > alpha:
                    alpha = stored_value
                elif stored_flag == "UPPERBOUND" and stored_value < beta:
                    beta = stored_value
                if alpha >= beta:
                    return stored_value

        if self.board.is_game_over():
            return self.evaluate_board()

        if depth == 0:
            return self.quiescence_search(alpha, beta,maximizing_player)

        original_alpha = alpha

        moves = list(self.order_moves(self.board.legal_moves))
        moves_searched = 0

        if maximizing_player:
            max_eval = float("-inf")
            for move in moves:
                is_check = self.board.is_check()
                gives_check = False
                self.board.push(move)
                gives_check = self.board.is_check()
                self.board.pop()

                self.current_ply = current_ply + 1

                reduced_depth = depth - 1
                do_full_depth = True

                if depth >= 3 and moves_searched >= 3 and not is_check and not gives_check and not self.board.is_capture(move):
                    reduced_depth = depth - 2

                    self.board.push(move)
                    eval = -self.minimax(reduced_depth, -beta, -alpha, False)
                    self.board.pop()

                    do_full_depth = (eval > alpha)

                if do_full_depth:
                    self.board.push(move)
                    eval = self.minimax(depth - 1, alpha, beta, False)
                    self.board.pop()

                self.current_ply = current_ply
                moves_searched += 1

                max_eval = max(max_eval, eval)
                alpha = max(alpha,eval)
                if beta <= alpha:
                    self.add_killer_move(move)
                    break

            if max_eval <= original_alpha:
                flag = "UPPERBOUND"
            elif max_eval >= beta:
                flag = "LOWERBOUND"
            else:
                flag = "EXACT"
            self.transposition_table[board_hash] = (depth, max_eval, flag)
            
            return max_eval
        else:
            min_eval = float("inf")
            for move in moves:
                is_check = self.board.is_check()
                gives_check = False
                self.board.push(move)
                gives_check = self.board.is_check()
                self.board.pop()

                self.current_ply = current_ply + 1

                reduced_depth = depth - 1

                do_full_depth = True

                if depth >= 3 and moves_searched >= 3 and not is_check and not gives_check and not self.board.is_capture(move):
                    reduced_depth = depth = 2

                    self.board.push(move)
                    eval = -self.minimax(reduced_depth, -beta, -alpha, True)
                    self.board.pop()

                    do_full_depth = (eval < beta)

                if do_full_depth:
                    self.board.push(move)
                    eval = self.minimax(depth - 1, alpha, beta, True)
                    self.board.pop()

                self.current_ply = current_ply
                moves_searched += 1

                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    self.add_killer_move(move)
                    break

            if min_eval <= original_alpha:
                flag = "UPPERBOUND"
            elif min_eval >= beta:
                flag = "LOWERBOUND"
            else:
                flag = "EXACT"
            self.transposition_table[board_hash] = (depth, min_eval, flag)
            
            return min_eval
    def quiescence_search(self, alpha, beta, maximizing_player, q_depth=0):
        self.nodes_searched += 1

        MAX_QDEPTH = 8
        if q_depth >= MAX_QDEPTH:
            return self.evaluate_board()

        if self.board.is_game_over():
            return self.evaluate_board()

        stand_pat_score = self.evaluate_board()

        if maximizing_player:
            if stand_pat_score >= beta:
                return beta
            alpha = max(alpha, stand_pat_score)
        else:
            if stand_pat_score <= alpha:
                return alpha
            beta = min(beta, stand_pat_score)

        captures = [move for move in self.board.legal_moves if self.board.is_capture(move)]
        captures = self.order_moves(captures)

        if maximizing_player:
            for move in captures:
                self.board.push(move)
                score = self.quiescence_search(alpha, beta, False, q_depth + 1)
                self.board.pop()

                if score >= beta:
                    return beta
                alpha = max(alpha, score)
        else:
            for move in captures:
                self.board.push(move)
                score = self.quiescence_search(alpha, beta, True, q_depth + 1)
                self.board.pop()

                if score <= alpha:
                    return alpha
                beta = min(beta, score)
        
        return alpha if maximizing_player else beta
                
    def order_moves(self, moves):
        scored_moves = []
        for move in moves:
            score = 0
            if move in self.killer_moves[self.current_ply]:
                if move == self.killer_moves[self.current_ply][0]:
                    score = 9000
                else:
                    score = 8000
            
            elif self.board.is_capture(move):
                victim_piece = self.board.piece_at(move.to_square)
                attacker_piece = self.board.piece_at(move.from_square)
                if victim_piece and attacker_piece:
                    victim_value = {
                        chess.PAWN: 10,
                        chess.KNIGHT: 30,
                        chess.BISHOP:30,
                        chess.ROOK: 50,
                        chess.QUEEN: 90,
                        chess.KING: 900
                    }
                    attacker_value = {
                        chess.PAWN: 1,
                        chess.KNIGHT: 3,
                        chess.BISHOP: 3,
                        chess.ROOK: 5,
                        chess.QUEEN: 9,
                        chess.KING: 90
                    }
                    score = 1000 + victim_value[victim_piece.piece_type] - attacker_value[attacker_piece.piece_type]

            if move.promotion:
                score += 900

            self.board.push(move)
            if self.board.is_check():
                score += 50
            self.board.pop()
            scored_moves.append((move, score))
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in scored_moves]
            
    def get_current_best_move(self):
        return self.current_best_move
    
    def get_best_move_iterative_deepening(self, max_depth=4, time_limit=5.0, stop_callback=None):
        self.killer_moves = [[None,None] for _ in range(100)]
        book_move = self.get_book_move()
        if book_move:
            print(f"info string book move {book_move.uci()}")
            self.board_score = 0
            return book_move
        
        start_time = time.time()
        best_move = None
        self.nodes_searched = 0

        legal_moves = list(self.board.legal_moves)
        if len(legal_moves) == 1:
            self.current_best_move = legal_moves[0]
            return legal_moves[0]

        for current_depth in range(1, max_depth + 1):
            if stop_callback and stop_callback():
                break
            
            elapsed = time.time() - start_time
            if elapsed > time_limit * 0.8:
                break

            # self.transposition_table = {}
            move = self.get_best_move(current_depth)
            elapsed = time.time() - start_time

            if move:
                best_move = move
                nps = int(self.nodes_searched / elapsed) if elapsed > 0 else 0

                if abs(self.board_score) >= 9900:
                    if self.board_score > 0:
                        mate_in = (10000 - self.board_score + 1) // 2
                        score_str = f"mate {mate_in}"
                    else:
                        mate_in = (10000 + self.board_score + 1) // 2
                        score_str = f"mate -{mate_in}"
                else:
                    score_str = f"cp {int(self.board_score)}"
                print(f"info depth {current_depth} score {score_str} nodes {self.nodes_searched} nps {nps} time {int(elapsed * 1000)}")
                print(f"info depth {current_depth} score {score_str} nodes {self.nodes_searched} nps {nps} time {int(elapsed * 1000)}", file=sys.stderr)

                if abs(self.board_score) >= 9900:
                    break

            if elapsed >= time_limit:
                break
            
        if best_move is None and list(self.board.legal_moves):
            best_move = self.get_random_move()

        return best_move
            
    def get_best_move(self, depth=3):
        best_move = None
        best_value = float("-inf") if self.board.turn == chess.WHITE else float("inf")
        alpha = float("-inf")
        beta = float("inf")

        self.board_score = 0

        for move in self.order_moves(self.board.legal_moves):
            self.board.push(move)

            if self.board.turn == chess.WHITE:
                board_value = self.minimax(depth - 1, alpha, beta, True)
            else:
                board_value = self.minimax(depth - 1, alpha, beta, False)

            self.board.pop()

            if self.board.turn == chess.WHITE:
                if board_value > best_value:
                    best_value = board_value
                    best_move = move
                    alpha = max(alpha, best_value)

            else:
                if board_value < best_value:
                    best_value = board_value
                    best_move = move
                    beta = min(beta, best_value)

        self.board_score = best_value
        return best_move

    def get_book_move(self):
        try:
            with chess.polyglot.open_reader(self.opening_book) as reader:
                entries = list(reader.find_all(self.board))

                if entries:
                    # print(entries, file=sys.stderr)
                    total = sum(entry.weight for entry in entries)
                    pick = random.randint(1, total)
                    current_sum = 0
                    for entry in entries:
                        current_sum += entry.weight
                        if current_sum >= pick:
                            return entry.move
                    return entries[0].move
            return None
        except Exception as e:
            print(f"info string Book error: {e}")
            print("Error" + e, file=sys.stderr)
            return None

    def evaluate_pawn_structure(self):
        score = 0
        white_pawn_files = [0] * 8
        black_pawn_files = [0] * 8

        # Count pawns on each file
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                file_index = chess.square_file(square)
                if piece.color == chess.WHITE:
                    white_pawn_files[file_index] += 1
                else:
                    black_pawn_files[file_index] += 1

        # Evaluate doubled pawns
        doubled_pawn_penalty = 10
        for file_index in range(8):
            if white_pawn_files[file_index] > 1:
                score -= (white_pawn_files[file_index] - 1) * doubled_pawn_penalty
            if black_pawn_files[file_index] > 1:
                score += (black_pawn_files[file_index] - 1) * doubled_pawn_penalty

        # Evaluate isolated pawn
        isolated_pawn_penalty = 15
        for file_index in range(8):
            if white_pawn_files[file_index] > 0:
                is_isolated = True
                if file_index > 0 and white_pawn_files[file_index - 1] > 0:
                    is_isolated = False
                if file_index < 7 and white_pawn_files[file_index + 1] > 0:
                    is_isolated = False
                if is_isolated:
                    score -= isolated_pawn_penalty

            if black_pawn_files[file_index] > 0:
                is_isolated = True
                if file_index > 0 and black_pawn_files[file_index - 1] > 0:
                    is_isolated = False
                if file_index < 7 and black_pawn_files[file_index + 1] > 0:
                    is_isolated = False
                if is_isolated:
                    score += isolated_pawn_penalty
                    
        # Evaluate passed pawns
        passed_pawn_bonus = [0, 10, 20, 30, 60, 100, 150, 0]

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                file_index = chess.square_file(square)
                rank_index = chess.square_rank(square)

                if piece.color == chess.WHITE:
                    is_passed = True
                    for r in range(rank_index + 1, 8):
                        for f in range(max(0, file_index - 1), min(8, file_index + 2)):
                            check_square = chess.square(f, r)
                            check_piece = self.board.piece_at(check_square)
                            if check_piece and check_piece.piece_type == chess.PAWN and check_piece.color == chess.BLACK:
                                is_passed = False
                                break

                        if not is_passed:
                            break

                    if is_passed:
                        score += passed_pawn_bonus[rank_index]
                else:
                    is_passed = True
                    for r in range(rank_index + 1, 8):
                        for f in range(max(0, file_index - 1), min(8, file_index + 2)):
                            check_square = chess.square(f, r)
                            check_piece = self.board.piece_at(check_square)
                            if check_piece and check_piece.piece_type == chess.PAWN and check_piece.color == chess.WHITE:
                                is_passed = False
                                break

                        if not is_passed:
                            break

                    if is_passed:
                        score -= passed_pawn_bonus[7 - rank_index]
                    
        return score

    def evaluate_king_safety(self):
        score = 0

        white_king_square = self.board.king(chess.WHITE)
        black_king_square = self.board.king(chess.BLACK)

        if white_king_square is None or black_king_square is None:
            return 0

        def evaluate_king_side(king_square, color):
            safety_score = 0
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)

            is_kingside = king_file >= 5
            is_queenside = king_file <= 2

            if color == chess.WHITE:
                base_rank = 0
                shield_ranks = [king_rank + 1, king_rank + 2]
                file_check_range = range(1, 8)
            else:
                base_rank = 7
                shield_ranks = [king_rank - 1, king_rank - 2]
                file_check_range = range(6, -1, - 1)

            if (is_kingside or is_queenside) and king_rank == base_rank:
                shield_count = 0
                for f in range(max(0, king_file - 1), min(8, king_file + 2)):
                    for r in shield_ranks:
                        if 0 <= r < 8:
                            shield_square = chess.square(f, r)
                            piece = self.board.piece_at(shield_square)
                            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                                shield_count +=1

                safety_score += shield_count * 10

                for f in range(max(0, king_file - 1), min(8, king_file + 2)):
                    file_open = True
                    for r in file_check_range:
                        square = chess.square(f, r)
                        if self.board.piece_at(square) and self.board.piece_at(square).piece_type == chess.PAWN:
                            file_open = False
                            break
                    if file_open:
                        safety_score -= 25
            return safety_score
        score += evaluate_king_side(white_king_square, chess.WHITE)
        score -= evaluate_king_side(black_king_square, chess.BLACK)

        return score
        
    def evaluate_mobility(self):
        mobility_score = 0
        mobility_weights = {
            chess.PAWN: 0.1,
            chess.KNIGHT: 0.5,
            chess.BISHOP: 0.6,
            chess.ROOK: 0.4,
            chess.QUEEN: 0.3,
            chess.KING: 0.0
        }

        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        extended_center = [
            chess.C3, chess.D3, chess.E3, chess.F3,
            chess.C4, chess.D4, chess.E4, chess.F4,
            chess.C5, chess.D5, chess.E5, chess.F5,
            chess.C6, chess.D6, chess.E6, chess.F6,
        ]

        original_turn = self.board.turn

        mobility = {chess.WHITE: 0, chess.BLACK: 0}
        center_control = {chess.WHITE: 0, chess.BLACK: 0}

        for color in [chess.WHITE, chess.BLACK]:
            self.board.turn = color

            for square in chess.SQUARES:
                piece = self.board.piece_at(square)
                if piece and piece.color == color:
                    moves = [move for move in self.board.legal_moves if move.from_square == square]
                    mobility[color] += len(moves) * mobility_weights.get(piece.piece_type, 0)

            for square in center_squares:
                if self.board.is_attacked_by(color, square):
                    center_control[color] += 3

            for square in extended_center:
                if self.board.is_attacked_by(color, square):               
                    center_control[color] += 1

        self.board.turn = original_turn

        mobility_score = (mobility[chess.WHITE] - mobility[chess.BLACK]) * 2
        center_score = (center_control[chess.WHITE] - center_control[chess.BLACK]) * 2

        return mobility_score + center_score

    def add_killer_move(self, move):
        if not self.board.is_capture(move) and move != self.killer_moves[self.current_ply][0]:
            self.killer_moves[self.current_ply][1] = self.killer_moves[self.current_ply][0]
            self.killer_moves[self.current_ply][0] = move

    def _board_to_vector(self, board):
        piece_map = {
            'P' : 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6, # WHite pieces
            'p' : 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6, # Black pieces
            '.': 0 # Empty square
        }
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

    def _evaluate_board_nn(self):
        if self.board.is_checkmate():
            return -10000 if self.board.turn else 10000
        if self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
            return 0

        if self.nn_model is None:
            print("Error: NN model not available", file=sys.stderr)

        board_vec = self._board_to_vector(self.board)

        input_batch = np.expand_dims(board_vec, axis=0)

        raw_prediction = self.nn_model.predict(input_batch, verbose=0)

        nn_score = raw_prediction[0][0]        

        scaled_score = nn_score * NN_SCORE_SCALING_FACTOR

        return float(scaled_score)
