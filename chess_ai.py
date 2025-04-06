import chess
import chess.polyglot
import random
import time
import sys

class ChessAI:
    def __init__(self):
        self.board = chess.Board()
        self.transposition_table = {}
        self.nodes_searched = 0
        self.opening_book = "baron30.bin"

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
                    pos_value = pawn_table[square]
                elif piece.piece_type == chess.KNIGHT:
                    pos_value = knight_table[square]
                elif piece.piece_type == chess.ROOK:
                    pos_value = rook_table[square]
                elif piece.piece_type == chess.BISHOP:
                    pos_value = bishop_table[square]
                elif piece.piece_type == chess.QUEEN:
                    pos_value = queen_table[square]
                elif piece.piece_type == chess.KING:
                    pos_value = king_middle_game_table[square]    

                if piece.color == chess.WHITE:
                    material_score += value
                    position_score += pos_value
                else:
                    material_score -= value
                    position_score -= pos_value

        try:
            original_turn = self.board.turn

            self.board.turn = chess.WHITE
            white_moves = len(list(self.board.legal_moves))

            self.board.turn = chess.BLACK
            black_moves = len(list(self.board.legal_moves))

            self.board.turn = original_turn

            mobility_score = (white_moves - black_moves) * 5
        except Exception:
            mobility_score = 0

        score = material_score + position_score * 0.3 + mobility_score

        return score

    def get_board_hash(self):
        return chess.polyglot.zobrist_hash(self.board)
    
    def minimax(self, depth, alpha, beta, maximizing_player):
        self.nodes_searched += 1

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

        if maximizing_player:
            max_eval = float("-inf")
            for move in self.order_moves(self.board.legal_moves):
                self.board.push(move)
                eval = self.minimax(depth - 1, alpha, beta, False)
                self.board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha,eval)
                if beta <= alpha:
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
            for move in self.order_moves(self.board.legal_moves):
                self.board.push(move)
                eval = self.minimax(depth - 1, alpha, beta, True)
                self.board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if (beta <= alpha):
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

                if score >= alpha:
                    return alpha
                alpha = max(beta, score)
        
        return alpha if maximizing_player else beta
                
    def order_moves(self, moves):
        scored_moves = []
        for move in moves:
            score = 0
            if self.board.is_capture(move):
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
            
    def get_best_move_iterative_deepening(self, max_depth=4, time_limit=5.0):
        book_move = self.get_book_move()
        if book_move:
            print(f"info string book move {book_move.uci()}")
            self.board_score = 0
            return book_move
        
        start_time = time.time()
        best_move = None
        self.nodes_searched = 0

        for current_depth in range(1, max_depth + 1):
            elapsed = time.time() - start_time
            if elapsed > time_limit * 0.8:
                break

            self.transposition_table = {}
            move = self.get_best_move(current_depth)
            elapsed = time.time() - start_time

            if move:
                best_move = move

            nps = int(self.nodes_searched /elapsed) if elapsed > 0 else 0

            print(f"info depth {current_depth} score cp {self.board_score} nodes {self.nodes_searched} nps {nps} time {int(elapsed * 1000)}")

            if elapsed >= time_limit:
                break

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
                print(entries, file=sys.stderr)

                print(f"book {self.opening_book}")

                if entries:
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
