import chess
import random

class ChessAI:
    def __init__(self):
        self.board = chess.Board()

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
    
    def minimax(self, depth, alpha, beta, maximizing_player):
        try:
            if depth == 0 or self.board.is_game_over():
                return self.evaluate_board()

            if maximizing_player:
                max_eval = float("-inf")
                for move in self.board.legal_moves:
                    self.board.push(move)
                    eval = self.minimax(depth - 1, alpha, beta, False)
                    self.board.pop()
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha,eval)
                    if beta <= alpha:
                        break
                return max_eval
            else:
                min_eval = float("inf")
                for move in self.board.legal_moves:
                    self.board.push(move)
                    eval = self.minimax(depth - 1, alpha, beta, True)
                    self.board.pop()
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if (beta <= alpha):
                        break
                return min_eval
        except Exception as e:
            return 0
                
    def get_best_move(self, depth=3):
        best_move = None
        best_value = float("-inf") if self.board.turn == chess.WHITE else float("inf")
        alpha = float("-inf")
        beta = float("inf")

        def move_value(move):
            if self.board.is_capture(move):
                return 10
            elif move.promotion:
                return 9
            else:
                return 0

        legal_moves = sorted(
            list(self.board.legal_moves),
            key=move_value,
            reverse=True
        )

        for move in legal_moves:
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

        if best_move is None and list(self.board.legal_moves):
            best_move = random.choice(list(self.board.legal_moves))
        return best_move
