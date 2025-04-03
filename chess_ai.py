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

        score = 0
        for piece_type in piece_values:
            score += len(self.board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
            score -= len(self.board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]

        original_turn = self.board.turn

        self.board.turn = chess.WHITE
        white_moves = len(list(self.board.legal_moves))

        self.board.turn = chess.BLACK
        black_moves = len(list(self.board.legal_moves))

        self.board.turn = original_turn

        score += (white_moves - black_moves) * 10

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

        for move in self.board.legal_moves:
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
