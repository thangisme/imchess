import chess
import chess.engine
import time
import os
import sys
from typing import Optional, Tuple, Dict, Any, Callable

class StockfishAI:
    def __init__(self, 
                 stockfish_path: str = None, 
                 depth: int = 15,
                 time_limit: float = 3.0,
                 elo_rating: int = 1800): 
        self.board = chess.Board()
        self.depth = depth
        self.time_limit = time_limit
        self.elo_rating = max(1320, min(2850, elo_rating))  # Clamp to valid range
        self.stockfish_path = stockfish_path
        self.engine = None
        self.current_best_move = None
        self.open_engine()
        
    def open_engine(self) -> None:
        try:
            path = self.stockfish_path
            if path is None:
                # Try to automatically locate Stockfish
                for candidate in ["stockfish", "stockfish.exe"]:
                    if chess.engine.which(candidate):
                        path = chess.engine.which(candidate)
                        break
            
            if not path:
                print("Error: Stockfish executable not found. Please specify the path.", file=sys.stderr)
                return
                
            self.engine = chess.engine.SimpleEngine.popen_uci(path)
            
            self.engine.configure({"UCI_LimitStrength": True})
            self.engine.configure({"UCI_Elo": self.elo_rating})
            
            print(f"Stockfish engine initialized with Elo rating {self.elo_rating}")
        except Exception as e:
            print(f"Error initializing Stockfish engine: {e}", file=sys.stderr)
            self.engine = None
            
    def reset_board(self) -> None:
        """Reset the chess board."""
        self.board.reset()
        
    def set_board_from_fen(self, fen: str) -> None:
        """Set the board position from FEN string."""
        try:
            self.board.set_fen(fen)
        except ValueError:
            print(f"Invalid FEN string: {fen}", file=sys.stderr)
            self.board.reset()
    
    def make_move(self, move_uci_or_object) -> bool:
        """Update internal board with the provided move."""
        try:
            if isinstance(move_uci_or_object, str):
                move = self.board.parse_uci(move_uci_or_object)
            else:
                move = move_uci_or_object
                
            if move in self.board.legal_moves:
                self.board.push(move)
                return True
            return False
        except Exception:
            return False
        
    def get_best_move_iterative_deepening(
        self, max_depth: int = None, time_limit: float = None, 
        external_stop_callback: Callable[[], bool] = None
    ) -> Optional[chess.Move]:
        if not self.engine:
            self.open_engine()
        if not self.engine:
            print("Error: Stockfish engine not available", file=sys.stderr)
            return None
            
        depth = max_depth or self.depth
        limit_time = time_limit or self.time_limit
        
        if self.board.is_game_over():
            return None
            
        try:
            limit = chess.engine.Limit(depth=depth, time=limit_time)
            
            start_time = time.time()
            
            result = self.engine.play(
                self.board, 
                limit, 
                info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
                ponder=False
            )
            
            self.current_best_move = result.move
            
            elapsed = time.time() - start_time
            
            score_info = ""
            if hasattr(result, "info") and "score" in result.info:
                score = result.info["score"].relative.score(mate_score=10000)
                if score is not None:
                    score_info = f"score cp {score}"
                elif result.info["score"].relative.mate():
                    mate_in = result.info["score"].relative.mate()
                    score_info = f"mate {mate_in}"
            
            print(f"info depth {depth} {score_info} time {int(elapsed * 1000)} pv {result.move.uci()}")
            
            return result.move
            
        except chess.engine.EngineTerminatedError:
            print("Engine process terminated unexpectedly", file=sys.stderr)
            self.engine = None
            return None
            
        except chess.engine.EngineError as e:
            print(f"Engine error: {e}", file=sys.stderr)
            return None
            
    def get_current_best_move(self) -> Optional[chess.Move]:
        """Get the most recently calculated best move."""
        return self.current_best_move
        
    def close(self) -> None:
        """Close the engine properly."""
        if self.engine:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None
            
    def __del__(self):
        """Ensure the engine is closed when the object is destroyed."""
        self.close()
