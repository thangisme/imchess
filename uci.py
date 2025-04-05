#!/usr/bin/env python3

import chess
from chess_ai import ChessAI
import sys
import time

class UCIEngine:
    def __init__(self):
        self.ai = ChessAI()
        self.name = "ImChess"
        self.author = "..."

    def uci_loop(self):
        while True:
            if sys.stdin.isatty():
                # If running in a terminal then prompt for input
                cmd = input("uci> ")
            else:
                cmd = input()

            if not cmd:
                continue

            print(f"Received input: {cmd}", file=sys.stderr)

            if cmd == "uci":
                self.uci_command()
            elif cmd == "isready":
                self.isready_command()
            elif cmd == "ucinewgame":
                self.ucinewgame_command()
            elif cmd.startswith("position"):
                self.position_command(cmd)
            elif cmd.startswith("go"):
                self.go_command(cmd)
            elif cmd == "quit":
                break
            else:
                print(f"Unknown command: {cmd}")

    def uci_command(self):
        print(f"id name {self.name}")
        print(f"id author {self.author}")
        print("uciok")

    def isready_command(self):
        print("readyok")

    def ucinewgame_command(self):
        self.ai.reset_board()

    def position_command(self, cmd):
        parts = cmd.split()

        try:
            moves_index = parts.index("moves")
            moves = parts[moves_index + 1:]
            parts = parts[:moves_index]
        except ValueError:
            moves = []

        if len(parts) >= 2:
            if parts[1] == "startpos":
                self.ai.reset_board()
            elif parts[1] == "fen" and len(parts) >= 8:
                fen = " ".join(parts[2:8])
                self.ai.set_board_from_fen(fen)

        for move_uci in moves:
            try:
                move = chess.Move.from_uci(move_uci)
                self.ai.make_move(move)
            except ValueError:
                print(f"info string Invalid move: {move_uci}")

    def go_command(self, cmd):
        parts = cmd.split()

        depth = 10
        time_limit = 10.0

        print(f"{depth} {time_limit}", file=sys.stderr)
        
        for i in range(len(parts)):
            if parts[i] == "depth" and i + 1 < len(parts):
                try:
                    depth = int(parts[i + 1])
                except ValueError:
                    pass
            elif parts[i] == "movetime" and i + 1 < len(parts):
                try:
                    time_limit = int(parts[i + 1]) / 1000.0
                except ValueError:
                    pass
            elif parts[i] == "wtime" and i + 1 < len(parts) and self.ai.board.turn == chess.WHITE:
                try:
                    time_limit = int(parts[i + 1]) / 1000.0 / 20.0
                except ValueError:
                    pass
            elif parts[i] == "btime" and i + 1 < len(parts) and self.ai.board.turn == chess.BLACK:
                try:
                    time_limit = int(parts[i + 1]) / 1000.0 / 20.0
                except ValueError:
                    pass


        best_move = self.ai.get_best_move_iterative_deepening(depth,time_limit)
        if best_move:
            print(f"bestmove {best_move}")
        else:
            print("bestmove 0000") # No legal moves

if __name__ == "__main__":
    engine = UCIEngine()
    engine.uci_loop()

