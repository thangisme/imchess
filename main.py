import asyncio
import chess
import chess.engine
import json
import os
import shutil
import sys
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional, Literal, Callable

try:
    from chess_engine.chess_ai_policy import ChessAI

    POLICY_AI_AVAILABLE = True
    print("PolicyChessAI imported successfully.")
except ImportError as e:
    POLICY_AI_AVAILABLE = False
    print(f"Error: Could not import PolicyChessAI: {e}", file=sys.stderr)
    print("Neural Network mode will not be available.", file=sys.stderr)

try:
    from chess_engine.stockfish_ai import StockfishAI
    STOCKFISH_PATH = shutil.which("stockfish")
    STOCKFISH_AVAILABLE = STOCKFISH_PATH is not None
    if STOCKFISH_AVAILABLE:
        print(f"Stockfish found at: {STOCKFISH_PATH}")
    else:
        print("Stockfish not found in PATH. Stockfish mode will not be available.", file=sys.stderr)
except ImportError as e:
    STOCKFISH_AVAILABLE = False
    print(f"Error importing StockfishAI: {e}", file=sys.stderr)
    print("Stockfish mode will not be available.", file=sys.stderr)

POLICY_MODEL_PATH = "policy_models/policy_value_model.onnx"
OPENING_BOOK_PATH = "baron30.bin"
DEFAULT_POLICY_TEMPERATURE = 0.2
DEFAULT_SEARCH_TIME_LIMIT = 5.0

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


class GameRequest(BaseModel):
    time_control: int
    white_side: Literal["human", "nn_policy", "stockfish"]
    black_side: Literal["human", "nn_policy", "stockfish"]
    stockfish_depth: int = 15
    stockfish_elo: int = 1800  # 0-20, 20 being strongest
    nn_temperature: float = DEFAULT_POLICY_TEMPERATURE


games: Dict[str, Any] = {}
computer_tasks: Dict[str, asyncio.Task] = {}


class GameSession:
    def __init__(
        self, game_id, white_side="human", black_side="human", time_control=300,
        stockfish_depth=15, stockfish_elo=1800, nn_temperature=DEFAULT_POLICY_TEMPERATURE
    ):
        self.game_id = game_id
        self.connected_clients = set()
        self.stop_requested = False
        self.board = chess.Board()
        self.white_side = white_side
        self.black_side = black_side
        self.time_control = time_control
        self.stockfish_depth = stockfish_depth
        self.stockfish_elo = stockfish_elo
        self.nn_temperature = nn_temperature
        self.engines: Dict[bool, Any] = {}
        self.last_move_made_info: Optional[Dict[str, Any]] = None
        self.is_engine_thinking = False

        print(
            f"Initializing GameSession {game_id}: White={white_side}, Black={black_side}"
        )

        if white_side == "nn_policy" and POLICY_AI_AVAILABLE:
            self.engines[chess.WHITE] = self._create_nn_instance("White")
        elif white_side == "stockfish" and STOCKFISH_AVAILABLE:
            self.engines[chess.WHITE] = self._create_stockfish_instance(
                "White", stockfish_depth, stockfish_elo)
            print(f"Initiated Stockfish engine on White side, depth: {stockfish_depth}, elo: {stockfish_elo}")

        if black_side == "nn_policy" and POLICY_AI_AVAILABLE:
            self.engines[chess.BLACK] = self._create_nn_instance("Black")
        elif black_side == "stockfish" and STOCKFISH_AVAILABLE:
            self.engines[chess.BLACK] = self._create_stockfish_instance(
                "Black", stockfish_depth, stockfish_elo)
            print(f"Initiated Stockfish engine on Black side, depth: {stockfish_depth}, elo: {stockfish_elo}")

        print(f"Engines initialized for game {game_id}: {self.engines}")

    def _create_nn_instance(self, color_str: str):
        if not POLICY_AI_AVAILABLE:
            print(
                f"Error: Neural network not available for {color_str}.", file=sys.stderr
            )
            return None
        if not os.path.exists(POLICY_MODEL_PATH):
            print(
                f"Error: Policy model file not found: {POLICY_MODEL_PATH}",
                file=sys.stderr,
            )
            return None
        return ChessAI(
            policy_model_path=POLICY_MODEL_PATH,
            opening_book_path=OPENING_BOOK_PATH,
            temperature=self.nn_temperature,
        )
        
    def _create_stockfish_instance(self, color_str: str, depth: int = 15, elo_rating: int = 1800):
        if not STOCKFISH_AVAILABLE:
            print(f"Error: Stockfish not available for {color_str}.", file=sys.stderr)
            return None
        return StockfishAI(
            stockfish_path=STOCKFISH_PATH,
            depth=depth,
            time_limit=DEFAULT_SEARCH_TIME_LIMIT,
            elo_rating=elo_rating
        )

    def push_move_and_record_info(self, move: chess.Move) -> bool:
        if move in self.board.legal_moves:
            self.last_move_made_info = {
                "san": self.board.san(move),
                "color": "w" if self.board.turn == chess.WHITE else "b",
                "from_uci": chess.square_name(move.from_square),
                "to_uci": chess.square_name(move.to_square),
                "promotion": chess.piece_symbol(move.promotion).lower()
                if move.promotion
                else None,
            }
            self.board.push(move)
            return True
        return False

    def reset_board(self):
        self.board.reset()
        self.stop_requested = False
        self.last_move_made_info = None
        self.is_engine_thinking = False
        for ai_instance in self.engines.values():
            if hasattr(ai_instance, "reset_board"):
                ai_instance.reset_board()
        print(f"Game {self.game_id} board reset.")

    def cleanup_engines(self):
        """Close any engine resources properly"""
        for color, engine in list(self.engines.items()):
            if hasattr(engine, 'close'):
                try:
                    engine.close()
                    print(f"Engine for {'White' if color else 'Black'} in game {self.game_id} closed.")
                except Exception as e:
                    print(f"Error closing engine: {e}", file=sys.stderr)


@app.get("/", response_class=HTMLResponse)
async def get_root_html():
    try:
        with open("templates/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: index.html not found</h1>", status_code=500
        )


@app.post("/api/new-game")
async def create_new_game_api(request: GameRequest):
    game_id = str(uuid.uuid4())
    print(
        f"Creating new game with ID: {game_id}, White: {request.white_side}, Black: {request.black_side}"
    )

    if game_id in computer_tasks:
        await _cancel_computer_task(game_id)

    games[game_id] = GameSession(
        game_id,
        white_side=request.white_side,
        black_side=request.black_side,
        time_control=request.time_control,
        stockfish_depth=request.stockfish_depth,
        stockfish_elo=request.stockfish_elo,
        nn_temperature=request.nn_temperature,
    )

    if games[game_id].white_side != "human" and games[game_id].black_side != "human":
        print(f"Starting Computer vs Computer game for {game_id}")
        computer_tasks[game_id] = asyncio.create_task(
            computer_vs_computer_game_loop(game_id)
        )
    elif (
        games[game_id].white_side != "human"
        and games[game_id].board.turn == chess.WHITE
    ):
        print(f"Starting initial Computer (White) move for {game_id}")
        computer_tasks[game_id] = asyncio.create_task(
            make_single_computer_move(game_id, chess.WHITE)
        )

    return {"game_id": game_id}


@app.post("/api/games/{game_id}/reset")
async def reset_game_api(game_id: str):
    print(f"Resetting game: {game_id}")
    if game_id not in games:
        print(f"Error: Game not found for reset: {game_id}")
        return {"error": "Game not found"}

    await _cancel_computer_task(game_id)

    game_session = games[game_id]
    game_session.reset_board()

    await broadcast_board_state_to_clients(game_id)

    if game_session.white_side != "human" and game_session.black_side != "human":
        print(f"Restarting Computer vs Computer game for {game_id} after reset.")
        computer_tasks[game_id] = asyncio.create_task(
            computer_vs_computer_game_loop(game_id)
        )
    elif game_session.white_side != "human" and game_session.board.turn == chess.WHITE:
        print(f"Restarting initial Computer (White) move for {game_id} after reset.")
        computer_tasks[game_id] = asyncio.create_task(
            make_single_computer_move(game_id, chess.WHITE)
        )
    return {"success": True}


async def _cancel_computer_task(game_id: str):
    if game_id in computer_tasks:
        task = computer_tasks[game_id]
        if not task.done():
            if game_id in games:
                games[game_id].stop_requested = True
                games[game_id].is_engine_thinking = False
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                print(f"Computer task {game_id} cancelled successfully.")
            except Exception as e:
                print(f"Exception while cancelling task {game_id}: {e}")
        del computer_tasks[game_id]


@app.websocket("/ws/{game_id}")
async def websocket_game_endpoint(websocket: WebSocket, game_id: str):
    await websocket.accept()
    print(f"WebSocket connection accepted for game: {game_id}")

    if game_id not in games:
        print(f"WebSocket: Game not found for ID {game_id}. Closing connection.")
        await websocket.close(code=1000, reason="Game not found")
        return

    game_session = games[game_id]
    game_session.connected_clients.add(websocket)
    print(
        f"Client added to game {game_id}. Total clients: {len(game_session.connected_clients)}"
    )

    try:
        await broadcast_board_state_to_clients(game_id)

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            print(f"Received message from client for game {game_id}: {message}")

            if message["type"] == "make_move":
                if game_id not in games:
                    break

                player_color = (
                    chess.WHITE
                    if game_session.board.turn == chess.WHITE
                    else chess.BLACK
                )
                is_human_turn = (
                    player_color == chess.WHITE and game_session.white_side == "human"
                ) or (
                    player_color == chess.BLACK and game_session.black_side == "human"
                )

                if game_session.board.is_game_over():
                    print("Human move attempted on game over.")
                    continue
                if game_id in computer_tasks and not computer_tasks[game_id].done():
                    print("Human move attempted while computer is thinking.")
                    continue
                if not is_human_turn:
                    print(f"Human move attempted on computer's turn.")
                    continue

                move_uci = message["move"]
                try:
                    move = game_session.board.parse_uci(move_uci)
                except ValueError:
                    print(f"Invalid UCI move string: {move_uci}")
                    continue

                if game_session.push_move_and_record_info(move):
                    print(
                        f"Human move {move.uci()} (SAN: {game_session.last_move_made_info['san']}) made in game {game_id}."
                    )

                    for ai_instance in game_session.engines.values():
                        if hasattr(ai_instance, "board"):
                            ai_instance.board = chess.Board(game_session.board.fen())

                    await broadcast_board_state_to_clients(game_id)

                    side_to_move_now = game_session.board.turn
                    if (
                        not game_session.board.is_game_over()
                        and side_to_move_now in game_session.engines
                    ):
                        if game_id in computer_tasks:
                            await _cancel_computer_task(game_id)
                        print(
                            f"Triggering computer move for color {'White' if side_to_move_now == chess.WHITE else 'Black'} in game {game_id}."
                        )
                        computer_tasks[game_id] = asyncio.create_task(
                            make_single_computer_move(game_id, side_to_move_now)
                        )
                else:
                    print(
                        f"Illegal human move attempted: {move.uci()} in game {game_id}"
                    )

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for game: {game_id}")
    except Exception as e:
        print(f"Error in WebSocket handler for game {game_id}: {e}")
    finally:
        if game_id in games and websocket in games[game_id].connected_clients:
            games[game_id].connected_clients.remove(websocket)
            print(
                f"Client removed from game {game_id}. Remaining clients: {len(games[game_id].connected_clients)}"
            )
        if game_id in games and not games[game_id].connected_clients:
            print(f"No clients connected for game {game_id}. Cleaning up.")
            await _cancel_computer_task(game_id)
            if game_id in games:
                games[game_id].cleanup_engines()
                del games[game_id]


async def broadcast_board_state_to_clients(game_id):
    if game_id not in games:
        return

    game_session = games[game_id]
    fen = game_session.board.fen()
    legal_moves = [move.uci() for move in game_session.board.legal_moves]
    is_over = game_session.board.is_game_over()

    message = {
        "type": "board_update",
        "fen": fen,
        "legal_moves": legal_moves,
        "is_game_over": is_over,
        "computer_thinking": game_session.is_engine_thinking,
        "white_player_type": game_session.white_side,
        "black_player_type": game_session.black_side,
        "last_move_info": game_session.last_move_made_info,
    }

    if is_over:
        if game_session.board.is_checkmate():
            winner = "Black" if game_session.board.turn == chess.WHITE else "White"
            message["result"] = "checkmate"
            message["winner"] = winner
        elif game_session.board.is_stalemate():
            message["result"] = "stalemate"
        elif game_session.board.is_insufficient_material():
            message["result"] = "insufficient material"
        elif game_session.board.is_fifty_moves():
            message["result"] = "fifty-move rule"
        elif game_session.board.is_repetition(3):
            message["result"] = "threefold repetition"
        else:
            message["result"] = "draw"

    disconnected_clients = set()
    for client in list(game_session.connected_clients):
        try:
            await client.send_json(message)
        except Exception:
            disconnected_clients.add(client)
    for client in disconnected_clients:
        if client in game_session.connected_clients:
            game_session.connected_clients.remove(client)


async def broadcast_thinking_status_to_clients(game_id, is_thinking):
    if game_id not in games:
        return

    games[game_id].is_engine_thinking = is_thinking

    message = {"type": "thinking_status", "computer_thinking": is_thinking}
    disconnected_clients = set()

    for client in list(games[game_id].connected_clients):
        try:
            await client.send_json(message)
        except Exception:
            disconnected_clients.add(client)

    for client in disconnected_clients:
        if client in games[game_id].connected_clients:
            games[game_id].connected_clients.remove(client)


async def make_single_computer_move(game_id: str, color_to_move: bool):
    if game_id not in games or games[game_id].stop_requested:
        return

    game_session = games[game_id]
    if color_to_move not in game_session.engines:
        print(
            f"No AI engine found for color {'White' if color_to_move else 'Black'} in game {game_id}."
        )
        return

    try:
        ai_instance = game_session.engines[color_to_move]
        ai_instance.board = chess.Board(game_session.board.fen())

        print(
            f"Computer (color: {'White' if color_to_move else 'Black'}) is thinking for game {game_id}..."
        )
        await broadcast_thinking_status_to_clients(game_id, True)

        move = None
        try:
            loop = asyncio.get_event_loop()
            move = await loop.run_in_executor(
                None,
                lambda: ai_instance.get_best_move_iterative_deepening(
                    time_limit=DEFAULT_SEARCH_TIME_LIMIT,
                    external_stop_callback=lambda: game_session.stop_requested,
                ),
            )
        except Exception as e:
            print(
                f"Error during AI move calculation for game {game_id}: {e}",
                file=sys.stderr,
            )
            move = None
        finally:
            await broadcast_thinking_status_to_clients(game_id, False)

        if game_session.stop_requested:
            print(f"Computer move stopped during calculation for game {game_id}.")
            return

        if move:
            if game_session.push_move_and_record_info(move):
                print(
                    f"Computer move {move.uci()} (SAN: {game_session.last_move_made_info['san']}) made in game {game_id}."
                )

                for other_color, other_ai in game_session.engines.items():
                    if other_color != color_to_move and hasattr(other_ai, "board"):
                        other_ai.board = chess.Board(game_session.board.fen())
            else:
                print(
                    f"ERROR: AI returned illegal move {move.uci()} for FEN {game_session.board.fen()}",
                    file=sys.stderr,
                )
        else:
            print(
                f"Warning: AI did not return a move for game {game_id}.",
                file=sys.stderr,
            )

        await broadcast_board_state_to_clients(game_id)

    except Exception as e:
        print(
            f"Unexpected error in make_single_computer_move for game {game_id}: {e}",
            file=sys.stderr,
        )
        if game_id in games:
            await broadcast_thinking_status_to_clients(game_id, False)

    finally:
        is_cvc = False
        if game_id in games:
            is_cvc = (
                games[game_id].white_side != "human"
                and games[game_id].black_side != "human"
            )
            is_cvc_continuing = (
                is_cvc
                and not games[game_id].board.is_game_over()
                and not games[game_id].stop_requested
            )

            if not is_cvc_continuing and game_id in computer_tasks:
                task = computer_tasks.pop(game_id, None)
                if task:
                    print(f"Computer task removed for game {game_id}.")


async def computer_vs_computer_game_loop(game_id: str):
    print(f"Starting Computer vs Computer loop for game {game_id}")
    if game_id not in games:
        return

    game_session = games[game_id]
    while not game_session.board.is_game_over() and not game_session.stop_requested:
        current_player_color = game_session.board.turn
        if current_player_color not in game_session.engines:
            print(
                f"Error in CvC: No engine for {'White' if current_player_color else 'Black'} in game {game_id}"
            )
            break

        await make_single_computer_move(game_id, current_player_color)

        if game_session.stop_requested:
            break
        await asyncio.sleep(0.5)

    print(f"Computer vs Computer loop finished for game {game_id}.")
    if game_id in computer_tasks:
        del computer_tasks[game_id]
    await broadcast_board_state_to_clients(game_id)


@app.on_event("shutdown")
async def application_shutdown_event():
    print("Application shutting down. Cleaning up active games and tasks...")
    for game_id in list(games.keys()):
        await _cancel_computer_task(game_id)
        if game_id in games:
            games[game_id].cleanup_engines()
            del games[game_id]
    print("Cleanup complete.")
