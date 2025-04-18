from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Literal
import chess
import uuid
import json
import asyncio
from chess_engine.chess_ai import ChessAI

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


class GameRequest(BaseModel):
    time_control: int
    white_side: Literal["human", "algo", "nn"]
    black_side: Literal["human", "algo", "nn"]


games = {}
computer_tasks = {}


class GameSession:
    def __init__(
        self, game_id, white_side="human", black_side="algo", time_control=300
    ):
        self.game_id = game_id
        self.connected_clients = set()
        self.stop_requested = False
        self.board = chess.Board()
        self.white_side = white_side
        self.black_side = black_side
        self.time_control = time_control
        self.engines = {}
        if white_side in ("algo", "nn"):
            self.engines[chess.WHITE] = ChessAI(
                evaluation_mode="nn" if white_side == "nn" else "algo"
            )
        if black_side in ("algo", "nn"):
            self.engines[chess.BLACK] = ChessAI(
                evaluation_mode="nn" if black_side == "nn" else "algo"
            )

    def reset_board(self):
        self.board.reset()
        self.stop_requested = False
        for ai in self.engines.values():
            ai.reset_board()


@app.get("/", response_class=HTMLResponse)
async def get_root():
    with open("templates/index.html", "r") as f:
        return f.read()


@app.post("/api/new-game")
async def create_game(request: GameRequest):
    game_id = str(uuid.uuid4())

    games[game_id] = GameSession(
        game_id,
        white_side=request.white_side,
        black_side=request.black_side,
        time_control=request.time_control,
    )

    if request.white_side != "human" and request.black_side != "human":
        computer_tasks[game_id] = asyncio.create_task(
            computer_vs_computer_game(game_id)
        )
    elif request.white_side in ("algo", "nn"):
        computer_tasks[game_id] = asyncio.create_task(
            make_computer_move(game_id, chess.WHITE)
        )

    return {"game_id": game_id}


@app.post("/api/games/{game_id}/reset")
async def reset_game(game_id: str):
    if game_id not in games:
        return {"error": "Game not found"}

    if game_id in computer_tasks:
        games[game_id].stop_requested = True
        computer_tasks[game_id].cancel()
        try:
            await computer_tasks[game_id]
        except asyncio.CancelledError:
            pass
        del computer_tasks[game_id]

    game_session = games[game_id]
    game_session.reset_board()

    await broadcast_board_state(game_id)

    if game_session.white_side != 'human' and game_session.black_side != 'human':
        computer_tasks[game_id] = asyncio.create_task(
            computer_vs_computer_game(game_id)
        )

    return {"success": True}


@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    await websocket.accept()

    if game_id not in games:
        await websocket.close(code=1000, reason="Game not found")
        return

    games[game_id].connected_clients.add(websocket)

    try:
        game_session = games[game_id]
        await broadcast_board_state(game_id)

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "make_move":
                move = chess.Move.from_uci(message["move"])
                if move in game_session.board.legal_moves:
                    game_session.board.push(move)
                    for ai in game_session.engines.values():
                        ai.board = game_session.board

                    await broadcast_board_state(game_id)

                    side_to_move = game_session.board.turn
                    if side_to_move in game_session.engines:
                        computer_tasks[game_id] = asyncio.create_task(
                            make_computer_move(game_id, side_to_move)
                        )

    except WebSocketDisconnect:
        games[game_id].connected_clients.remove(websocket)

        if len(games[game_id].connected_clients) == 0:
            if game_id in computer_tasks and not computer_tasks[game_id].done():
                games[game_id].stop_requested = True
                computer_tasks[game_id].cancel()
                try:
                    await computer_tasks[game_id]
                except asyncio.CancelledError:
                    pass
            if game_id in computer_tasks:
                del computer_tasks[game_id]
            if game_id in games:
                del games[game_id]


async def broadcast_board_state(game_id):
    if game_id in games:
        game_session = games[game_id]

        fen = game_session.board.fen()
        legal_moves = [move.uci() for move in game_session.board.legal_moves]
        is_over = game_session.board.is_game_over()

        message = {
            "type": "board_update",
            "fen": fen,
            "legal_moves": legal_moves,
            "is_game_over": is_over,
            "computer_thinking": game_id in computer_tasks
            and not computer_tasks[game_id].done(),
        }

        if is_over:
            if game_session.board.is_checkmate():
                message["result"] = "checkmate"
                message["winner"] = (
                    "Black" if game_session.board.turn == chess.WHITE else "White"
                )
            elif game_session.board.is_stalemate():
                message["result"] = "stalemate"
            elif game_session.board.is_insufficient_material():
                message["result"] = "insufficient material"
            elif game_session.board.is_fifty_moves():
                message["result"] = "fifty-move rule"
            elif game_session.board.is_repetition():
                message["result"] = "threefold repetition"

        for client in game_session.connected_clients:
            try:
                await client.send_json(message)
            except Exception as e:
                print(f"Error sending to client: {e}")


async def broadcast_thinking_status(game_id, is_thinking):
    if game_id in games:
        message = {"computer_thinking": is_thinking}

        for client in games[game_id].connected_clients:
            try:
                await client.send_json(message)
            except Exception as e:
                print(f"Error sending thinking status: {e}")


async def make_computer_move(game_id:str, color:bool):
    if game_id not in games:
        return

    game_session = games[game_id]
    ai = game_session.engines[color]
    ai.board = chess.Board(game_session.board.fen())

    try:
        await broadcast_thinking_status(game_id, True)

        loop = asyncio.get_event_loop()
        move = await loop.run_in_executor(
            None,
            lambda: ai.get_best_move_iterative_deepening(
                5, 10.0, stop_callback=lambda: game_session.stop_requested
            ),
        )

        if game_session.stop_requested or move is None:
            return

        game_session.board.push(move)

        for other in game_session.engines.values():
            other.board = game_session.board

        await broadcast_board_state(game_id)

    except Exception as e:
        print(f"Error in computer move: {e}")
    finally:
        await broadcast_thinking_status(game_id, False)

        if game_id in computer_tasks:
            del computer_tasks[game_id]


async def computer_vs_computer_game(game_id):
    session = games[game_id]
    while not session.board.is_game_over() and not session.stop_requested:
        side = session.board.turn
        if side not in session.engines:
            break
        await make_computer_move(game_id, side)
        await asyncio.sleep(0.5)

@app.on_event("shutdown")
async def shutdown_event():
    for game_id in list(games.keys()):
        games[game_id].stop_requested = True
        if game_id in computer_tasks:
            computer_tasks[game_id].cancel()
