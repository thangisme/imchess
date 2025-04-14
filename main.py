from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chess
import uuid
import json
import asyncio
import os
from chess_engine.chess_ai import ChessAI

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Models
class GameRequest(BaseModel):
    game_type: str

# Track active games and tasks
games = {}
computer_tasks = {}

# Game session class for better organization
class GameSession:
    def __init__(self, game_id, game_type="human_vs_computer"):
        self.game_id = game_id
        self.ai = ChessAI()
        self.type = game_type
        self.connected_clients = set()
        self.stop_requested = False
    
    def reset_board(self):
        self.ai.reset_board()
        self.stop_requested = False

# Routes and endpoints
@app.get("/", response_class=HTMLResponse)
async def get_root():
    with open("templates/index.html", "r") as f:
        return f.read()

@app.post("/api/new-game")
async def create_game(request: GameRequest):
    game_id = str(uuid.uuid4())
    
    # Create a new game session
    games[game_id] = GameSession(game_id, request.game_type)
    
    # Start computer vs computer game if needed
    if request.game_type == "computer_vs_computer":
        computer_tasks[game_id] = asyncio.create_task(computer_vs_computer_game(game_id))
    
    return {"game_id": game_id}

@app.post("/api/games/{game_id}/reset")
async def reset_game(game_id: str):
    if game_id not in games:
        return {"error": "Game not found"}
    
    # Cancel any ongoing computer task
    if game_id in computer_tasks:
        games[game_id].stop_requested = True
        computer_tasks[game_id].cancel()
        try:
            await computer_tasks[game_id]
        except asyncio.CancelledError:
            pass
        del computer_tasks[game_id]
    
    # Reset the game
    game_type = games[game_id].type
    games[game_id].reset_board()
    
    # Broadcast the reset state
    await broadcast_board_state(game_id)
    
    # Start a new computer vs computer game if needed
    if game_type == "computer_vs_computer":
        computer_tasks[game_id] = asyncio.create_task(computer_vs_computer_game(game_id))
    
    return {"success": True}

@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    await websocket.accept()
    
    if game_id not in games:
        await websocket.close(code=1000, reason="Game not found")
        return
    
    # Add this client to the game
    games[game_id].connected_clients.add(websocket)
    
    try:
        # Send initial state
        game_session = games[game_id]
        await websocket.send_json({
            "type": "board_update",
            "fen": game_session.ai.board.fen(),
            "legal_moves": [move.uci() for move in game_session.ai.board.legal_moves],
            "is_game_over": game_session.ai.board.is_game_over(),
            "computer_thinking": game_id in computer_tasks and not computer_tasks[game_id].done()
        })
        
        # Process incoming messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "make_move":
                move = chess.Move.from_uci(message["move"])
                if move in game_session.ai.board.legal_moves:
                    # Make the player's move
                    game_session.ai.make_move(move)
                    
                    # Broadcast updated state
                    await broadcast_board_state(game_id)
                    
                    # If it's human vs computer and not game over, make computer move
                    if game_session.type == "human_vs_computer" and not game_session.ai.board.is_game_over():
                        # Create a task for the computer's move
                        computer_tasks[game_id] = asyncio.create_task(make_computer_move(game_id))
    
    except WebSocketDisconnect:
        # Remove this client when disconnected
        games[game_id].connected_clients.remove(websocket)
        
        # If no clients are connected, consider cleaning up the game
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

# Helper functions for game management
async def broadcast_board_state(game_id):
    if game_id in games:
        game_session = games[game_id]
        
        message = {
            "type": "board_update",
            "fen": game_session.ai.board.fen(),
            "legal_moves": [move.uci() for move in game_session.ai.board.legal_moves],
            "is_game_over": game_session.ai.board.is_game_over(),
            "computer_thinking": game_id in computer_tasks and not computer_tasks[game_id].done()
        }
        
        if game_session.ai.board.is_game_over():
            if game_session.ai.board.is_checkmate():
                message["result"] = "checkmate"
                message["winner"] = "Black" if game_session.ai.board.turn == chess.WHITE else "White"
            elif game_session.ai.board.is_stalemate():
                message["result"] = "stalemate"
            elif game_session.ai.board.is_insufficient_material():
                message["result"] = "insufficient material"
            elif game_session.ai.board.is_fifty_moves():
                message["result"] = "fifty-move rule"
            elif game_session.ai.board.is_repetition():
                message["result"] = "threefold repetition"
        
        # Send to all connected clients
        for client in game_session.connected_clients:
            try:
                await client.send_json(message)
            except Exception as e:
                print(f"Error sending to client: {e}")

async def broadcast_thinking_status(game_id, is_thinking):
    if game_id in games:
        message = {
            "computer_thinking": is_thinking
        }
        
        for client in games[game_id].connected_clients:
            try:
                await client.send_json(message)
            except Exception as e:
                print(f"Error sending thinking status: {e}")

async def make_computer_move(game_id):
    if game_id not in games:
        return
    
    game_session = games[game_id]
    
    try:
        await broadcast_thinking_status(game_id, True)
        
        loop = asyncio.get_event_loop()
        move = await loop.run_in_executor(
            None, 
            lambda: game_session.ai.get_best_move_iterative_deepening(
                3, 2.0, 
                stop_callback=lambda: game_session.stop_requested
            )
        )
        
        if game_session.stop_requested:
            await broadcast_thinking_status(game_id, False)
            return
                
        if move:
            game_session.ai.make_move(move)
            
            await broadcast_board_state(game_id)
            
    except Exception as e:
        print(f"Error in computer move: {e}")
    finally:
        await broadcast_thinking_status(game_id, False)

        if game_id in computer_tasks:
            del computer_tasks[game_id]

async def computer_vs_computer_game(game_id):
    if game_id not in games:
        return
    
    game_session = games[game_id]
    
    try:
        while game_id in games and not game_session.ai.board.is_game_over() and not game_session.stop_requested:
            # Signal that computer is thinking
            await broadcast_thinking_status(game_id, True)
            
            # Run the chess engine in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            move = await loop.run_in_executor(
                None,
                lambda: game_session.ai.get_best_move_iterative_deepening(
                    3, 2.0,  # depth=3, time_limit=2.0
                    stop_callback=lambda: game_session.stop_requested
                )
            )
            
            # Check if a stop was requested during computation
            if game_session.stop_requested:
                break
                
            # Signal that computer is done thinking
            await broadcast_thinking_status(game_id, False)
            
            if move:
                # Make the computer's move
                game_session.ai.make_move(move)
                
                # Broadcast the updated board state
                await broadcast_board_state(game_id)
                
                # Add a delay between moves to make the game watchable
                await asyncio.sleep(1.5)
            else:
                break
                
    except asyncio.CancelledError:
        # Handle cancellation cleanly
        await broadcast_thinking_status(game_id, False)
    except Exception as e:
        print(f"Error in computer vs computer game: {e}")
    finally:
        # Make sure to clean up
        if game_id in computer_tasks:
            del computer_tasks[game_id]

@app.on_event("shutdown")
async def shutdown_event():
    # Clean up all games and tasks on server shutdown
    for game_id in list(games.keys()):
        games[game_id].stop_requested = True
        if game_id in computer_tasks:
            computer_tasks[game_id].cancel()
