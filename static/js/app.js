let board = null;
let game = new Chess();
let gameId = null;
let socket = null;
let gameStarted = false;
let gameMode = 'human_vs_computer';
let timeControl = 300; 
let moveCount = 0;
let gameOver = false;
let computerThinking = false;

const setupCard = document.getElementById('setup-card');
const gameOptionsCard = document.getElementById('game-options-card');
const gameStatusCard = document.getElementById('game-status-card');
const timerCard = document.getElementById('timer-card');
const startGameBtn = document.getElementById('start-game-btn');
const newGameBtn = document.getElementById('new-game-btn');
const playAgainBtn = document.getElementById('play-again-btn');
const backToSetupBtn = document.getElementById('back-to-setup-btn');
const timeControlSelect = document.getElementById('time-control');
const gameModeRadios = document.querySelectorAll('input[name="game-mode"]');
const whiteTimeDisplay = document.getElementById('white-time');
const blackTimeDisplay = document.getElementById('black-time');
const whitePlayerLabel = document.getElementById('white-player-label');
const blackPlayerLabel = document.getElementById('black-player-label');
const gameModeDisplay = document.getElementById('game-mode-display');
const timeControlDisplay = document.getElementById('time-control-display');
const currentTurnDisplay = document.getElementById('current-turn-display');
const moveCountDisplay = document.getElementById('move-count-display');
const gameOverNotification = document.getElementById('game-over-notification');
const gameOverResult = document.getElementById('game-over-result');
const computerThinkingIndicator = document.getElementById('computer-thinking');

function initializeBoard() {
    const config = {
        draggable: true,
        position: 'start',
        onDragStart: onDragStart,
        onDrop: onDrop,
        onSnapEnd: onSnapEnd,
        pieceTheme: '/static/img/chesspieces/wikipedia/{piece}.png',
    };
    board = Chessboard('board', config);
    updateStatus();
    $(window).resize(() => board.resize());
}

function onDragStart(source, piece) {
    if (game.game_over()) return false;
    
    if (gameMode === 'computer_vs_computer') return false;
    
    if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
        (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
        return false;
    }
    
    if (computerThinking) {
        return false;
    }
}

function onDrop(source, target) {
    const move = game.move({
        from: source,
        to: target,
        promotion: 'q' 
    });

    if (move === null) return 'snapback';
    
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            type: 'make_move',
            move: move.from + move.to + (move.promotion || '')
        }));
    }
    
    updateStatus();
}

function onSnapEnd() {
    board.position(game.fen());
}

function updateStatus() {
    let status = '';
    
    if (game.in_checkmate()) {
        status = `Game over: ${game.turn() === 'w' ? 'Black' : 'White'} wins by checkmate`;
        endGame(game.turn() === 'w' ? 'Black' : 'White');
    } else if (game.in_draw()) {
        status = 'Game over: Draw';
        endGame('Draw');
    } else {
        status = `${game.turn() === 'w' ? 'White' : 'Black'} to move`;
        if (game.in_check()) {
            status += `, ${game.turn() === 'w' ? 'White' : 'Black'} is in check`;
        }
    }
    
    currentTurnDisplay.textContent = game.turn() === 'w' ? 'White' : 'Black';
    
    moveCount = game.history().length;
    moveCountDisplay.textContent = Math.floor(moveCount / 2);
    
    if (game.turn() === 'w') {
        document.getElementById('white-timer').classList.add('timer-active');
        document.getElementById('black-timer').classList.remove('timer-active');
    } else {
        document.getElementById('white-timer').classList.remove('timer-active');
        document.getElementById('black-timer').classList.add('timer-active');
    }
}

function connectWebSocket(id) {
    socket = new WebSocket(`ws://${window.location.host}/ws/${id}`);
    
    socket.onopen = function(event) {
        console.log('WebSocket connection established');
    };
    
    socket.onmessage = function(event) {
        const message = JSON.parse(event.data);
        console.log(message);
        
        if (message.type === 'board_update') {
            game.load(message.fen);
            board.position(message.fen);
            updateStatus();
            
            if (message.hasOwnProperty('computer_thinking')) {
                showComputerThinking(message.computer_thinking);
            }
            
            if (message.is_game_over) {
                let resultMessage = 'Game over: ';
                if (message.result === 'checkmate') {
                    resultMessage += `${message.winner} wins by checkmate`;
                    endGame(message.winner);
                } else if (message.result === 'stalemate') {
                    resultMessage += 'Draw by stalemate';
                    endGame('Draw');
                } else if (message.result === 'insufficient material') {
                    resultMessage += 'Draw by insufficient material';
                    endGame('Draw');
                } else if (message.result === 'fifty-move rule') {
                    resultMessage += 'Draw by fifty-move rule';
                    endGame('Draw');
                } else if (message.result === 'threefold repetition') {
                    resultMessage += 'Draw by threefold repetition';
                    endGame('Draw');
                } else {
                    resultMessage += 'Draw';
                    endGame('Draw');
                }
            }
        } else if (message.hasOwnProperty('computer_thinking')) {
          showComputerThinking(message.computer_thinking);
        }
    };
    
    socket.onclose = function(event) {
        console.log('WebSocket connection closed');
    };
    
    socket.onerror = function(error) {
        console.error('WebSocket error:', error);
    };
}

async function startGame() {
    try {
        const response = await fetch('/api/new-game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ game_type: gameMode })
        });
        
        if (!response.ok) {
            throw new Error('Failed to create game');
        }
        
        const data = await response.json();
        gameId = data.game_id;
        
        setupCard.classList.add('hidden');
        gameOptionsCard.classList.remove('hidden');
        gameStatusCard.classList.remove('hidden');
        timerCard.classList.remove('hidden');
        gameOverNotification.classList.add('hidden');
        computerThinkingIndicator.classList.add('hidden');
        
        gameStarted = true;
        gameOver = false;
        game = new Chess();
        
        initializeBoard();
        
        connectWebSocket(gameId);
        
        gameModeDisplay.textContent = gameMode === 'human_vs_computer' ? 'Human vs Computer' : 'Computer vs Computer';
        timeControlDisplay.textContent = timeControlSelect.options[timeControlSelect.selectedIndex].text;
        currentTurnDisplay.textContent = 'White';
        moveCountDisplay.textContent = '0';
        
        updatePlayerLabels();
        
        updateTimerDisplays();
    } catch (error) {
        console.error('Error starting game:', error);
        alert('Failed to start game');
    }
}

async function resetGame() {
    try {
        if (!gameId) return;
        
        const response = await fetch(`/api/games/${gameId}/reset`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to reset game');
        }
        
        game = new Chess();
        board.position('start');
        
        gameOver = false;
        gameOverNotification.classList.add('hidden');
        computerThinkingIndicator.classList.add('hidden');
        
        currentTurnDisplay.textContent = 'White';
        moveCountDisplay.textContent = '0';
        
        updateTimerDisplays();
    } catch (error) {
        console.error('Error resetting game:', error);
        alert('Failed to reset game: ' + error.message);
    }
}

function backToSetup() {
    gameStarted = false;
    gameOver = false;
    
    if (socket) {
        socket.close();
        socket = null;
    }
    
    setupCard.classList.remove('hidden');
    gameOptionsCard.classList.add('hidden');
    gameStatusCard.classList.add('hidden');
    timerCard.classList.add('hidden');
}

function showComputerThinking(isThinking) {
    computerThinking = isThinking;
    if (isThinking) {
        computerThinkingIndicator.classList.remove('hidden');
    } else {
        computerThinkingIndicator.classList.add('hidden');
    }
}

function endGame(result) {
    if (gameOver) return;
    
    gameOver = true;
    
    gameOverNotification.classList.remove('hidden');
    gameOverResult.textContent = result === 'Draw' ? 'Draw!' : `${result} wins!`;
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function updateTimerDisplays() {
    whiteTimeDisplay.textContent = formatTime(timeControl);
    blackTimeDisplay.textContent = formatTime(timeControl);
}

function updatePlayerLabels() {
    if (gameMode === 'human_vs_computer') {
        whitePlayerLabel.textContent = 'White (Human)';
        blackPlayerLabel.textContent = 'Black (Computer)';
    } else if (gameMode === 'computer_vs_computer') {
        whitePlayerLabel.textContent = 'White (Computer)';
        blackPlayerLabel.textContent = 'Black (Computer)';
    }
}

document.addEventListener('DOMContentLoaded', function() {
    gameModeRadios.forEach(radio => {
        radio.addEventListener('change', () => {
            gameMode = radio.value;
            updatePlayerLabels();
        });
    });
    
    timeControlSelect.addEventListener('change', () => {
        timeControl = parseInt(timeControlSelect.value);
        updateTimerDisplays();
    });
    
    startGameBtn.addEventListener('click', startGame);
    newGameBtn.addEventListener('click', startGame);
    playAgainBtn.addEventListener('click', resetGame);
    backToSetupBtn.addEventListener('click', backToSetup);
    
    updateTimerDisplays();
});
