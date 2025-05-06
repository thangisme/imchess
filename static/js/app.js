class ChessGame {
  constructor() {
    this.board = null;
    this.game = new Chess();
    this.gameId = null;
    this.socket = null;
    
    this.whiteSide = 'human';
    this.blackSide = 'nn_policy';
    this.timeControl = 300; 
    
    this.gameStarted = false;
    this.gameOver = false;
    this.computerThinking = false;
    this.whiteTimeRemaining = null;
    this.blackTimeRemaining = null;
    this.timerInterval = null;
    this.moveCount = 0;

    this.initDomElements();
    
    this.setupEventListeners();
  }

  initDomElements() {
    this.setupCard = document.getElementById('setup-card');
    this.gameOptionsCard = document.getElementById('game-options-card');
    this.gameStatusCard = document.getElementById('game-status-card');
    this.timerCard = document.getElementById('timer-card');
    
    this.startGameBtn = document.getElementById('start-game-btn');
    this.newGameBtn = document.getElementById('new-game-btn');
    this.playAgainBtn = document.getElementById('play-again-btn');
    this.backToSetupBtn = document.getElementById('back-to-setup-btn');
    
    this.timeControlSelect = document.getElementById('time-control');
    this.whiteTimeDisplay = document.getElementById('white-time');
    this.blackTimeDisplay = document.getElementById('black-time');
    this.whitePlayerLabel = document.getElementById('white-player-label');
    this.blackPlayerLabel = document.getElementById('black-player-label');
    this.whiteSideSelect = document.getElementById('white-side');
    this.blackSideSelect = document.getElementById('black-side');
    
    this.gameModeDisplay = document.getElementById('game-mode-display');
    this.timeControlDisplay = document.getElementById('time-control-display');
    this.currentTurnDisplay = document.getElementById('current-turn-display');
    this.moveCountDisplay = document.getElementById('move-count-display');
    this.gameOverNotification = document.getElementById('game-over-notification');
    this.gameOverResult = document.getElementById('game-over-result');
    this.computerThinkingIndicator = document.getElementById('computer-thinking');
  }

  setupEventListeners() {
    this.timeControl = parseInt(this.timeControlSelect.value, 10);
    this.updatePlayerLabels();
    this.updateTimerDisplays();
    
    this.whiteSideSelect.addEventListener('change', () => {
      this.whiteSide = this.whiteSideSelect.value;
      this.updatePlayerLabels();
    });
    
    this.blackSideSelect.addEventListener('change', () => {
      this.blackSide = this.blackSideSelect.value;
      this.updatePlayerLabels();
    });
    
    this.timeControlSelect.addEventListener('change', () => {
      this.timeControl = parseInt(this.timeControlSelect.value, 10);
      this.updateTimerDisplays();
    });
    
    this.startGameBtn.addEventListener('click', () => this.startGame());
    this.newGameBtn.addEventListener('click', () => this.startGame());
    this.playAgainBtn.addEventListener('click', () => this.resetGame());
    this.backToSetupBtn.addEventListener('click', () => this.backToSetup());
  }

  initializeBoard() {
    const config = {
      draggable: true,
      position: 'start',
      onDragStart: (source, piece) => this.onDragStart(source, piece),
      onDrop: (source, target) => this.onDrop(source, target),
      onSnapEnd: () => this.onSnapEnd(),
      pieceTheme: '/static/img/chesspieces/wikipedia/{piece}.png',
      orientation: this.getBoardOrientation()
    };
    
    this.board = Chessboard('board', config);
    this.updateStatus();
    
    $(window).resize(() => this.board.resize());
  }

  onDragStart(source, piece) {
    if (this.game.game_over()) return false;
    
    if (this.whiteSide !== 'human' && this.blackSide !== 'human') return false;
    
    if ((this.game.turn() === 'w' && piece.search(/^b/) !== -1) ||
        (this.game.turn() === 'b' && piece.search(/^w/) !== -1)) {
      return false;
    }
    
    if (this.computerThinking) {
      return false;
    }
    
    return true;
  }

  onDrop(source, target) {
    const move = this.game.move({
      from: source,
      to: target,
      promotion: 'q' 
    });
    
    if (move === null) return 'snapback';
    
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({
        type: 'make_move',
        move: move.from + move.to + (move.promotion || '')
      }));
    }
    
    this.updateStatus();
    return true;
  }

  onSnapEnd() {
    this.board.position(this.game.fen());
  }

  updateStatus() {
    this.currentTurnDisplay.textContent = this.game.turn() === 'w' ? 'White' : 'Black';
    
    const whiteTimer = document.getElementById('white-timer');
    const blackTimer = document.getElementById('black-timer');
    
    if (this.game.turn() === 'w') {
      whiteTimer.classList.add('timer-active');
      blackTimer.classList.remove('timer-active');
    } else {
      whiteTimer.classList.remove('timer-active');
      blackTimer.classList.add('timer-active');
    }
    
    if (this.game.in_checkmate()) {
      const winner = this.game.turn() === 'w' ? 'Black' : 'White';
      this.endGame(winner);
    } else if (this.game.in_draw()) {
      this.endGame('Draw');
    }
  }

  connectWebSocket(id) {
    this.socket = new WebSocket(`ws://${window.location.host}/ws/${id}`);
    
    this.socket.onopen = () => {
      console.log('WebSocket connection established');
    };
    
    this.socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      console.log('Received message:', message);
      
      if (message.type === 'board_update') {
        this.game.load(message.fen);
        this.board.position(message.fen);
        this.board.orientation(this.getBoardOrientation());
        this.updateMoveCountFromFEN(message.fen);
        this.updateStatus();
        
        if (!message.is_game_over) {
          this.startClock();
        } else {
          this.stopClock();
        }
        
        if (message.hasOwnProperty('computer_thinking')) {
          this.showComputerThinking(message.computer_thinking);
        }
        
        if (message.is_game_over) {
          let resultMessage = 'Game over: ';
          if (message.result === 'checkmate') {
            resultMessage += `${message.winner} wins by checkmate`;
            this.endGame(message.winner);
          } else if (message.result === 'stalemate' || 
                     message.result === 'insufficient material' || 
                     message.result === 'fifty-move rule' || 
                     message.result === 'threefold repetition') {
            resultMessage += `Draw by ${message.result}`;
            this.endGame('Draw');
          } else {
            resultMessage += 'Draw';
            this.endGame('Draw');
          }
        }
      } else if (message.hasOwnProperty('computer_thinking')) {
        this.showComputerThinking(message.computer_thinking);
      }
    };
    
    this.socket.onclose = () => {
      console.log('WebSocket connection closed');
    };
    
    this.socket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  async startGame() {
    try {
      this.whiteSide = this.whiteSideSelect.value;
      this.blackSide = this.blackSideSelect.value;
      this.timeControl = parseInt(this.timeControlSelect.value, 10);
      this.resetTimers();
      
      const response = await fetch('/api/new-game', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          time_control: this.timeControl,
          white_side: this.whiteSide,
          black_side: this.blackSide
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to create game');
      }
      
      const data = await response.json();
      this.gameId = data.game_id;
      
      this.setupCard.classList.add('hidden');
      this.gameOptionsCard.classList.remove('hidden');
      this.gameStatusCard.classList.remove('hidden');
      this.timerCard.classList.remove('hidden');
      this.gameOverNotification.classList.add('hidden');
      this.computerThinkingIndicator.classList.add('hidden');
      
      this.gameStarted = true;
      this.gameOver = false;
      this.game = new Chess();
      
      this.initializeBoard();
      this.connectWebSocket(this.gameId);
      
      const playerTypes = { 'human': 'Human', 'nn_policy': 'Neural Network' };
      this.gameModeDisplay.textContent = playerTypes[this.whiteSide] + ' vs ' + playerTypes[this.blackSide];
      this.timeControlDisplay.textContent = this.timeControlSelect.options[this.timeControlSelect.selectedIndex].text;
      this.currentTurnDisplay.textContent = 'White';
      this.moveCountDisplay.textContent = '0';
      
      this.updatePlayerLabels();
      this.updateTimerDisplays();
      
    } catch (error) {
      console.error('Error starting game:', error);
      alert('Failed to start game');
    }
  }

  async resetGame() {
    try {
      if (!this.gameId) return;
      
      this.resetTimers();
      
      const response = await fetch(`/api/games/${this.gameId}/reset`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to reset game');
      }
      
      this.game = new Chess();
      this.board.position('start');
      this.board.orientation(this.getBoardOrientation());
      
      this.gameOver = false;
      this.gameOverNotification.classList.add('hidden');
      this.computerThinkingIndicator.classList.add('hidden');
      
      this.currentTurnDisplay.textContent = 'White';
      this.moveCountDisplay.textContent = '0';
      
      this.updateTimerDisplays();
      
    } catch (error) {
      console.error('Error resetting game:', error);
      alert('Failed to reset game: ' + error.message);
    }
  }

  backToSetup() {
    this.gameStarted = false;
    this.gameOver = false;
    
    this.stopClock();
    
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    
    this.setupCard.classList.remove('hidden');
    this.gameOptionsCard.classList.add('hidden');
    this.gameStatusCard.classList.add('hidden');
    this.timerCard.classList.add('hidden');
  }

  showComputerThinking(isThinking) {
    this.computerThinking = isThinking;
    if (isThinking) {
      this.computerThinkingIndicator.classList.remove('hidden');
    } else {
      this.computerThinkingIndicator.classList.add('hidden');
    }
  }

  endGame(result) {
    if (this.gameOver) return;
    
    this.gameOver = true;
    this.gameOverNotification.classList.remove('hidden');
    this.gameOverResult.textContent = result === 'Draw' ? 'Draw!' : `${result} wins!`;
    this.stopClock();
  }

  tickClock() {
    if (this.gameOver) {
      this.stopClock();
      return;
    }
    
    if (this.game.turn() === 'w') {
      this.whiteTimeRemaining--;
      if (this.whiteTimeRemaining <= 0) {
        this.whiteTimeRemaining = 0;
        this.onTimeOut('White');
      }
    } else {
      this.blackTimeRemaining--;
      if (this.blackTimeRemaining <= 0) {
        this.blackTimeRemaining = 0;
        this.onTimeOut('Black');
      }
    }
    
    this.updateTimerDisplays();
  }

  onTimeOut(losingSide) {
    this.stopClock();
    this.gameOver = true;
    if (losingSide === 'White') {
      this.endGame('Black');
    } else {
      this.endGame('White');
    }
  }

  startClock() {
    if (this.timerInterval) clearInterval(this.timerInterval);
    this.timerInterval = setInterval(() => this.tickClock(), 1000);
  }

  stopClock() {
    if (this.timerInterval) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }
  }

  resetTimers() {
    this.stopClock();
    this.whiteTimeRemaining = this.timeControl;
    this.blackTimeRemaining = this.timeControl;
    this.updateTimerDisplays();
  }

  formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }

  updateTimerDisplays() {
    const white = this.whiteTimeRemaining != null ? this.whiteTimeRemaining : this.timeControl;
    const black = this.blackTimeRemaining != null ? this.blackTimeRemaining : this.timeControl;
    this.whiteTimeDisplay.textContent = this.formatTime(white);
    this.blackTimeDisplay.textContent = this.formatTime(black);
  }

  updatePlayerLabels() {
    const playerTypes = {
      'human': 'Human',
      'nn_policy': 'Neural Network'
    };
    
    this.whitePlayerLabel.textContent = `White (${playerTypes[this.whiteSide]})`;
    this.blackPlayerLabel.textContent = `Black (${playerTypes[this.blackSide]})`;
  }

  getBoardOrientation() {
    if (this.whiteSide !== 'human' && this.blackSide === 'human') {
      return 'black';
    }
    return 'white';
  }

  updateMoveCountFromFEN(fen) {
    const parts = fen.split(' ');
    if (parts.length >= 6) {
      const fullMoveNumber = parseInt(parts[5]);
      this.moveCountDisplay.textContent = fullMoveNumber - 1;
    }
  }
}

document.addEventListener('DOMContentLoaded', function() {
  window.chessGame = new ChessGame();
});
