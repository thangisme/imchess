class ChessGame {
  constructor() {
    this.board = null;
    this.game = new Chess();
    this.gameId = null;
    this.socket = null;
    this.whiteSide = 'human';
    this.blackSide = 'nn_policy';
    this.timeControl = 300;
    this.stockfishElo = 1800;
    this.gameStarted = false;
    this.gameOver = false;
    this.computerThinking = false;
    this.whiteTimeRemaining = null;
    this.blackTimeRemaining = null;
    this.timerInterval = null;
    this.moveCount = 0;
    this.moveHistory = [];
    this.currentViewIndex = -1;
    this.viewingHistory = false;
    this.waitingForServerUpdate = false;
    this.squareHighlights = {
      selectedSquare: null,
      legalMoves: []
    };
    this.lastMoveHighlight = {
      from: null,
      to: null
    };
    this.initDomElements();
    this.setupEventListeners();
  }

  initDomElements() {
    this.setupCard = document.getElementById('setup-card');
    this.gameOptionsCard = document.getElementById('game-options-card');
    this.gameStatusCard = document.getElementById('game-status-card');
    this.timerCard = document.getElementById('timer-card');
    this.gameHistoryCard = document.getElementById('game-history-card');
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
    this.moveList = document.getElementById('move-list');
    this.firstMoveBtn = document.getElementById('first-move-btn');
    this.prevMoveBtn = document.getElementById('prev-move-btn');
    this.nextMoveBtn = document.getElementById('next-move-btn');
    this.lastMoveBtn = document.getElementById('last-move-btn');
    this.viewingHistoryIndicator = document.getElementById('viewing-history-indicator');
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
    this.firstMoveBtn.addEventListener('click', () => this.goToFirstMove());
    this.prevMoveBtn.addEventListener('click', () => this.goToPreviousMove());
    this.nextMoveBtn.addEventListener('click', () => this.goToNextMove());
    this.lastMoveBtn.addEventListener('click', () => this.goToLastMove());
    // document.getElementById('stockfish-depth').addEventListener('input', (e) => {
    //   this.stockfishDepth = parseInt(e.target.value, 10);
    //   document.getElementById('stockfish-depth-value').textContent = this.stockfishDepth;
    // });

    document.getElementById('stockfish-elo').addEventListener('input', (e) => {
      this.stockfishElo = parseInt(e.target.value, 10);
      document.getElementById('stockfish-elo-value').textContent = this.stockfishElo;
    });

    this.whiteSideSelect.addEventListener('change', () => {
      this.whiteSide = this.whiteSideSelect.value;
      this.updatePlayerLabels();
      this.updateStockfishSettingsVisibility();
    });

    this.blackSideSelect.addEventListener('change', () => {
      this.blackSide = this.blackSideSelect.value;
      this.updatePlayerLabels();
      this.updateStockfishSettingsVisibility();
    });
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
      (this.game.turn() === 'b' && piece.search(/^w/) !== -1)) return false;
    if (this.computerThinking) return false;
    if (this.viewingHistory) return false;
    if (this.waitingForServerUpdate) return false;

    this.removeHighlights();

    $(`#board .square-${source}`).addClass('highlight-selected');

    const moves = this.game.moves({
      square: source,
      verbose: true
    });

    this.squareHighlights.selectedSquare = source;
    this.squareHighlights.legalMoves = moves.map(move => move.to);

    this.squareHighlights.legalMoves.forEach(square => {
      $(`#board .square-${square}`).addClass('highlight-legal');
    });

    return true;
  }

  onDrop(source, target) {
    this.removeHighlights();

    if (this.viewingHistory) {
      this.goToLastMove();
      return 'snapback';
    }

    if (this.waitingForServerUpdate) {
      return 'snapback';
    }

    const tmpGame = new Chess(this.game.fen());
    const move = tmpGame.move({ from: source, to: target, promotion: 'q' });

    if (move === null) {
      return 'snapback';
    }

    this.waitingForServerUpdate = true;

    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({
        type: 'make_move',
        move: source + target + (move.promotion || '')
      }));
    }

    return true;
  }

  onSnapEnd() {
    this.removeHighlights();
  }

  updateStockfishSettingsVisibility() {
    const stockfishSettings = document.getElementById('stockfish-settings');
    if (this.whiteSide === 'stockfish' || this.blackSide === 'stockfish') {
      stockfishSettings.classList.remove('hidden');
    } else {
      stockfishSettings.classList.add('hidden');
    }
  }

  removeHighlights() {
    if (this.squareHighlights.selectedSquare) {
      $(`#board .square-${this.squareHighlights.selectedSquare}`).removeClass('highlight-selected');
    }

    this.squareHighlights.legalMoves.forEach(square => {
      $(`#board .square-${square}`).removeClass('highlight-legal');
    });

    this.squareHighlights.selectedSquare = null;
    this.squareHighlights.legalMoves = [];
  }

  updateLastMoveHighlight() {
    if (this.lastMoveHighlight.from) {
      $(`#board .square-${this.lastMoveHighlight.from}`).removeClass('highlight-lastmove');
    }
    if (this.lastMoveHighlight.to) {
      $(`#board .square-${this.lastMoveHighlight.to}`).removeClass('highlight-lastmove');
    }

    let lastMove = null;
    if (this.currentViewIndex >= 0 && this.moveHistory.length > 0) {
      lastMove = this.moveHistory[this.currentViewIndex];
    } else if (this.moveHistory.length > 0 && !this.viewingHistory) {
      lastMove = this.moveHistory[this.moveHistory.length - 1];
    }

    if (lastMove) {
      this.lastMoveHighlight.from = lastMove.from;
      this.lastMoveHighlight.to = lastMove.to;

      $(`#board .square-${lastMove.from}`).addClass('highlight-lastmove');
      $(`#board .square-${lastMove.to}`).addClass('highlight-lastmove');
    } else {
      this.lastMoveHighlight.from = null;
      this.lastMoveHighlight.to = null;
    }
  }

  updateStatus() {
    this.currentTurnDisplay.textContent = this.game.turn() === 'w' ? 'White' : 'Black';

    if (!this.viewingHistory) {
      const whiteTimer = document.getElementById('white-timer');
      const blackTimer = document.getElementById('black-timer');

      if (this.game.turn() === 'w') {
        whiteTimer.classList.add('timer-active');
        blackTimer.classList.remove('timer-active');
      } else {
        whiteTimer.classList.remove('timer-active');
        blackTimer.classList.add('timer-active');
      }
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
    this.socket.onopen = () => { };
    this.socket.onmessage = (event) => {
      const message = JSON.parse(event.data);

      if (message.type === 'board_update') {
        if (message.last_move_info) {
          const serverMove = message.last_move_info;
          const moveForHistory = {
            san: serverMove.san,
            color: serverMove.color,
            from: serverMove.from_uci,
            to: serverMove.to_uci,
            promotion: serverMove.promotion,
            fen: message.fen
          };

          if (!this.hasMoveInHistory(moveForHistory)) {
            this.addMoveToHistory(moveForHistory);
          }
        }

        this.game.load(message.fen);

        if (!this.viewingHistory) {
          this.board.position(message.fen);
          if (this.moveHistory.length > 0) {
            this.currentViewIndex = this.moveHistory.length - 1;
          } else {
            this.currentViewIndex = -1;
          }
        }

        this.waitingForServerUpdate = false;
        this.board.orientation(this.getBoardOrientation());
        this.updateMoveCountFromFEN(message.fen);
        this.updateStatus();
        this.updateLastMoveHighlight();

        if (!message.is_game_over) {
          this.startClock();
        } else {
          this.stopClock();
        }

        if (message.hasOwnProperty('computer_thinking')) {
          this.showComputerThinking(message.computer_thinking);
        }

        if (message.is_game_over) {
          if (message.result === 'checkmate') {
            this.endGame(message.winner);
          } else if (message.result === 'stalemate' ||
            message.result === 'insufficient material' ||
            message.result === 'fifty-move rule' ||
            message.result === 'threefold repetition') {
            this.endGame('Draw');
          } else {
            this.endGame('Draw');
          }
        }

        this.updateMoveList();
        this.updateHistoryNavButtons();

      } else if (message.type === 'thinking_status' || message.hasOwnProperty('computer_thinking')) {
        this.showComputerThinking(message.computer_thinking);
      }
    };
    this.socket.onclose = () => { };
    this.socket.onerror = (error) => { };
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
          black_side: this.blackSide,
          stockfish_elo: this.stockfishElo,
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
      this.gameHistoryCard.classList.remove('hidden');
      this.gameOverNotification.classList.add('hidden');
      this.computerThinkingIndicator.classList.add('hidden');
      this.gameStarted = true;
      this.gameOver = false;
      this.waitingForServerUpdate = false;
      this.game = new Chess();
      this.moveHistory = [];
      this.currentViewIndex = -1;
      this.viewingHistory = false;
      this.lastMoveHighlight = { from: null, to: null };
      this.squareHighlights = { selectedSquare: null, legalMoves: [] };
      this.updateMoveList();
      this.updateHistoryNavButtons();
      this.updateHistoryView();
      this.initializeBoard();
      this.connectWebSocket(this.gameId);
      const playerTypes = { 'human': 'Human', 'nn_policy': 'Neural Network', 'stockfish': 'Stockfish' + `(${this.stockfishElo})` };
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
      const response = await fetch(`/api/games/${this.gameId}/reset`, { method: 'POST' });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to reset game');
      }
      this.game = new Chess();
      this.board.position('start');
      this.board.orientation(this.getBoardOrientation());
      this.moveHistory = [];
      this.currentViewIndex = -1;
      this.viewingHistory = false;
      this.waitingForServerUpdate = false;
      this.lastMoveHighlight = { from: null, to: null };
      this.squareHighlights = { selectedSquare: null, legalMoves: [] };
      this.removeHighlights();
      this.updateMoveList();
      this.updateHistoryNavButtons();
      this.updateHistoryView();
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
    this.waitingForServerUpdate = false;
    this.stopClock();
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    this.setupCard.classList.remove('hidden');
    this.gameOptionsCard.classList.add('hidden');
    this.gameStatusCard.classList.add('hidden');
    this.timerCard.classList.add('hidden');
    this.gameHistoryCard.classList.add('hidden');
    this.moveHistory = [];
    this.currentViewIndex = -1;
    this.viewingHistory = false;
    this.lastMoveHighlight = { from: null, to: null };
    this.squareHighlights = { selectedSquare: null, legalMoves: [] };
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
    if (!this.viewingHistory) {
      this.timerInterval = setInterval(() => this.tickClock(), 1000);
    }
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
    const playerTypes = { 'human': 'Human', 'nn_policy': 'Neural Network', 'stockfish': 'Stockfish' };
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
      this.moveCountDisplay.textContent = Math.floor(this.moveHistory.length / 2);
    } else {
      this.moveCountDisplay.textContent = Math.floor(this.moveHistory.length / 2);
    }
  }

  hasMoveInHistory(move) {
    if (this.moveHistory.length === 0) {
      return false;
    }

    const lastMove = this.moveHistory[this.moveHistory.length - 1];
    return (
      lastMove.san === move.san &&
      lastMove.color === move.color &&
      lastMove.from === move.from &&
      lastMove.to === move.to
    );
  }

  addMoveToHistory(move) {
    if (!move || !move.san || !move.color || !move.fen) {
      return;
    }

    this.moveHistory.push({
      san: move.san,
      from: move.from,
      to: move.to,
      promotion: move.promotion,
      fen: move.fen,
      color: move.color
    });

    if (!this.viewingHistory) {
      this.currentViewIndex = this.moveHistory.length - 1;
    }

    this.updateLastMoveHighlight();
  }

  updateMoveList() {
    this.moveList.innerHTML = '';
    for (let i = 0; i < this.moveHistory.length; i += 2) {
      const moveNumber = Math.floor(i / 2) + 1;
      const whiteMoveData = this.moveHistory[i];
      const blackMoveData = (i + 1 < this.moveHistory.length) ? this.moveHistory[i + 1] : null;

      const li = document.createElement('li');
      li.className = 'move-item';

      const moveNumberSpan = document.createElement('span');
      moveNumberSpan.className = 'move-number';
      moveNumberSpan.textContent = moveNumber + '.';
      li.appendChild(moveNumberSpan);

      const whiteMoveSpan = document.createElement('span');
      whiteMoveSpan.className = 'move-white';
      if (whiteMoveData) {
        whiteMoveSpan.textContent = whiteMoveData.san;
        whiteMoveSpan.dataset.index = i;
        whiteMoveSpan.addEventListener('click', () => this.goToMove(i));
        if (i === this.currentViewIndex) {
          whiteMoveSpan.classList.add('active');
        }
      }
      li.appendChild(whiteMoveSpan);

      const blackMoveSpan = document.createElement('span');
      blackMoveSpan.className = 'move-black';
      if (blackMoveData) {
        blackMoveSpan.textContent = blackMoveData.san;
        blackMoveSpan.dataset.index = i + 1;
        blackMoveSpan.addEventListener('click', () => this.goToMove(i + 1));
        if ((i + 1) === this.currentViewIndex) {
          blackMoveSpan.classList.add('active');
        }
      }
      li.appendChild(blackMoveSpan);
      this.moveList.appendChild(li);
    }

    if (this.currentViewIndex >= 0 && this.moveHistory.length > 0) {
      const activeElement = this.moveList.querySelector('.active');
      if (activeElement) {
        activeElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
  }

  goToMove(index) {
    if (index < -1 || index >= this.moveHistory.length) {
      if (index === -1 && this.moveHistory.length === 0) { }
      else {
        return;
      }
    }

    const wasViewingHistory = this.viewingHistory;

    this.viewingHistory = (index < this.moveHistory.length - 1 && index !== -1) ||
      (index === -1 && this.moveHistory.length > 0);
    this.currentViewIndex = index;

    this.removeHighlights();

    if (index === -1) {
      this.game.reset();
      this.board.position('start');
    } else {
      const fenToLoad = this.moveHistory[index].fen;
      this.game.load(fenToLoad);
      this.board.position(fenToLoad);
    }

    this.updateLastMoveHighlight();

    if (!wasViewingHistory && this.viewingHistory) {
      this.stopClock();
    } else if (wasViewingHistory && !this.viewingHistory) {
      this.startClock();
    }

    this.updateStatus();
    this.updateHistoryView();
    this.updateHistoryNavButtons();
    this.updateMoveList();
  }

  goToFirstMove() {
    this.goToMove(-1);
  }

  goToPreviousMove() {
    this.goToMove(this.currentViewIndex - 1);
  }

  goToNextMove() {
    this.goToMove(this.currentViewIndex + 1);
  }

  goToLastMove() {
    if (this.moveHistory.length === 0) {
      this.goToMove(-1);
    } else {
      this.goToMove(this.moveHistory.length - 1);
    }
  }

  updateHistoryView() {
    if (this.viewingHistory) {
      this.viewingHistoryIndicator.classList.remove('hidden');
    } else {
      this.viewingHistoryIndicator.classList.add('hidden');
    }
  }

  updateHistoryNavButtons() {
    const noMoves = this.moveHistory.length === 0;
    this.firstMoveBtn.disabled = this.currentViewIndex <= -1;
    this.prevMoveBtn.disabled = this.currentViewIndex <= -1;
    this.nextMoveBtn.disabled = noMoves || this.currentViewIndex >= this.moveHistory.length - 1;
    this.lastMoveBtn.disabled = noMoves || this.currentViewIndex >= this.moveHistory.length - 1;
  }
}

document.addEventListener('DOMContentLoaded', function() {
  window.chessGame = new ChessGame();
});
