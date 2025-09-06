class QueensGame {
    constructor() {
        this.gridSize = 0;
        this.colors = [];
        this.clusters = [];
        this.cellStates = [];
        this.moveHistory = [];
        this.startTime = 0;
        this.elapsedTime = 0;
        this.isComplete = false;
        this.autoX = false;
        this.timerInterval = null;
        
        // Touch handling properties
        this.touchStartTime = 0;
        this.lastTouchCell = null;
        this.touchTimeout = null;
        this.isDoubleTap = false;
        
        // DOM elements
        this.gameBoard = document.getElementById('game-board');
        this.timerDisplay = document.getElementById('timer');
        this.uploadForm = document.getElementById('upload-form');
        this.uploadSection = document.querySelector('.upload-section');
        this.gameSection = document.querySelector('.game-section');
        this.autoXButton = document.getElementById('auto-x-btn');
        this.clearButton = document.getElementById('clear-btn');
        this.undoButton = document.getElementById('undo-btn');
        this.completionPopup = document.getElementById('completion-popup');
        this.completionTime = document.getElementById('completion-time');
        this.closePopupButton = document.getElementById('close-popup');
        
        this.initializeEventListeners();
        this.loadSettings();
    }
    
    initializeEventListeners() {
        console.log("==================================================================");
        console.log("DEBUG js: Initializing event listeners");
        this.uploadForm.addEventListener('submit', (e) => {
            e.preventDefault(); // Prevent default form submission
            this.handleUpload(e);
        });
        this.clearButton.addEventListener('click', () => this.clearBoard());
        this.undoButton.addEventListener('click', () => this.undoMove());
        this.autoXButton.addEventListener('click', () => this.toggleAutoX());
        this.closePopupButton.addEventListener('click', () => this.hideCompletionPopup());
    }

    async loadSettings() {
        try {
            const response = await fetch('/load_settings');
            const settings = await response.json();
            this.autoX = settings.auto_x;
            this.updateAutoXButton();
        } catch (error) {
            console.error('Error loading settings:', error);
        }
    }
    
    async saveSettings() {
        try {
            await fetch('/save_settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ auto_x: this.autoX }),
            });
        } catch (error) {
            console.error('Error saving settings:', error);
        }
    }
    
    async handleUpload(e) {
        e.preventDefault(); // This is crucial
        console.log("DEBUG: handleUpload fired");
        const fileInput = document.getElementById('file-input');
        const file = fileInput.files[0];
        if (!file) {
            alert('Please select a file first.');
            return;
        }
        console.log("DEBUG - Uploading file:", file.name, "Type:", file.type);
        const formData = new FormData();
        formData.append('file', file);
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });
            console.log("DEBUG - Response status:", response.status);
            const data = await response.json();
            console.log("DEBUG - Server response:", data);
            if (data.success) {
                console.log("DEBUG - Initializing game with data:", data);
                this.initializeGame(data);
                this.uploadSection.style.display = 'none';
                this.gameSection.style.display = 'block';
            } else {
                alert("Upload failed: " + data.error);
            }
        } catch (error) {
            console.error('Error during upload:', error);
            alert('Error uploading file');
        }
    }
    
    initializeGame(data) {
        this.gridSize = data.grid_size;
        this.colors = data.colors;
        this.clusters = data.clusters;
        this.cellStates = Array(this.gridSize).fill().map(() => Array(this.gridSize).fill(0));
        this.moveHistory = [];
        this.startTime = Date.now();
        this.isComplete = false;
        this.createBoard();
        this.startTimer();
        this.addTouchSupport();
    }
    
    createBoard() {
        // Calculate cell size based on viewport
        const viewportWidth = Math.min(window.innerWidth - 40, window.innerHeight - 120);
        const cellSize = Math.floor((viewportWidth - (this.gridSize + 1) * 2) / this.gridSize);
        const borderSize = 2;
        const totalSize = this.gridSize * (cellSize + borderSize) + borderSize;
        
        this.gameBoard.style.width = totalSize + 'px';
        this.gameBoard.style.height = totalSize + 'px';
        this.gameBoard.innerHTML = '';
        
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                const cell = document.createElement('div');
                cell.className = 'game-cell';
                cell.style.width = cellSize + 'px';
                cell.style.height = cellSize + 'px';
                cell.style.left = (j * (cellSize + borderSize) + borderSize) + 'px';
                cell.style.top = (i * (cellSize + borderSize) + borderSize) + 'px';
                
                const color = this.colors[i][j];
                cell.style.backgroundColor = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
                
                cell.addEventListener('mousedown', (e) => this.handleCellMouseDown(e, i, j));
                cell.addEventListener('mousemove', (e) => this.handleCellMouseMove(e, i, j));
                cell.addEventListener('mouseup', () => this.handleCellMouseUp());
                
                this.gameBoard.appendChild(cell);
            }
        }
    }
    
    startTimer() {
        if (this.timerInterval) clearInterval(this.timerInterval);
        this.timerInterval = setInterval(() => {
            if (!this.isComplete) {
                this.elapsedTime = Math.floor((Date.now() - this.startTime) / 1000);
                this.timerDisplay.textContent = `Time: ${this.elapsedTime}s`;
            }
        }, 1000);
    }
    
    handleCellMouseDown(e, row, col) {
        if (this.isComplete) return;
        
        const currentTime = Date.now();
        if (e.button === 0) { // Left click
            if (this.lastClickCell && 
                this.lastClickCell.row === row && 
                this.lastClickCell.col === col &&
                currentTime - this.lastClickTime < 500) {
                // Double click - place queen
                this.placeQueen(row, col);
            } else {
                // Start swipe or remove queen
                if (this.cellStates[row][col] === 2) { // Queen
                    this.toggleCell(row, col);
                } else {
                    this.startSwipe(row, col);
                }
            }
            this.lastClickTime = currentTime;
            this.lastClickCell = { row, col };
        } else if (e.button === 2) { // Right click
            this.toggleCell(row, col);
        }
    }
    
    handleCellMouseMove(e, row, col) {
        if (this.isSwiping && this.lastSwipedCell !== `${row},${col}`) {
            this.handleSwipe(row, col);
        }
    }
    
    handleCellMouseUp() {
        this.endSwipe();
    }
    
    startSwipe(row, col) {
        const currentState = this.cellStates[row][col];
        if (currentState !== 2) { // Not a queen
            this.isSwiping = true;
            this.swipeState = currentState === 0 ? 1 : 0;
            this.lastSwipedCell = `${row},${col}`;
            this.setCellState(row, col, this.swipeState);
        }
    }
    
    handleSwipe(row, col) {
        if (this.cellStates[row][col] !== 2) { // Not a queen
            this.lastSwipedCell = `${row},${col}`;
            this.setCellState(row, col, this.swipeState);
        }
    }
    
    endSwipe() {
        this.isSwiping = false;
        this.swipeState = null;
        this.lastSwipedCell = null;
    }
    
    placeQueen(row, col) {
        const oldState = this.cellStates[row][col];
        this.cellStates[row][col] = 2; // Queen
        this.moveHistory.push({
            row,
            col,
            oldState,
            affectedCells: this.autoX ? this.markRowColCells(row, col) : []
        });
        this.updateCell(row, col);
        this.checkWinCondition();
    }
    
    toggleCell(row, col) {
        const oldState = this.cellStates[row][col];
        let newState;
        
        if (oldState === 0) newState = 1; // Empty to Marked
        else if (oldState === 1) newState = 2; // Marked to Queen
        else newState = 0; // Queen to Empty
        
        this.cellStates[row][col] = newState;
        
        const affectedCells = [];
        if (newState === 2 && this.autoX) { // If placing a queen and auto-X is enabled
            affectedCells.push(...this.markRowColCells(row, col));
        }
        
        this.moveHistory.push({ row, col, oldState, affectedCells });
        this.updateCell(row, col);
        
        if (newState === 2) this.checkWinCondition();
    }
    
    setCellState(row, col, state) {
        if (this.cellStates[row][col] !== state) {
            const oldState = this.cellStates[row][col];
            this.cellStates[row][col] = state;
            this.moveHistory.push({ row, col, oldState, affectedCells: [] });
            this.updateCell(row, col);
        }
    }
    
    markRowColCells(row, col) {
        const affectedCells = [];
        
        // Helper function to mark a cell
        const markCell = (r, c) => {
            if (r >= 0 && r < this.gridSize && c >= 0 && c < this.gridSize &&
                this.cellStates[r][c] === 0) {
                affectedCells.push({ row: r, col: c, oldState: 0 });
                this.cellStates[r][c] = 1;
                this.updateCell(r, c);
            }
        };
        
        // Mark row and column
        for (let i = 0; i < this.gridSize; i++) {
            markCell(row, i); // Mark row
            markCell(i, col); // Mark column
        }
        
        // Mark surrounding cells
        for (let r = Math.max(0, row-1); r <= Math.min(this.gridSize-1, row+1); r++) {
            for (let c = Math.max(0, col-1); c <= Math.min(this.gridSize-1, col+1); c++) {
                if (r !== row || c !== col) {
                    markCell(r, c);
                }
            }
        }
        
        // Mark cells in the same cluster
        const cluster = this.clusters.find(c => c.some(([r, c]) => r === row && c === col));
        if (cluster) {
            for (const [r, c] of cluster) {
                if (r !== row || c !== col) {
                    markCell(r, c);
                }
            }
        }
        
        return affectedCells;
    }
    
    updateCell(row, col) {
        const cell = this.gameBoard.children[row * this.gridSize + col];
        cell.className = 'game-cell';
        
        if (this.cellStates[row][col] === 1) {
            cell.classList.add('marked');
        } else if (this.cellStates[row][col] === 2) {
            cell.innerHTML = '<div class="queen"></div>';
        } else {
            cell.innerHTML = '';
        }
    }
    
    clearBoard() {
        this.cellStates = Array(this.gridSize).fill().map(() => Array(this.gridSize).fill(0));
        this.moveHistory = [];
        this.isComplete = false;
        this.hideCompletionPopup();
        
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                this.updateCell(i, j);
            }
        }
    }
    
    undoMove() {
        if (this.moveHistory.length === 0) return;
        
        const move = this.moveHistory.pop();
        this.cellStates[move.row][move.col] = move.oldState;
        this.updateCell(move.row, move.col);
        
        // Restore affected cells from auto-X
        for (const cell of move.affectedCells) {
            this.cellStates[cell.row][cell.col] = cell.oldState;
            this.updateCell(cell.row, cell.col);
        }
        
        this.isComplete = false;
        this.hideCompletionPopup();
    }
    
    toggleAutoX() {
        this.autoX = !this.autoX;
        this.updateAutoXButton();
        this.saveSettings();
    }
    
    updateAutoXButton() {
        this.autoXButton.textContent = `Auto-X: ${this.autoX ? 'On' : 'Off'}`;
        this.autoXButton.className = this.autoX ? 'active' : 'inactive';
    }
    
    showCompletionPopup() {
        // If time elapsed os > than 60 s show minutes and seconds
        // else show only seconds
        if (this.elapsedTime > 60) {
            const minutes = Math.floor(this.elapsedTime / 60);
            const seconds = this.elapsedTime % 60;
            this.completionTime.textContent = `Time: ${minutes}m ${seconds}s`;
        } else {
            this.completionTime.textContent = `Time: ${this.elapsedTime}s`;
        }
        this.completionPopup.style.display = 'block';
    }
    
    hideCompletionPopup() {
        this.completionPopup.style.display = 'none';
    }

    addTouchSupport() {
        this.gameBoard.addEventListener('touchstart', (e) => this.handleTouchStart(e), { passive: false });
        this.gameBoard.addEventListener('touchmove', (e) => this.handleTouchMove(e), { passive: false });
        this.gameBoard.addEventListener('touchend', (e) => this.handleTouchEnd(e), { passive: false });
    }

    handleTouchStart(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const cell = this.getCellFromTouch(touch);

        if (!cell) return;

        // Highlight the touched cell
        const cellElement = this.gameBoard.children[cell.row * this.gridSize + cell.col];
        cellElement.style.boxShadow = '0 0 10px 5px rgba(255, 255, 0, 0.7)';
        
        if (this.lastTouchCell && 
            this.lastTouchCell.row === cell.row && 
            this.lastTouchCell.col === cell.col && 
            currentTime - this.touchStartTime < 300) {
            // Double tap detected
            this.isDoubleTap = true;
            clearTimeout(this.touchTimeout);
            this.placeQueen(cell.row, cell.col);
        } else {
            // Single tap - wait to see if it's a double tap or start of swipe
            this.touchStartTime = currentTime;
            this.lastTouchCell = cell;
            this.isDoubleTap = false;
            
            this.touchTimeout = setTimeout(() => {
                if (!this.isDoubleTap) {
                    if (this.cellStates[cell.row][cell.col] === 2) {
                        this.toggleCell(cell.row, cell.col);
                    } else {
                        this.startSwipe(cell.row, cell.col);
                    }
                }
            }, 300);
        }
    }

    handleTouchMove(e) {
        e.preventDefault();
        if (!this.isSwiping) return;
        
        const touch = e.touches[0];
        const cell = this.getCellFromTouch(touch);
        
        if (cell) {
            this.handleSwipe(cell.row, cell.col);
        }
    }

    handleTouchEnd(e) {
        e.preventDefault();
        this.endSwipe();
        clearTimeout(this.touchTimeout);

        // Remove highlight from all cells
        for (let i = 0; i < this.gridSize * this.gridSize; i++) {
            this.gameBoard.children[i].style.boxShadow = 'none';
        }
    }

    getCellFromTouch(touch) {
        const rect = this.gameBoard.getBoundingClientRect();
        const x = touch.clientX - rect.left;
        const y = touch.clientY - rect.top;

        const cellSize = rect.width / this.gridSize;
        const row = Math.floor(y / cellSize);
        const col = Math.floor(x / cellSize);

        // Add a small tolerance for touch accuracy
        if (row >=   0 && row < this.gridSize && col >= 0 && col < this.gridSize) {
            return { row, col };
        }
        return null;
    }
        
    checkWinCondition() {
        // Check rows
        for (let i = 0; i < this.gridSize; i++) {
            let queensInRow = this.cellStates[i].filter(state => state === 2).length;
            if (queensInRow !== 1) return false;
        }
        
        // Check columns
        for (let j = 0; j < this.gridSize; j++) {
            let queensInCol = 0;
            for (let i = 0; i < this.gridSize; i++) {
                if (this.cellStates[i][j] === 2) queensInCol++;
            }
            if (queensInCol !== 1) return false;
        }
        
        // Check clusters
        for (const cluster of this.clusters) {
            let queensInCluster = 0;
            for (const [row, col] of cluster) {
                if (this.cellStates[row][col] === 2) queensInCluster++;
            }
            if (queensInCluster !== 1) return false;
        }
        
        // Check for adjacent queens
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                if (this.cellStates[i][j] === 2) {
                    // Check surrounding cells
                    for (let r = Math.max(0, i-1); r <= Math.min(this.gridSize-1, i+1); r++) {
                        for (let c = Math.max(0, j-1); c <= Math.min(this.gridSize-1, j+1); c++) {
                            if ((r !== i || c !== j) && this.cellStates[r][c] === 2) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        
        this.isComplete = true;
        this.showCompletionPopup();
        return true;
    }

    addTouchSupport() {
        this.gameBoard.addEventListener('touchstart', (e) => this.handleTouchStart(e), { passive: false });
        this.gameBoard.addEventListener('touchmove', (e) => this.handleTouchMove(e), { passive: false });
        this.gameBoard.addEventListener('touchend', (e) => this.handleTouchEnd(e), { passive: false });
    }

    handleTouchStart(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const cell = this.getCellFromTouch(touch);

        if (!cell) return;

        const currentTime = Date.now();

        if (this.lastTouchCell &&
            this.lastTouchCell.row === cell.row &&
            this.lastTouchCell.col === cell.col &&
            currentTime - this.touchStartTime < 500) { // Increased from 300ms
            // Double tap detected
            this.isDoubleTap = true;
            clearTimeout(this.touchTimeout);
            this.placeQueen(cell.row, cell.col);
        } else {
            // Single tap - wait to see if it's a double tap or start of swipe
            this.touchStartTime = currentTime;
            this.lastTouchCell = cell;
            this.isDoubleTap = false;

            this.touchTimeout = setTimeout(() => {
                if (!this.isDoubleTap) {
                    if (this.cellStates[cell.row][cell.col] === 2) {
                        this.toggleCell(cell.row, cell.col);
                    } else {
                        this.startSwipe(cell.row, cell.col);
                    }
                }
            }, 500); // Increased from 300ms
        }
    }

    handleTouchMove(e) {
        e.preventDefault();
        if (!this.isSwiping) return;
        
        const touch = e.touches[0];
        const cell = this.getCellFromTouch(touch);
        
        if (cell) {
            this.handleSwipe(cell.row, cell.col);
        }
    }

    handleTouchEnd(e) {
        e.preventDefault();
        this.endSwipe();
        clearTimeout(this.touchTimeout);
    }

    getCellFromTouch(touch) {
        const rect = this.gameBoard.getBoundingClientRect();
        const x = touch.clientX - rect.left;
        const y = touch.clientY - rect.top;
        
        const cellSize = rect.width / this.gridSize;
        const row = Math.floor(y / cellSize);
        const col = Math.floor(x / cellSize);
        
        if (row >= 0 && row < this.gridSize && col >= 0 && col < this.gridSize) {
            return { row, col };
        }
        return null;
    }
}

// Prevent right-click context menu on the game board
document.getElementById('game-board').addEventListener('contextmenu', (e) => {
    e.preventDefault();
});

document.addEventListener('DOMContentLoaded', () => {
    const game = new QueensGame();
});
