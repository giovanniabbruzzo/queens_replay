import cv2
import numpy as np
import pygame
import argparse
import time
import json
import os
from typing import List, Tuple, Set, Optional
from enum import Enum
from collections import defaultdict

DEBUG = 1  # Set to 1 to enable debug output, 0 to disable, Set
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')

class CellState(Enum):
    EMPTY = 0
    MARKED = 1
    QUEEN = 2

class Move:
    def __init__(self, row: int, col: int, old_state: CellState, affected_cells: List[Tuple[int, int, CellState]] = None):
        self.row = row
        self.col = col
        self.old_state = old_state
        self.affected_cells = affected_cells or []  # For auto-X moves

class Game:
    def __init__(self, grid_size: int, cell_colors: List[List[Tuple[int, int, int]]], clusters: List[Set[Tuple[int, int]]]):
        self.grid_size = grid_size
        self.cell_colors = cell_colors
        self.clusters = clusters
        self.cell_states = [[CellState.EMPTY for _ in range(grid_size)] for _ in range(grid_size)]
        self.start_time = time.time()
        self.elapsed_time = 0
        self.is_complete = False
        self.move_history: List[Move] = []
        self.auto_x = self.load_settings().get('auto_x', False)  # Load auto-X setting from file
        
    def save_settings(self) -> None:
        """Save game settings to file"""
        settings = {
            'auto_x': self.auto_x
        }
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings, f)
        except Exception as e:
            print(f"Error saving settings: {e}")
            
    def load_settings(self) -> dict:
        """Load game settings from file"""
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading settings: {e}")
        return {}
        
    def clear_board(self) -> None:
        """Clear all cells on the board"""
        self.cell_states = [[CellState.EMPTY for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.is_complete = False
        self.move_history.clear()
        
    def set_cell_state(self, row: int, col: int, state: CellState) -> None:
        """Set a cell's state directly for swiping"""
        if (0 <= row < self.grid_size and 0 <= col < self.grid_size and 
            self.cell_states[row][col] != state and
            state in [CellState.EMPTY, CellState.MARKED]):  # Only allow empty or marked states
            old_state = self.cell_states[row][col]
            self.cell_states[row][col] = state
            # Record the move for undo
            self.move_history.append(Move(row, col, old_state))
        
    def undo_move(self) -> bool:
        """Undo the last move if possible"""
        if not self.move_history:
            return False
            
        move = self.move_history.pop()
        self.cell_states[move.row][move.col] = move.old_state
        
        # Restore affected cells (from auto-X)
        for row, col, state in move.affected_cells:
            self.cell_states[row][col] = state
            
        self.is_complete = False
        return True
        
    def mark_row_col_cells(self, row: int, col: int) -> List[Tuple[int, int, CellState]]:
        """Mark cells in the same row, column, surrounding cells, and cluster with X"""
        affected_cells = []
        
        # Function to mark a cell if it's empty
        def mark_cell(r: int, c: int):
            if (0 <= r < self.grid_size and 0 <= c < self.grid_size and 
                self.cell_states[r][c] == CellState.EMPTY):
                affected_cells.append((r, c, CellState.EMPTY))
                self.cell_states[r][c] = CellState.MARKED
        
        # Mark entire row and column
        for i in range(self.grid_size):
            mark_cell(row, i)  # Mark row
            mark_cell(i, col)  # Mark column
            
        # Mark surrounding cells (including diagonals)
        for r in range(max(0, row-1), min(self.grid_size, row+2)):
            for c in range(max(0, col-1), min(self.grid_size, col+2)):
                if (r != row or c != col):  # Don't mark the queen's cell
                    mark_cell(r, c)
                    
        # Mark all cells in the same cluster
        cluster = self.get_cluster_for_cell(row, col)
        if cluster:
            for r, c in cluster:
                if (r != row or c != col):  # Don't mark the queen's cell
                    mark_cell(r, c)
            
        return affected_cells
        
    def toggle_cell(self, row: int, col: int) -> None:
        """Cycle through cell states: EMPTY -> MARKED -> QUEEN -> EMPTY"""
        current_state = self.cell_states[row][col]
        affected_cells = []
        
        if current_state == CellState.EMPTY:
            new_state = CellState.MARKED
        elif current_state == CellState.MARKED:
            new_state = CellState.QUEEN
            if self.auto_x:
                affected_cells = self.mark_row_col_cells(row, col)
        else:  # QUEEN
            new_state = CellState.EMPTY
            
        # Record the move
        move = Move(row, col, current_state, affected_cells)
        self.move_history.append(move)
        
        # Apply the move
        self.cell_states[row][col] = new_state
    
    def has_adjacent_queens(self, row: int, col: int) -> bool:
        """Check if there are any queens in adjacent cells (including diagonals)"""
        for i in range(max(0, row-1), min(self.grid_size, row+2)):
            for j in range(max(0, col-1), min(self.grid_size, col+2)):
                if (i != row or j != col) and self.cell_states[i][j] == CellState.QUEEN:
                    return True
        return False
    
    def get_cluster_for_cell(self, row: int, col: int) -> Optional[Set[Tuple[int, int]]]:
        """Find which cluster contains the given cell"""
        for cluster in self.clusters:
            if (row, col) in cluster:
                return cluster
        return None
    
    def check_win_condition(self) -> bool:
        """Check if all rules are satisfied"""
        # Check rows
        for i in range(self.grid_size):
            queens_in_row = sum(1 for j in range(self.grid_size) 
                              if self.cell_states[i][j] == CellState.QUEEN)
            if queens_in_row != 1:
                return False
        
        # Check columns
        for j in range(self.grid_size):
            queens_in_col = sum(1 for i in range(self.grid_size) 
                              if self.cell_states[i][j] == CellState.QUEEN)
            if queens_in_col != 1:
                return False
        
        # Check clusters
        for cluster in self.clusters:
            queens_in_cluster = sum(1 for i, j in cluster 
                                  if self.cell_states[i][j] == CellState.QUEEN)
            if queens_in_cluster != 1:
                return False
        
        # Check for adjacent queens
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.cell_states[i][j] == CellState.QUEEN:
                    # Remove current queen temporarily to not count it as adjacent to itself
                    self.cell_states[i][j] = CellState.EMPTY
                    if self.has_adjacent_queens(i, j):
                        self.cell_states[i][j] = CellState.QUEEN
                        return False
                    self.cell_states[i][j] = CellState.QUEEN
        
        return True

def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    # Convert to HSV for better color comparison
    c1_hsv = cv2.cvtColor(np.uint8([[list(c1)[::-1]]]), cv2.COLOR_BGR2HSV)[0][0]
    c2_hsv = cv2.cvtColor(np.uint8([[list(c2)[::-1]]]), cv2.COLOR_BGR2HSV)[0][0]
    
    # Calculate distance in HSV space
    h1, s1, v1 = map(float, c1_hsv)
    h2, s2, v2 = map(float, c2_hsv)
    
    # Check if either color is grayscale (low saturation)
    is_gray1 = s1 < 30
    is_gray2 = s2 < 30
    
    # For grayscale colors, use primarily value-based comparison
    if is_gray1 and is_gray2:
        v_diff = abs(v1 - v2)
        # Be very strict with gray values
        if v_diff > 30:  # Reduced from 60
            return 1.0
        return v_diff / 255.0
    
    # If one is gray and one isn't, they're different
    if is_gray1 != is_gray2:
        return 1.0
    
    # Handle hue wraparound
    h_diff = min(abs(h1 - h2), 180.0 - abs(h1 - h2))
    
    # Special handling for yellow-green range (hue values around 40-80)
    if (40 <= h1 <= 80 and 40 <= h2 <= 80):
        if h_diff > 20:  # Allow more hue difference for yellow-green
            return 1.0
    else:
        if h_diff > 10:  # Strict for other colors
            return 1.0
        
    # Calculate saturation and value differences
    s_diff = abs(s1 - s2)
    v_diff = abs(v1 - v2)
    
    # More strict thresholds for all colors
    if s_diff > 50:  # Reduced from 60
        return 1.0
    if v_diff > 50:  # Reduced from 60
        return 1.0
    
    # Weighted average based on color properties
    if (40 <= h1 <= 80 and 40 <= h2 <= 80):  # Yellow-green
        h_weight = 0.4
        s_weight = 0.4
        v_weight = 0.2
    else:
        h_weight = 0.5
        s_weight = 0.3
        v_weight = 0.2
    
    # Normalize differences to [0,1]
    h_diff = h_diff / 180.0
    s_diff = s_diff / 255.0
    v_diff = v_diff / 255.0
    
    return h_diff * h_weight + s_diff * s_weight + v_diff * v_weight

def find_color_clusters(cell_colors: List[List[Tuple[int, int, int]]], similarity_threshold: float = 0.08) -> List[Set[Tuple[int, int]]]:
    grid_size = len(cell_colors)
    visited = set()
    clusters = []
    
    def get_neighbors(i: int, j: int) -> List[Tuple[int, int]]:
        neighbors = []
        # Only consider direct neighbors (no diagonals)
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < grid_size and 0 <= nj < grid_size:
                # Get color distance
                dist = color_distance(cell_colors[i][j], cell_colors[ni][nj])
                # Check if colors are similar enough
                if dist < similarity_threshold:
                    neighbors.append((ni, nj))
        return neighbors
    
    def find_cluster(i: int, j: int) -> Set[Tuple[int, int]]:
        cluster = {(i, j)}
        stack = [(i, j)]
        while stack:
            curr_i, curr_j = stack.pop()
            for ni, nj in get_neighbors(curr_i, curr_j):
                if (ni, nj) not in cluster:
                    cluster.add((ni, nj))
                    stack.append((ni, nj))
        return cluster
    
    # Find all clusters
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) not in visited:
                cluster = find_cluster(i, j)
                clusters.append(cluster)
                visited.update(cluster)
    
    return clusters

def normalize_colors(cell_colors: List[List[Tuple[int, int, int]]]) -> List[List[Tuple[int, int, int]]]:
    grid_size = len(cell_colors)
    normalized = [[cell_colors[i][j] for j in range(grid_size)] for i in range(grid_size)]
    
    # First pass: Find initial clusters with strict threshold
    initial_clusters = find_color_clusters(cell_colors, similarity_threshold=0.12)
    
    def merge_clusters(clusters_with_colors, merge_threshold, require_adjacent=True):
        merged = [False] * len(clusters_with_colors)
        result_clusters = []
        
        for i, (color1, cluster1) in enumerate(clusters_with_colors):
            if merged[i]:
                continue
                
            merged[i] = True
            current_cluster = set(cluster1)
            current_color = np.array(color1)
            changed = True
            
            # Keep trying to merge until no more merges are possible
            while changed:
                changed = False
                for j, (color2, cluster2) in enumerate(clusters_with_colors):
                    if merged[j]:
                        continue
                        
                    # Check if clusters should be merged
                    should_merge = color_distance(tuple(map(int, current_color)), color2) < merge_threshold
                    
                    if require_adjacent and should_merge:
                        # Check if clusters are adjacent
                        adjacent = False
                        for cell1 in current_cluster:
                            if adjacent:
                                break
                            for cell2 in cluster2:
                                if abs(cell1[0] - cell2[0]) <= 1 and abs(cell1[1] - cell2[1]) <= 1:
                                    adjacent = True
                                    break
                        should_merge = adjacent
                    
                    if should_merge:
                        current_cluster.update(cluster2)
                        # Update color as weighted average
                        w1 = len(current_cluster)
                        w2 = len(cluster2)
                        current_color = (current_color * w1 + np.array(color2) * w2) / (w1 + w2)
                        merged[j] = True
                        changed = True
            
            result_clusters.append(current_cluster)
        return result_clusters
    
    # Calculate initial cluster colors
    cluster_colors = []
    for cluster in initial_clusters:
        avg_color = tuple(map(int, np.mean([cell_colors[i][j] for i, j in cluster], axis=0)))
        cluster_colors.append((avg_color, cluster))
    
    # Second pass: Merge adjacent similar clusters
    clusters = merge_clusters(cluster_colors, merge_threshold=0.2, require_adjacent=True)
    
    # Third pass: Merge very similar clusters even if not adjacent
    cluster_colors = []
    for cluster in clusters:
        avg_color = tuple(map(int, np.mean([cell_colors[i][j] for i, j in cluster], axis=0)))
        cluster_colors.append((avg_color, cluster))
    
    final_clusters = merge_clusters(cluster_colors, merge_threshold=0.15, require_adjacent=False)
    
    # Apply final colors
    for cluster in final_clusters:
        avg_color = tuple(map(int, np.mean([cell_colors[i][j] for i, j in cluster], axis=0)))
        for i, j in cluster:
            normalized[i][j] = avg_color
    
    return normalized, final_clusters

def get_dominant_color(img: np.ndarray, x: int, y: int, w: int, h: int) -> Tuple[int, int, int]:
    # Extract the cell
    cell = img[y:y+h, x:x+w]
    
    # Convert to HSV for better color segmentation
    cell_hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    
    # Create masks for different elements:
    
    # 1. Mask for the queen crown (yellow/golden color)
    # Yellow/golden in HSV space
    queen_mask = cv2.inRange(cell_hsv,
                           np.array([20, 100, 100]),  # Golden yellow lower bound
                           np.array([40, 255, 255]))  # Golden yellow upper bound
    
    # 2. Mask for very dark pixels (borders)
    dark_mask = cv2.inRange(cell_hsv,
                          np.array([0, 0, 0]),
                          np.array([180, 255, 30]))
    
    # 3. Mask for very bright pixels (reflections/highlights)
    bright_mask = cv2.inRange(cell_hsv,
                            np.array([0, 0, 225]),
                            np.array([180, 30, 255]))
    
    # Combine all masks to get pixels to exclude
    exclude_mask = cv2.bitwise_or(queen_mask, dark_mask)
    exclude_mask = cv2.bitwise_or(exclude_mask, bright_mask)
    
    # Invert mask to get valid pixels
    valid_mask = cv2.bitwise_not(exclude_mask)
    
    # Convert to RGB for k-means
    cell_rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
    
    # Apply mask to RGB image
    masked_cell = cv2.bitwise_and(cell_rgb, cell_rgb, mask=valid_mask)
    
    # Get only the valid pixels
    valid_pixels = masked_cell[valid_mask > 0]
    
    if len(valid_pixels) == 0:
        # If all pixels were masked, try without excluding the queen
        # Just exclude very dark/bright pixels
        valid_mask = cv2.bitwise_not(cv2.bitwise_or(dark_mask, bright_mask))
        masked_cell = cv2.bitwise_and(cell_rgb, cell_rgb, mask=valid_mask)
        valid_pixels = masked_cell[valid_mask > 0]
        
        if len(valid_pixels) == 0:
            # If still no valid pixels, return white
            return (255, 255, 255)
        
    # Reshape to a list of RGB pixels
    pixels = valid_pixels.reshape(-1, 3)
    
    # Convert to float32 for k-means
    pixels = np.float32(pixels)
    
    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    # Determine number of clusters based on available pixels
    n_pixels = len(pixels)
    k = min(3, max(1, n_pixels))  # Use fewer clusters since we've removed queen pixels
    
    # Apply k-means clustering
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, flags)
    
    # Get the counts of each cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Sort by frequency
    sorted_indices = np.argsort(-counts)
    
    # Get the most common color after filtering
    dominant_color = centers[sorted_indices[0]]
    return tuple(map(int, dominant_color))

def draw_clean_grid(grid_size: int, cell_colors: List[List[Tuple[int, int, int]]], 
                  cell_size: int = 50, clusters: List[Set[Tuple[int, int]]] = None) -> np.ndarray:
    # Create a blank image
    inner_border_size = 2
    outer_border_size = 4
    total_size = (grid_size * cell_size + 
                 (grid_size - 1) * inner_border_size + 
                 2 * outer_border_size)
    grid_img = np.zeros((total_size, total_size, 3), dtype=np.uint8)
    
    # Fill with black (for borders)
    grid_img.fill(0)
    
    # Draw colored cells
    for i in range(grid_size):
        for j in range(grid_size):
            color = cell_colors[i][j]
            # Calculate position with borders
            x1 = (j * (cell_size + inner_border_size)) + outer_border_size
            y1 = (i * (cell_size + inner_border_size)) + outer_border_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            
            # Fill cell with its color
            grid_img[y1:y2, x1:x2] = color[::-1]  # Convert RGB to BGR
    
    # If clusters are provided and DEBUG is enabled, draw cluster numbers
    if DEBUG and clusters is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        # Draw cluster numbers
        for cluster_idx, cluster in enumerate(clusters):
            # Get center cell of cluster for label placement
            center_i, center_j = list(cluster)[len(cluster)//2]
            
            # Calculate text position
            x1 = (center_j * (cell_size + inner_border_size)) + outer_border_size
            y1 = (center_i * (cell_size + inner_border_size)) + outer_border_size
            text_x = x1 + cell_size // 4
            text_y = y1 + cell_size // 2
            
            # Draw white background for better visibility
            text = str(cluster_idx + 1)
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            cv2.rectangle(grid_img, 
                         (text_x - 2, text_y - text_height - 2),
                         (text_x + text_width + 2, text_y + 2),
                         (255, 255, 255), -1)
            
            # Draw cluster number
            cv2.putText(grid_img, text, (text_x, text_y),
                       font, font_scale, (0, 0, 0), font_thickness)
    
    return grid_img

def parse_args():
    parser = argparse.ArgumentParser(description='Process Queens game screenshot')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('-d','--debug', help='Enable debug output', default=False, action='store_true')
    return parser.parse_args()

def detect_grid(image_path: str, debug: bool = False) -> Tuple[int, np.ndarray, List[List[Tuple[int, int, int]]], List[Set[Tuple[int, int]]]]:
    # Set debug flag
    global DEBUG
    DEBUG = 1 if debug else 0
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to ensure grid lines are connected
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours in the edge image
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (should be the grid)
    grid_contour = max(contours, key=cv2.contourArea)
    grid_x, grid_y, grid_w, grid_h = cv2.boundingRect(grid_contour)
    
    # Create a mask for the grid area
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [grid_contour], -1, (255, 255, 255), -1)
    
    # Apply the mask to the original image
    masked = cv2.bitwise_and(gray, mask)
    
    # Save debug images
    if DEBUG:
        cv2.imwrite('debug_edges.png', edges)
        cv2.imwrite('debug_dilated.png', dilated)
        cv2.imwrite('debug_masked.png', masked)
    
    # Crop to the grid area
    cropped = img[grid_y:grid_y+grid_h, grid_x:grid_x+grid_w]
    cropped_gray = gray[grid_y:grid_y+grid_h, grid_x:grid_x+grid_w]
    
    # Apply thresholding to the cropped image to find cells
    _, binary = cv2.threshold(cropped_gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Find contours of cells
    cell_contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and sort cell contours
    cells = []
    min_area = (grid_w * grid_h) / (15 * 15)  # Assuming grid is at most 15x15
    max_area = (grid_w * grid_h) / (5 * 5)    # Assuming grid is at least 5x5
    
    for contour in cell_contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            # Check if it's roughly square
            if 0.8 < w/h < 1.2:
                cells.append((x, y, w, h))
    
    # Get unique x and y coordinates (with some tolerance)
    def get_unique_coords(coords, tolerance=10):
        unique = []
        for coord in sorted(coords):
            if not unique or abs(coord - unique[-1]) > tolerance:
                unique.append(coord)
        return unique
    
    x_coords = get_unique_coords([x for x, _, _, _ in cells])
    y_coords = get_unique_coords([y for _, y, _, _ in cells])
    
    grid_size = max(len(x_coords), len(y_coords))
    
    # Create a grid to store cell colors
    cell_colors = [[None for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Calculate average cell size
    avg_cell_width = grid_w / grid_size
    avg_cell_height = grid_h / grid_size
    
    # For each grid position, find the dominant color
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate the expected position of this cell
            x = int(j * avg_cell_width)
            y = int(i * avg_cell_height)
            w = int(avg_cell_width)
            h = int(avg_cell_height)
            
            # Get the dominant color for this cell
            color = get_dominant_color(cropped, x, y, w, h)
            cell_colors[i][j] = color
    
    # Create clean grid image with normalized colors
    normalized_colors, clusters = normalize_colors(cell_colors)
    
    if DEBUG:
        print(f"Number of color clusters detected: {len(clusters)}")
        # Create debug grid with cluster numbers
        debug_grid = draw_clean_grid(grid_size, normalized_colors, clusters=clusters)
        cv2.imwrite('debug_clusters.png', debug_grid)
    
    # Create clean grid without cluster numbers
    clean_grid = draw_clean_grid(grid_size, normalized_colors, clusters=None)
    cv2.imwrite('clean_grid.png', clean_grid)
    
    return grid_size, clean_grid, normalized_colors, clusters

class GameUI:
    def __init__(self, game: Game):
        pygame.init()
        self.game = game
        self.cell_size = 60
        self.border_size = 2
        self.screen_size = (game.grid_size * (self.cell_size + self.border_size) + self.border_size,
                          game.grid_size * (self.cell_size + self.border_size) + self.border_size + 40)  # Extra height for timer
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Queens Puzzle")
        self.font = pygame.font.Font(None, 36)
        
        # Swipe tracking
        self.is_swiping = False
        self.swipe_state = None  # Will be set to MARKED or EMPTY when swiping
        self.last_swiped_cell = None
        
        # Double click tracking
        self.last_click_time = 0
        self.last_click_pos = None
        self.double_click_time = 500  # Maximum milliseconds between clicks for double click
        
        # Popup dimensions
        self.popup_width = 400
        self.popup_height = 200
        self.popup_x = (self.screen_size[0] - self.popup_width) // 2
        self.popup_y = (self.screen_size[1] - self.popup_height) // 2
        self.show_popup = True  # Control whether to show the popup when game is complete
        
        # Button dimensions
        self.button_width = 100
        self.button_height = 30
        self.button_spacing = 10
        
        # Position buttons
        total_buttons_width = 3 * self.button_width + 2 * self.button_spacing
        self.buttons_start_x = self.screen_size[0] - total_buttons_width - self.button_spacing
        self.buttons_y = self.screen_size[1] - 35  # Same y as timer text
        
        # Define button positions
        self.clear_button_x = self.buttons_start_x
        self.undo_button_x = self.clear_button_x + self.button_width + self.button_spacing
        self.auto_x_button_x = self.undo_button_x + self.button_width + self.button_spacing
        
        # Load and scale crown image
        crown_surface = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.polygon(crown_surface, (255, 215, 0), [
            (20, 0),    # Top point
            (35, 20),   # Right point
            (40, 15),   # Right crown
            (30, 35),   # Right base
            (10, 35),   # Left base
            (0, 15),    # Left crown
            (5, 20),    # Left point
        ])
        self.crown_image = crown_surface
        
        # Load and scale X mark
        x_surface = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.line(x_surface, (255, 0, 0), (5, 5), (35, 35), 3)
        pygame.draw.line(x_surface, (255, 0, 0), (5, 35), (35, 5), 3)
        self.x_mark = x_surface
    
    def draw(self) -> None:
        self.screen.fill((0, 0, 0))  # Black background
        
        # Draw grid
        for i in range(self.game.grid_size):
            for j in range(self.game.grid_size):
                x = j * (self.cell_size + self.border_size) + self.border_size
                y = i * (self.cell_size + self.border_size) + self.border_size
                
                # Draw cell with its color
                color = self.game.cell_colors[i][j]
                pygame.draw.rect(self.screen, color, 
                               (x, y, self.cell_size, self.cell_size))
                
                # Draw cell state
                if self.game.cell_states[i][j] == CellState.QUEEN:
                    # Center the crown in the cell
                    crown_x = x + (self.cell_size - self.crown_image.get_width()) // 2
                    crown_y = y + (self.cell_size - self.crown_image.get_height()) // 2
                    self.screen.blit(self.crown_image, (crown_x, crown_y))
                elif self.game.cell_states[i][j] == CellState.MARKED:
                    # Center the X mark in the cell
                    x_x = x + (self.cell_size - self.x_mark.get_width()) // 2
                    x_y = y + (self.cell_size - self.x_mark.get_height()) // 2
                    self.screen.blit(self.x_mark, (x_x, x_y))
        
        # Draw timer
        if not self.game.is_complete:
            self.game.elapsed_time = time.time() - self.game.start_time
        timer_text = f"Time: {int(self.game.elapsed_time)}s"
        timer_surface = self.font.render(timer_text, True, (255, 255, 255))
        self.screen.blit(timer_surface, (10, self.screen_size[1] - 35))
        
        # Draw completion popup if game is complete and popup should be shown
        if self.game.is_complete and self.show_popup:
            self.draw_completion_popup()
        
        # Draw Clear button
        pygame.draw.rect(self.screen, (100, 100, 100), 
                        (self.clear_button_x, self.buttons_y, self.button_width, self.button_height))
        clear_text = self.font.render("Clear", True, (255, 255, 255))
        text_x = self.clear_button_x + (self.button_width - clear_text.get_width()) // 2
        text_y = self.buttons_y + (self.button_height - clear_text.get_height()) // 2
        self.screen.blit(clear_text, (text_x, text_y))
        
        # Draw Undo button
        pygame.draw.rect(self.screen, (100, 100, 100), 
                        (self.undo_button_x, self.buttons_y, self.button_width, self.button_height))
        undo_text = self.font.render("Undo", True, (255, 255, 255))
        text_x = self.undo_button_x + (self.button_width - undo_text.get_width()) // 2
        text_y = self.buttons_y + (self.button_height - undo_text.get_height()) // 2
        self.screen.blit(undo_text, (text_x, text_y))
        
        # Draw Auto-X toggle button
        button_color = (0, 150, 0) if self.game.auto_x else (100, 100, 100)
        pygame.draw.rect(self.screen, button_color,
                        (self.auto_x_button_x, self.buttons_y, self.button_width, self.button_height))
        auto_x_text = self.font.render("Auto-X", True, (255, 255, 255))
        text_x = self.auto_x_button_x + (self.button_width - auto_x_text.get_width()) // 2
        text_y = self.buttons_y + (self.button_height - auto_x_text.get_height()) // 2
        self.screen.blit(auto_x_text, (text_x, text_y))
        
        pygame.display.flip()
    
    def get_cell_from_pos(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        x, y = pos
        # Ignore clicks in the timer/button area
        if y >= self.game.grid_size * (self.cell_size + self.border_size) + self.border_size:
            return None
        
        # Calculate grid coordinates
        row = y // (self.cell_size + self.border_size)
        col = x // (self.cell_size + self.border_size)
        
        # Check if click was within grid bounds
        if 0 <= row < self.game.grid_size and 0 <= col < self.game.grid_size:
            return (row, col)
        return None
        
    def start_swipe(self, cell: Tuple[int, int]) -> None:
        """Start a swipe operation"""
        row, col = cell
        current_state = self.game.cell_states[row][col]
        # Only start swiping if the cell is empty or marked
        if current_state in [CellState.EMPTY, CellState.MARKED]:
            self.is_swiping = True
            # Set the swipe state to the opposite of the current cell state
            self.swipe_state = (CellState.MARKED if current_state == CellState.EMPTY 
                              else CellState.EMPTY)
            self.last_swiped_cell = cell
            # Apply the state change to the first cell
            self.game.set_cell_state(row, col, self.swipe_state)
            
    def handle_swipe(self, cell: Tuple[int, int]) -> None:
        """Handle swipe movement"""
        if not self.is_swiping or cell == self.last_swiped_cell:
            return
            
        row, col = cell
        current_state = self.game.cell_states[row][col]
        # Only affect empty or marked cells
        if current_state in [CellState.EMPTY, CellState.MARKED]:
            self.game.set_cell_state(row, col, self.swipe_state)
            self.last_swiped_cell = cell
            
    def end_swipe(self) -> None:
        """End the swipe operation"""
        self.is_swiping = False
        self.swipe_state = None
        self.last_swiped_cell = None
        
    def is_button_click(self, pos: Tuple[int, int], button_x: int) -> bool:
        """Check if the given position is within a button"""
        x, y = pos
        return (button_x <= x <= button_x + self.button_width and 
                self.buttons_y <= y <= self.buttons_y + self.button_height)
    
    def is_clear_button_click(self, pos: Tuple[int, int]) -> bool:
        """Check if the given position is within the clear button"""
        return self.is_button_click(pos, self.clear_button_x)
        
    def is_undo_button_click(self, pos: Tuple[int, int]) -> bool:
        """Check if the given position is within the undo button"""
        return self.is_button_click(pos, self.undo_button_x)
        
    def is_auto_x_button_click(self, pos: Tuple[int, int]) -> bool:
        """Check if the given position is within the auto-x button"""
        return self.is_button_click(pos, self.auto_x_button_x)
        
    def draw_completion_popup(self) -> None:
        """Draw the completion popup with congratulations message and time"""
        # Create a semi-transparent overlay
        overlay = pygame.Surface(self.screen_size)
        overlay.fill((0, 0, 0))
        overlay.set_alpha(128)
        self.screen.blit(overlay, (0, 0))
        
        # Create the popup rectangle
        popup_width = 400
        popup_height = 200
        popup_x = (self.screen_size[0] - popup_width) // 2
        popup_y = (self.screen_size[1] - popup_height) // 2
        
        # Draw popup background
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (popup_x, popup_y, popup_width, popup_height))
        pygame.draw.rect(self.screen, (0, 0, 0),
                        (popup_x, popup_y, popup_width, popup_height), 2)
        
        # Draw congratulations text
        congrats_font = pygame.font.Font(None, 48)
        time_font = pygame.font.Font(None, 36)
        
        congrats_text = "Congratulations!"
        time_text = f"Time: {int(self.game.elapsed_time)}s"
        
        # Render text
        congrats_surface = congrats_font.render(congrats_text, True, (0, 0, 0))
        time_surface = time_font.render(time_text, True, (0, 0, 0))
        
        # Position text
        congrats_x = popup_x + (popup_width - congrats_surface.get_width()) // 2
        congrats_y = popup_y + 50
        time_x = popup_x + (popup_width - time_surface.get_width()) // 2
        time_y = popup_y + 120
        
        # Draw text
        self.screen.blit(congrats_surface, (congrats_x, congrats_y))
        self.screen.blit(time_surface, (time_x, time_y))

def main():
    args = parse_args()
    try:
        # Get grid info from the image
        grid_size, clean_grid, normalized_colors, clusters = detect_grid(args.image, args.debug)
        print(f"Detected grid size: {grid_size}x{grid_size}")
        
        # Initialize game
        game = Game(grid_size, normalized_colors, clusters)
        ui = GameUI(game)
        
        # Game loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if game.is_complete and ui.show_popup:
                        # Check if click is inside popup
                        x, y = event.pos
                        if (ui.popup_x <= x <= ui.popup_x + ui.popup_width and
                            ui.popup_y <= y <= ui.popup_y + ui.popup_height):
                            ui.show_popup = False  # Dismiss popup
                    elif ui.is_clear_button_click(event.pos):
                        game.clear_board()
                        ui.show_popup = True  # Reset popup state
                    elif ui.is_undo_button_click(event.pos):
                        game.undo_move()
                    elif ui.is_auto_x_button_click(event.pos):
                        game.auto_x = not game.auto_x
                        game.save_settings()  # Save the setting when changed
                    elif not game.is_complete:
                        cell = ui.get_cell_from_pos(event.pos)
                        if cell is not None:
                            if event.button == 1:  # Left mouse button
                                current_time = pygame.time.get_ticks()
                                # Check for double click
                                if (ui.last_click_pos == cell and 
                                    current_time - ui.last_click_time < ui.double_click_time):
                                    # Double click detected - place queen
                                    old_state = game.cell_states[cell[0]][cell[1]]
                                    game.cell_states[cell[0]][cell[1]] = CellState.QUEEN
                                    # Add to move history
                                    affected_cells = []
                                    if game.auto_x:
                                        affected_cells = game.mark_row_col_cells(cell[0], cell[1])
                                    game.move_history.append(Move(cell[0], cell[1], old_state, affected_cells))
                                    if game.check_win_condition():
                                        game.is_complete = True
                                    # Reset double click tracking
                                    ui.last_click_time = 0
                                    ui.last_click_pos = None
                                else:
                                    # Single click - update tracking
                                    ui.last_click_time = current_time
                                    ui.last_click_pos = cell
                                    # Handle normal click
                                    if game.cell_states[cell[0]][cell[1]] == CellState.QUEEN:
                                        game.toggle_cell(cell[0], cell[1])
                                        if game.check_win_condition():
                                            game.is_complete = True
                                    else:
                                        ui.start_swipe(cell)
                            elif event.button == 3:  # Right mouse button
                                game.toggle_cell(cell[0], cell[1])
                                if game.check_win_condition():
                                    game.is_complete = True
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    ui.end_swipe()
                    
                elif event.type == pygame.MOUSEMOTION:
                    if not game.is_complete and ui.is_swiping:
                        cell = ui.get_cell_from_pos(event.pos)
                        if cell is not None:
                            ui.handle_swipe(cell)
            
            ui.draw()
        
        pygame.quit()
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise  # Re-raise the exception to see the full traceback

if __name__ == "__main__":
    main()
