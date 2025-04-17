import os
import json
import torch
import random
import numpy as np
import re
from tqdm import tqdm
import traceback
import time

# --- Import necessary components from train_transformer ---
# Assuming train_transformer.py is in the same directory or accessible
try:
    from train_transformer import TakTransformer, board_state_to_tensor, decode_move_vector
except ImportError:
    print("Error: Could not import from train_transformer.py.")
    print("Make sure train_transformer.py is in the Python path.")
    # Define dummy classes/functions if import fails, to allow script structure to load
    class TakTransformer: pass
    def board_state_to_tensor(*args, **kwargs): return torch.zeros(1, 14, 5, 5) # Dummy tensor
    def decode_move_vector(*args, **kwargs): return "a1" # Dummy move

# --- Tak Game Engine (Copied from self_play_w_board.py for self-containment) ---
# Consider moving this to a separate tak_engine.py file for better organization
class TakGameEngine:
    """Implements Tak game rules for self-play"""

    def __init__(self, board_size=5):
        self.board_size = board_size
        self.reset()

        # Define piece types
        self.FLAT = 0
        self.WALL = 1
        self.CAPSTONE = 2

        # Define directions
        self.DIRECTIONS = {
            '+': (0, 1),   # North (Up row index)
            '-': (0, -1),  # South (Down row index)
            '>': (1, 0),   # East (Up col index)
            '<': (-1, 0)   # West (Down col index)
        }
        # Define squares using a1 notation (a1 is bottom-left, index (0,0))
        self.square_to_coords = {}
        self.coords_to_square = {}
        for r in range(board_size): # row index 0 to N-1
            for c in range(board_size): # col index 0 to N-1
                square = f"{chr(ord('a') + c)}{r + 1}"
                self.square_to_coords[square] = (r, c)
                self.coords_to_square[(r, c)] = square

    def reset(self):
        """Reset the game to initial state"""
        # Board representation: list of lists, each cell is a list (stack) of (player, piece_type) tuples
        self.board = [[[] for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = 1 # Player 1 starts
        self.move_count = 0
        self.game_over = False
        self.winner = None # 0 for draw, 1 or 2 for winner

        # Initialize piece counts (adjust based on board size)
        if self.board_size == 5:
            flats = 21
            caps = 1
        elif self.board_size == 4:
            flats = 15
            caps = 0 # No capstones in 4x4 standard
        elif self.board_size == 6:
            flats = 30
            caps = 1 # Or 2 depending on ruleset
        else: # Default/fallback
            flats = (self.board_size**2) # Rough estimate
            caps = 1
            print(f"Warning: Using default piece counts for board size {self.board_size}")

        self.pieces = {
            1: {'F': flats, 'C': caps},
            2: {'F': flats, 'C': caps}
        }
        self.move_history = []

    def get_board_state_dict(self):
        """Convert current board state to the dictionary format used for tensor conversion."""
        board_squares_list = []
        # Iterate in the order expected by board_state_to_tensor (a5..e5, a4..e4, ..., a1..e1)
        for r_from_top in range(self.board_size - 1, -1, -1): # 4 down to 0 for 5x5
            for c in range(self.board_size): # 0 to 4 for 5x5
                # Convert tensor row/col back to engine row/col (r=0 is bottom row)
                r = r_from_top # Engine uses r=0 as bottom row
                stack_list = self.board[r][c]
                square_data = {"stack": []}
                if stack_list:
                    # Map player (1, 2) to (0, 1) and piece type (0,1,2) to ('F','S','C')
                    piece_map_rev = {self.FLAT: 'F', self.WALL: 'S', self.CAPSTONE: 'C'}
                    for player, piece_type in stack_list:
                        square_data["stack"].append({
                            "player": player - 1, # Convert 1/2 to 0/1
                            "type": piece_map_rev.get(piece_type, '?')
                        })
                board_squares_list.append(square_data)

        board_state = {
            "board": board_squares_list,
            "board_size": self.board_size,
            "player1_flats": self.pieces[1]['F'],
            "player1_capstones": self.pieces[1]['C'],
            "player2_flats": self.pieces[2]['F'],
            "player2_capstones": self.pieces[2]['C'],
            "turn": self.current_player - 1, # Convert 1/2 to 0/1
        }
        return board_state

    def count_flats(self, player):
        """Count flat stones on the board for a player (top of stack only)."""
        count = 0
        for r in range(self.board_size):
            for c in range(self.board_size):
                stack = self.board[r][c]
                if stack and stack[-1][0] == player and stack[-1][1] == self.FLAT:
                    count += 1
        return count

    def get_valid_moves(self):
        """Get all valid moves for the current player in PTN format."""
        valid_moves = []
        player = self.current_player

        # --- First Turn Placement Rules ---
        if self.move_count < 2:
            opponent_piece = 3 - player # The other player
            piece_type = self.FLAT # Must place opponent's flat
            # Check if player has flats left (should always be true initially)
            # Note: PTN doesn't specify piece type for first two moves
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if not self.board[r][c]: # If square is empty
                        square = self.coords_to_square.get((r, c))
                        if square:
                            valid_moves.append(square)
            return valid_moves

        # --- Regular Move Rules ---

        # 1. Place a stone (Flat, Wall, or Capstone)
        can_place_flat = self.pieces[player]['F'] > 0
        can_place_cap = self.pieces[player]['C'] > 0

        if can_place_flat or can_place_cap:
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if not self.board[r][c]: # Empty square
                        square = self.coords_to_square.get((r, c))
                        if square:
                            if can_place_flat:
                                valid_moves.append(f"{square}") # Default place is Flat
                                valid_moves.append(f"F{square}")
                                valid_moves.append(f"S{square}") # Place Wall
                            if can_place_cap:
                                valid_moves.append(f"C{square}") # Place Capstone

        # 2. Move a stack
        for r in range(self.board_size):
            for c in range(self.board_size):
                stack = self.board[r][c]
                if stack and stack[-1][0] == player: # Stack controlled by player
                    square = self.coords_to_square.get((r, c))
                    if not square: continue # Should not happen

                    stack_height = len(stack)
                    max_carry = min(stack_height, self.board_size) # Carry limit is board size

                    for carry_count in range(1, max_carry + 1):
                        for direction_symbol, (dc, dr) in self.DIRECTIONS.items(): # dc, dr based on engine coords
                            # Check validity of moving in this direction with this carry count
                            # This requires simulating the drops, which is complex.
                            # We need to check each possible drop sequence.
                            # Example: carry=3, direction=+
                            # Possible drops: [3], [1,2], [2,1], [1,1,1]
                            # Let's generate all valid drop sequences for this move start
                            possible_drop_sequences = self._generate_drop_sequences(carry_count)

                            for drops in possible_drop_sequences:
                                is_valid_stack_move, _ = self._check_stack_move(r, c, dr, dc, carry_count, drops, stack)
                                if is_valid_stack_move:
                                    # Format the move string
                                    move_str = f"{carry_count if carry_count > 1 else ''}{square}{direction_symbol}{''.join(map(str, drops))}"
                                    valid_moves.append(move_str)

        return list(set(valid_moves)) # Return unique moves

    def _generate_drop_sequences(self, carry_count):
        """Generates all possible valid drop sequences for a given carry count."""
        if carry_count == 0:
            return [[]]
        if carry_count == 1:
            return [[1]]

        sequences = []
        # Generate partitions of carry_count
        def get_partitions(target, current_partition=[]):
            s = sum(current_partition)
            if s == target:
                sequences.append(current_partition)
                return
            if s > target:
                return

            # Next number can be from 1 up to target - s
            # And must be <= the last number added if partition exists (optional constraint for standard partitions)
            start = 1 # Allow any drop amount >= 1
            # start = current_partition[-1] if current_partition else 1 # Use this for non-decreasing partitions

            for i in range(start, target - s + 1):
                 get_partitions(target, current_partition + [i])

        get_partitions(carry_count)
        # Filter out sequences longer than board_size (though technically handled by _check_stack_move)
        # sequences = [seq for seq in sequences if len(seq) <= self.board_size]
        return sequences


    def _check_stack_move(self, r_start, c_start, dr, dc, carry_count, drops, original_stack):
        """
        Checks if a specific stack move (r, c, direction, carry, drops) is valid.
        Does NOT modify the board state. Returns (bool: is_valid, str: error_reason).
        """
        current_r, current_c = r_start, c_start
        top_piece_type = original_stack[-1][1] if original_stack else None
        # print(f"DEBUG _check_stack_move: Start at ({r_start},{c_start}), dir ({dr},{dc}), drops {drops}") # Add start debug

        for i, drop_count in enumerate(drops):
            next_r, next_c = current_r + dr, current_c + dc
            # print(f"DEBUG _check_stack_move: Drop {i+1}, checking dest ({next_r},{next_c})") # Add destination debug

            # 1. Check Bounds
            # print(f"DEBUG _check_stack_move: Checking bounds: 0 <= {next_r} < {self.board_size} and 0 <= {next_c} < {self.board_size}") # Add bounds values debug
            row_in_bounds = (0 <= next_r < self.board_size)
            col_in_bounds = (0 <= next_c < self.board_size)
            # print(f"DEBUG _check_stack_move: Row Check ({next_r}): {row_in_bounds}, Col Check ({next_c}): {col_in_bounds}") # Detailed check results
            if not (row_in_bounds and col_in_bounds):
                # print(f"DEBUG _check_stack_move: Bounds check FAILED for ({next_r},{next_c})") # Add failure confirmation
                return False, "Out of bounds"
            # print(f"DEBUG _check_stack_move: Bounds check PASSED for ({next_r},{next_c})") # Add success confirmation


            # 2. Check Destination Square Obstructions
            # print(f"DEBUG _check_stack_move: Checking obstructions at ({next_r},{next_c})") # Add obstruction check debug
            dest_stack = self.board[next_r][next_c]
            if dest_stack:
                dest_top_type = dest_stack[-1][1]
                # Cannot move onto a Capstone
                if dest_top_type == self.CAPSTONE:
                    # print(f"DEBUG _check_stack_move: Obstruction FAILED - Capstone at ({next_r},{next_c})") # Add capstone fail debug
                    return False, "Cannot move onto Capstone"
                # Cannot move onto a Wall UNLESS carrying only a Capstone and it's the last drop
                if dest_top_type == self.WALL:
                    is_last_drop = (i == len(drops) - 1)
                    # Simplified check: is the top piece of the *original* stack a capstone?
                    # A more robust check would track the piece being dropped.
                    is_carrying_capstone = (top_piece_type == self.CAPSTONE)

                    # A single capstone drop can flatten a wall.
                    can_flatten = (is_carrying_capstone and drop_count == 1 and is_last_drop)
                    # print(f"DEBUG _check_stack_move: Wall at ({next_r},{next_c}). Can flatten: {can_flatten} (is_cap={is_carrying_capstone}, drop_count={drop_count}, is_last={is_last_drop})") # Add wall check debug

                    if not can_flatten:
                        # print(f"DEBUG _check_stack_move: Obstruction FAILED - Blocked by Wall at ({next_r},{next_c})") # Add wall fail debug
                        return False, "Blocked by Wall"
            # else:
                # print(f"DEBUG _check_stack_move: Obstruction check PASSED - Square ({next_r},{next_c}) is empty or has flat top") # Add obstruction pass debug


            # If checks pass, update current position for the next drop check
            current_r, current_c = next_r, next_c
            # print(f"DEBUG _check_stack_move: Updated position to ({current_r},{current_c})") # Add position update debug


        # If all drops are valid according to rules
        # print(f"DEBUG _check_stack_move: All checks passed.") # Add final success debug
        return True, ""


    def apply_move(self, move):
        """Apply a move string (PTN) to the board state."""
        if self.game_over: return False
        player = self.current_player
        # print(f"Attempting move: '{move}' for Player {player}") # Basic debug
        # piece_type = self.FLAT # REMOVE THIS LINE - Unnecessary and causes bug

        # --- Parse Move ---
        parsed_info = self._parse_ptn_move(move)
        if not parsed_info:
            print(f"DEBUG: Failed to parse move '{move}'") # Debug parse failure
            return False # Invalid format

        move_type = parsed_info['type']
        r, c = parsed_info['coords']
        # print(f"DEBUG: Parsed move: {parsed_info}") # Debug parsed info

        # --- Apply Placement ---
        if move_type == 'place':
            # Get the correctly parsed piece type
            placement_piece_type = parsed_info['piece_type']

            # Check if square is empty
            if self.board[r][c]:
                print(f"DEBUG: Cannot place on occupied square ({r},{c}) for move '{move}'") # Debug placement fail
                return False
            # Check piece availability using the correct piece type
            # Note: Walls use Flat count
            piece_code_lookup = {self.FLAT: 'F', self.WALL: 'F', self.CAPSTONE: 'C'}
            piece_code = piece_code_lookup.get(placement_piece_type)
            if piece_code is None: # Should not happen if parsing is correct
                 print(f"DEBUG: Invalid placement_piece_type {placement_piece_type} for move '{move}'")
                 return False
            if self.pieces[player][piece_code] <= 0:
                print(f"DEBUG: No pieces of type {piece_code} (for {placement_piece_type}) left for player {player} for move '{move}'") # Debug placement fail
                return False

            # Apply using the correct piece type
            self.board[r][c].append((player, placement_piece_type))
            self.pieces[player][piece_code] -= 1
            if self.move_count < 2: # First two moves place opponent's color
                 # Ensure first moves are Flats, overriding S or C if parsed (though parser handles this too)
                 self.board[r][c] = [(3-player, self.FLAT)] # Overwrite with opponent's FLAT piece

        # --- Apply Stack Movement ---
        elif move_type == 'move':
            # ... (rest of the move logic remains the same) ...
            carry_count = parsed_info['carry']
            # dr, dc = parsed_info['direction_vector'] # Incorrect unpacking
            dc, dr = parsed_info['direction_vector'] # Correct unpacking (dc, dr)
            drops = parsed_info['drops']
            square = parsed_info['square'] # Get square for logging
            source_stack = self.board[r][c]
            # Check source stack control and height (Add these checks if missing)
            if not source_stack or source_stack[-1][0] != player:
                print(f"DEBUG: Cannot move from {square}({r},{c}): Not controlled by player {player} for move '{move}'")
                return False
            if len(source_stack) < carry_count:
                print(f"DEBUG: Cannot carry {carry_count} from stack of size {len(source_stack)} at {square}({r},{c}) for move '{move}'")
                return False


            # Check move validity (redundant if get_valid_moves is perfect, but good safeguard)
            # print(f"DEBUG: Checking validity for move '{move}' with r={r}, c={c}, dr={dr}, dc={dc}, carry={carry_count}, drops={drops}") # Debug check call - Note dr/dc order here now matches _check_stack_move args
            is_valid_check, reason = self._check_stack_move(r, c, dr, dc, carry_count, drops, source_stack) # Pass dr, dc in the order expected by the function
            if not is_valid_check:
                # *** This is a likely place the error occurs ***
                print(f"DEBUG: Internal validity check failed for move '{move}': {reason}") # Debug validity fail
                return False

            # Apply
            # print(f"DEBUG: Applying move '{move}'. Removing {carry_count} from {square}({r},{c}).") # Debug apply start
            pieces_to_move = source_stack[-carry_count:]
            self.board[r][c] = source_stack[:-carry_count] # Remove pieces from source

            current_r, current_c = r, c
            pieces_remaining_idx = 0
            for i, drop_count in enumerate(drops):
                # Use the correct dr and dc for calculating next position
                current_r, current_c = current_r + dr, current_c + dc
                dest_square = self.coords_to_square.get((current_r, current_c), "Invalid Coords")
                # print(f"DEBUG: Dropping {drop_count} pieces onto {dest_square}({current_r},{current_c}) for move '{move}' (Drop {i+1}/{len(drops)})") # Debug drop step
                pieces_to_drop = pieces_to_move[pieces_remaining_idx : pieces_remaining_idx + drop_count]
                pieces_remaining_idx += drop_count

                dest_stack = self.board[current_r][current_c]
                # Check for Capstone flattening Wall
                if dest_stack and dest_stack[-1][1] == self.WALL:
                    if pieces_to_drop and pieces_to_drop[-1][1] == self.CAPSTONE: # Check if piece exists before indexing
                        # print(f"DEBUG: Capstone flattening wall at {dest_square}({current_r},{current_c})") # Debug flatten
                        dest_stack[-1] = (dest_stack[-1][0], self.FLAT) # Flatten wall

                # Add dropped pieces
                self.board[current_r][current_c].extend(pieces_to_drop)

        else: # Should not happen if parsing is correct
             print(f"DEBUG: Unknown move type '{move_type}' for move '{move}'") # Debug unknown type
             return False

        # --- Update Game State ---
        self.move_history.append(move)
        self.move_count += 1
        self.check_win_conditions() # Check win AFTER move is applied

        if not self.game_over:
            self.current_player = 3 - player # Switch player

        return True

    def _parse_ptn_move(self, move):
        """Parses a PTN move string into structured info or None."""
        move = move.strip()
        if not move: return None

        # Pattern for placement: Optional[FSC], Square (e.g., Fa1, Sa1, Ca1, a1)
        # Note: PTN allows omitting F for flat placement.
        m_place = re.match(r"^([FSC]?)([a-h][1-8])$", move, re.IGNORECASE)
        if m_place:
            piece_char = m_place.group(1).upper()
            square = m_place.group(2).lower()
            coords = self.square_to_coords.get(square)
            if not coords: return None

            piece_type = self.FLAT # Default
            if piece_char == 'S': piece_type = self.WALL
            elif piece_char == 'C': piece_type = self.CAPSTONE

            # Special first/second move handling (always opponent's flat)
            if self.move_count < 2:
                piece_type = self.FLAT

            return {'type': 'place', 'coords': coords, 'piece_type': piece_type, 'square': square}

        # Pattern for movement: Optional[Carry], Square, Direction, Drops (e.g., 3a1+12, a1>1, 5c3<212)
        m_move = re.match(r"^(\d*)([a-h][1-8])([<>+-])(\d+)$", move, re.IGNORECASE)
        if m_move:
            carry_str = m_move.group(1)
            square = m_move.group(2).lower()
            direction_char = m_move.group(3)
            drops_str = m_move.group(4)

            coords = self.square_to_coords.get(square)
            direction_vector = self.DIRECTIONS.get(direction_char)
            if not coords or not direction_vector: return None

            carry_count = int(carry_str) if carry_str else 1 # Default carry is 1
            if carry_count == 0: return None # Carry must be > 0

            try:
                drops = [int(d) for d in drops_str]
            except ValueError:
                return None # Invalid drop numbers

            if not drops or sum(drops) != carry_count:
                # print(f"Debug: Parsed drops {drops} sum != carry count {carry_count} for move {move}")
                return None # Drops must sum to carry count

            return {
                'type': 'move', 'coords': coords, 'carry': carry_count,
                'direction_vector': direction_vector, 'drops': drops,
                'square': square, 'direction_char': direction_char
            }

        return None # Does not match known patterns


    def check_win_conditions(self):
        """Check for road wins, flat wins, or draws."""
        if self.game_over: return # Don't re-check

        # 1. Road Win Check
        for player in [1, 2]:
            if self.has_road(player):
                self.game_over = True
                self.winner = player
                # print(f"Debug: Player {player} wins by road!")
                return

        # 2. Out of Pieces / Board Full Check (Flat Win)
        board_full = all(self.board[r][c] for r in range(self.board_size) for c in range(self.board_size))
        p1_out = self.pieces[1]['F'] <= 0 and self.pieces[1]['C'] <= 0
        p2_out = self.pieces[2]['F'] <= 0 and self.pieces[2]['C'] <= 0

        if board_full or p1_out or p2_out:
            self.game_over = True
            flats_p1 = self.count_flats(1)
            flats_p2 = self.count_flats(2)
            # print(f"Debug: Game end condition met. Board Full: {board_full}, P1 Out: {p1_out}, P2 Out: {p2_out}")
            # print(f"Debug: Flat counts: P1={flats_p1}, P2={flats_p2}")

            if flats_p1 > flats_p2:
                self.winner = 1
            elif flats_p2 > flats_p1:
                self.winner = 2
            else:
                self.winner = 0 # Draw
            return

    def has_road(self, player):
        """Check if player has a complete road (simplified DFS/BFS)."""
        q = []
        visited = set()

        # Check horizontal (start from left edge, target right edge)
        for r in range(self.board_size):
            if self.is_road_piece(player, r, 0):
                q.append((r, 0))
                visited.add((r, 0))

        target_col = self.board_size - 1
        found_h_road = self._check_road_bfs(player, q, visited, lambda r, c: c == target_col)
        if found_h_road: return True

        # Check vertical (start from bottom edge, target top edge)
        q = []
        visited = set()
        for c in range(self.board_size):
             if self.is_road_piece(player, 0, c):
                 q.append((0, c))
                 visited.add((0, c))

        target_row = self.board_size - 1
        found_v_road = self._check_road_bfs(player, q, visited, lambda r, c: r == target_row)
        if found_v_road: return True

        return False

    def _check_road_bfs(self, player, queue, visited, target_condition):
        """Helper BFS for road checking."""
        while queue:
            r, c = queue.pop(0)

            if target_condition(r, c):
                return True # Reached the target edge

            # Explore neighbors (N, S, E, W)
            for dc, dr in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and \
                   (nr, nc) not in visited and self.is_road_piece(player, nr, nc):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    def is_road_piece(self, player, r, c):
        """Check if the top piece at (r, c) is a flat or capstone owned by player."""
        stack = self.board[r][c]
        if stack and stack[-1][0] == player:
            piece_type = stack[-1][1]
            return piece_type == self.FLAT or piece_type == self.CAPSTONE
        return False

    def print_board(self):
        """Prints the current board state with full stacks for each square."""
        piece_symbols = {0: 'F', 1: 'S', 2: 'C'}
        print("   " + " ".join([chr(97 + c) for c in range(self.board_size)]))
        for row in range(self.board_size-1, -1, -1):
            row_str = f"{row+1} "
            for col in range(self.board_size):
                stack = self.board[row][col]
                if not stack:
                    row_str += ".   "
                else:
                    # Show full stack, bottom to top
                    stack_str = ""
                    for player, piece_type in stack:
                        symbol = piece_symbols.get(piece_type, '?')
                        stack_str += f"{symbol}{player}"
                    # Pad to 3 chars for alignment
                    row_str += stack_str.ljust(3)
                    row_str += " "
            print(row_str)
        print()


    def get_heuristic(self):
        """Calculate a heuristic value for the current board state (simplified version)."""
        flat = 3
        wall = 2.98
        cap = 2.99
        diff = [0, 35, 75, 120, 170, 235, 285, 400]
        board_size = self.board_size

        # Count pieces per player
        arr = [[0 for _ in range(6)] for _ in range(2 * board_size)]
        for i in range(board_size):
            for j in range(board_size):
                stack = self.board[i][j]
                if not stack:
                    continue
                top = stack[-1]
                # Encoding: 0=F1, 1=S1, 2=C1, 3=F2, 4=S2, 5=C2
                idx = top[1] + (0 if top[0] == 1 else 3)
                arr[i][idx] += 1
                arr[j + board_size][idx] += 1

        # Captured advantage
        captured = 0.0
        for i in range(board_size):
            captured += (arr[i][0] - arr[i][3]) * 50 + (arr[i][2] - arr[i][5]) * 80

        # Composition value and center control
        composition_value = 0.0
        center_value = 0.0
        for i in range(2 * board_size):
            flat_capt_me = arr[i][0]
            wall_capt_me = arr[i][1]
            cap_capt_me = arr[i][2]
            my_capt = flat_capt_me + cap_capt_me

            flat_capt_you = arr[i][3]
            wall_capt_you = arr[i][4]
            cap_capt_you = arr[i][5]
            your_capt = flat_capt_you + cap_capt_you

            capt_diff = my_capt - your_capt
            capture_advantage = diff[min(my_capt, len(diff)-1)]
            capture_disadvantage = diff[min(your_capt, len(diff)-1)]

            wallFactor = 0.9  # Simplified
            if capt_diff > 0:
                wall_disadvantage = wall_capt_me * 32 + wall_capt_you * 40 + diff[abs(capt_diff)] * (wall_capt_me + wall_capt_you) * 2 / board_size
            elif capt_diff < 0:
                wall_disadvantage = wall_capt_me * 32 + wall_capt_you * 40 - diff[abs(capt_diff)] * (wall_capt_me + wall_capt_you) * 2 / board_size
            else:
                wall_disadvantage = wall_capt_me * 32 + wall_capt_you * 32

            composition_value += capture_advantage - capture_disadvantage - wallFactor * wall_disadvantage

            # Center control (simplified)
            if i < board_size:
                center_value += (cap_capt_me - cap_capt_you) * (board_size - i - 1) * i * 5
            else:
                center_value += (cap_capt_me - cap_capt_you) * (2 * board_size - i - 1) * (i - board_size) * 5

        # Piece holdings
        piece_val = 0.0
        piece_val += 60 * self.pieces[1]['C']
        piece_val -= 60 * self.pieces[2]['C']
        piece_val -= 24 * self.pieces[1]['F']
        piece_val += 24 * self.pieces[2]['F']

        # Influence (very simplified: count of pieces)
        infl_value = 0.0
        for i in range(board_size):
            for j in range(board_size):
                stack = self.board[i][j]
                if stack:
                    infl_value += len(stack) if stack[-1][0] == 1 else -len(stack)

        # Move advantage (not tracked here, set to 0)
        move_advantage = 0.0

        # Combine (weights as in C++ for 5x5)
        heuristic_value = (
            move_advantage +
            1.4 * captured +
            1.55 * composition_value +
            piece_val +
            0.9 * infl_value +
            1.1 * center_value
        )
        return heuristic_value # Higher value is better for the current player

# --- End TakGameEngine ---


def load_transformer_model(model_path, board_size, device="cuda"):
    """Load a trained TakTransformer model from a state dict file."""
    # Infer model parameters from common practice or add them if saved in checkpoint
    # These should match the parameters used during training
    d_model = 256 # Example, adjust if needed
    nhead = 8     # Example, adjust if needed
    num_layers = 6 # Example, adjust if needed
    dropout = 0.1  # Example, adjust if needed
    input_channels = 14 # Should be fixed based on board_state_to_tensor

    model = TakTransformer(
        board_size=board_size,
        input_channels=input_channels,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        dropout=dropout
    )
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"Successfully loaded model from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model state_dict from {model_path}: {e}")
        traceback.print_exc()
        return None


def self_play_transformers(model1, model2, board_size=5, num_games=10, max_moves=150, device="cuda"):
    """
    Let two TakTransformer models play against each other.

    Args:
        model1: Trained TakTransformer model for Player 1.
        model2: Trained TakTransformer model for Player 2.
        board_size: Size of the Tak board.
        num_games: Number of games to play.
        max_moves: Maximum moves per game.
        device: Device to run models on ("cuda" or "cpu").

    Returns:
        List of dictionaries containing game data.
    """
    if not model1 or not model2:
        print("Error: One or both models are not loaded.")
        return []

    model1.eval()
    model2.eval()
    models = {1: model1, 2: model2} # Map player number to model

    games_data = []

    for game_id in tqdm(range(num_games), desc="Self-Play Games (Transformer vs Transformer)"):
        game_engine = TakGameEngine(board_size=board_size)
        game_moves_data = []
        move_count_internal = 0 # Use internal counter matching engine's move_count

        print(f"\n--- Starting Game {game_id} ---")
        game_engine.print_board()

        while not game_engine.game_over and move_count_internal < max_moves:
            current_player = game_engine.current_player
            current_model = models[current_player]

            # 1. Get current board state dictionary
            board_state_dict = game_engine.get_board_state_dict()

            # 2. Convert to tensor (use [0.0, 0.0] for heuristic channels during inference)
            # Ensure board_state_to_tensor handles the format from get_board_state_dict correctly
            try:
                # Heuristic values are not used by the model during inference/play
                input_tensor = board_state_to_tensor(board_state_dict, board_size, ai_heuristic_value=[0.0, 0.0])
                input_tensor = input_tensor.unsqueeze(0).to(device) # Add batch dimension
            except Exception as e:
                print(f"\nError converting board state to tensor for player {current_player} at move {move_count_internal}: {e}")
                print("Board State Dict:", board_state_dict)
                traceback.print_exc()
                game_engine.game_over = True # Stop game on error
                game_engine.winner = "Error"
                break

            # 3. Get model prediction (list of 9 logit tensors)
            predicted_move = None
            try:
                with torch.no_grad():
                    output_logits = current_model(input_tensor) # List of 9 tensors: [(B, H0_dim), ...]

                # 4. Decode prediction to move vector
                predicted_vector = []
                for head_logits in output_logits:
                    # Get the index with the highest probability for each head
                    prediction = head_logits.argmax(dim=1).squeeze().item() # Squeeze removes batch dim
                    predicted_vector.append(prediction)

                # 5. Decode vector to PTN move string
                predicted_move = decode_move_vector(predicted_vector)
                # print(f"Debug: Player {current_player} predicted vector: {predicted_vector} -> move: {predicted_move}")

            except Exception as e:
                print(f"\nError during model prediction or decoding for player {current_player} at move {move_count_internal}: {e}")
                traceback.print_exc()
                predicted_move = None # Ensure fallback is triggered

            # 6. Validate and Apply Move
            valid_moves = game_engine.get_valid_moves()
            if not valid_moves:
                print(f"Player {current_player} has no valid moves. Game might end.")
                break # No moves possible

            selected_move = None
            if predicted_move and predicted_move in valid_moves:
                selected_move = predicted_move
                # print(f"Player {current_player} plays predicted valid move: {selected_move}")
            else:
                # Fallback: Choose a random valid move
                selected_move = random.choice(valid_moves)
                # print(f"Player {current_player} predicted move '{predicted_move}' is invalid or failed. Playing random valid move: {selected_move}")


            # 7. Apply the selected move to the engine
            move_successful = game_engine.apply_move(selected_move)

            # 8. Record and Print
            if move_successful:
                move_count_internal = game_engine.move_count # Sync internal counter
                # Calculate heuristic AFTER the move is applied
                current_heuristic = game_engine.get_heuristic() # From perspective of the player whose turn it *now* is

                print(f"\nMove {move_count_internal}: Player {3 - game_engine.current_player} played: {selected_move}") # Player who just moved
                game_engine.print_board()
                # Heuristic value reflects the state *after* the move, evaluated for the *next* player.
                # To show heuristic for the player who *just* moved, calculate before switching player or negate.
                # Let's show the heuristic from the perspective of the player who just made the move.
                heuristic_after_move = game_engine.get_heuristic()
                if game_engine.current_player == 1: # If next player is 1, the move was by P2
                    heuristic_perspective_player_who_moved = -heuristic_after_move
                else: # If next player is 2, the move was by P1
                    heuristic_perspective_player_who_moved = heuristic_after_move

                print(f"Heuristic (for Player {3 - game_engine.current_player} after move): {heuristic_perspective_player_who_moved:.2f}")


                # Store move data (board state *before* the move was made)
                move_data = {
                    "move_number": move_count_internal, # Move number that resulted in this state
                    "player_who_moved": 3 - game_engine.current_player, # Player who made the move
                    "move": selected_move,
                    "board_state_before_move": board_state_dict, # State before the move
                    "heuristic_after_move": heuristic_perspective_player_who_moved # Heuristic after move from mover's perspective
                }
                game_moves_data.append(move_data)
            else:
                # This should ideally not happen if validation/fallback works
                print(f"!!! Error: Failed to apply selected move '{selected_move}' for player {current_player}. Stopping game.")
                game_engine.game_over = True
                game_engine.winner = "Error Applying Move"
                break # Stop the game loop

        # --- Game End ---
        winner = game_engine.winner
        print(f"\n--- Game {game_id} Ended ---")
        print(f"Total Moves: {game_engine.move_count}")
        if winner == "Error":
             print("Result: Error during game.")
        elif winner == 0:
             print("Result: Draw")
        elif winner is not None:
             print(f"Result: Player {winner} wins!")
        else:
             print("Result: Game ended by max moves limit.") # Or other condition

        game_data = {
            "game_id": game_id,
            "board_size": game_engine.board_size,
            "total_moves": game_engine.move_count,
            "winner": winner,
            "moves": game_moves_data,
            "final_board_state": game_engine.get_board_state_dict() # Record final state
        }
        games_data.append(game_data)

    return games_data

def save_games(games_data, output_dir="self_play_transformer_games"):
    """Save generated games to files."""
    os.makedirs(output_dir, exist_ok=True)
    num_saved = 0
    for game in games_data:
        game_id = game["game_id"]
        filename = os.path.join(output_dir, f"game_{game_id}.json")
        try:
            with open(filename, "w") as f:
                json.dump(game, f, indent=2)
            num_saved += 1
        except Exception as e:
            print(f"Error saving game {game_id} to {filename}: {e}")

    if num_saved > 0:
        print(f"Saved {num_saved} games to {output_dir}")
    else:
        print("No games were saved.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run self-play between two trained Tak Transformer models")
    parser.add_argument("--model1", type=str, required=True, help="Path to trained model state_dict file for Player 1")
    parser.add_argument("--model2", type=str, required=True, help="Path to trained model state_dict file for Player 2")
    parser.add_argument("--board_size", type=int, default=5, help="Board size (must match models' training)")
    parser.add_argument("--games", type=int, default=5, help="Number of games to play")
    parser.add_argument("--max_moves", type=int, default=150, help="Maximum moves per game")
    parser.add_argument("--output", type=str, default="self_play_transformer_games", help="Directory to save game JSON files")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run models on (cuda or cpu)")

    args = argparse.ArgumentParser()
    args = parser.parse_args()
    # Load models
    print(f"Loading model for Player 1 from {args.model1}...")
    model1 = load_transformer_model(args.model1, args.board_size, args.device)
    print(f"Loading model for Player 2 from {args.model2}...")
    model2 = load_transformer_model(args.model2, args.board_size, args.device)

    if model1 and model2:
        # Run self-play
        games_data = self_play_transformers(
            model1, model2,
            board_size=args.board_size,
            num_games=args.games,
            max_moves=args.max_moves,
            device=args.device
        )

        # Save games
        if games_data:
            save_games(games_data, args.output)
        else:
            print("No games were generated.")
    else:
        print("Could not run self-play due to model loading errors.")

    print("Self-play finished.")