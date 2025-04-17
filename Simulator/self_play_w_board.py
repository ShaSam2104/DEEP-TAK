import os
import json
import torch
import random
import numpy as np
import re
from tqdm import tqdm

# Import from the transformer model implementation
# Assuming the model code is in tak_transformer.py
from seq_transformer_train import (
    TakBoard, TakTransformer, mask_invalid_moves,
    encode_move_history, tokenize_move
)

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
            '+': (0, 1),   # North
            '-': (0, -1),  # South
            '>': (1, 0),   # East
            '<': (-1, 0)   # West
        }
        
        # Define squares using a1 notation
        self.square_to_coords = {}
        self.coords_to_square = {}
        for row in range(board_size):
            for col in range(board_size):
                square = f"{chr(97 + col)}{row + 1}"  # a1, b1, etc.
                self.square_to_coords[square] = (row, col)
                self.coords_to_square[(row, col)] = square

    def reset(self):
        """Reset the game to initial state"""
        # Board representation: None for empty squares, or (player, piece_type, is_top)
        self.board = [[[] for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = 1
        self.move_count = 0
        self.game_over = False
        self.winner = None
        
        # Initialize piece counts
        if self.board_size == 5:
            self.pieces = {
                1: {'F': 21, 'C': 1},  # Player 1: 21 flats, 1 capstone
                2: {'F': 21, 'C': 1}   # Player 2: 21 flats, 1 capstone
            }
        else:
            # Adjust for different board sizes
            flats = 15 if self.board_size == 4 else 30  # 4x4 or 6x6
            self.pieces = {
                1: {'F': flats, 'C': 1},
                2: {'F': flats, 'C': 1}
            }
        
        # Game history
        self.move_history = []
    
    def get_board_state_dict(self):
        """Convert current board state to the dictionary format used in the dataset"""
        # Create board squares list
        board_squares = []
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                stack = self.board[row][col]
                
                if not stack:  # Empty square
                    square_data = {
                        "color_encoding": "000000",
                        "type_encoding": "00",
                        "stack_height": 0.0
                    }
                else:
                    # Get top piece
                    player, piece_type = stack[-1]
                    
                    # Set color encoding
                    color_encoding = "100000" if player == 1 else "010000"
                    
                    # Set type encoding
                    if piece_type == self.FLAT:
                        type_encoding = "00"
                    elif piece_type == self.WALL:
                        type_encoding = "01"
                    elif piece_type == self.CAPSTONE:
                        type_encoding = "10"
                    
                    square_data = {
                        "color_encoding": color_encoding,
                        "type_encoding": type_encoding,
                        "stack_height": float(len(stack))
                    }
                
                board_squares.append(square_data)
        
        # Calculate heuristics
        flat_diff = self.count_flats(1) - self.count_flats(2)
        
        # Create board state dictionary
        board_state = {
            "board": board_squares,
            "board_size": self.board_size,
            "player1_flats": self.pieces[1]['F'],
            "player1_capstones": self.pieces[1]['C'],
            "player2_flats": self.pieces[2]['F'],
            "player2_capstones": self.pieces[2]['C'],
            "turn": self.current_player,
            "heuristics": {
                "flat_diff": flat_diff,
                "center_control": 0,  # Simplified
                "stack_control": 0,   # Simplified
                "road_potential": 0,  # Simplified
                "capstone_value": 0   # Simplified
            }
        }
        
        return board_state
    
    def count_flats(self, player):
        """Count flat stones on the board for a player"""
        count = 0
        for row in range(self.board_size):
            for col in range(self.board_size):
                stack = self.board[row][col]
                if stack and stack[-1][0] == player and stack[-1][1] == self.FLAT:
                    count += 1
        return count
    
    def get_valid_moves(self):
        """Get all valid moves for the current player"""
        valid_moves = []
        
        # First turn placement rules
        if self.move_count == 0:
            # First player can only place a flat stone
            for row in range(self.board_size):
                for col in range(self.board_size):
                    square = self.coords_to_square[(row, col)]
                    valid_moves.append(f"{square}")
            return valid_moves
        
        if self.move_count == 1:
            # Second player can only place a flat stone, but not in the first player's position
            for row in range(self.board_size):
                for col in range(self.board_size):
                    if not self.board[row][col]:  # Empty square
                        square = self.coords_to_square[(row, col)]
                        valid_moves.append(f"{square}")
            return valid_moves
        
        # Regular move rules
        # 1. Place a stone
        if self.pieces[self.current_player]['F'] > 0:
            for row in range(self.board_size):
                for col in range(self.board_size):
                    if not self.board[row][col]:  # Empty square
                        square = self.coords_to_square[(row, col)]
                        # Place flat
                        valid_moves.append(f"{square}")
                        valid_moves.append(f"F{square}")
                        # Place wall
                        valid_moves.append(f"S{square}")
                        valid_moves.append(f"{square}S")
        
        # Place capstone if available
        if self.pieces[self.current_player]['C'] > 0:
            for row in range(self.board_size):
                for col in range(self.board_size):
                    if not self.board[row][col]:  # Empty square
                        square = self.coords_to_square[(row, col)]
                        valid_moves.append(f"C{square}")
                        valid_moves.append(f"{square}C")
        
        # 2. Move a stack
        for row in range(self.board_size):
            for col in range(self.board_size):
                stack = self.board[row][col]
                if stack and stack[-1][0] == self.current_player:  # Player's piece on top
                    square = self.coords_to_square[(row, col)]
                    stack_height = len(stack)
                    
                    # Can move up to stack_height pieces, but max stack_size is board_size
                    max_carry = min(stack_height, self.board_size)
                    
                    for carry in range(1, max_carry + 1):
                        # For each direction
                        for direction, (dr, dc) in self.DIRECTIONS.items():
                            # Try different drop combinations
                            # For simplicity, we'll just do single drops (not all combinations)
                            for drop_dist in range(1, carry + 1):
                                new_row, new_col = row + dr * drop_dist, col + dc * drop_dist
                                
                                # Check if in bounds
                                if not (0 <= new_row < self.board_size and 0 <= new_col < self.board_size):
                                    continue
                                
                                # Check if we can move on top of this square
                                dest_stack = self.board[new_row][new_col]
                                if dest_stack and dest_stack[-1][1] == self.WALL:
                                    # Can't stack on a wall unless moving a capstone
                                    if not (stack[-1][1] == self.CAPSTONE and drop_dist == 1):
                                        continue
                                
                                # Valid move
                                if carry == 1:
                                    valid_moves.append(f"{square}{direction}{drop_dist}")
                                else:
                                    valid_moves.append(f"{carry}{square}{direction}{drop_dist}")
        
        return valid_moves
    
    def apply_move(self, move):
        """Apply a move to the board state"""
        # Parse the move
        if not move:
            return False
        
        # Flat placement
        normal_pattern1 = r"^([a-e][1-5])([FSC]?)$"     # e.g., d2F, a1
        normal_pattern2 = r"^([FSC])([a-e][1-5])$"      # e.g., Fd2
        
        # Stack movement
        stack_pattern = r"^(\d?[a-e][1-5])([><\+\-])(\d+)$"  # e.g., a1+1, 2a1+1
        
        # Try to match the move patterns
        if re.match(normal_pattern1, move) or re.match(normal_pattern2, move):
            # Place a stone
            square = None
            piece_type = self.FLAT  # Default to flat
            
            if re.match(normal_pattern1, move):
                square, piece_code = re.findall(normal_pattern1, move)[0]
                if piece_code == 'S':
                    piece_type = self.WALL
                elif piece_code == 'C':
                    piece_type = self.CAPSTONE
            else:
                piece_code, square = re.findall(normal_pattern2, move)[0]
                if piece_code == 'S':
                    piece_type = self.WALL
                elif piece_code == 'C':
                    piece_type = self.CAPSTONE
            
            # Check if we have the piece
            if piece_type == self.FLAT and self.pieces[self.current_player]['F'] <= 0:
                return False
            if piece_type == self.CAPSTONE and self.pieces[self.current_player]['C'] <= 0:
                return False
            
            # Get coordinates
            try:
                row, col = self.square_to_coords[square]
            except KeyError:
                return False
            
            # Check if square is empty
            if self.board[row][col]:
                return False
            
            # Place the piece
            self.board[row][col].append((self.current_player, piece_type))
            
            # Update piece count
            if piece_type == self.FLAT or piece_type == self.WALL:
                self.pieces[self.current_player]['F'] -= 1
            elif piece_type == self.CAPSTONE:
                self.pieces[self.current_player]['C'] -= 1
            
        elif re.match(stack_pattern, move):
            # Move a stack
            stack_desc, direction, drops = re.findall(stack_pattern, move)[0]
            
            # Parse carry count and square
            if stack_desc[0].isdigit():
                carry_count = int(stack_desc[0])
                square = stack_desc[1:]
            else:
                carry_count = 1
                square = stack_desc
            
            # Get coordinates
            try:
                row, col = self.square_to_coords[square]
            except KeyError:
                return False
            
            # Check if square has pieces
            if not self.board[row][col]:
                return False
            
            # Check if top piece belongs to current player
            if self.board[row][col][-1][0] != self.current_player:
                return False
            
            # Check if there are enough pieces to carry
            if len(self.board[row][col]) < carry_count:
                return False
            
            # Get direction vector
            dr, dc = self.DIRECTIONS.get(direction, (0, 0))
            if (dr, dc) == (0, 0):
                return False
            
            # Check drops - simplified for now
            drop_distance = int(drops)
            total_distance = drop_distance
            
            # Check if move is within bounds
            end_row, end_col = row + dr * total_distance, col + dc * total_distance
            if not (0 <= end_row < self.board_size and 0 <= end_col < self.board_size):
                return False
            
            # Get pieces to move
            pieces_to_move = self.board[row][col][-carry_count:]
            self.board[row][col] = self.board[row][col][:-carry_count]
            
            # Move pieces
            curr_row, curr_col = row + dr * drop_distance, col + dc * drop_distance
            
            # Check if destination has a wall or capstone
            dest_stack = self.board[curr_row][curr_col]
            if dest_stack and dest_stack[-1][1] == self.WALL:
                # Only a capstone can flatten a wall
                if not (pieces_to_move[0][1] == self.CAPSTONE and len(pieces_to_move) == 1):
                    return False
                
                # Flatten the wall
                dest_stack[-1] = (dest_stack[-1][0], self.FLAT)
            
            # Add pieces to destination
            self.board[curr_row][curr_col].extend(pieces_to_move)
        
        else:
            # Invalid move format
            return False
        
        # Update game state
        self.move_count += 1
        self.move_history.append(move)
        
        # Check for win conditions
        self.check_win_conditions()
        
        # Switch players if game is not over
        if not self.game_over:
            self.current_player = 3 - self.current_player  # Switch between 1 and 2
        
        return True
    
    def check_win_conditions(self):
        """Check if the game is over"""
        # Check for road wins
        for player in [1, 2]:
            if self.has_road(player):
                self.game_over = True
                self.winner = player
                return
        
        # Check if board is full
        board_full = True
        for row in range(self.board_size):
            for col in range(self.board_size):
                if not self.board[row][col]:
                    board_full = False
                    break
            if not board_full:
                break
        
        if board_full:
            # Count flats and determine winner
            flats_p1 = self.count_flats(1)
            flats_p2 = self.count_flats(2)
            
            self.game_over = True
            if flats_p1 > flats_p2:
                self.winner = 1
            elif flats_p2 > flats_p1:
                self.winner = 2
            else:
                self.winner = 0  # Draw
    
    def has_road(self, player):
        """Check if player has a road from one side to the other"""
        # Check for horizontal roads
        for start_row in range(self.board_size):
            # Start from leftmost column
            if self.has_road_from(player, start_row, 0, "horizontal"):
                return True
        
        # Check for vertical roads
        for start_col in range(self.board_size):
            # Start from bottom row
            if self.has_road_from(player, 0, start_col, "vertical"):
                return True
        
        return False
    
    def has_road_from(self, player, start_row, start_col, direction):
        """
        Check if there's a road starting from the given position
        This is a simplified implementation - a proper one would use DFS/BFS
        """
        # For simplicity, we'll just check direct paths
        if direction == "horizontal":
            # Check if there's a horizontal path from left to right
            for col in range(self.board_size):
                if not self.is_player_controlled(player, start_row, col):
                    return False
            return True
        else:  # vertical
            # Check if there's a vertical path from bottom to top
            for row in range(self.board_size):
                if not self.is_player_controlled(player, row, start_col):
                    return False
            return True
    
    def is_player_controlled(self, player, row, col):
        """Check if a square is controlled by the player (has flat or capstone)"""
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            stack = self.board[row][col]
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
        return heuristic_value

def self_play(model, token_to_id, id_to_token, vocab_size,
              board_feature_size, num_games=10, max_moves=100,
              temperature=1.0, device="cuda"):
    """
    Let the model play against itself to generate games and evaluate performance
    
    Args:
        model: The trained TakTransformer model
        token_to_id: Dictionary mapping tokens to ids
        id_to_token: Dictionary mapping ids to tokens
        vocab_size: Size of vocabulary
        board_feature_size: Size of board features vector
        num_games: Number of games to play
        max_moves: Maximum moves per game
        temperature: Temperature for sampling moves (higher = more random)
        device: Device to run model on ("cuda" or "cpu")
        
    Returns:
        List of dictionaries containing game data
    """
    model.eval()
    all_tokens = list(token_to_id.keys())
    
    games_data = []
    
    for game_id in tqdm(range(num_games), desc="Self-Play Games"):
        game_engine = TakGameEngine()
        game_moves = []
        move_count = 0
        
        while not game_engine.game_over and move_count < max_moves:
            # Get valid moves
            valid_moves = game_engine.get_valid_moves()
            if not valid_moves:
                break
                
            # Create mask for valid moves
            valid_moves_mask = torch.zeros(vocab_size)
            for vm in valid_moves:
                try:
                    tokens = tokenize_move(vm)
                    main_token = tokens[0]  # Use first token as move identifier
                    if main_token in token_to_id:
                        valid_moves_mask[token_to_id[main_token]] = 1.0
                except:
                    continue
            
            # Get current board state
            board_state_dict = game_engine.get_board_state_dict()
            board_state = TakBoard.encode_board_state_from_dict(board_state_dict)
            
            # Prepare move history
            if game_engine.move_history:
                move_history = encode_move_history(
                    game_engine.move_history[-10:] if len(game_engine.move_history) >= 10 else game_engine.move_history,
                    token_to_id, vocab_size
                )
            else:
                # If no moves yet, create empty history tensor
                move_history = torch.zeros(10, vocab_size)
            
            # Add batch dimension
            board_state = board_state.unsqueeze(0).to(device)
            move_history = move_history.unsqueeze(0).to(device)
            valid_moves_mask = valid_moves_mask.to(device)
            
            # Get model prediction
            with torch.no_grad():
                logits = model(board_state, move_history)
                
                # Apply temperature
                logits = logits / temperature
                
                # Mask invalid moves
                masked_logits = mask_invalid_moves(logits, valid_moves_mask.unsqueeze(0))
                
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(masked_logits, dim=-1)
                
                # Sample move based on probabilities
                move_id = torch.multinomial(probs[0], 1).item()
                
                # Get move token
                try:
                    main_token = id_to_token[move_id]
                    
                    # Find a complete move that starts with this token
                    possible_moves = [m for m in valid_moves if tokenize_move(m)[0] == main_token]
                    if possible_moves:
                        selected_move = random.choice(possible_moves)
                    else:
                        # Fallback to random valid move
                        print("Fallback")
                        selected_move = random.choice(valid_moves)
                        
                except:
                    # Fallback to random valid move
                    selected_move = random.choice(valid_moves)
            
            # Apply move
            if game_engine.apply_move(selected_move):
                game_engine.print_board()
                heuristic = game_engine.get_heuristic()
                print(f"Heuristic after move {move_count}: {heuristic:.2f}")
                # Store move data
                move_data = {
                    "move_number": move_count,
                    "player": game_engine.current_player,  # This is already updated, so it's the next player
                    "move": selected_move,
                    "board_state": board_state_dict,
                    "heuristic": heuristic
                }
                game_moves.append(move_data)
                move_count += 1
            else:
                print(f"Invalid move generated: {selected_move}")
                # Select random valid move as fallback
                fallback_move = random.choice(valid_moves)
                if game_engine.apply_move(fallback_move):
                    game_engine.print_board()
                    move_data = {
                        "move_number": move_count,
                        "player": game_engine.current_player,
                        "move": fallback_move,
                        "board_state": board_state_dict
                    }
                    game_moves.append(move_data)
                    move_count += 1
        
        # Game is over, record results
        game_data = {
            "game_id": game_id,
            "board_size": game_engine.board_size,
            "total_moves": move_count,
            "winner": game_engine.winner,
            "moves": game_moves
        }
        
        # Pretty print game moves
        print(f"Game {game_id} completed in {move_count} moves")
        print(f"Winner: Player {game_engine.winner}")
        print("Move sequence:", " ".join([m["move"] for m in game_moves]))
        print("-" * 40)
        
        games_data.append(game_data)
    
    return games_data

def load_model(model_path, device="cuda"):
    """Load a trained model from a checkpoint file"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model parameters
    vocab_size = checkpoint["vocab_size"]
    token_to_id = checkpoint["token_to_id"]
    id_to_token = checkpoint["id_to_token"]
    board_feature_size = checkpoint["board_feature_size"]
    
    # Create model
    model = TakTransformer(vocab_size, board_feature_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, token_to_id, id_to_token, vocab_size, board_feature_size

def save_games(games_data, output_dir="self_play_games"):
    """Save generated games to files"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each game to a separate file
    for game in games_data:
        game_id = game["game_id"]
        filename = f"{output_dir}/game_{game_id}.json"
        
        with open(filename, "w") as f:
            json.dump(game, f, indent=2)
    
    print(f"Saved {len(games_data)} games to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run self-play with trained Tak model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model file")
    parser.add_argument("--games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--max-moves", type=int, default=100, help="Maximum moves per game")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for move sampling")
    parser.add_argument("--output", type=str, default="self_play_games", help="Directory to save games")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run on (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Load model
    model, token_to_id, id_to_token, vocab_size, board_feature_size = load_model(args.model, args.device)
    
    # Run self-play
    games_data = self_play(
        model, token_to_id, id_to_token, vocab_size, board_feature_size,
        num_games=args.games, max_moves=args.max_moves,
        temperature=args.temperature, device=args.device
    )
    
    # Save games
    save_games(games_data, args.output)