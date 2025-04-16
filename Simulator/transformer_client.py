import torch
import numpy as np
import sys
import argparse
from Game import Game
from train_transformer import TakTransformer, decode_move_vector, board_state_to_tensor
import traceback # For detailed error logging
import copy # For deep copying in visualization if needed

# ANSI Color Codes
COLOR_RED = '\033[91m' # Player 1 (Transformer)
COLOR_YELLOW = '\033[93m' # Player 2 (C++ AI)
COLOR_RESET = '\033[0m'
CELL_WIDTH = 6 # Width for each cell in the visualization

def format_piece(piece_data, height):
    """Formats a single piece or stack top for visualization."""
    if not piece_data:
        return "." * CELL_WIDTH

    player = piece_data.get('player', -1)
    ptype = piece_data.get('type', '?')
    color = COLOR_RED if player == 0 else COLOR_YELLOW if player == 1 else ""
    reset = COLOR_RESET if color else ""

    if height == 1:
        return f"{color}{ptype:<{CELL_WIDTH}}{reset}"
    else:
        # Show top piece type and height for stacks
        return f"{color}{ptype}({height}){' '*(CELL_WIDTH-len(ptype)-len(str(height))-2)}{reset}"

def visualize_board_state(board_state_dict, file=sys.stderr):
    """Prints a human-readable grid visualization of the board state."""
    if not board_state_dict:
        print("Transformer Client: Cannot visualize empty board state dict.", file=file, flush=True)
        return

    board_size = board_state_dict.get('board_size', 0)
    board_list = board_state_dict.get('board', [])
    turn = board_state_dict.get('turn', -1)
    p1_flats = board_state_dict.get('player1_flats', '?')
    p1_caps = board_state_dict.get('player1_capstones', '?')
    p2_flats = board_state_dict.get('player2_flats', '?')
    p2_caps = board_state_dict.get('player2_capstones', '?')

    if board_size == 0 or len(board_list) != board_size * board_size:
        print(f"Transformer Client: Invalid board data for visualization (size={board_size}, list_len={len(board_list)}).", file=file, flush=True)
        # Fallback to printing the raw dict if visualization fails
        print(f"Transformer Client: Raw board_state_dict:\n{board_state_dict}", file=file, flush=True)
        return

    turn_color = COLOR_RED if turn == 0 else COLOR_YELLOW if turn == 1 else ""
    turn_reset = COLOR_RESET if turn_color else ""
    print(f"Transformer Client: Board State Visualization (Turn: {turn_color}Player {turn+1}{turn_reset})", file=file, flush=True)
    print(f"  P1 ({COLOR_RED}Red{COLOR_RESET}): {p1_flats}F, {p1_caps}C | P2 ({COLOR_YELLOW}Yellow{COLOR_RESET}): {p2_flats}F, {p2_caps}C", file=file, flush=True)
    print("  +" + ("-" * CELL_WIDTH + "+") * board_size, file=file, flush=True)

    for r_idx in range(board_size): # Print from top row (board_size) down to 1
        row_num = board_size - r_idx
        row_str = f"{row_num:<2}|"
        for c_idx in range(board_size):
            # Calculate index in the flat board_list
            # board_list index 0 corresponds to a5, index 4 to e5, ..., index 20 to a1, index 24 to e1
            list_idx = r_idx * board_size + c_idx
            square_data = board_list[list_idx]
            stack = square_data.get('stack', [])
            height = len(stack)
            top_piece = stack[-1] if height > 0 else None
            row_str += format_piece(top_piece, height) + "|"
        print(row_str, file=file, flush=True)
        print("  +" + ("-" * CELL_WIDTH + "+") * board_size, file=file, flush=True)

    # Print column labels (a, b, c, ...)
    col_label_str = "   "
    for c_idx in range(board_size):
        col_char = chr(ord('a') + c_idx)
        col_label_str += f"{col_char:^{CELL_WIDTH}} " # Center column label
    print(col_label_str, file=file, flush=True)
    print("-" * 30, file=file, flush=True) # Separator


class TransformerTakPlayer:
    def __init__(self, model_path, board_size=5, device='cpu'):
        self.board_size = board_size
        self.device = torch.device(device)

        # Load the model (Ensure architecture matches saved model)
        self.model = TakTransformer(
            board_size=board_size, input_channels=14, d_model=256, nhead=8,
            num_encoder_layers=6, dim_feedforward=1024, dropout=0.1
        ).to(self.device)

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Transformer Client: Model loaded from {model_path}", file=sys.stderr, flush=True)
        except FileNotFoundError:
             print(f"Transformer Client: Error - Model file not found at {model_path}", file=sys.stderr, flush=True)
             raise
        except Exception as e:
             print(f"Transformer Client: Error loading model state_dict: {e}", file=sys.stderr, flush=True)
             raise

        # Initialize game state
        self.game = Game(self.board_size, mode='script')
        print(f"Transformer Client: Game initialized for {self.board_size}x{self.board_size} board", file=sys.stderr, flush=True)

    def apply_opponent_move(self, opponent_move):
        """Applies opponent's move to the internal game state."""
        if opponent_move:
            # print(f"Transformer Client: Applying opponent move '{opponent_move}'...", file=sys.stderr, flush=True) # Keep minimal
            try:
                result = self.game.execute_move(opponent_move)
                # print(f"Transformer Client: self.game.execute_move('{opponent_move}') returned: {result}", file=sys.stderr, flush=True) # Can be noisy
                # print(f"Transformer Client: self.game.turn AFTER opponent execute_move: {self.game.turn}", file=sys.stderr, flush=True) # Keep for debugging turn issues
            except Exception as e:
                 print(f"Transformer Client: EXCEPTION during opponent self.game.execute_move('{opponent_move}'): {e}", file=sys.stderr, flush=True)
                 traceback.print_exc(file=sys.stderr)
                 return False # Indicate failure on exception

            if result == 0:
                print(f"Transformer Client: Error applying invalid opponent move: {opponent_move}", file=sys.stderr, flush=True)
                return False # Indicate failure
            elif result in [2, 3]: # Game ended
                # print(f"Transformer Client: Game ended by opponent move {opponent_move}", file=sys.stderr, flush=True) # Covered by main loop
                return "GAME_OVER"
            # print(f"Transformer Client: Successfully applied opponent move '{opponent_move}'.", file=sys.stderr, flush=True) # Can be noisy
            return True # Success
        return True # No move to apply

    def generate_model_move(self):
        """Generates a move using the transformer model. Does NOT execute it."""
        # Convert game state to model input format
        board_state_dict = self._game_to_board_state_dict()

        # --- Visualize Board State ---
        visualize_board_state(board_state_dict)
        # -----------------------------

        if board_state_dict is None: # Check if conversion failed
             print("Transformer Client: _game_to_board_state_dict returned None", file=sys.stderr, flush=True)
             return self._generate_backup_move()

        # Get AI heuristic values (use dummy values for inference)
        ai_heuristic_value = [0.0, 0.0]

        # Convert to tensor
        try:
            board_tensor = board_state_to_tensor(board_state_dict, self.board_size, ai_heuristic_value)
            if board_tensor is None:
                 print("Transformer Client: board_state_to_tensor returned None", file=sys.stderr, flush=True)
                 return self._generate_backup_move() # Try backup

            # print(f"Transformer Client: board_tensor created with shape: {board_tensor.shape}", file=sys.stderr, flush=True) # Removed

            board_tensor = board_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        except Exception as e:
            print(f"Transformer Client: Error creating tensor: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return self._generate_backup_move() # Try backup

        # Get model prediction
        try:
            with torch.no_grad():
                outputs = self.model(board_tensor)
        except Exception as e:
             print(f"Transformer Client: Error during model inference: {e}", file=sys.stderr, flush=True)
             traceback.print_exc(file=sys.stderr)
             return self._generate_backup_move() # Try backup

        # Process outputs
        move_components = []
        try:
            if not isinstance(outputs, (list, tuple)) or len(outputs) != 9:
                 print(f"Transformer Client: Unexpected model output format. Expected tuple/list of 9 tensors, got {type(outputs)} len {len(outputs) if isinstance(outputs, (list, tuple)) else 'N/A'}", file=sys.stderr, flush=True)
                 return self._generate_backup_move()

            for i, output_head in enumerate(outputs):
                if not isinstance(output_head, torch.Tensor) or output_head.dim() != 2 or output_head.shape[0] != 1:
                     print(f"Transformer Client: Invalid output head #{i}. Expected shape [1, N], got {output_head.shape if isinstance(output_head, torch.Tensor) else type(output_head)}", file=sys.stderr, flush=True)
                     return self._generate_backup_move() # Try backup

                probs = torch.softmax(output_head, dim=1)
                component = torch.argmax(probs, dim=1).item()
                move_components.append(component)
                # print(f"Transformer Client: Model output component {i}: {component} (from argmax)", file=sys.stderr, flush=True) # Removed

        except Exception as e:
            print(f"Transformer Client: Error processing model outputs: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return self._generate_backup_move() # Try backup

        # Decode move vector to move string
        move_str = decode_move_vector(move_components)
        # print(f"Transformer Client: Decoded move components {move_components} to: '{move_str}'", file=sys.stderr, flush=True) # Removed

        if not move_str:
            print(f"Transformer Client: Decoding failed for components {move_components}", file=sys.stderr, flush=True)
            return self._generate_backup_move() # Try backup

        # --- Basic Validation (Revised) ---
        try:
            can_clone = hasattr(self.game, 'clone') and callable(self.game.clone)
            is_valid_result = 0 # Default to invalid

            if can_clone:
                game_to_validate = self.game.clone()
                # print(f"Transformer Client: Validating '{move_str}' using cloned game state.", file=sys.stderr, flush=True) # Removed
                is_valid_result = game_to_validate.execute_move(move_str)
                # print(f"Transformer Client: Validation on clone returned: {is_valid_result}", file=sys.stderr, flush=True) # Removed
            else:
                # If clone() is not available, skip pre-validation here.
                print(f"Transformer Client: Warning - Game.clone() not found. Skipping pre-validation for '{move_str}'.", file=sys.stderr, flush=True)
                is_valid_result = 1 # Tentatively assume valid

            if is_valid_result != 0: # If validation passed (or was skipped)
                # print(f"Transformer Client: Model generated potentially valid move: {move_str}", file=sys.stderr, flush=True) # Removed
                return move_str # Return the potentially valid move string
            else: # If validation on clone failed
                print(f"Transformer Client: Model generated invalid move (validation on clone): {move_str}. Trying backup.", file=sys.stderr, flush=True)
                return self._generate_backup_move() # Try backup if model move is invalid

        except Exception as e:
             print(f"Transformer Client: Error during move validation/cloning for '{move_str}': {e}", file=sys.stderr, flush=True)
             traceback.print_exc(file=sys.stderr)
             return self._generate_backup_move() # Try backup on validation error

    def _game_to_board_state_dict(self):
        """Convert Game object state to board_state dictionary for the model"""
        if not self.game or not hasattr(self.game, 'board') or not hasattr(self.game, 'players'):
             print("Transformer Client: Error - Game object not properly initialized in _game_to_board_state_dict", file=sys.stderr, flush=True)
             return None

        board_state = {
            'board': [],
            'board_size': self.board_size,
            'player1_flats': self.game.players[0].flats if len(self.game.players) > 0 else 0,
            'player1_capstones': self.game.players[0].capstones if len(self.game.players) > 0 else 0,
            'player2_flats': self.game.players[1].flats if len(self.game.players) > 1 else 0,
            'player2_capstones': self.game.players[1].capstones if len(self.game.players) > 1 else 0,
            'turn': self.game.turn if hasattr(self.game, 'turn') else 0
        }

        # Encode board squares
        for i in range(self.game.total_squares):
            square_dict = {'stack': []}
            if 0 <= i < len(self.game.board):
                 square_content = self.game.board[i]
                 if isinstance(square_content, list) and len(square_content) > 0:
                    # Ensure pieces are in the expected format (list/tuple of [player, type])
                    for piece in square_content:
                        if isinstance(piece, (list, tuple)) and len(piece) == 2:
                             square_dict['stack'].append({
                                'player': piece[0], # Should be 0 or 1
                                'type': piece[1]    # Should be 'F', 'S', or 'C'
                             })
                        # else: # Avoid excessive warnings if format is slightly off but recoverable
                             # print(f"Transformer Client: Warning - Unexpected piece format at index {i}: {piece}", file=sys.stderr, flush=True)
            # else: # Should not happen if total_squares is correct
                 # print(f"Transformer Client: Warning - Invalid square index {i} accessed in _game_to_board_state_dict", file=sys.stderr, flush=True)
            board_state['board'].append(square_dict)
        return board_state


    def _generate_backup_move(self):
        """Generate a simple valid placement move when the model fails. Does NOT execute the move."""
        print("Transformer Client: Attempting backup move generation...", file=sys.stderr, flush=True)
        # Try placing a flatstone ('F') on the first valid empty square
        for i in range(self.game.total_squares):
            if 0 <= i < len(self.game.board) and isinstance(self.game.board[i], list) and len(self.game.board[i]) == 0:
                col_idx = i % self.board_size
                row_idx = i // self.board_size
                col = chr(97 + col_idx)
                # Tak board rows are 1-based, bottom-up
                row = str(self.board_size - row_idx)
                backup_move = f"F{col}{row}"

                # Validate the backup move before returning (using clone if possible)
                try:
                    can_clone = hasattr(self.game, 'clone') and callable(self.game.clone)
                    is_valid_result = 0

                    if can_clone:
                        game_to_validate = self.game.clone()
                        # print(f"Transformer Client: Validating backup move '{backup_move}' using cloned game state.", file=sys.stderr, flush=True) # Removed
                        is_valid_result = game_to_validate.execute_move(backup_move)
                        # print(f"Transformer Client: Backup validation on clone returned: {is_valid_result}", file=sys.stderr, flush=True) # Removed
                    else:
                        # Skip pre-validation if no clone
                        print(f"Transformer Client: Warning - Game.clone() not found. Skipping pre-validation for backup move '{backup_move}'.", file=sys.stderr, flush=True)
                        is_valid_result = 1 # Tentatively assume valid

                    if is_valid_result != 0:
                        print(f"Transformer Client: Generated valid backup move: {backup_move}", file=sys.stderr, flush=True)
                        return backup_move
                    # else: continue searching for another empty square

                except Exception as e:
                     print(f"Transformer Client: Error validating backup move {backup_move}: {e}", file=sys.stderr, flush=True)
                     traceback.print_exc(file=sys.stderr)
                     # Continue search

        print("Transformer Client: Could not find a valid backup placement move.", file=sys.stderr, flush=True)
        return None # Indicate failure

    def apply_move(self, move_str):
        """Applies a move string (previously validated OR first attempt) to the internal game state."""
        if move_str:
             # print(f"Transformer Client: Applying self-generated move '{move_str}'...", file=sys.stderr, flush=True) # Keep minimal
             try:
                 result = self.game.execute_move(move_str)
                 # print(f"Transformer Client: self.game.execute_move('{move_str}') returned: {result}", file=sys.stderr, flush=True) # Can be noisy
             except Exception as e:
                 print(f"Transformer Client: EXCEPTION during self self.game.execute_move('{move_str}'): {e}", file=sys.stderr, flush=True)
                 traceback.print_exc(file=sys.stderr)
                 result = 0 # Treat exception as invalid move

             if result == 0:
                 print(f"Transformer Client: CRITICAL Error applying move {move_str} to internal state (returned 0).", file=sys.stderr, flush=True)
                 return False
             elif result in [2, 3]:
                 # print(f"Transformer Client: Game ended by self-generated move '{move_str}'.", file=sys.stderr, flush=True) # Covered by main loop
                 return "GAME_OVER"
             # print(f"Transformer Client: Successfully applied self-generated move '{move_str}'.", file=sys.stderr, flush=True) # Can be noisy
             return True
        return False # Should not happen if move_str is valid


def main():
    """Main function to handle communication with the game server via client.py"""
    player = None
    player_id = None

    try:
        # Read the initialization message from stdin
        init_line = input().strip()
        print(f"Transformer Client: Received init: {init_line}", file=sys.stderr, flush=True)
        parts = init_line.split()

        if len(parts) >= 3:
            player_id = parts[0]
            board_size = int(parts[1])
            time_limit = int(parts[2]) # Not used here

            print(f"Transformer Client: Initializing as Player {player_id} ({COLOR_RED if player_id == '1' else COLOR_YELLOW}Color{COLOR_RESET}) on {board_size}x{board_size} board", file=sys.stderr, flush=True)

            # Initialize the model and player state
            player = TransformerTakPlayer(
                model_path="tak_transformer.pth",
                board_size=board_size
            )

            # Player 1 makes the first move
            if player_id == '1':
                move = player.generate_model_move()
                if move:
                    apply_result = player.apply_move(move)
                    if apply_result and apply_result != "GAME_OVER":
                        print(move) # Send move to client.py via stdout
                        sys.stdout.flush()
                        print(f"Transformer Client: Sent move {move}", file=sys.stderr, flush=True)
                    else:
                         print(f"Transformer Client: Failed to apply generated move {move}. Exiting.", file=sys.stderr, flush=True)
                         sys.exit(1)
                else:
                    print("Transformer Client: Failed to generate initial move. Exiting.", file=sys.stderr, flush=True)
                    sys.exit(1)

            # Main game loop
            while True:
                # Read opponent's move from stdin
                try:
                    opponent_move = input().strip()
                    if not opponent_move:
                        print("Transformer Client: Received empty move, assuming game end.", file=sys.stderr, flush=True)
                        break
                    print(f"Transformer Client: Received opponent move: {opponent_move}", file=sys.stderr, flush=True)
                except EOFError:
                    print("Transformer Client: EOF received, assuming game end.", file=sys.stderr, flush=True)
                    break

                # Apply opponent's move
                apply_status = player.apply_opponent_move(opponent_move)
                if apply_status == "GAME_OVER":
                    print(f"Transformer Client: Game ended after opponent move {opponent_move}.", file=sys.stderr, flush=True)
                    break
                elif not apply_status:
                    print(f"Transformer Client: Failed to apply opponent move {opponent_move}. Exiting.", file=sys.stderr, flush=True)
                    break

                # Generate our move
                move = player.generate_model_move()
                if move:
                    apply_result = player.apply_move(move)
                    if apply_result and apply_result != "GAME_OVER":
                        print(move) # Send move to client.py via stdout
                        sys.stdout.flush()
                        print(f"Transformer Client: Sent move {move}", file=sys.stderr, flush=True)
                    elif apply_result == "GAME_OVER":
                         print(move) # Send final move
                         sys.stdout.flush()
                         print(f"Transformer Client: Sent final move {move}. Game Over.", file=sys.stderr, flush=True)
                         break
                    else:
                         print(f"Transformer Client: Failed to apply self-generated move {move}. Exiting.", file=sys.stderr, flush=True)
                         break
                else:
                    print("Transformer Client: Failed to generate a valid move. Exiting.", file=sys.stderr, flush=True)
                    break

        else:
            print(f"Transformer Client: Invalid initialization string received: '{init_line}'", file=sys.stderr, flush=True)
            sys.exit(1)

    except EOFError:
        print("Transformer Client: EOFError during game loop. Parent process likely closed pipe. Exiting.", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"Transformer Client Error: {str(e)}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        print("Transformer Client: Shutting down.", file=sys.stderr, flush=True)

if __name__ == "__main__":
    # Basic import checks
    if 'Game' not in globals():
         print("Transformer Client: Error - Game class not imported correctly.", file=sys.stderr); sys.exit(1)
    if 'TakTransformer' not in globals():
         print("Transformer Client: Error - TakTransformer class not imported correctly.", file=sys.stderr); sys.exit(1)
    if 'decode_move_vector' not in globals() or 'board_state_to_tensor' not in globals():
         print("Transformer Client: Error - decode_move_vector or board_state_to_tensor not imported correctly.", file=sys.stderr); sys.exit(1)

    main()
