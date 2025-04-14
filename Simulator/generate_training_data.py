import numpy as np
import os
import sys
import json
import time
import random
import argparse
from subprocess import Popen, PIPE
from nbstreamreader import NonBlockingStreamReader as NBSR
from Game import Game

class TakDataGenerator:
    def __init__(self, board_size=5, time_limit=600, games=100, output_dir="training_data"):
        self.board_size = board_size
        self.time_limit = time_limit
        self.games = games
        self.output_dir = output_dir
        self.compile_cpp()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create directory for transformer-ready data
        self.transformer_dir = f"{output_dir}/transformer_format"
        if not os.path.exists(self.transformer_dir):
            os.makedirs(self.transformer_dir)

    def compile_cpp(self):
        """Compile the C++ Tak AI if not already present"""
        if not os.path.exists("./tak_ai"):
            print("Compiling main.cpp...")
            os.system("g++ -O3 -o tak_ai main.cpp")

    def create_child_process(self, player_id):
        """Create a child process running the Tak AI"""
        process = Popen(["./tak_ai"], stdin=PIPE, stdout=PIPE, bufsize=0)
        return process, NBSR(process.stdout)

    def structured_encode_move(self, move: str) -> np.ndarray:
        """
        Encode a move string into a fixed-length integer vector of length 9.
        
        For placement moves (e.g., "Ff3", "Ss2", "Cc4"):
          - Index 0: 0 (placement indicator)
          - Index 1: Piece type: 0 for F, 1 for S, 2 for C
          - Index 2: Destination column (0-indexed, e.g., 'a' -> 0)
          - Index 3: Destination row (digit converted to int then subtracted by 1)
          - Index 4–8: Zeros
          
        For movement moves (e.g., "3a1>12"):
          - Index 0: 1 (movement indicator)
          - Index 1: Number of pieces picked up (first character as int)
          - Index 2: Origin column (0-indexed)
          - Index 3: Origin row (digit converted to int then subtracted by 1)
          - Index 4: Direction mapping: ">": 0, "<": 1, "+": 2, "-": 3
          - Index 5–8: Drop-partition tokens (each token as int; padded with zeros)
        """
        encoded = np.zeros(9, dtype=np.int32)
        # Check if the move is a placement move based on the first character
        if move[0] in ['F', 'S', 'C']:
            # Placement move indicator
            encoded[0] = 0
            piece_mapping = {'F': 0, 'S': 1, 'C': 2}
            encoded[1] = piece_mapping[move[0]]
            # The next two characters encode the destination coordinate:
            # Assume move[1] is the column letter and the remaining character(s) represent the row.
            col_letter = move[1]
            encoded[2] = ord(col_letter) - ord('a')
            # For a board of size 5-7, assume row is a single digit.
            # Adjust if needed for multi-digit rows.
            row_str = move[2:]
            encoded[3] = int(row_str) - 1
            # Indices 4-8 remain 0 as placeholders.
        else:
            # Movement move indicator
            encoded[0] = 1
            # The first character is the number of pieces picked up.
            encoded[1] = int(move[0])
            # Next is the origin coordinate: move[1] is column letter, move[2] is row digit.
            encoded[2] = ord(move[1]) - ord('a')
            encoded[3] = int(move[2]) - 1
            # The direction symbol is next (at index 3 in the move string, e.g., '>' in "3a1>12")
            direction_mapping = {">": 0, "<": 1, "+": 2, "-": 3}
            direction_char = move[3]
            encoded[4] = direction_mapping[direction_char]
            # The remaining part of the string contains the drop-partition tokens.
            drop_tokens_str = move[4:]
            # Store up to 4 drop tokens at indices 5 to 8, padding with zeros if necessary.
            for i, ch in enumerate(drop_tokens_str):
                if i < 4:
                    encoded[5 + i] = int(ch)
        return encoded

    def calculate_heuristics(self, game):
        """Calculate various heuristics for the current board state"""
        heuristics = {}
        # Flatstone count difference
        p1_flats = sum(1 for i in range(game.total_squares) 
                       for piece in game.board[i] if piece[0] == 0 and piece[1] == 'F')
        p2_flats = sum(1 for i in range(game.total_squares) 
                       for piece in game.board[i] if piece[0] == 1 and piece[1] == 'F')
        heuristics["flat_diff"] = p1_flats - p2_flats

        # Center control: use center square(s)
        center_squares = []
        mid = game.n // 2
        if game.n % 2 == 1:
            center_squares = [mid * game.n + mid]
        else:
            center_squares = [(mid-1) * game.n + (mid-1), 
                              (mid-1) * game.n + mid,
                              mid * game.n + (mid-1), 
                              mid * game.n + mid]
        p1_center = 0
        p2_center = 0
        for sq in center_squares:
            if game.board[sq] and game.board[sq][-1][0] == 0:
                p1_center += 1
            elif game.board[sq] and game.board[sq][-1][0] == 1:
                p2_center += 1
        heuristics["center_control"] = p1_center - p2_center

        # Stack control: total height controlled by each player
        p1_stack_height = sum(len(game.board[i]) for i in range(game.total_squares) 
                              if game.board[i] and game.board[i][-1][0] == 0)
        p2_stack_height = sum(len(game.board[i]) for i in range(game.total_squares) 
                              if game.board[i] and game.board[i][-1][0] == 1)
        heuristics["stack_control"] = p1_stack_height - p2_stack_height

        # Road potential: simplified connected pieces count
        p1_connected = self.count_connected_pieces(game, 0)
        p2_connected = self.count_connected_pieces(game, 1)
        heuristics["road_potential"] = p1_connected - p2_connected

        # Capstone position value: higher when capstone is closer to center
        p1_cap_value = self.evaluate_capstone_position(game, 0)
        p2_cap_value = self.evaluate_capstone_position(game, 1)
        heuristics["capstone_value"] = p1_cap_value - p2_cap_value

        return heuristics

    def count_connected_pieces(self, game, player):
        """Count connected pieces for road potential heuristic"""
        connected = 0
        for i in range(game.total_squares):
            if not game.board[i] or game.board[i][-1][0] != player:
                continue
            neighbours = game.get_neighbours(i)
            for neighbour in neighbours:
                if (game.board[neighbour] and 
                    game.board[neighbour][-1][0] == player and 
                    game.board[neighbour][-1][1] in ['F', 'C']):
                    connected += 1
        return connected // 2  # Each connection is counted twice

    def evaluate_capstone_position(self, game, player):
        """Evaluate how valuable a capstone's position is (closer to center gives higher value)"""
        value = 0
        for i in range(game.total_squares):
            if not game.board[i]:
                continue
            for piece in game.board[i]:
                if piece[0] == player and piece[1] == 'C':
                    row = i // game.n
                    col = i % game.n
                    center = game.n // 2
                    dist_from_center = abs(row - center) + abs(col - center)
                    value = game.n - dist_from_center
        return value

    def encode_board(self, game):
        """
        Convert game board to a format suitable for training.
        Each tile is encoded into:
          - a 6-bit binary number representing the colors of the top 6 pieces
          - a 2-bit binary number representing the type of the top piece
          - a normalized stack height.
        """
        encoded_board = []
        for i in range(game.total_squares):
            stack = game.board[i]
            if not stack:
                color_encoding = "000000"
                type_encoding = "00"
                stack_height = 0.0
            else:
                num_pieces = len(stack)
                bits = []
                pad_length = max(0, 6 - num_pieces)
                bits.extend(['0'] * pad_length)
                for piece in stack[-min(6, num_pieces):]:
                    bits.append(str(piece[0]))
                color_encoding = "".join(bits)
                top_piece_type = stack[-1][1]
                if top_piece_type == 'F':
                    type_encoding = "00"
                elif top_piece_type == 'S':
                    type_encoding = "01"
                elif top_piece_type == 'C':
                    type_encoding = "10"
                else:
                    type_encoding = "00"  # Fallback
                stack_height = len(stack) / float(game.n)
            encoded_board.append({
                "color_encoding": color_encoding,
                "type_encoding": type_encoding,
                "stack_height": stack_height
            })
        heuristics = self.calculate_heuristics(game)
        return {
            "board": encoded_board,
            "board_size": game.n,
            "player1_flats": game.players[0].flats,
            "player1_capstones": game.players[0].capstones,
            "player2_flats": game.players[1].flats,
            "player2_capstones": game.players[1].capstones,
            "turn": game.turn,
            "heuristics": heuristics
        }
    
    def board_to_tensor(self, board_state):
        """
        Convert a board state to a tensor representation for transformers.
        Channels:
         - Channels 0-5: Each bit of the 6-bit color encoding.
         - Channels 6-7: Each bit of the 2-bit type encoding.
         - Channel 8: Normalized stack height.
         - Channels 9-13: Five broadcasted heuristic values.
        Final tensor shape: (14, board_size, board_size)
        """
        n = board_state["board_size"]
        tensor = np.zeros((8, n, n), dtype=np.float32)
        for idx, tile in enumerate(board_state["board"]):
            row = idx // n
            col = idx % n
            color_str = tile["color_encoding"]
            if len(color_str) < 6:
                color_str = "0" * (6 - len(color_str)) + color_str
            for bit_idx in range(6):
                tensor[bit_idx, row, col] = float(color_str[bit_idx])
            type_str = tile["type_encoding"]
            if len(type_str) < 2:
                type_str = "0" * (2 - len(type_str)) + type_str
            for bit_idx in range(2):
                tensor[6 + bit_idx, row, col] = float(type_str[bit_idx])
        stack_channel = np.zeros((1, n, n), dtype=np.float32)
        for idx, tile in enumerate(board_state["board"]):
            row = idx // n
            col = idx % n
            stack_channel[0, row, col] = tile["stack_height"]
        tensor = np.vstack((tensor, stack_channel))
        heuristics = np.array([
            board_state["heuristics"]["flat_diff"] / n,
            board_state["heuristics"]["center_control"] / n,
            board_state["heuristics"]["stack_control"] / (n * n),
            board_state["heuristics"]["road_potential"] / (n * n),
            board_state["heuristics"]["capstone_value"] / n
        ], dtype=np.float32).reshape(5, 1, 1)
        heuristics_broadcasted = np.tile(heuristics, (1, n, n))
        tensor = np.vstack((tensor, heuristics_broadcasted))
        return tensor
    
    def create_transformer_training_sample(self, state, next_move):
        """
        Create a single training sample for transformer training.
        This now includes the encoded move as a 9-element vector.
        """
        input_tensor = self.board_to_tensor(state)
        encoded_move = self.structured_encode_move(next_move)
        
        # The following keys (move_type, move_dest) can be kept for debugging or further processing.
        move_type = None
        move_dest = None
        if next_move[0] in ['F', 'S', 'C']:
            move_type = next_move[0]
            move_dest = next_move[1:]
        else:
            parts = next_move.split('>')
            if len(parts) == 2:
                move_type = 'M'
                move_dest = parts[1]
        
        return {
            "input_tensor": input_tensor.tolist(),
            "encoded_move": encoded_move.tolist(),  # New encoded move vector (length 9)
            "move_type": move_type,
            "move_dest": move_dest,
            "full_move": next_move,
            "turn": state["turn"]
        }
    
    def run_games(self):
        """Run multiple games and collect transformer training samples."""
        transformer_samples = []
        
        for game_id in range(self.games):
            print(f"Starting game {game_id+1}/{self.games}...")
            game = Game(self.board_size, "CUI")
            p1_process, p1_reader = self.create_child_process(1)
            p2_process, p2_reader = self.create_child_process(2)
            
            p1_process.stdin.write(f"1 {self.board_size} {self.time_limit}\n".encode('utf-8'))
            p1_process.stdin.flush()
            
            p2_process.stdin.write(f"2 {self.board_size} {self.time_limit}\n".encode('utf-8'))
            p2_process.stdin.flush()
            
            game_data = {
                "game_id": game_id,
                "board_size": self.board_size,
                "time_limit": self.time_limit,
                "moves": [],
                "winner": None,
                "win_type": None
            }
            
            game_over = False
            move_count = 0
            
            try:
                # Player 1 makes the first move.
                p1_move = p1_reader.readline(timeout=self.time_limit).decode('utf-8').strip()
                if not p1_move:
                    print("Player 1 timed out on first move")
                    break
                
                result = game.execute_move(p1_move)
                board_state = self.encode_board(game)
                game_data["moves"].append({
                    "move_number": move_count,
                    "player": 1,
                    "move": p1_move,
                    "move_vector": self.structured_encode_move(p1_move).tolist(),
                    "board_state": board_state
                })
                move_count += 1
                
                if result > 1:
                    game_data["winner"] = result - 2
                    game_data["win_type"] = game.winner["type"]
                    game_over = True
                
                if not game_over:
                    p2_process.stdin.write(f"{p1_move}\n".encode('utf-8'))
                    p2_process.stdin.flush()
                
                last_state = board_state
                
                # Main game loop.
                while not game_over:
                    p2_move = p2_reader.readline(timeout=self.time_limit).decode('utf-8').strip()
                    if not p2_move:
                        print("Player 2 timed out")
                        game_data["winner"] = 0
                        break
                    
                    transformer_samples.append(
                        self.create_transformer_training_sample(last_state, p2_move)
                    )
                    
                    result = game.execute_move(p2_move)
                    board_state = self.encode_board(game)
                    last_state = board_state
                    game_data["moves"].append({
                        "move_number": move_count,
                        "player": 2,
                        "move": p2_move,
                        "move_vector": self.structured_encode_move(p2_move).tolist(),
                        "board_state": board_state
                    })
                    move_count += 1
                    
                    if result > 1:
                        game_data["winner"] = result - 2
                        game_data["win_type"] = game.winner["type"]
                        break
                    
                    p1_process.stdin.write(f"{p2_move}\n".encode('utf-8'))
                    p1_process.stdin.flush()
                    
                    p1_move = p1_reader.readline(timeout=self.time_limit).decode('utf-8').strip()
                    if not p1_move:
                        print("Player 1 timed out")
                        game_data["winner"] = 1
                        break
                    
                    transformer_samples.append(
                        self.create_transformer_training_sample(last_state, p1_move)
                    )
                    
                    result = game.execute_move(p1_move)
                    board_state = self.encode_board(game)
                    last_state = board_state
                    game_data["moves"].append({
                        "move_number": move_count,
                        "player": 1,
                        "move": p1_move,
                        "move_vector": self.structured_encode_move(p1_move).tolist(),
                        "board_state": board_state
                    })
                    move_count += 1
                    
                    if result > 1:
                        game_data["winner"] = result - 2
                        game_data["win_type"] = game.winner["type"]
                        break
                    
                    p2_process.stdin.write(f"{p1_move}\n".encode('utf-8'))
                    p2_process.stdin.flush()
                
                with open(f"{self.output_dir}/game_{game_id}.json", "w") as f:
                    json.dump(game_data, f, indent=2)
                
                if game_data["winner"] is not None:
                    for sample in transformer_samples[-move_count:]:
                        player = 0 if sample["turn"] == 0 else 1
                        sample["from_winner"] = (player == game_data["winner"])
                
                print(f"Game {game_id+1} completed. Winner: Player {game_data['winner']+1 if game_data['winner'] is not None else 'None'} " + 
                      (f"({game_data['win_type']} win)" if game_data["win_type"] else ""))
                
            except Exception as e:
                print(f"Error in game {game_id}: {str(e)}")
            finally:
                try:
                    p1_process.kill()
                    p2_process.kill()
                except:
                    pass
                time.sleep(1)
        
        with open(f"{self.transformer_dir}/transformer_data.json", "w") as f:
            json.dump(transformer_samples, f)
        
        print(f"Saved {len(transformer_samples)} training samples in transformer format")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Tak training data")
    parser.add_argument("--board_size", type=int, default=5, help="Board size (5-7)")
    parser.add_argument("--time_limit", type=int, default=300, help="Time limit per player in seconds")
    parser.add_argument("--games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--output_dir", type=str, default="tak_training_data", help="Output directory for training data")
    
    args = parser.parse_args()
    
    generator = TakDataGenerator(
        board_size=args.board_size,
        time_limit=args.time_limit,
        games=args.games,
        output_dir=args.output_dir
    )
    
    generator.run_games()
