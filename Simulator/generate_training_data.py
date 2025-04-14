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

    def compile_cpp(self):
        """Compile the C++ program if needed"""
        if not os.path.exists("./tak_ai"):
            print("Compiling main.cpp...")
            os.system("g++ -O3 -o tak_ai main.cpp")

    def create_child_process(self, player_id):
        """Create a child process running the Tak AI"""
        process = Popen(["./tak_ai"], stdin=PIPE, stdout=PIPE, bufsize=0)
        return process, NBSR(process.stdout)

    def encode_board(self, game):
        """Convert game board to a format suitable for training"""
        encoded_board = []
        for i in range(game.total_squares):
            square = []
            for piece in game.board[i]:
                player, piece_type = piece
                # Encode: player (0/1), piece type (F/S/C)
                square.append({"player": player, "type": piece_type})
            encoded_board.append(square)
        
        return {
            "board": encoded_board,
            "board_size": game.n,
            "player1_flats": game.players[0].flats,
            "player1_capstones": game.players[0].capstones,
            "player2_flats": game.players[1].flats,
            "player2_capstones": game.players[1].capstones,
            "turn": game.turn
        }

    def run_games(self):
        """Run multiple games and collect data"""
        for game_id in range(self.games):
            print(f"Starting game {game_id+1}/{self.games}...")
            
            # Create game instance and child processes
            game = Game(self.board_size, "CUI")
            p1_process, p1_reader = self.create_child_process(1)
            p2_process, p2_reader = self.create_child_process(2)
            
            # Initialize player 1 (sends: player_id, board_size, time_limit)
            p1_process.stdin.write(f"1 {self.board_size} {self.time_limit}\n".encode('utf-8'))
            p1_process.stdin.flush()
            
            # Initialize player 2 (sends: player_id, board_size, time_limit)
            p2_process.stdin.write(f"2 {self.board_size} {self.time_limit}\n".encode('utf-8'))
            p2_process.stdin.flush()
            
            # Store the complete game data
            game_data = {
                "game_id": game_id,
                "board_size": self.board_size,
                "time_limit": self.time_limit,
                "moves": [],
                "winner": None,
                "win_type": None
            }
            
            # Game loop
            game_over = False
            move_count = 0
            
            try:
                # Player 1 makes the first move
                p1_move = p1_reader.readline(timeout=self.time_limit).decode('utf-8').strip()
                if not p1_move:
                    print("Player 1 timed out on first move")
                    break
                
                # Update game state with the move
                result = game.execute_move(p1_move)
                
                # Record the move
                game_data["moves"].append({
                    "move_number": move_count,
                    "player": 1,
                    "move": p1_move,
                    "board_state": self.encode_board(game)
                })
                move_count += 1
                
                # Check if game ended
                if result > 1:
                    game_data["winner"] = result - 2
                    game_data["win_type"] = game.winner["type"]
                    game_over = True
                
                # Send the move to player 2
                if not game_over:
                    p2_process.stdin.write(f"{p1_move}\n".encode('utf-8'))
                    p2_process.stdin.flush()
                
                # Main game loop
                while not game_over:
                    # Player 2's turn
                    p2_move = p2_reader.readline(timeout=self.time_limit).decode('utf-8').strip()
                    if not p2_move:
                        print("Player 2 timed out")
                        game_data["winner"] = 0  # Player 1 wins by default
                        break
                    
                    # Update game state
                    result = game.execute_move(p2_move)
                    
                    # Record the move
                    game_data["moves"].append({
                        "move_number": move_count,
                        "player": 2,
                        "move": p2_move,
                        "board_state": self.encode_board(game)
                    })
                    move_count += 1
                    
                    # Check if game ended
                    if result > 1:
                        game_data["winner"] = result - 2
                        game_data["win_type"] = game.winner["type"]
                        break
                    
                    # Send the move to player 1
                    p1_process.stdin.write(f"{p2_move}\n".encode('utf-8'))
                    p1_process.stdin.flush()
                    
                    # Player 1's turn
                    p1_move = p1_reader.readline(timeout=self.time_limit).decode('utf-8').strip()
                    if not p1_move:
                        print("Player 1 timed out")
                        game_data["winner"] = 1  # Player 2 wins by default
                        break
                    
                    # Update game state
                    result = game.execute_move(p1_move)
                    
                    # Record the move
                    game_data["moves"].append({
                        "move_number": move_count,
                        "player": 1,
                        "move": p1_move,
                        "board_state": self.encode_board(game)
                    })
                    move_count += 1
                    
                    # Check if game ended
                    if result > 1:
                        game_data["winner"] = result - 2
                        game_data["win_type"] = game.winner["type"]
                        break
                    
                    # Send the move to player 2
                    p2_process.stdin.write(f"{p1_move}\n".encode('utf-8'))
                    p2_process.stdin.flush()
                
                # Save game data
                with open(f"{self.output_dir}/game_{game_id}.json", "w") as f:
                    json.dump(game_data, f, indent=2)
                
                print(f"Game {game_id+1} completed. Winner: Player {game_data['winner']+1} ({game_data['win_type']} win)")
                
            except Exception as e:
                print(f"Error in game {game_id}: {str(e)}")
            finally:
                # Clean up processes
                try:
                    p1_process.kill()
                    p2_process.kill()
                except:
                    pass
                
                # Short delay between games
                time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Tak training data")
    parser.add_argument("--board_size", type=int, default=5, help="Board size (5-7)")
    parser.add_argument("--time_limit", type=int, default=300, help="Time limit per player in seconds")
    parser.add_argument("--games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--output_dir", type=str, default="training_data", help="Output directory for training data")
    
    args = parser.parse_args()
    
    generator = TakDataGenerator(
        board_size=args.board_size,
        time_limit=args.time_limit,
        games=args.games,
        output_dir=args.output_dir
    )
    
    generator.run_games()