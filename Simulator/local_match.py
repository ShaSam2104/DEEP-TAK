import subprocess
import sys
import time
import argparse
import os
from threading import Thread
from queue import Queue, Empty
from nbstreamreader import NonBlockingStreamReader as NBSR # Assuming nbstreamreader.py is in the same directory

def start_player_process(player_script_path, player_id, board_size, time_limit):
    """Starts a player process and returns the process object and its non-blocking reader."""
    if not os.path.exists(player_script_path):
        print(f"Error: Player script not found at {player_script_path}", file=sys.stderr)
        return None, None
        
    # Determine execution command based on script extension
    if player_script_path.endswith('.sh'):
        execution_command = ['sh', player_script_path]
    elif player_script_path.endswith('.py'):
        execution_command = ['python', player_script_path]
    elif player_script_path.endswith('.cpp'): # Assuming compiled to 'a.out' or similar if run directly
         print(f"Warning: Direct execution of .cpp not standard. Assuming pre-compiled executable at {player_script_path}", file=sys.stderr)
         execution_command = [player_script_path] # Or adjust if compilation needed
    else: # Default to shell execution for other types or executables
        execution_command = [player_script_path]

    print(f"Starting Player {player_id} with command: {' '.join(execution_command)}", file=sys.stderr)
    
    try:
        # Use bufsize=0 for unbuffered I/O, text=True for automatic encoding/decoding
        process = subprocess.Popen(
            execution_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, # Capture stderr for debugging
            bufsize=0,
            text=True, # Use text mode for stdin/stdout
            universal_newlines=True # Ensure consistent line endings
        )
    except Exception as e:
        print(f"Error starting process for {player_script_path}: {e}", file=sys.stderr)
        return None, None

    # Send initialization string immediately after starting
    init_string = f"{player_id} {board_size} {time_limit}\n"
    try:
        process.stdin.write(init_string)
        process.stdin.flush()
        print(f"Sent init to Player {player_id}: {init_string.strip()}", file=sys.stderr)
    except (IOError, BrokenPipeError) as e:
         print(f"Error sending init string to Player {player_id}: {e}", file=sys.stderr)
         process.terminate()
         return None, None

    # Create non-blocking readers for stdout and stderr
    stdout_reader = NBSR(process.stdout)
    stderr_reader = NBSR(process.stderr)

    # Start a thread to print stderr for debugging
    def print_stderr(reader, p_id):
        while True:
            line = reader.readline(0.1) # Small timeout
            if line:
                print(f"[Player {p_id} ERR]: {line.strip()}", file=sys.stderr, flush=True)
            elif process.poll() is not None: # Check if process ended
                # Read any remaining lines
                while True:
                    line = reader.readline(0.01)
                    if line:
                        print(f"[Player {p_id} ERR]: {line.strip()}", file=sys.stderr, flush=True)
                    else:
                        break
                break # Exit thread when process ends

    stderr_thread = Thread(target=print_stderr, args=(stderr_reader, player_id), daemon=True)
    stderr_thread.start()


    return process, stdout_reader

def play_local_match(player1_script, player2_script, board_size, time_limit):
    """Plays a local match between two AI scripts."""

    p1_proc, p1_reader = start_player_process(player1_script, 1, board_size, time_limit)
    p2_proc, p2_reader = start_player_process(player2_script, 2, board_size, time_limit)

    if not p1_proc or not p2_proc:
        print("Failed to start one or both player processes. Exiting.", file=sys.stderr)
        if p1_proc: p1_proc.terminate()
        if p2_proc: p2_proc.terminate()
        return

    current_player_proc = p1_proc
    current_player_reader = p1_reader
    other_player_proc = p2_proc
    current_player_id = 1
    move_count = 0

    try:
        while True:
            # Check if processes are still running
            if current_player_proc.poll() is not None:
                print(f"Player {current_player_id} process terminated unexpectedly.", file=sys.stderr)
                break
            if other_player_proc.poll() is not None:
                print(f"Player {3 - current_player_id} process terminated unexpectedly.", file=sys.stderr)
                break

            # Read move from the current player
            # Use a timeout slightly longer than expected move time, or loop with smaller timeouts
            print(f"\nWaiting for Player {current_player_id}'s move...", file=sys.stderr)
            move = None
            read_start_time = time.time()
            # Allow generous time for reading, actual time limit is handled by the AI (or ignored here)
            read_timeout = time_limit + 60 # Seconds
            
            while time.time() - read_start_time < read_timeout:
                 line = current_player_reader.readline(0.5) # Read with 0.5s timeout
                 if line:
                     move = line.strip()
                     if move: # Got a non-empty move
                         break
                 if current_player_proc.poll() is not None: # Check if process died while waiting
                     print(f"Player {current_player_id} process terminated while waiting for move.", file=sys.stderr)
                     move = None # Ensure move is None
                     break
            
            if move is None:
                 if current_player_proc.poll() is None:
                     print(f"Timeout or empty move received from Player {current_player_id}. Terminating.", file=sys.stderr)
                 break # Exit loop if no move received or process died

            move_count += 1
            print(f"Move {move_count} (Player {current_player_id}): {move}", file=sys.stderr)
            print(f"Move {move_count} (Player {current_player_id}): {move}") # Also print to main stdout for visibility

            # Send move to the other player
            try:
                other_player_proc.stdin.write(move + '\n')
                other_player_proc.stdin.flush()
            except (IOError, BrokenPipeError) as e:
                print(f"Error sending move to Player {3 - current_player_id}: {e}. Terminating.", file=sys.stderr)
                break

            # Switch turns
            current_player_proc, other_player_proc = other_player_proc, current_player_proc
            current_player_reader = p2_reader if current_player_id == 1 else p1_reader
            current_player_id = 3 - current_player_id

    except KeyboardInterrupt:
        print("\nGame interrupted by user.", file=sys.stderr)
    finally:
        print("Cleaning up processes...", file=sys.stderr)
        if p1_proc and p1_proc.poll() is None:
            p1_proc.terminate()
        if p2_proc and p2_proc.poll() is None:
            p2_proc.terminate()
        # Wait briefly for processes to terminate
        time.sleep(1)
        if p1_proc and p1_proc.poll() is None:
            p1_proc.kill()
            print("Killed Player 1 process.", file=sys.stderr)
        if p2_proc and p2_proc.poll() is None:
            p2_proc.kill()
            print("Killed Player 2 process.", file=sys.stderr)
        print("Game finished.", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a local Tak match between two AI scripts.")
    parser.add_argument("player1_script", help="Path to the script for Player 1 (e.g., run.sh or run_transformer.sh)")
    parser.add_argument("player2_script", help="Path to the script for Player 2 (e.g., run.sh or run_transformer.sh)")
    parser.add_argument("-n", "--board_size", type=int, default=5, help="Board size (default: 5)")
    parser.add_argument("-t", "--time_limit", type=int, default=300, help="Time limit per player in seconds (Note: enforcement depends on AI implementation)")

    args = parser.parse_args()

    play_local_match(args.player1_script, args.player2_script, args.board_size, args.time_limit)