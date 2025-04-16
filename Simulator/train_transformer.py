import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np # Make sure numpy is imported
import math
import glob
import argparse
import time
import traceback # For better error reporting

# --- Decoder Function ---
def decode_move_vector(encoded_vector: np.ndarray) -> str:
    """
    Decodes a 9-element integer vector back into a Tak move string.
    Inverse of structured_encode_move.

    Args:
        encoded_vector: A numpy array or list of 9 integers.

    Returns:
        The decoded Tak move string, or None if decoding fails.
    """
    if not isinstance(encoded_vector, (np.ndarray, list)) or len(encoded_vector) != 9:
        # print("Error: Input must be a 9-element vector/list.") # Reduce noise during training
        return None

    move_indicator = encoded_vector[0]

    try:
        if move_indicator == 0:  # Placement move
            piece_map_inv = {0: 'F', 1: 'S', 2: 'C'}
            piece_type_idx = encoded_vector[1]
            if piece_type_idx not in piece_map_inv:
                # print(f"Error: Invalid piece type index {piece_type_idx} for placement.")
                return None
            piece_type = piece_map_inv[piece_type_idx]

            col_idx = encoded_vector[2]
            row_idx = encoded_vector[3]

            col_char = chr(ord('a') + col_idx)
            row_num = row_idx + 1 # Tak rows are 1-based

            # Basic validation for coordinates (adjust if board size is known and needed)
            # Example: board_size = 5
            # if not (0 <= col_idx < 5 and 0 <= row_idx < 5): return None

            return f"{piece_type}{col_char}{row_num}"

        elif move_indicator == 1:  # Movement move
            num_pieces = encoded_vector[1]
            if num_pieces <= 0: return None # Number of pieces must be > 0

            col_idx = encoded_vector[2]
            row_idx = encoded_vector[3]

            col_char = chr(ord('a') + col_idx)
            row_num = row_idx + 1 # Tak rows are 1-based

            # Basic validation for coordinates
            # Example: board_size = 5
            # if not (0 <= col_idx < 5 and 0 <= row_idx < 5): return None

            direction_map_inv = {0: '>', 1: '<', 2: '+', 3: '-'}
            direction_idx = encoded_vector[4]
            if direction_idx not in direction_map_inv: return None
            direction_char = direction_map_inv[direction_idx]

            drop_tokens = []
            total_dropped = 0
            for i in range(5, 9): # Indices 5, 6, 7, 8 correspond to drop counts
                drop_count = encoded_vector[i]
                if drop_count > 0:
                    drop_tokens.append(str(drop_count))
                    total_dropped += drop_count
                else:
                    # Allow for implicit full drop at the end if drop_tokens is empty
                    if not drop_tokens: # If no drops specified yet, assume full drop at end
                         pass # Will be handled later if total_dropped == 0
                    # If some drops were specified, a 0 means padding, stop here.
                    elif drop_tokens:
                         break

            # If no drops specified explicitly, assume all pieces dropped on last square
            if total_dropped == 0:
                 drop_string = str(num_pieces) # Implicit full drop
            # If drops were specified, ensure they sum to num_pieces
            elif total_dropped != num_pieces:
                 # print(f"Warning: Decoded drop counts ({total_dropped}) != num_pieces ({num_pieces})")
                 # Allow this for now, game engine should validate
                 drop_string = "".join(drop_tokens)
            else:
                 drop_string = "".join(drop_tokens)


            if not drop_string: return None # Should have drops if moving

            return f"{num_pieces}{col_char}{row_num}{direction_char}{drop_string}"

        else:
            # print(f"Error: Invalid move indicator {move_indicator}. Must be 0 or 1.")
            return None

    except (IndexError, ValueError, TypeError) as e:
        # print(f"Error during decoding: {e}") # Reduce noise
        return None

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: [max_len, 1, d_model] -> [max_len, 1, d_model]
        self.register_buffer('pe', pe) # Register as buffer so it's part of state_dict but not parameters

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# --- Board Encoding Helper (Corrected Logic) ---
def board_state_to_tensor(board_state_dict, board_size, ai_heuristic_value):
    """
    Converts the board_state dictionary (matching _game_to_board_state_dict format)
    into the 14-channel tensor, using the provided ai_heuristic_value for the last two channels.
    """
    # Handle potential None or invalid ai_heuristic_value
    if not isinstance(ai_heuristic_value, (list, tuple)) or len(ai_heuristic_value) != 2:
        # print(f"Warning: Invalid ai_heuristic_value received: {ai_heuristic_value}. Using [0.0, 0.0].")
        ai_heuristic_value = [0.0, 0.0]
    try:
        # Basic normalization/scaling - adjust the divisor as needed based on typical value ranges
        # These values are used ONLY during training. Inference uses [0.0, 0.0].
        heur_val_1 = float(ai_heuristic_value[0]) / 100.0 # Example scaling
        heur_val_2 = float(ai_heuristic_value[1]) / 100.0 # Example scaling
    except (ValueError, TypeError):
        # print(f"Warning: Could not convert ai_heuristic_value {ai_heuristic_value} to float. Using [0.0, 0.0].")
        heur_val_1 = 0.0
        heur_val_2 = 0.0

    # Validate input dictionary structure
    if not board_state_dict or 'board' not in board_state_dict or \
       not isinstance(board_state_dict['board'], list) or \
       len(board_state_dict['board']) != board_size * board_size:
        print(f"Warning: Invalid or missing board state dict structure. Board size {board_size}. Dict: {board_state_dict}")
        # Return mostly zeros but keep the heuristic channels (as inference expects 14 channels)
        tensor_data = np.zeros((14, board_size, board_size), dtype=np.float32)
        tensor_data[12, :, :] = heur_val_1 # Populated during training, 0 during inference
        tensor_data[13, :, :] = heur_val_2 # Populated during training, 0 during inference
        return torch.from_numpy(tensor_data)

    # Initialize channels
    player1_flat = np.zeros((board_size, board_size), dtype=np.float32)
    player1_wall = np.zeros((board_size, board_size), dtype=np.float32)
    player1_cap = np.zeros((board_size, board_size), dtype=np.float32)
    player2_flat = np.zeros((board_size, board_size), dtype=np.float32)
    player2_wall = np.zeros((board_size, board_size), dtype=np.float32)
    player2_cap = np.zeros((board_size, board_size), dtype=np.float32)
    stack_height = np.zeros((board_size, board_size), dtype=np.float32) # Use raw count

    max_possible_height = board_size # Example for potential normalization if needed later

    # --- Board Piece Encoding ---
    for idx, square_data in enumerate(board_state_dict['board']):
        # Calculate row and column (Tak board: a1 is bottom-left)
        # idx 0 = a5, idx 4 = e5
        # idx 20 = a1, idx 24 = e1
        row_from_top = idx // board_size # 0-based row from top
        col = idx % board_size          # 0-based col from left
        r = (board_size - 1) - row_from_top # 0-based row from bottom for tensor indexing

        stack = square_data.get('stack', [])
        height = len(stack)
        stack_height[r, col] = float(height) # Store raw height

        if height > 0:
            top_piece = stack[-1] # Get the last piece in the stack (top piece)
            top_player = top_piece.get('player', -1) # 0 or 1
            top_type_char = top_piece.get('type', '') # 'F', 'S', or 'C'

            # Fill channels based on top piece
            if top_player == 0: # Player 1
                if top_type_char == 'F': player1_flat[r, col] = 1.0
                elif top_type_char == 'S': player1_wall[r, col] = 1.0
                elif top_type_char == 'C': player1_cap[r, col] = 1.0
            elif top_player == 1: # Player 2
                if top_type_char == 'F': player2_flat[r, col] = 1.0
                elif top_type_char == 'S': player2_wall[r, col] = 1.0
                elif top_type_char == 'C': player2_cap[r, col] = 1.0
    # --- End Board Piece Encoding ---

    # Player info channels (normalize counts)
    # Calculate max possible flats based on board size (adjust if rules differ)
    max_flats_per_player = {3:10, 4:15, 5:21, 6:30, 7:40, 8:50}.get(board_size, 21) # Default to 5x5
    max_caps_per_player = {3:0, 4:0, 5:1, 6:1, 7:2, 8:2}.get(board_size, 1) # Default to 5x5

    p1_flats_norm = board_state_dict.get('player1_flats', 0) / max_flats_per_player if max_flats_per_player > 0 else 0.0
    p1_caps_norm = board_state_dict.get('player1_capstones', 0) / max_caps_per_player if max_caps_per_player > 0 else 0.0
    p2_flats_norm = board_state_dict.get('player2_flats', 0) / max_flats_per_player if max_flats_per_player > 0 else 0.0
    p2_caps_norm = board_state_dict.get('player2_capstones', 0) / max_caps_per_player if max_caps_per_player > 0 else 0.0
    turn = board_state_dict.get('turn', 0) # 0 or 1

    player1_flats_channel = np.full((board_size, board_size), p1_flats_norm, dtype=np.float32)
    player1_caps_channel = np.full((board_size, board_size), p1_caps_norm, dtype=np.float32)
    player2_flats_channel = np.full((board_size, board_size), p2_flats_norm, dtype=np.float32)
    player2_caps_channel = np.full((board_size, board_size), p2_caps_norm, dtype=np.float32)
    turn_channel = np.full((board_size, board_size), turn, dtype=np.float32) # Channel indicates whose turn it is

    # AI Heuristic channels (using the passed-in, potentially scaled values)
    ai_heur_1_channel = np.full((board_size, board_size), heur_val_1, dtype=np.float32)
    ai_heur_2_channel = np.full((board_size, board_size), heur_val_2, dtype=np.float32)

    # Stack channels - Order must be consistent! (14 channels total)
    tensor_data = np.stack([
        player1_flat, player1_wall, player1_cap,           # 0, 1, 2
        player2_flat, player2_wall, player2_cap,           # 3, 4, 5
        stack_height,                                      # 6
        player1_flats_channel, player1_caps_channel,       # 7, 8
        player2_flats_channel, player2_caps_channel,       # 9, 10
        turn_channel,                                      # 11
        ai_heur_1_channel, ai_heur_2_channel               # 12, 13
    ], axis=0)

    tensor = torch.from_numpy(tensor_data)

    # Ensure correct shape (14, H, W)
    if tensor.shape != (14, board_size, board_size):
         raise ValueError(f"Generated tensor has shape {tensor.shape}, expected {(14, board_size, board_size)}.")

    return tensor


# --- Tak Dataset (Modified to handle ai_heuristic_value) ---
class TakDataset(Dataset):
    def __init__(self, data_dir, board_size, max_game_id=9): # Default max_game_id
        self.data_files = []
        print(f"Looking for game files from game_0.json to game_{max_game_id}.json in {data_dir}...")
        for i in range(max_game_id + 1): # Loop from 0 to max_game_id inclusive
            filename = f"game_{i}.json"
            file_path = os.path.join(data_dir, filename)
            if os.path.exists(file_path):
                self.data_files.append(file_path)
            # else:
                # print(f"Warning: File not found: {file_path}") # Optional warning

        if not self.data_files:
             raise ValueError(f"No game files found in the range 0-{max_game_id} in {data_dir}")

        self.samples = []
        self.board_size = board_size
        print(f"Found {len(self.data_files)} game files to process.")

        # Define initial empty board state dictionary structure
        initial_board_state = {
            "board": [{"stack": []}] * (board_size * board_size),
            "board_size": board_size,
            "player1_flats": {3:10, 4:15, 5:21, 6:30, 7:40, 8:50}.get(board_size, 21),
            "player1_capstones": {3:0, 4:0, 5:1, 6:1, 7:2, 8:2}.get(board_size, 1),
            "player2_flats": {3:10, 4:15, 5:21, 6:30, 7:40, 8:50}.get(board_size, 21),
            "player2_capstones": {3:0, 4:0, 5:1, 6:1, 7:2, 8:2}.get(board_size, 1),
            "turn": 0, # Player 1 starts
        }
        # Initial heuristic value before the first move (defaults to [0, 0])
        initial_ai_heuristic = [0.0, 0.0]

        for file_path in self.data_files:
            try:
                with open(file_path, 'r') as f:
                    game_data = json.load(f)

                    if not game_data or 'moves' not in game_data or not isinstance(game_data['moves'], list):
                        print(f"Warning: Skipping file {file_path} due to missing/invalid 'moves' list.")
                        continue

                    last_state_dict = initial_board_state
                    last_ai_heuristic = initial_ai_heuristic # Heuristic corresponding to last_state_dict

                    for move_data in game_data['moves']:
                        # Check if essential keys exist
                        if 'board_state' in move_data and 'move_vector' in move_data and 'ai_heuristic_value' in move_data:
                            current_target_vector = move_data['move_vector']
                            current_board_state = move_data['board_state'] # State *after* the move
                            # Heuristic value *before* the move (associated with last_state_dict)
                            heuristic_for_last_state = move_data.get('ai_heuristic_value', [0.0, 0.0])
                            if heuristic_for_last_state is None: # Handle explicit null
                                heuristic_for_last_state = [0.0, 0.0]

                            # Ensure target vector is valid
                            if isinstance(current_target_vector, (list, tuple)) and len(current_target_vector) == 9:
                                # Add the pair:
                                # (state *before* this move, heuristic *before* this move, move_vector *of* this move)
                                self.samples.append((last_state_dict, heuristic_for_last_state, current_target_vector))

                                # Update state for the next iteration
                                last_state_dict = current_board_state
                                # The heuristic for the *next* state isn't directly available here,
                                # it will be read from the *next* move_data entry.
                            else:
                                print(f"Warning: Invalid move_vector found in {file_path}, move {move_data.get('move_number', '?')}")
                        else:
                             print(f"Warning: Missing board_state, move_vector, or ai_heuristic_value in {file_path}, move {move_data.get('move_number', '?')}")
                             # Don't break, just skip this move maybe? Or break if critical. Let's skip.
                             # break

            except json.JSONDecodeError:
                print(f"Warning: Skipping corrupted JSON file: {file_path}")
            except Exception as e:
                print(f"Warning: Error processing file {file_path}: {e}")
                traceback.print_exc() # Print full traceback for debugging

        if not self.samples:
             raise ValueError(f"No valid training samples could be extracted from the specified game files.")
        print(f"Processed {len(self.samples)} total state-action samples.")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Unpack state, heuristic, and target
        state_dict, ai_heuristic, target_move_vector = self.samples[idx]

        # Convert state dict and heuristic to tensor
        # ai_heuristic is the value *before* the target_move_vector was made
        input_tensor = board_state_to_tensor(state_dict, self.board_size, ai_heuristic)

        # Target move vector
        encoded_move = torch.tensor(target_move_vector, dtype=torch.long)

        return input_tensor, encoded_move


# --- Tak Transformer Model ---
class TakTransformer(nn.Module):
    def __init__(self, board_size, input_channels=14, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=1024, dropout=0.1):
        super(TakTransformer, self).__init__()
        self.board_size = board_size
        self.d_model = d_model
        self.input_channels = input_channels # Should be 14

        # --- CNN Stem ---
        self.cnn_stem = nn.Sequential(
            nn.Conv2d(self.input_channels, d_model // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(d_model // 4),
            nn.Conv2d(d_model // 4, d_model // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(d_model // 2),
            nn.Conv2d(d_model // 2, d_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(d_model)
            # Output shape: (Batch, d_model, board_size, board_size)
        )

        # --- Positional Encoding & Transformer ---
        # self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=board_size*board_size) # Optional explicit
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # --- Output Heads ---
        self.output_dims = [
            2,                      # Head 0: Move Type
            board_size + 1,         # Head 1: Piece/Count
            board_size,             # Head 2: Column
            board_size,             # Head 3: Row
            4,                      # Head 4: Direction
            board_size + 1,         # Head 5: Drop 1
            board_size + 1,         # Head 6: Drop 2
            board_size + 1,         # Head 7: Drop 3
            board_size + 1          # Head 8: Drop 4
        ]

        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, dim) for dim in self.output_dims
        ])

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for head in self.output_heads:
            head.bias.data.zero_()
            head.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src shape: (Batch, Channels=14, Height, Width)
        if src.shape[1] != self.input_channels:
             raise ValueError(f"Input tensor has {src.shape[1]} channels, but model expects {self.input_channels}")

        # 1. CNN Stem
        src = self.cnn_stem(src) # (B, d_model, H, W)

        # 2. Flatten and Permute for Transformer
        batch_size = src.shape[0]
        src = src.flatten(2) # (B, d_model, H*W)
        src = src.permute(0, 2, 1) # (B, H*W, d_model) - batch_first=True format

        # 3. Transformer Encoder (expects batch_first=True)
        transformer_output = self.transformer_encoder(src) # (B, SeqLen, d_model)

        # 4. Aggregate Transformer Output (e.g., mean pooling over sequence length)
        pooled_output = transformer_output.mean(dim=1) # (B, d_model)

        # 5. Output Heads
        outputs = [head(pooled_output) for head in self.output_heads]

        return outputs

# --- Training Function ---
def train_epoch(model, dataloader, criterion, optimizer, device, epoch, log_interval=100):
    model.train()
    total_loss = 0.
    start_time = time.time()

    for i, (data, targets) in enumerate(dataloader):
        data, targets = data.to(device), targets.to(device) # Move batch to device

        optimizer.zero_grad()
        outputs = model(data) # List of 9 outputs

        # Calculate loss for each head and sum them up
        losses = []
        for head_idx in range(9):
            loss = criterion(outputs[head_idx], targets[:, head_idx])
            losses.append(loss)

        total_batch_loss = sum(losses)
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Gradient clipping
        optimizer.step()

        total_loss += total_batch_loss.item()

        if i % log_interval == 0 and i > 0:
            lr = optimizer.param_groups[0]['lr']
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            print(f'| Epoch {epoch:3d} | {i:5d}/{len(dataloader):5d} batches | '
                  f'lr {lr:02.2e} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f}')
            total_loss = 0
            start_time = time.time()

# --- Evaluation Function ---
def evaluate(model, dataloader, criterion, device, print_samples=2):
    model.eval()
    total_loss = 0.
    total_correct = [0] * 9 # Track correct predictions per head
    total_samples = 0
    samples_printed = 0

    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data) # List of 9 logit tensors [(B, H0_dim), (B, H1_dim), ...]
            batch_size = data.size(0)
            total_samples += batch_size

            losses = []
            predictions_list = [] # Store predictions for decoding
            for head_idx in range(9):
                loss = criterion(outputs[head_idx], targets[:, head_idx])
                losses.append(loss.item())
                # Calculate accuracy for this head
                predictions = outputs[head_idx].argmax(dim=1) # Shape: (B)
                predictions_list.append(predictions)
                total_correct[head_idx] += (predictions == targets[:, head_idx]).sum().item()

            total_loss += sum(losses) # Sum loss over heads for the batch

            # --- Decode and Print Sample Predictions ---
            if samples_printed < print_samples:
                predicted_vectors = torch.stack(predictions_list, dim=1).cpu().numpy()
                target_vectors = targets.cpu().numpy()

                num_to_print = min(print_samples - samples_printed, batch_size)
                for i in range(num_to_print):
                    pred_vec = predicted_vectors[i]
                    target_vec = target_vectors[i]
                    decoded_pred = decode_move_vector(pred_vec)
                    decoded_target = decode_move_vector(target_vec) # Decode target for comparison
                    print(f"  Sample {samples_printed + i}:")
                    print(f"    Target Vec : {target_vec} -> Decoded: {decoded_target}")
                    print(f"    Predict Vec: {pred_vec} -> Decoded: {decoded_pred}")
                samples_printed += num_to_print
            # --- End Decode/Print ---

    # Calculate average loss per sample (average over batches and heads)
    avg_loss = total_loss / len(dataloader) # Average loss per batch
    accuracies = [(correct / total_samples) * 100 for correct in total_correct]
    return avg_loss, accuracies


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tak Transformer Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing raw game_*.json files")
    parser.add_argument("--board_size", type=int, required=True, help="Board size used during data generation (e.g., 5)")
    parser.add_argument("--max_game_id", type=int, default=9, help="Maximum game ID to load (e.g., 9 loads game_0.json to game_9.json)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension (embedding size)")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of Transformer encoder layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--save_path", type=str, default="tak_transformer.pth", help="Path to save the trained model")
    parser.add_argument("--log_interval", type=int, default=100, help="Log training progress every N batches")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--validation_split", type=float, default=0.1, help="Fraction of data to use for validation (0.0 to 1.0)")


    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset and DataLoader ---
    full_dataset = TakDataset(args.data_dir, args.board_size, max_game_id=args.max_game_id)

    # Split dataset
    val_size = int(args.validation_split * len(full_dataset))
    train_size = len(full_dataset) - val_size

    if val_size == 0 and train_size > 0:
        print("Warning: Validation set size is 0 based on split ratio. Using entire dataset for training.")
        train_dataset = full_dataset
        val_dataloader = None
    elif train_size == 0:
         raise ValueError("Dataset is too small to create even a training set.")
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True if device=='cuda' else False)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True if device=='cuda' else False)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {val_size}")

    # --- Model, Loss, Optimizer ---
    model = TakTransformer(
        board_size=args.board_size,
        input_channels=14, # Fixed at 14
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1.0, gamma=0.95) # Example scheduler

    # --- Training Loop ---
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_epoch(model, train_dataloader, criterion, optimizer, device, epoch, args.log_interval)

        val_loss = float('inf') # Default if no validation
        val_accuracies = [0.0] * 9 # Default if no validation

        if val_dataloader: # Only evaluate if validation set exists
            val_loss, val_accuracies = evaluate(model, val_dataloader, criterion, device, print_samples=2)
            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | valid loss {val_loss:5.2f}')
            print(f'| Validation Accuracies per Head: {[f"{acc:.2f}%" for acc in val_accuracies]}')
            print('-' * 89)

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), args.save_path)
                print(f"Saved best model to {args.save_path}")

        else: # No validation set
            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | (No validation set)')
            print('-' * 89)
            # Save the model from the last epoch if no validation is performed
            if epoch == args.epochs:
                 torch.save(model.state_dict(), args.save_path)
                 print(f"Saved final model (no validation) to {args.save_path}")

        scheduler.step() # Step the scheduler each epoch

    print("Training finished.")