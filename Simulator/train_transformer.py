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
            row_num = row_idx + 1

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
            row_num = row_idx + 1

            # Basic validation for coordinates
            # Example: board_size = 5
            # if not (0 <= col_idx < 5 and 0 <= row_idx < 5): return None

            direction_map_inv = {0: '>', 1: '<', 2: '+', 3: '-'}
            direction_idx = encoded_vector[4]
            if direction_idx not in direction_map_inv: return None
            direction_char = direction_map_inv[direction_idx]

            drop_tokens = []
            total_dropped = 0
            for i in range(5, 9):
                drop_count = encoded_vector[i]
                if drop_count > 0:
                    drop_tokens.append(str(drop_count))
                    total_dropped += drop_count
                else:
                    break # Stop at first zero padding

            # Optional validation
            # if total_dropped != num_pieces: return None

            drop_string = "".join(drop_tokens)
            if not drop_string: return None # Must have drops if moving

            return f"{num_pieces}{col_char}{row_num}{direction_char}{drop_string}"

        else:
            # print(f"Error: Invalid move indicator {move_indicator}. Must be 0 or 1.")
            return None

    except (IndexError, ValueError, TypeError) as e:
        # print(f"Error during decoding: {e}") # Reduce noise
        return None

# --- Positional Encoding ---
# Optional: PyTorch's TransformerEncoder includes positional encoding,
# but if using a custom setup or wanting explicit control:
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

# --- Board Encoding Helper (Modified to use ai_heuristic_value) ---
def board_state_to_tensor(board_state_dict, board_size, ai_heuristic_value):
    """
    Converts the board_state dictionary from game_*.json into the 14-channel tensor,
    using the provided ai_heuristic_value for the last two channels.
    """
    # Handle potential None or invalid ai_heuristic_value
    if not isinstance(ai_heuristic_value, (list, tuple)) or len(ai_heuristic_value) != 2:
        # print(f"Warning: Invalid ai_heuristic_value received: {ai_heuristic_value}. Using [0.0, 0.0].")
        ai_heuristic_value = [0.0, 0.0]
    try:
        # Basic normalization/scaling - adjust the divisor as needed based on typical value ranges
        heur_val_1 = float(ai_heuristic_value[0]) / 100.0
        heur_val_2 = float(ai_heuristic_value[1]) / 100.0
    except (ValueError, TypeError):
        # print(f"Warning: Could not convert ai_heuristic_value {ai_heuristic_value} to float. Using [0.0, 0.0].")
        heur_val_1 = 0.0
        heur_val_2 = 0.0


    if not board_state_dict or 'board' not in board_state_dict or len(board_state_dict['board']) != board_size * board_size:
        # print("Warning: Invalid or missing board state dict, returning zero tensor.")
        # Still include heuristic values even if board state is bad? Or return all zeros?
        # Let's return mostly zeros but keep the heuristic channels
        tensor_data = np.zeros((14, board_size, board_size), dtype=np.float32)
        tensor_data[12, :, :] = heur_val_1
        tensor_data[13, :, :] = heur_val_2
        return torch.from_numpy(tensor_data)


    # Initialize channels
    player1_flat = np.zeros((board_size, board_size), dtype=np.float32)
    player1_wall = np.zeros((board_size, board_size), dtype=np.float32)
    player1_cap = np.zeros((board_size, board_size), dtype=np.float32)
    player2_flat = np.zeros((board_size, board_size), dtype=np.float32)
    player2_wall = np.zeros((board_size, board_size), dtype=np.float32)
    player2_cap = np.zeros((board_size, board_size), dtype=np.float32)
    stack_height = np.zeros((board_size, board_size), dtype=np.float32) # Normalized 0-1

    # --- Board Piece Encoding ---
    # This placeholder logic MUST be verified/corrected against generate_training_data.py
    for idx, square_dict in enumerate(board_state_dict['board']):
        r, c = divmod(idx, board_size)
        color_enc = square_dict.get("color_encoding", "000000")
        type_enc = square_dict.get("type_encoding", "00")
        height = square_dict.get("stack_height", 0.0)

        stack_height[r, c] = height

        top_piece_color = -1
        top_piece_type = -1

        if height > 0:
             # *** Placeholder logic - MUST BE VERIFIED/CORRECTED ***
             # Example: Check the last bit of color_encoding for player 2?
             if color_enc.endswith('1'):
                 top_piece_color = 1 # Assume Player 2
             elif '1' in color_enc: # If not ending in 1, but has a 1, assume Player 1?
                 top_piece_color = 0 # Assume Player 1
             # If only '0's, it's an empty square effectively, color/type remain -1

             if type_enc == "00": top_piece_type = 0 # Flat
             elif type_enc == "01": top_piece_type = 1 # Wall
             elif type_enc == "10": top_piece_type = 2 # Capstone

        # Fill channels based on top piece
        if top_piece_color == 0: # Player 1
            if top_piece_type == 0: player1_flat[r, c] = 1.0
            elif top_piece_type == 1: player1_wall[r, c] = 1.0
            elif top_piece_type == 2: player1_cap[r, c] = 1.0
        elif top_piece_color == 1: # Player 2
            if top_piece_type == 0: player2_flat[r, c] = 1.0
            elif top_piece_type == 1: player2_wall[r, c] = 1.0
            elif top_piece_type == 2: player2_cap[r, c] = 1.0
    # --- End Board Piece Encoding ---

    # Player info channels
    total_pieces = (board_size * board_size) // 2 + 1
    p1_flats_norm = board_state_dict.get('player1_flats', 0) / total_pieces
    p1_caps_norm = board_state_dict.get('player1_capstones', 0)
    p2_flats_norm = board_state_dict.get('player2_flats', 0) / total_pieces
    p2_caps_norm = board_state_dict.get('player2_capstones', 0)
    turn = board_state_dict.get('turn', 0)

    player1_flats_channel = np.full((board_size, board_size), p1_flats_norm, dtype=np.float32)
    player1_caps_channel = np.full((board_size, board_size), p1_caps_norm, dtype=np.float32)
    player2_flats_channel = np.full((board_size, board_size), p2_flats_norm, dtype=np.float32)
    player2_caps_channel = np.full((board_size, board_size), p2_caps_norm, dtype=np.float32)
    turn_channel = np.full((board_size, board_size), turn, dtype=np.float32)

    # AI Heuristic channels (using the passed-in values)
    ai_heur_1_channel = np.full((board_size, board_size), heur_val_1, dtype=np.float32)
    ai_heur_2_channel = np.full((board_size, board_size), heur_val_2, dtype=np.float32)


    # Stack channels - Order must be consistent! (Now 14 channels)
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
    if tensor.shape[0] != 14:
         raise ValueError(f"Generated tensor has {tensor.shape[0]} channels, expected 14.")

    return tensor


# --- Tak Dataset (Modified to handle ai_heuristic_value) ---
class TakDataset(Dataset):
    def __init__(self, data_dir, board_size, max_game_id=68): # Added max_game_id
        # Load specific range of game_*.json files
        self.data_files = []
        print(f"Looking for game files from game_0.json to game_{max_game_id}.json in {data_dir}...")
        for i in range(max_game_id + 1): # Loop from 0 to max_game_id inclusive
            filename = f"game_{i}.json"
            file_path = os.path.join(data_dir, filename)
            if os.path.exists(file_path):
                self.data_files.append(file_path)
            # else:
                # Optional: print a warning if a file in the range is missing
                # print(f"Warning: File not found: {file_path}")

        if not self.data_files:
             raise ValueError(f"No game files found in the range 0-{max_game_id} in {data_dir}")

        # Store (input_state_dict, ai_heuristic_before_move, target_move_vector) tuples
        self.samples = []
        self.board_size = board_size
        print(f"Found {len(self.data_files)} game files to process.")

        # Define initial empty board state dictionary structure (matching generator output)
        initial_board_state = {
            "board": [{"color_encoding": "000000", "type_encoding": "00", "stack_height": 0.0}] * (board_size * board_size),
            "board_size": board_size,
            "player1_flats": (board_size*board_size)//2 + 1, # Example starting counts
            "player1_capstones": 1,
            "player2_flats": (board_size*board_size)//2 + 1,
            "player2_capstones": 1,
            "turn": 0, # Player 1 starts
            "heuristics": {} # This 'heuristics' key is no longer used for tensor input
        }
        # Initial heuristic value before the first move (defaults to [0, 0])
        initial_ai_heuristic = [0.0, 0.0]

        for file_path in self.data_files:
            try:
                with open(file_path, 'r') as f:
                    game_data = json.load(f)
                    # ... (rest of the file processing logic) ...
                    if not game_data or 'moves' not in game_data or not isinstance(game_data['moves'], list):
                        print(f"Warning: Skipping file {file_path} due to missing/invalid 'moves' list.")
                        continue
                    last_state_dict = initial_board_state
                    last_ai_heuristic = initial_ai_heuristic # Heuristic corresponding to last_state_dict

                    for move_data in game_data['moves']:
                        if 'board_state' in move_data and 'move_vector' in move_data:
                            current_target_vector = move_data['move_vector']
                            # Get the AI heuristic calculated *before* this move was made
                            current_ai_heuristic = move_data.get('ai_heuristic_value', [0.0, 0.0])
                            if current_ai_heuristic is None: # Handle explicit null
                                current_ai_heuristic = [0.0, 0.0]

                            # Ensure target vector is valid
                            if isinstance(current_target_vector, (list, tuple)) and len(current_target_vector) == 9:
                                # Add the pair:
                                # (state *before* this move, heuristic *before* this move, move_vector *of* this move)
                                self.samples.append((last_state_dict, last_ai_heuristic, current_target_vector))

                                # Update state and heuristic for the next iteration
                                last_state_dict = move_data['board_state']
                                last_ai_heuristic = current_ai_heuristic
                            else:
                                print(f"Warning: Invalid move_vector found in {file_path}, move {move_data.get('move_number', '?')}")
                        else:
                             print(f"Warning: Missing board_state or move_vector in {file_path}, move {move_data.get('move_number', '?')}")
                             break # Stop processing this game if data is inconsistent

            except json.JSONDecodeError:
                print(f"Warning: Skipping corrupted JSON file: {file_path}")
            except Exception as e:
                print(f"Warning: Error processing file {file_path}: {e}")

        if not self.samples:
             raise ValueError(f"No valid training samples could be extracted from the specified game files.")
        print(f"Processed {len(self.samples)} total state-action samples.")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Unpack state, heuristic, and target
        state_dict, ai_heuristic, target_move_vector = self.samples[idx]

        # Convert state dict and heuristic to tensor
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

        # --- CNN Stem ---
        # Simple CNN stem - adjust complexity as needed
        self.cnn_stem = nn.Sequential(
            nn.Conv2d(input_channels, d_model // 4, kernel_size=3, stride=1, padding=1),
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
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=board_size*board_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # --- Output Heads ---
        # Calculate output dimensions based on board_size
        # Head 0: Move type (0=place, 1=move) -> Size 2
        # Head 1: Piece type (0,1,2) or Count (1..board_size) -> Size board_size + 1 (0 to board_size)
        # Head 2: Column (0..board_size-1) -> Size board_size
        # Head 3: Row (0..board_size-1) -> Size board_size
        # Head 4: Direction (0..3) -> Size 4
        # Head 5-8: Drop counts (0..board_size) -> Size board_size + 1
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
        # Initialize weights for better convergence
        initrange = 0.1
        for head in self.output_heads:
            head.bias.data.zero_()
            head.weight.data.uniform_(-initrange, initrange)
        # Consider initializing CNN and Transformer layers too if needed

    def forward(self, src):
        # src shape: (Batch, Channels, Height, Width) e.g., (B, 14, 5, 5)
        
        # 1. CNN Stem
        src = self.cnn_stem(src) # (B, d_model, H, W) e.g., (B, 256, 5, 5)

        # 2. Flatten and Permute for Transformer
        batch_size = src.shape[0]
        src = src.flatten(2) # (B, d_model, H*W) e.g., (B, 256, 25)
        src = src.permute(0, 2, 1) # (B, H*W, d_model) e.g., (B, 25, 256)

        # 3. Positional Encoding (if using explicit one)
        # Note: PyTorch TransformerEncoder expects (SeqLen, Batch, Dim) by default if batch_first=False
        # Since we use batch_first=True, input is (Batch, SeqLen, Dim)
        # If using explicit PositionalEncoding class above, it expects (SeqLen, Batch, Dim)
        # src = src.permute(1, 0, 2) # (SeqLen, B, d_model)
        # src = self.pos_encoder(src)
        # src = src.permute(1, 0, 2) # (B, SeqLen, d_model) - back to batch_first
        # OR adapt PositionalEncoding class for batch_first

        # 4. Transformer Encoder
        transformer_output = self.transformer_encoder(src) # (B, SeqLen, d_model)

        # 5. Aggregate Transformer Output (e.g., mean pooling)
        pooled_output = transformer_output.mean(dim=1) # (B, d_model)

        # 6. Output Heads
        outputs = [head(pooled_output) for head in self.output_heads]
        # outputs is a list of 9 tensors, e.g., [(B, 2), (B, 6), (B, 5), ..., (B, 6)]

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
            # targets[:, head_idx] selects the target value for the current head across the batch
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

# --- Evaluation Function (Modified) ---
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

            total_loss += sum(losses) * batch_size # Loss per sample

            # --- Decode and Print Sample Predictions ---
            if samples_printed < print_samples:
                # Stack predictions along a new dimension: (9, B) -> (B, 9)
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


    avg_loss = total_loss / total_samples
    accuracies = [(correct / total_samples) * 100 for correct in total_correct]
    return avg_loss, accuracies


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tak Transformer Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing raw game_*.json files")
    parser.add_argument("--board_size", type=int, required=True, help="Board size used during data generation (e.g., 5)")
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


    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset and DataLoader ---
    # Instantiate TakDataset specifying the max_game_id
    full_dataset = TakDataset(args.data_dir, args.board_size, max_game_id=9) # <-- CHANGE HERE

    # --- The rest of the main execution block remains the same ---
    # Simple split example (adjust ratio as needed)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    # Handle case where val_size might be 0 if dataset is very small
    if val_size == 0 and train_size > 0:
        print("Warning: Validation set size is 0. Using entire dataset for training.")
        train_dataset = full_dataset
        # Create a dummy val_dataloader or skip evaluation if val_size is 0
        val_dataloader = None # Or DataLoader with a small subset if needed for code structure
    elif train_size == 0:
         raise ValueError("Dataset is too small to create even a training set.")
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True if device=='cuda' else False)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True if device=='cuda' else False)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {val_size}")

    # --- Model, Loss, Optimizer ---
    # ... (Model, Criterion, Optimizer, Scheduler setup remains the same) ...
    model = TakTransformer(
        board_size=args.board_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1.0, gamma=0.95)


    # --- Training Loop (Handle potential lack of validation data) ---
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
        else:
            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | (No validation set)')
            print('-' * 89)
            # If no validation, save based on training loss or just save last epoch?
            # For simplicity, let's just save the last model if no validation
            if epoch == args.epochs:
                 torch.save(model.state_dict(), args.save_path)
                 print(f"Saved final model (no validation) to {args.save_path}")


        # Save best model based on validation loss if validation exists
        if val_dataloader and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model to {args.save_path}")

        scheduler.step()

    print("Training finished.")