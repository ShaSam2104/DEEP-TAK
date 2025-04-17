import re
import os
import sys
import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ---------------------- BOARD STATE AND MOVE ENCODING ----------------------
class TakBoard:
    def __init__(self, size=5):
        self.size = size
        # We'll use the board_state from the dataset instead of implementing full game logic
        # This is a placeholder class that will be populated with state from dataset
        
    @staticmethod
    def encode_board_state_from_dict(board_state_dict):
        """
        Encodes the board state from the dictionary format provided in the dataset
        Returns a tensor representation of the board state
        """
        board_size = board_state_dict["board_size"]
        board_squares = board_state_dict["board"]
        
        # Encode each square with its properties
        # We'll create a tensor with channels for different properties
        # Channels: player1_flat, player2_flat, player1_cap, player2_cap, player1_wall, player2_wall, stack_height
        board_tensor = torch.zeros(7, board_size, board_size)
        
        # Process each square
        for i, square in enumerate(board_squares):
            row = i // board_size
            col = i % board_size
            
            color_encoding = square["color_encoding"]
            type_encoding = square["type_encoding"]
            stack_height = square["stack_height"]
            
            # Set stack height
            board_tensor[6, row, col] = stack_height
            
            # Skip empty squares
            if stack_height == 0:
                continue
                
            # Determine piece type and color
            if color_encoding == "100000":  # Player 1
                if type_encoding == "00":  # Flat
                    board_tensor[0, row, col] = 1
                elif type_encoding == "01":  # Wall
                    board_tensor[4, row, col] = 1
                elif type_encoding == "10":  # Capstone
                    board_tensor[2, row, col] = 1
            elif color_encoding == "010000":  # Player 2
                if type_encoding == "00":  # Flat
                    board_tensor[1, row, col] = 1
                elif type_encoding == "01":  # Wall
                    board_tensor[5, row, col] = 1
                elif type_encoding == "10":  # Capstone
                    board_tensor[3, row, col] = 1
        
        # Add additional game state information
        player_turn = board_state_dict["turn"]
        player1_flats = board_state_dict["player1_flats"] / 21.0  # Normalize by max flats
        player1_caps = board_state_dict["player1_capstones"] / 1.0  # Normalize by max caps
        player2_flats = board_state_dict["player2_flats"] / 21.0
        player2_caps = board_state_dict["player2_capstones"] / 1.0
        
        # Create a feature vector with global state information
        global_features = torch.tensor([
            player_turn == 1,  # Current player is player 1
            player_turn == 2,  # Current player is player 2
            player1_flats,
            player1_caps,
            player2_flats,
            player2_caps
        ])
        
        # Flatten the board tensor for easier processing
        flattened_board = board_tensor.flatten()
        
        # Combine flattened board with global features
        combined_state = torch.cat([flattened_board, global_features])
        
        return combined_state

def encode_ptn_move(games):
    """Get vocabulary for PTN moves"""
    tokens = set()
    for game in games:
        for move in game:
            token = tokenize_move(move)
            if "F" in token:
                token.remove("F")
            tokens.update(token)
    vocab = sorted(tokens)
    token_to_id = {tok: i for i, tok in enumerate(vocab)}
    id_to_token = {i: tok for tok, i in token_to_id.items()}
    return token_to_id, id_to_token, vocab

def tokenize_move(move):
    stack_pattern = r"^(\d?[a-e][1-5])([><\+\-])(\d+)$"
    normal_pattern1 = r"^([a-e][1-5])([FSC]?)$"     # e.g., d2F, a1
    normal_pattern2 = r"^([FSC])([a-e][1-5])$"      # e.g., Fd2
    flat_drop_pattern = r"^(\d)([a-e][1-5])([><\+\-])(\d+)$"  # e.g., 1c1+1

    if re.match(stack_pattern, move):
        square, direction, drops = re.findall(stack_pattern, move)[0]
        return [square, direction] + list(drops)

    elif re.match(flat_drop_pattern, move):
        count, square, direction, drops = re.findall(flat_drop_pattern, move)[0]
        return [count, square, direction] + list(drops)

    elif re.match(normal_pattern1, move):
        square, piece = re.findall(normal_pattern1, move)[0]
        return [square] + ([piece] if piece else [])

    elif re.match(normal_pattern2, move):
        piece, square = re.findall(normal_pattern2, move)[0]
        return [square, piece]

    else:
        raise ValueError(f"Unrecognized PTN move: {move}")

def encode_move(move, token_to_id, vocab_size):
    """One-hot encode a single move"""
    tokens = tokenize_move(move)
    vec = torch.zeros(vocab_size)
    for tok in tokens:
        if tok == "F":
            continue
        vec[token_to_id[tok]] = 1.0
    return vec

def encode_move_history(moves, token_to_id, vocab_size, max_history_len=10):
    """Encode move history with padding"""
    # Take last max_history_len moves
    recent_moves = moves[-max_history_len:] if len(moves) > max_history_len else moves
    
    # Initialize history tensor with padding
    history_tensor = torch.zeros(max_history_len, vocab_size)
    
    # Encode each move
    for i, move in enumerate(recent_moves):
        idx = max_history_len - len(recent_moves) + i  # Start from appropriate position
        history_tensor[idx] = encode_move(move, token_to_id, vocab_size)
    
    return history_tensor

def get_valid_moves_mask(board_state, all_possible_moves, token_to_id):
    """
    Creates a mask of valid moves based on the board state
    This would require full Tak game rules implementation
    
    For now, we'll use a simplified approach based on available pieces and board constraints
    """
    # In a real implementation, this would analyze the board state to determine legal moves
    # For this prototype, we'll return a mask where all moves are valid
    # A proper implementation would check for:
    # - Piece placement rules
    # - Stack movement rules
    # - Available pieces in player's inventory
    
    mask = torch.zeros(len(token_to_id))
    for move in all_possible_moves:
        try:
            tokens = tokenize_move(move)
            main_token = tokens[0]  # Use first token as move identifier
            if main_token in token_to_id:
                mask[token_to_id[main_token]] = 1.0
        except ValueError:
            continue
    
    return mask

# ---------------------- MODEL ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            # Extend the positional encoding on-the-fly if needed
            device = self.pe.device
            d_model = self.pe.size(2)
            new_max_len = x.size(1)
            pe = torch.zeros(1, new_max_len, d_model, device=device)
            position = torch.arange(0, new_max_len, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-np.log(10000.0) / d_model))
            pe[:, :, 0::2] = torch.sin(position * div_term)
            pe[:, :, 1::2] = torch.cos(position * div_term)
            x = x + pe
        else:
            # Use the pre-computed positional encoding
            x = x + self.pe[:, :x.size(1)]
        return x

class TakTransformer(nn.Module):
    def __init__(self, vocab_size, board_feature_size, d_model=128, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        # Board state encoding
        self.board_embedder = nn.Sequential(
            nn.Linear(board_feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )
        
        # Move history processing
        self.move_embedder = nn.Linear(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=150)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection - combines board state and move history
        self.output_proj = nn.Linear(d_model * 2, vocab_size)
        
    def forward(self, board_state, move_history):
        batch_size = board_state.size(0)
        
        # Process board state
        board_features = self.board_embedder(board_state)  # [batch_size, d_model]
        
        # Process move history
        move_features = self.move_embedder(move_history)  # [batch_size, seq_len, d_model]
        move_features = self.pos_enc(move_features)
        
        # Transformer processing (requires [seq_len, batch_size, d_model])
        move_features = move_features.transpose(0, 1)
        move_features = self.transformer_encoder(move_features)
        move_features = move_features.transpose(0, 1)
        
        # Get the representation of the last move
        last_move_features = move_features[:, -1, :]  # [batch_size, d_model]
        
        # Combine board and move features
        combined_features = torch.cat([board_features, last_move_features], dim=1)  # [batch_size, d_model*2]
        
        # Generate move predictions
        logits = self.output_proj(combined_features)  # [batch_size, vocab_size]
        
        return logits

def mask_invalid_moves(logits, valid_moves_mask):
    """Apply mask to logits to ensure only valid moves get predicted"""
    # Set logits for invalid moves to a very low value
    masked_logits = logits.clone()
    
    # Handle both batch and non-batch cases
    if logits.dim() == 2 and valid_moves_mask.dim() == 2:
        # Batch case: [batch_size, vocab_size] and [batch_size, vocab_size]
        invalid_mask = ~valid_moves_mask.bool()
        masked_logits[invalid_mask] = -1e9
    elif logits.dim() == 2 and valid_moves_mask.dim() == 1:
        # Single mask for all batches: [batch_size, vocab_size] and [vocab_size]
        invalid_mask = ~valid_moves_mask.bool()
        for i in range(logits.size(0)):
            masked_logits[i, invalid_mask] = -1e9
    else:
        # Handle other dimensionality cases
        raise ValueError(f"Incompatible dimensions: logits {logits.shape}, mask {valid_moves_mask.shape}")
    
    return masked_logits

# ---------------------- DATASET & TRAINING ----------------------
class TakDatasetWithBoardState(Dataset):
    def __init__(self, game_data, token_to_id, vocab_size, max_history_len=10):
        self.samples = []
        self.token_to_id = token_to_id
        self.vocab_size = vocab_size
        self.max_history_len = max_history_len
        
        # Process each game
        for game in game_data:
            moves = []
            
            # Process each move in the game
            for move_data in game["moves"]:
                move = move_data["move"]
                board_state_dict = move_data["board_state"]
                
                # Get board state encoding
                board_state = TakBoard.encode_board_state_from_dict(board_state_dict)
                
                # Only add samples where we have move history
                if moves:
                    # Create move history tensor
                    move_history = encode_move_history(moves, token_to_id, vocab_size, max_history_len)
                    
                    # Encode target move
                    target_move = encode_move(move, token_to_id, vocab_size)
                    
                    # Create sample
                    self.samples.append({
                        'board_state': board_state,
                        'move_history': move_history,
                        'target_move': target_move
                    })
                
                # Add current move to history
                moves.append(move)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['board_state'], sample['move_history'], sample['target_move']

def format_winner_token(player_id, winner):
    # player_id: 0 or 1, winner: 0 or 1
    if player_id == winner:
        return f"{player_id}W{1-player_id}L"
    else:
        return f"{player_id}L{1-player_id}W"

# When loading a sample:
move_history = ... # list of moves
player_id = ...    # 0 or 1
winner = ...       # from JSON
move_history.append(format_winner_token(player_id, winner))

# ---------------------- TRAINING FUNCTIONS ----------------------
def compute_loss(logits, targets, valid_moves_mask=None, board_states=None, moves=None, winner=None, player_id=None, alpha=0.7):
    """Compute loss combining BCE with heuristic evaluation"""
    # Standard BCE loss
    bce_loss = nn.BCEWithLogitsLoss()(logits, targets)
    
    # If we don't have board states for heuristic calculation
    if board_states is None:
        return bce_loss
        
    # Add heuristic-based loss component
    batch_size = logits.size(0)
    heuristic_loss = 0.0
    
    for i in range(batch_size):
        # Get current board state
        board = board_states[i]
        
        # Calculate heuristic components similar to main.cpp
        captured_value = calculate_captured_advantage(board)
        composition_value = calculate_composition_value(board) 
        center_control = calculate_center_control(board)
        influence_value = calculate_influence(board)
        
        # Combine heuristic components
        heuristic_value = 1.4*captured_value + 1.55*composition_value + 0.9*influence_value + 1.1*center_control
        
        # Normalize heuristic value to appropriate scale
        normalized_heuristic = torch.tanh(torch.tensor(heuristic_value/1000.0))
        
        # Add to total heuristic loss
        heuristic_loss += (1.0 - normalized_heuristic)
    
    heuristic_loss = heuristic_loss / batch_size
    
    # Combine losses with weighting parameter alpha
    combined_loss = alpha * bce_loss + (1 - alpha) * heuristic_loss

    # If this sample is a losing game for the current player, increase loss
    if winner is not None and player_id is not None and winner != player_id:
        combined_loss = combined_loss * 10  # or another large factor

    return combined_loss

def count_pieces(board_tensor, channels):
    """Sum the number of pieces for the given channels."""
    # board_tensor: [flattened_board, global_features]
    # First, extract the board part (7, board_size, board_size)
    board_size = int(((board_tensor.shape[0] - 6) // 7) ** 0.5)
    board = board_tensor[:7 * board_size * board_size].reshape(7, board_size, board_size)
    return board[channels].sum().item()

def calculate_captured_advantage(board_tensor):
    """Advantage based on number of flatstones and capstones controlled."""
    my_pieces = count_pieces(board_tensor, [0, 2])  # player1_flat, player1_cap
    opponent_pieces = count_pieces(board_tensor, [1, 3])  # player2_flat, player2_cap
    return (my_pieces - opponent_pieces) * 50

def calculate_composition_value(board_tensor):
    """Reward for having more flats/caps, penalize for opponent's walls."""
    board_size = int(((board_tensor.shape[0] - 6) // 7) ** 0.5)
    board = board_tensor[:7 * board_size * board_size].reshape(7, board_size, board_size)
    my_flats = board[0].sum().item()
    my_caps = board[2].sum().item()
    my_walls = board[4].sum().item()
    opp_flats = board[1].sum().item()
    opp_caps = board[3].sum().item()
    opp_walls = board[5].sum().item()
    return (
        3.0 * my_flats + 2.99 * my_caps + 2.98 * my_walls
        - 3.0 * opp_flats - 2.99 * opp_caps - 2.98 * opp_walls
    )

def calculate_center_control(board_tensor):
    """Reward for controlling center squares."""
    board_size = int(((board_tensor.shape[0] - 6) // 7) ** 0.5)
    board = board_tensor[:7 * board_size * board_size].reshape(7, board_size, board_size)
    center = (board_size - 1) / 2
    y, x = torch.meshgrid(torch.arange(board_size), torch.arange(board_size), indexing='ij')
    dist = ((x - center) ** 2 + (y - center) ** 2).sqrt()
    max_dist = dist.max()
    weights = 1.0 - dist / max_dist  # 1 at center, 0 at edge
    my_control = (board[0] + board[2]) * weights
    opp_control = (board[1] + board[3]) * weights
    return my_control.sum().item() - opp_control.sum().item()

def calculate_influence(board_tensor):
    """Influence: reward for having more pieces adjacent to empty or opponent squares."""
    board_size = int(((board_tensor.shape[0] - 6) // 7) ** 0.5)
    board = board_tensor[:7 * board_size * board_size].reshape(7, board_size, board_size)
    my_presence = (board[0] + board[2] + board[4])
    opp_presence = (board[1] + board[3] + board[5])
    influence = 0.0
    my_pad = torch.nn.functional.pad(my_presence, (1,1,1,1))
    opp_pad = torch.nn.functional.pad(opp_presence, (1,1,1,1))
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        my_neighbors = my_pad[1+dy:1+dy+board_size, 1+dx:1+dx+board_size]
        opp_neighbors = opp_pad[1+dy:1+dy+board_size, 1+dx:1+dx+board_size]
        influence += (my_neighbors * (1 - opp_presence)).sum().item()
        influence -= (opp_neighbors * (1 - my_presence)).sum().item()
    return influence

def train_step(model, optimizer, board_state, move_history, target_move, valid_moves_mask=None):
    """Execute a single training step"""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(board_state, move_history)
    
    # Compute loss (with optional move masking)
    loss = compute_loss(logits, target_move, valid_moves_mask)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate(model, dataloader, device, all_valid_moves_mask=None):
    """Evaluate model on a dataset"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for board_state, move_history, target_move in dataloader:
            board_state = board_state.to(device)
            move_history = move_history.to(device)
            target_move = target_move.to(device)
            
            # Forward pass
            logits = model(board_state, move_history)
            
            # Compute loss
            loss = compute_loss(logits, target_move, all_valid_moves_mask)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train(model, train_dataset, test_dataset, epochs=10, batch_size=32, lr=1e-4, device="cuda"):
    """Train the model"""
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for board_state, move_history, target_move in train_dataloader:
            board_state = board_state.to(device)
            move_history = move_history.to(device)
            target_move = target_move.to(device)
            
            # Train step
            loss = train_step(model, optimizer, board_state, move_history, target_move)
            train_loss += loss
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Evaluation phase
        avg_test_loss = evaluate(model, test_dataloader, device)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Test Loss: {avg_test_loss:.4f}")
    
    return model

# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 1:
        raise ValueError("Please provide the path to the dir with game files.")
    
    # Load game data
    game_data = []
    outs = os.listdir(args[0])
    for out in outs:
        if out.endswith(".json"):
            with open(f"{args[0]}/{out}", "r") as fd:
                game_data.append(json.load(fd))
    
    if not game_data:
        raise ValueError("No valid game files found in the provided directory.")
    
    print(f"Loaded {len(game_data)} games.")
    
    # Extract moves for vocabulary building
    moves_dataset = []
    for game in game_data:
        moves = []
        for move_data in game["moves"]:
            moves.append(move_data["move"])
        moves_dataset.append(moves)
    
    # Build vocabulary
    token_to_id, id_to_token, vocab = encode_ptn_move(moves_dataset)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Calculate board feature size
    # Board tensor (7 channels × 5×5 board) + global features (6)
    board_size = game_data[0]["moves"][0]["board_state"]["board_size"]
    board_feature_size = (7 * board_size * board_size) + 6
    
    # Create dataset
    full_dataset = TakDatasetWithBoardState(game_data, token_to_id, vocab_size)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create model
    model = TakTransformer(vocab_size, board_feature_size)
    
    # Train model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = train(model, train_dataset, test_dataset, epochs=50, batch_size=32, device=device)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'token_to_id': token_to_id,
        'id_to_token': id_to_token,
        'board_feature_size': board_feature_size
    }, "tak_transformer_with_board_state.pth")
    
    print("Model trained and saved as tak_transformer_with_board_state.pth")