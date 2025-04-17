import re
import os
import sys
import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ---------------------- PTN ENCODING ----------------------
def encode_ptn_move(games):
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

def encode_game(game, token_to_id, vocab_size):
    encoded = []
    for move in game:
        tokens = tokenize_move(move)
        vec = np.zeros(vocab_size, dtype=np.float32)
        for tok in tokens:
            if tok == "F":
                continue
            vec[token_to_id[tok]] = 1.0
        encoded.append(vec)
    return np.stack(encoded)

def pad_batch(game_batch, max_len=None):
    tensor_batch = [torch.tensor(g) for g in game_batch]
    if not max_len:
        max_len = max(g.shape[0] for g in tensor_batch)
    padded = torch.zeros((len(tensor_batch), max_len, tensor_batch[0].shape[1]))
    for i, t in enumerate(tensor_batch):
        end = min(max_len, t.shape[0])
        padded[i, :end, :] = t[:end, :]
    return padded

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
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=150)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = x.float()
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)
        logits = self.output_proj(x)
        return logits

# ---------------------- DATASET & TRAINING ----------------------
class TakDataset(Dataset):
    def __init__(self, onehot_games, max_len):
        self.games = onehot_games
        self.max_len = max_len

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        game = self.games[idx]
        T = min(len(game), self.max_len)
        x = game[:T - 1]
        y = game[1:T]
        pad_size = self.max_len - 1 - x.shape[0]
        if pad_size > 0:
            x = np.vstack([x, np.zeros((pad_size, x.shape[1]))])
            y = np.vstack([y, np.zeros((pad_size, y.shape[1]))])
        return torch.tensor(x), torch.tensor(y)

def compute_loss(logits, targets):
    target_ids = torch.argmax(targets, dim=-1)
    logits = logits.view(-1, logits.size(-1))
    target_ids = target_ids.view(-1)
    return nn.CrossEntropyLoss()(logits, target_ids)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = compute_loss(logits, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train(model, train_dataset, test_dataset, epochs=10, batch_size=32, lr=1e-4, device="cuda"):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = compute_loss(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        
        # Evaluation phase
        avg_test_loss = evaluate(model, test_dataloader, device)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Test Loss: {avg_test_loss:.4f}")

# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":

    args = sys.argv[1:]
    if len(args) != 1:
        raise ValueError("Please provide the path to the dir with game files.")
    
    outs = os.listdir(args[0])
    dataset = []
    for out in outs:
        if out.endswith(".json"):
            with open(f"{args[0]}/{out}", "r") as fd:
                fc = json.load(fd)

                moves = fc.get("moves", None)
                if not moves:
                    print("Invalid File format: Expected moves, but not found")

                mvs = []
                for idx, move in enumerate(moves):
                    mv = move.get("move", None)
                    if not mv:
                        print(f"Invalid File format: Expected move in moves but not found in {idx}th iteration")

                    mvs.append(mv)
                dataset.append(mvs)
    if not dataset:
        raise ValueError("No valid game files found in the provided directory.")
    print(f"Loaded {len(dataset)} games.")
    print(f"Dataset size: {len(dataset)}")

    token_to_id, id_to_token, vocab = encode_ptn_move(dataset)
    vocab_size = len(vocab)
    encoded_games = [encode_game(g, token_to_id, vocab_size) for g in dataset]

    max_len = max(len(g) for g in encoded_games)
    full_dataset = TakDataset(encoded_games, max_len=max_len)
    
    # Split dataset into train and test sets (80% train, 20% test)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    model = TakTransformer(vocab_size)
    train(model, train_dataset, test_dataset, epochs=100, batch_size=32, 
          device="cuda" if torch.cuda.is_available() else "cpu")

    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'token_to_id': token_to_id,
        'id_to_token': id_to_token
    }, "new_tak_transformer_complete.pth")
    print("Model trained and saved as new_tak_transformer_complete.pth")