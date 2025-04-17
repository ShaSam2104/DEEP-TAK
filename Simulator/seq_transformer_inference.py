import os
import sys
import json
import torch
from seq_transformer_train import TakTransformer, encode_game, encode_ptn_move

# ---------------------- INFERENCE ----------------------
def generate_next_move(model, input_moves, token_to_id, id_to_token, vocab_size, beam_width=3, max_tokens=4, device="cuda"):
    model.eval()
    encoded = encode_game(input_moves, token_to_id, vocab_size)
    input_seq = torch.tensor(encoded).unsqueeze(0).to(device)

    beams = [(input_seq, [], 0.0)]

    for _ in range(max_tokens):
        new_beams = []
        for seq, tok_ids, log_prob in beams:
            with torch.no_grad():
                logits = model(seq)
                last_logits = logits[0, -1]
                probs = torch.softmax(last_logits, dim=-1)
                topk = torch.topk(probs, beam_width)

                for i in range(beam_width):
                    next_id = topk.indices[i].item()
                    next_prob = topk.values[i].item()
                    new_tok_ids = tok_ids + [next_id]

                    next_vec = torch.zeros(vocab_size, dtype=torch.float32)
                    next_vec[next_id] = 1.0
                    new_seq = torch.cat([seq, next_vec.unsqueeze(0).unsqueeze(0).to(device)], dim=1)
                    new_beams.append((new_seq, new_tok_ids, log_prob + torch.log(torch.tensor(next_prob))))

        beams = sorted(new_beams, key=lambda x: -x[2])[:beam_width]

    best_seq = beams[0][1]
    return [id_to_token[i] for i in best_seq]

def generate_next_move_2(model, input_moves, token_to_id, id_to_token, vocab_size, device="cuda"):
    """
    Predict the next move token(s) from the current game sequence
    input_moves: list of PTN move strings
    """
    model.eval()
    with torch.no_grad():
        encoded = encode_game(input_moves, token_to_id, vocab_size)
        x = torch.tensor(encoded).unsqueeze(0).to(device)  # [1, T, V]
        logits = model(x)
        next_token_logits = logits[0, -1]  # last step
        probs = torch.softmax(next_token_logits, dim=-1)
        top_id = torch.argmax(probs).item()
        return id_to_token[top_id]

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 1:
        print(f"Invalid Usage. Valid Usage: python3 {sys.argv[0]} model_path")
        exit()

    model_path = args[0]
    checkpoint = torch.load(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    token_to_id = checkpoint['token_to_id']
    id_to_token = checkpoint['id_to_token']
    vocab_size = checkpoint['vocab_size']
    model = TakTransformer(vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])

    input_moves = []
    while True:
        input_moves.append(input("Enter moves: ").strip())
        # input_moves = [move.strip() for move in input_moves if move.strip()]
        if not input_moves:
            break
        next_moves = generate_next_move_2(model, input_moves, token_to_id, id_to_token, vocab_size, device=device)
        print(f"Input moves: {input_moves}")
        print(f"Predicted next moves: {next_moves}")