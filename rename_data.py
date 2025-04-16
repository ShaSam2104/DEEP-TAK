import os
import shutil
from pathlib import Path

def copy_and_rename_incrementally(input_dir, output_dir, prefix="game_", start=1):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = [f for f in input_path.iterdir() if f.is_file() and f.name.startswith(prefix) and f.suffix == ".json"]

    # Sort by number after prefix
    def extract_number(f):
        try:
            return int(f.stem[len(prefix):])
        except ValueError:
            return float('inf')

    files.sort(key=extract_number)

    for i, f in enumerate(files, start=start):
        new_name = f"{prefix}{i}{f.suffix}"
        dest = output_path / new_name
        shutil.copy2(f, dest)
        print(f"Copied {f.name} -> {new_name}")

if __name__ == "__main__":
    copy_and_rename_incrementally("Simulator/data", "Simulator/final_data")

