
import torch
import sys
import argparse
from mace import data
from mace.tools import torch_geometric, torch_tools, utils

MODEL_PATH = "/home/phanim/harshitrawat/summer/mace_models/universal/2024-01-07-mace-128-L2_epoch-199.model"

def inspect(model_path):
    try:
        # Safe load setup
        _original_load = torch.load
        def _safe_load(*args, **kwargs):
            if 'weights_only' not in kwargs: kwargs['weights_only'] = False
            return _original_load(*args, **kwargs)
        torch.load = _safe_load

        print(f"Loading {model_path}...", flush=True)
        model = torch.load(model_path, map_location='cpu')
        
        print("\n--- Model Architecture Info ---", flush=True)
        if hasattr(model, 'hidden_irreps'):
            print(f"Hidden Irreps: {model.hidden_irreps}", flush=True)
        if hasattr(model, 'readout_irreps'):
            print(f"Readout Irreps: {model.readout_irreps}", flush=True)
        
        # Check atomic energies or scale/shift
        pass
        
        print(f"\nModel Class: {type(model)}", flush=True)
        # print(str(model)[:500] + "...") # Print start of string rep

    except Exception as e:
        print(f"Error: {e}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs='?', default=MODEL_PATH, help="Path to model")
    args = parser.parse_args()
    inspect(args.model_path)
