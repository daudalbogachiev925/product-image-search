"""Export CLIP image encoder to ONNX — WIP.

Goal: run image encoding without PyTorch, ~50% faster on CPU.
"""
from pathlib import Path
import torch
from src.models.clip_encoder import CLIPEncoder

# TODO: implement proper export with dummy input
# TODO: verify ONNX runtime output matches PyTorch within tolerance
# TODO: benchmark CPU latency vs PyTorch

print("ONNX export not yet implemented.")
