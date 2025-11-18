import torch 

def quant_any(logits: torch.tensor) -> tuple[torch.Tensor, float]:
    max = torch.max(torch.abs(logits))
    if max == 0:
        scale = 1
    else:
        scale = max/127
    quantized = torch.clamp(torch.round(logits / scale), -128, 127).to(torch.int8)

    return quantized , scale.item()

def dequantize(quantized_tensor: torch.Tensor, scale: float) -> torch.Tensor:
    return (quantized_tensor.to(torch.float32) * scale)

logits = torch.tensor([0.5, -1.0, 0.8, 0.0], dtype=torch.float32)

q_logits, scale = quant_any(logits)
reconstructed = dequantize(q_logits, scale)

print("Original:", logits)
print("Quantized:", q_logits)
print("Scale:", scale)
print("Reconstructed:", reconstructed)
