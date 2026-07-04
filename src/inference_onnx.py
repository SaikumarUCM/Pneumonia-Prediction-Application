import torch

from src.pneumonia_model import load_model

model = load_model()
model.eval()

# Dummy input must match real inference:[batch_size, channels, height, width]
dummy_input = torch.randn(1, 3, 224, 224)

onnx_path = "modelWeights/pneumonia_model.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)
print(f"Exported to {onnx_path}")