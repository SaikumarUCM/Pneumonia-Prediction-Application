import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def preprocess_image(image_bytes):
    """PyTorch preprocessing for local training/scripts (requires torch)."""
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = Image.open(image_bytes).convert("RGB")
    transformed_image: torch.Tensor = transform(image)  # type: ignore
    return transformed_image.unsqueeze(0)


def preprocess_image_to_numpy(image_bytes) -> np.ndarray:
    """NumPy preprocessing for ONNX inference (no PyTorch required)."""
    image = Image.open(image_bytes).convert("RGB").resize((224, 224))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr[np.newaxis, ...]
