
from torchvision import transforms
from PIL import Image
import torch

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image_bytes):
    image = Image.open(image_bytes).convert("RGB")
    transformed_image: torch.Tensor = transform(image)  # type: ignore
    transformed_image = transformed_image.unsqueeze(0)  # add batch dimension
    return transformed_image
