
import torch
import torch.nn as nn
from torchvision import models

def load_model(model_path= "modelWeights/pneumonia_model.pth", device=None):


    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load a pre-trained ResNet model
    model = models.resnet18(weights=None)

    # Modify final layer for 2 classes
    model.fc = nn.Linear(model.fc.in_features, 2)

    # Load trained weights
    weights = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(weights)

    
    # Move model to device
    model = model.to(device)

    # Set evaluation mode
    model.eval()

    return model