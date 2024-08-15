import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models.mobilenetv2
from torchvision.models.mobilenetv3 

# Define models
class MultiClassMobileNetV2(nn.Module):
    def __init__(self):
        super(MultiClassMobileNetV2, self).__init__()
        base_model = models.mobilenet_v2()
        base_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        base_model.classifier[-1] = nn.Linear(in_features=1280, out_features=5)
        self.base_model = base_model

    def forward(self, x):
        return self.base_model(x)

class MultiClassMobileNetV3Small(nn.Module):
    def __init__(self):
        super(MultiClassMobileNetV3Small, self).__init__()
        base_model = models.mobilenet_v3_small()
        base_model.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        base_model.classifier = nn.Linear(in_features=576, out_features=5)
        self.base_model = base_model

    def forward(self, x):
        return self.base_model(x)

# Define loading functions
def load_standard_model_weights(model, checkpoint_path, map_location='cpu'):
    """Load model weights from a checkpoint in the standard format with 'model_state_dict'."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    print("Standard Checkpoint loaded successfully.")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise KeyError(f"'model_state_dict' not found in the checkpoint file: {checkpoint_path}")
    return model
