import torch
import numpy as np
from torch.utils.data import DataLoader
from Models import load_standard_model_weights

def load_models(MN2_model, MN2_model, models_folder, device):
    MN2_model = load_standard_model_weights(MN2_model, f"{models_folder}/MobileNetV2_4.pth", map_location='cpu')
    MN3S_model = load_standard_model_weights(MN3S_model, f"{models_folder}/MobileNetV3Small_1.pth", map_location='cpu')

    MN2_model.to(device)
    MN3S_model.to(device)

    return MN2_model, MN3S_model

def make_predictions(model, dataloader, device):
    model.eval()
    predictions = []

    for inputs in dataloader:
        inputs = inputs.to(device)
        with torch.no_grad():
            output = model(inputs)
            pred = (output > 0.5).view(-1).long()
            predictions.extend(pred.cpu().numpy())

    return predictions

def predict_pneumonia(MN2_model, MN3S_model, dataloader_MN, dataloader_swdsgd, device):
