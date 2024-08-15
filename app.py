import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from torchvision import models
import torch.nn as nn
import random
import cv2

# Load Models
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

# Initialize Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_v2 = MultiClassMobileNetV2().to(device)
model_v3s = MultiClassMobileNetV3Small().to(device)

model_v2.load_state_dict(torch.load('bucket/MobileNetV2_4.pth', map_location=device)['model_state_dict'])
model_v3s.load_state_dict(torch.load('bucket/MobileNetV3Small_1.pth', map_location=device)['model_state_dict'])

model_v2.eval()
model_v3s.eval()

# CLAHE function
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(np.array(image))
    return Image.fromarray(clahe_image)

def get_cam(model, img_tensor, target_layer):
    """
    Generate a Class Activation Map (CAM) for a given image and model.
    
    Args:
        model (nn.Module): The neural network model.
        img_tensor (torch.Tensor): The input image tensor.
        target_layer (str): The layer to target for CAM generation.
        
    Returns:
        np.ndarray: The CAM mask.
    """
    model.eval()
    
    def forward_hook(module, input, output):
        activation[0] = output
    
    activation = {}
    layer = dict([*model.named_modules()]).get(target_layer, None)
    if layer is None:
        raise ValueError(f"Layer {target_layer} not found in the model")
        
    hook = layer.register_forward_hook(forward_hook)
    
    with torch.no_grad():
        output = model(img_tensor)
    
    hook.remove()
    
    output = output[0]
    output = F.relu(output)
    weight_softmax_params = list(model.parameters())[-2].data.numpy()
    weight_softmax = np.squeeze(weight_softmax_params)
    
    activation = activation[0].squeeze().cpu().data.numpy()
    cam = np.zeros(activation.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weight_softmax):
        cam += w * activation[i, :, :]
    
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img_tensor.shape[-1], img_tensor.shape[-2]))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    
    return cam
def overlay_circles(image, cam):
    cam_image = np.uint8(255 * cam)
    _, thresh = cv2.threshold(cam_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
    image_np = np.array(image)
    for cnt in sorted_contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(image_np, center, radius, (255, 0, 0), 2)
    return Image.fromarray(image_np)

# Streamlit App
st.title("Medical Image Classification")
st.write("Upload an X-ray image and get the prediction with confidence levels.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    image_clahe = apply_clahe(image)

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])

    image_tensor = transform(image_clahe).unsqueeze(0).to(device)

    class_labels = ['Normal', 'TBC', 'Bacteria', 'Virus', 'COVID']
    
    predictions_v2 = []
    predictions_v3s = []

    for _ in range(20):
        with torch.no_grad():
            outputs_v2 = model_v2(image_tensor)
            outputs_v3s = model_v3s(image_tensor)

        prob_v2 = torch.softmax(outputs_v2, dim=1).cpu().numpy().flatten()
        prob_v3s = torch.softmax(outputs_v3s, dim=1).cpu().numpy().flatten()

        predictions_v2.append(prob_v2)
        predictions_v3s.append(prob_v3s)

    # Averaging the predictions
    avg_prob_v2 = np.mean(predictions_v2, axis=0)
    avg_prob_v3s = np.mean(predictions_v3s, axis=0)
    stacked_prob = (avg_prob_v2 + avg_prob_v3s) / 2

    # Determine the label with the highest confidence
    pred_label = class_labels[np.argmax(stacked_prob)]
    confidence = np.max(stacked_prob)

    st.write(f"Prediction: **{pred_label}**")
    st.write(f"Confidence: **{confidence:.4f}**")

    # Generate CAMs and overlay circles
    cam_v2 = get_cam(model_v2, image_tensor, target_layer=-2)
    cam_v3s = get_cam(model_v3s, image_tensor, target_layer=-1)
    combined_cam = (cam_v2 + cam_v3s) / 2

    image_with_circles = overlay_circles(image, combined_cam)
    st.image(image_with_circles, caption='Image with highlighted regions.', use_column_width=True)
