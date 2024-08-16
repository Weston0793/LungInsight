import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
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

# Function to get CAM
def get_cam(model, img_tensor, target_layer_name):
    model.eval()
    
    def forward_hook(module, input, output):
        activation[0] = output

    activation = {}
    layer = dict([*model.named_modules()]).get(target_layer_name, None)
    if layer is None:
        raise ValueError(f"Layer {target_layer_name} not found in the model")
        
    hook = layer.register_forward_hook(forward_hook)
    
    with torch.no_grad():
        output = model(img_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    
    hook.remove()
    
    weight_softmax_params = list(model.parameters())[-2].detach().numpy()
    weight_softmax = np.squeeze(weight_softmax_params)
    
    activation = activation[0].squeeze().cpu().data.numpy()
    cam = np.zeros(activation.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weight_softmax[predicted_class]):
        cam += w * activation[i, :, :]
    
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img_tensor.shape[-1], img_tensor.shape[-2]))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    
    return cam

# Function to overlay hexagons on the image
# Function to overlay hexagons on the image
def overlay_hexagons(image, cam):
    # Scale cam to the range [0, 255] to highlight the lowest activation points
    cam_image = np.uint8(255 * cam)
    
    # Threshold to isolate the lowest activation points
    _, thresh = cv2.threshold(cam_image, 205, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours from the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area in descending order
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Convert PIL image to numpy array
    image_np = np.array(image)
    
    hexagons = []
    total_activation_points = np.sum(cam_image > 205)
    covered_activation_points = 0
    
    # Split the image into two halves along the sagittal plane
    mid_x = image_np.shape[1] // 2
    left_half = image_np[:, :mid_x]
    right_half = image_np[:, mid_x:]
    
    # Process each half separately
    for half, half_cam in zip([left_half, right_half], [cam_image[:, :mid_x], cam_image[:, mid_x:]]):
        half_hexagons = []
        half_covered_points = 0
        for cnt in sorted_contours:
            if (len(half_hexagons) == 1 and half_covered_points >= 0.2 * total_activation_points) or \
               (len(half_hexagons) == 2 and half_covered_points >= 0.1 * total_activation_points) or \
               len(half_hexagons) >= 3:
                break

            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(cnt)

            # Skip contours that do not lie within the current half
            if x >= mid_x and half is left_half:
                continue
            if x + w <= mid_x and half is right_half:
                continue

            # Adjust x to be relative to the current half
            if half is right_half:
                x -= mid_x

            # Calculate the center and size for the hexagon
            center_x, center_y = x + w // 2, y + h // 2
            size = int(0.45 * max(w, h) // 2)

            # Generate points for a 6-sided polygon (hexagon)
            hexagon = np.array([
                (center_x + size * np.cos(theta), center_y + size * np.sin(theta))
                for theta in np.linspace(0, 2 * np.pi, 6, endpoint=False)
            ], np.int32)

            # Check for overlaps with existing hexagons
            overlaps = False
            for existing_hexagon in half_hexagons:
                if cv2.pointPolygonTest(existing_hexagon, (center_x, center_y), False) >= 0:
                    overlaps = True
                    break

            if not overlaps:
                # Calculate the area of the current hexagon
                mask = np.zeros_like(half_cam)
                cv2.fillPoly(mask, [hexagon], 1)
                hexagon_activation_points = np.sum(mask * (half_cam > 205))

                half_hexagons.append(hexagon)
                half_covered_points += hexagon_activation_points

        hexagons.extend(half_hexagons)
        covered_activation_points += half_covered_points
    
    # Draw hexagons on the original image
    for hexagon in hexagons:
        if hexagon[0][0] < mid_x:
            cv2.polylines(left_half, [hexagon], isClosed=True, color=(255, 0, 0), thickness=2)
        else:
            cv2.polylines(right_half, [hexagon], isClosed=True, color=(255, 0, 0), thickness=2)
    
    # Merge the two halves back into the original image
    image_np[:, :mid_x] = left_half
    image_np[:, mid_x:] = right_half
    
    # Convert numpy array back to PIL image
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

    # Generate CAMs
    cam_v2 = get_cam(model_v2, image_tensor, target_layer_name='base_model.features.18.2')
    cam_v3s = get_cam(model_v3s, image_tensor, target_layer_name='base_model.features.12')
    combined_cam = (cam_v2 + cam_v3s) / 2

    # Overlay hexagons on the original image
    image_with_hexagons = overlay_hexagons(image, combined_cam)
    
    # Create a heatmap of the combined CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * (1-combined_cam)), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    image_np = np.array(image.resize((300, 300)))
    image_np = np.float32(image_np) / 255
    cam_overlay = heatmap + np.expand_dims(image_np, axis=2)
    cam_overlay = cam_overlay / np.max(cam_overlay)
    cam_overlay_image = Image.fromarray(np.uint8(255 * cam_overlay))

    st.image(image_with_hexagons, caption='Image with highlighted regions.', use_column_width=True)
    st.image(cam_overlay_image, caption='Stacked CAM overlay.', use_column_width=True)
