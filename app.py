import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import cv2
from Models import MultiClassMobileNetV2, MultiClassMobileNetV3Small
from CAM import get_cam
from Preprocess import apply_clahe

# Initialize Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_v2 = MultiClassMobileNetV2().to(device)
model_v3s = MultiClassMobileNetV3Small().to(device)

model_v2.load_state_dict(torch.load('bucket/MobileNetV2_4.pth', map_location=device)['model_state_dict'])
model_v3s.load_state_dict(torch.load('bucket/MobileNetV3Small_1.pth', map_location=device)['model_state_dict'])

model_v2.eval()
model_v3s.eval()

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
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    
    return cam

# Function to overlay rotated ellipses on the image
def overlay_ellipses(image, cam):
    try:
        # Convert the original image to a numpy array
        image_np = np.array(image)
        original_height, original_width = image_np.shape[:2]
        
        # Scale CAM to [0, 255] range and resize it to match the original image size
        cam_resized = cv2.resize(cam, (original_width, original_height))
        cam_image = np.uint8(255 * cam_resized)
        
        # Threshold to isolate the lowest activation points
        _, thresh = cv2.threshold(cam_image, 0, 50, cv2.THRESH_BINARY_INV)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area in descending order (largest first)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Define max area for bounding ellipses (15% of the original image area)
        max_area = 0.15 * original_width * original_height
        
        for cnt in sorted_contours:
            # Get the ellipse bounding box
            if len(cnt) >= 5:  # FitEllipse requires at least 5 points
                ellipse = cv2.fitEllipse(cnt)
                (x, y), (MA, ma), angle = ellipse
                
                # Ensure ellipse stays within the image bounds
                if (x - MA / 2) < 0 or (x + MA / 2) > original_width or (y - ma / 2) < 0 or (y + ma / 2) > original_height:
                    continue

                # Check if the bounding box exceeds the allowed area
                if MA * ma > max_area:
                    scale_factor = (max_area / (MA * ma)) ** 0.8
                    MA *= scale_factor
                    ma *= scale_factor
                
                # Draw the rotated ellipse on the image
                cv2.ellipse(image_np, ((int(x), int(y)), (int(MA), int(ma)), angle), color=(0, 255, 0), thickness=2)
                
                # Stop after drawing the first valid ellipse
                break
    except Exception as e:
        st.write(f"Error in overlay_ellipses: {e}")
    
    # Convert numpy array back to PIL image and return
    return Image.fromarray(image_np)

# Function to overlay CAM and ellipses on the original image
def overlay_cam_and_ellipses(image, cam):
    try:
        # Convert image to numpy array
        image_np = np.array(image)
        original_height, original_width = image_np.shape[:2]
        
        # Ensure image_np has three channels by stacking if it's single-channel
        if len(image_np.shape) == 2 or image_np.shape[2] == 1:
            image_np = np.stack((image_np,) * 3, axis=-1)
        
        # Resize CAM to match original image size
        cam_resized = cv2.resize(cam, (original_width, original_height))
        
        # Apply color map to CAM
        heatmap = cv2.applyColorMap(np.uint8(255 * (1 - cam_resized)), cv2.COLORMAP_JET)
        
        # Combine CAM overlay with the original image
        cam_overlay = np.float32(heatmap) / 255 + np.float32(image_np) / 255
        cam_overlay = cam_overlay / np.max(cam_overlay)
        
        # Convert back to PIL image
        cam_overlay_image = Image.fromarray(np.uint8(255 * cam_overlay))
        
        # Draw the ellipses on the combined image
        image_with_ellipses = overlay_ellipses(cam_overlay_image, cam_resized)
        
        return image_with_ellipses
    except Exception as e:
        st.write(f"Error in overlay_cam_and_ellipses: {e}")
        return image

# Streamlit App
st.title("LungInsight for X-ray Classification")
st.write("Upload an X-ray image and get the prediction with confidence levels.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    image_clahe = apply_clahe(image)

    transform = transforms.Compose([
        transforms.Resize((300, 300)),  # Keep this to match the model input size
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

    # Overlay ellipses on the original image
    image_with_ellipses = overlay_ellipses(image, combined_cam)
    st.image(image_with_ellipses, caption='Image with highlighted regions (ellipses).', use_column_width=True)
    
    # Overlay CAM with ellipses for debugging
    cam_overlay_with_ellipses = overlay_cam_and_ellipses(image, combined_cam)
    st.image(cam_overlay_with_ellipses, caption='CAM overlay with ellipses.', use_column_width=True)

