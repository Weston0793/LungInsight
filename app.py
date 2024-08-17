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
import operator

# Initialize Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_v2 = MultiClassMobileNetV2().to(device)
model_v3s = MultiClassMobileNetV3Small().to(device)

model_v2.load_state_dict(torch.load('bucket/MobileNetV2_4.pth', map_location=device)['model_state_dict'])
model_v3s.load_state_dict(torch.load('bucket/MobileNetV3Small_1.pth', map_location=device)['model_state_dict'])

model_v2.eval()
model_v3s.eval()

def overlay_rotated_rectangle(image, heatmap):
    # Convert the original image to a numpy array
    image_np = np.array(image)
    original_height, original_width = image_np.shape[:2]
    
    # Split the heatmap into left and right halves
    midline = heatmap.shape[1] // 2
    heatmap_left = heatmap[:, :midline]
    heatmap_right = heatmap[:, midline:]
    
    # Scaling factors for the original image dimensions
    cms = heatmap.shape[0]  # The heatmap is assumed to be square
    max_area = 0.3 * (original_width // 2) * original_height  # 30% of half image
    
    def process_and_draw(heatmap_half, origin_x):
        # Since heatmap_half is already a NumPy array, no need to convert
        heatmap_np = heatmap_half
        
        # Threshold to isolate the highest activation points
        _, thresh = cv2.threshold(heatmap_np, np.max(heatmap_np) * 0.7, 255, cv2.THRESH_BINARY)
        thresh = np.uint8(thresh)
        
        # Find contours in the thresholded heatmap
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Sort contours by area, descending
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get the minimum area rectangle around the largest contour
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Calculate the area of the rectangle
            width = rect[1][0]
            height = rect[1][1]
            rect_area = width * height
            
            # Scale down the rectangle if it exceeds the allowed maximum area
            if rect_area > max_area:
                scale_factor = (max_area / rect_area) ** 0.5
                width = int(width * scale_factor)
                height = int(height * scale_factor)
                rect = ((rect[0][0], rect[0][1]), (width, height), rect[2])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
            
            # Adjust the box coordinates to account for the half-image offset
            box[:, 0] += origin_x
            
            # Draw the rotated rectangle on the image
            cv2.drawContours(image_np, [box], 0, (255, 0, 0), 2)
    
    # Process and draw rectangles on the left and right halves
    process_and_draw(heatmap_left, origin_x=0)            # Process left half
    process_and_draw(heatmap_right, origin_x=midline * original_width // cms)  # Process right half
    
    # Convert numpy array back to PIL image and return
    return Image.fromarray(image_np)

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

    # Generate the heatmap (modify this part if you have a different method to generate heatmaps)
    heatmap = cv2.applyColorMap(np.uint8(255 * (1-combined_cam)), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    st.image(heatmap, caption='Heatmap', use_column_width=True)

    # Overlay rectangles on the original image using heatmap analysis
    image_with_rectangles = overlay_rectangles(image, combined_cam)
    st.image(image_with_rectangles, caption='Image with highlighted regions.', use_column_width=True)
    
    # Create a heatmap-overlayed image
    image_np = np.array(image.resize((300, 300)))  # Resize image for the CAM overlay
    image_np = np.float32(image_np) / 255
    cam_overlay = heatmap + np.expand_dims(image_np, axis=2)
    cam_overlay = cam_overlay / np.max(cam_overlay)
    cam_overlay_image = Image.fromarray(np.uint8(255 * cam_overlay))

    st.image(cam_overlay_image, caption='Stacked CAM overlay.', use_column_width=True)
