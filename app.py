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

def overlay_rectangles(image, cam):
    # Convert the original image to a numpy array
    image_np = np.array(image)
    original_height, original_width = image_np.shape[:2]
    
    # Ensure CAM is grayscale
    if len(cam.shape) == 3:
        cam = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
    
    # Scale CAM to [0, 255] range
    cam_image = np.uint8(255 * (1 - cam))
    
    # Split the CAM into left and right halves
    midline = cam_image.shape[1] // 2
    cam_left = cam_image[:, :midline]
    cam_right = cam_image[:, midline:]
    
    # Function to process a CAM half and draw rectangles on the original image
    def process_and_draw(cam_half, origin_x):
        # Threshold to isolate the lowest activation points
        _, thresh = cv2.threshold(cam_half, 230, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area in descending order (largest first)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Calculate scaling factors to map CAM half coordinates to original image size
        scale_x = (original_width / 2) / cam_half.shape[1]
        scale_y = original_height / cam_half.shape[0]
        
        # Define max area for bounding boxes (50% of the original image area)
        max_area = 0.50 * original_width * original_height
        
        for cnt in sorted_contours:
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Debugging: log the raw bounding box values
            st.write(f"Raw bounding box - x: {x}, y: {y}, w: {w}, h: {h}")
            
            # Scale bounding box to original image size
            center_x = (x + w / 2) * scale_x + origin_x
            center_y = (y + h / 2) * scale_y
            w = int(w * scale_x)
            h = int(h * scale_y)
            
            # Calculate new x and y based on the scaled center
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            
            # Debugging: log the scaled bounding box values
            st.write(f"Scaled bounding box - x: {x}, y: {y}, w: {w}, h: {h}")
            
            # Check if the bounding box exceeds the allowed area
            if w * h > max_area:
                scale_factor = (max_area / (w * h)) ** 0.5
                w = int(w * scale_factor)
                h = int(h * scale_factor)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
            
            # Debugging: log the final bounding box values
            st.write(f"Final bounding box - x: {x}, y: {y}, w: {w}, h: {h}")
            
            # Draw the rectangle on the image, ensuring all coordinates are within bounds
            x = max(0, min(x, original_width - 1))
            y = max(0, min(y, original_height - 1))
            w = max(1, min(w, original_width - x))
            h = max(1, min(h, original_height - y))
            
            cv2.rectangle(image_np, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
            
            # Stop after drawing the first valid rectangle
            break
    
    # Process and draw rectangles on the left and right halves
    process_and_draw(cam_left, origin_x=0)            # Process left half
    process_and_draw(cam_right, origin_x=midline)     # Process right half
    
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

    # Overlay rectangles on the original image
    image_with_rectangles = overlay_rectangles(image, combined_cam)
    st.image(image_with_rectangles, caption='Image with highlighted regions.', use_column_width=True)
    
    # Create a heatmap of the combined CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * (1-combined_cam)), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    image_np = np.array(image.resize((300, 300)))  # Resize image for the CAM overlay
    image_np = np.float32(image_np) / 255
    cam_overlay = heatmap + np.expand_dims(image_np, axis=2)
    cam_overlay = cam_overlay / np.max(cam_overlay)
    cam_overlay_image = Image.fromarray(np.uint8(255 * cam_overlay))


    st.image(cam_overlay_image, caption='Stacked CAM overlay.', use_column_width=True)
