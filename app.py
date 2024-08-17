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
    
    # Ensure the CAM is in grayscale
    if len(cam.shape) == 3:  # If CAM is not grayscale, convert it
        cam = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
    
    # Scale CAM to [0, 255] range
    cam_image = np.uint8(255 * -cam)  # Inverting CAM for better visual contrast
    
    # Split the CAM into left and right halves
    midline = cam_image.shape[1] // 2
    cam_left = cam_image[:, :midline]
    cam_right = cam_image[:, midline:]
    
    # Scaling factors for the original image dimensions
    half_width = original_width // 2  # Width of one side of the split image
    
    # Function to process a CAM half and draw rectangles on the original image
    def process_and_draw(cam_half, origin_x):
        # Threshold to isolate the lowest activation points
        _, thresh = cv2.threshold(cam_half, 0, 45, cv2.THRESH_BINARY_INV)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area in descending order (largest first)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Calculate scaling factors to map CAM half coordinates to original image size
        scale_x = half_width / cam_half.shape[1]
        scale_y = original_height / cam_half.shape[0]
        
        # Define max area for bounding boxes (50% of the original image area)
        max_area = 0.50 * half_width * original_height
        
        for cnt in sorted_contours:
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(cnt)
            # Debugging: log the raw bounding box values
            st.write(f"Raw bounding box - x: {x}, y: {y}, w: {w}, h: {h}")            
            # Calculate the center of the bounding box
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Scale bounding box to original image size, including center shift
            x_scaled = int((x + origin_x) * scale_x)
            y_scaled = int(y * scale_y)
            w_scaled = int(w * scale_x*2)
            h_scaled = int(h * scale_y*2)
            
            # Calculate new center and shift the bounding box accordingly
            center_x_scaled = int((center_x + origin_x) * scale_x)
            center_y_scaled = int(center_y * scale_y)
            
            # Adjust bounding box based on new center (for more accurate scaling)
            x_scaled = center_x_scaled - w_scaled // 2
            y_scaled = center_y_scaled - h_scaled // 2
            # Debugging: log the scaled bounding box values
            st.write(f"Final bounding box - x: {x_scaled}, y: {y_scaled}, w: {w_scaled}, h: {h_scaled}")
            # Check if the bounding box exceeds the allowed area
            if w_scaled * h_scaled > max_area:
                scale_factor = (max_area / (w_scaled * h_scaled)) ** 0.9
                w_scaled = int(w_scaled * scale_factor)
                h_scaled = int(h_scaled * scale_factor)
            
            # Error checking for bounds
            if x_scaled < 0 or y_scaled < 0 or x_scaled + w_scaled > original_width or y_scaled + h_scaled > original_height:
                print(f"Warning: Bounding box out of bounds: x: {x_scaled}, y: {y_scaled}, w: {w_scaled}, h: {h_scaled}")
                continue  # Skip this bounding box if out of bounds
            
            # Draw the rectangle on the image, ensuring all coordinates are integers
            cv2.rectangle(image_np, (int(x_scaled), int(y_scaled)), (int(x_scaled + w_scaled), int(y_scaled + h_scaled)), color=(255, 0, 0), thickness=2)
            # Debugging: log the final bounding box values
            st.write(f"Final bounding box - x: {x_scaled}, y: {y_scaled}, w: {w_scaled}, h: {h_scaled}")
            # Stop after drawing the first valid rectangle
            break

    # Process and draw rectangles on the left and right halves
    process_and_draw(cam_left, origin_x=1)            # Process left half
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
