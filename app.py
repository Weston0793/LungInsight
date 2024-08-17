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

def find_largest_similar_rectangle(heatmap, origin_x, origin_y, threshold=0.6):
    """
    Finds the largest rectangle around the origin point that contains similarly
    highly activated points, moving up, down, left, and right from the origin.
    """
    height, width = heatmap.shape
    origin_value = heatmap[origin_y, origin_x]

    if origin_value < threshold:
        return origin_x, origin_y, origin_x, origin_y

    # Initialize boundaries of the rectangle
    left = origin_x
    right = origin_x
    top = origin_y
    bottom = origin_y

    # Expand the rectangle leftwards
    while left > 0 and heatmap[origin_y, left - 1] >= origin_value * threshold:
        left -= 1

    # Expand the rectangle rightwards
    while right < width - 1 and heatmap[origin_y, right + 1] >= origin_value * threshold:
        right += 1

    # Expand the rectangle upwards
    while top > 0 and heatmap[top - 1, origin_x] >= origin_value * threshold:
        top -= 1

    # Expand the rectangle downwards
    while bottom < height - 1 and heatmap[bottom + 1, origin_x] >= origin_value * threshold:
        bottom += 1

    return left, top, right, bottom

def overlay_rectangles(image, heatmap):
    # Convert the original image to a numpy array
    image_np = np.array(image)
    original_height, original_width = image_np.shape[:2]
    
    # Split the heatmap into left and right halves
    midline = heatmap.shape[1] // 2
    heatmap_left = heatmap[:, :midline]
    heatmap_right = heatmap[:, midline:]
    
    # Scaling factors for the original image dimensions
    cms = heatmap.shape[0]  # The heatmap is assumed to be square
    
    def process_and_draw(heatmap_half, origin_x):
        # Find the maximum value and its index in each row
        val = []
        for i in range(0, heatmap_half.shape[0]):
            index, value = max(enumerate(heatmap_half[i]), key=operator.itemgetter(1))
            val.append(value)
        
        # Find the index of the row with the highest activation
        y_index, y_value = max(enumerate(val), key=operator.itemgetter(1))
        
        # Find the x index of the highest activation in that row
        x_index, x_value = max(enumerate(heatmap_half[y_index]), key=operator.itemgetter(1))
        
        # Use the new function to find the largest rectangle
        left, top, right, bottom = find_largest_similar_rectangle(heatmap_half, x_index, y_index)
        
        # Convert coordinates to original image space
        x1 = origin_x + left * (original_width // (2 * cms))
        y1 = top * (original_height // cms)
        x2 = origin_x + right * (original_width // (2 * cms))
        y2 = bottom * (original_height // cms)
        
        # Draw the rectangle on the image
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
    
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
