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

def overlay_rectangles(image, cam):
    # Convert the original image to a numpy array
    image_np = np.array(image)
    original_height, original_width = image_np.shape[:2]
    
    # Resize the CAM to the original image dimensions to ensure proper scaling
    cam_resized = cv2.resize(cam, (original_width, original_height))
    cam_image = np.uint8(255 * cam_resized)
    
    # Split the CAM into left and right halves
    midline = cam_image.shape[1] // 2
    cam_left = cam_image[:, :midline]
    cam_right = cam_image[:, midline:]
    
    def find_bounding_box(cam_half, origin_x):
        # Threshold to isolate the lowest activation points
        _, thresh = cv2.threshold(cam_half, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours of the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (smallest first to capture least activation points)
        sorted_contours = sorted(contours, key=cv2.contourArea)
        
        max_area = 0.3 * original_width * original_height
        for cnt in sorted_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Scale bounding box coordinates back to the original image size
            x = int((x + origin_x))
            y = int(y)
            w = int(w)
            h = int(h)
            
            if w * h <= max_area:
                # Draw the rectangle on the original image
                cv2.rectangle(image_np, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
                break
    
    # Analyze and draw bounding boxes for left and right halves
    find_bounding_box(cam_left, origin_x=0)             # Process left half
    find_bounding_box(cam_right, origin_x=midline)      # Process right half
    
    # Return the image with bounding boxes as a PIL image
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
    
    # Create a heatmap of the combined CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * (1-combined_cam)), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    image_np = np.array(image.resize((300, 300)))  # Resize image for the CAM overlay
    image_np = np.float32(image_np) / 255
    cam_overlay = heatmap + np.expand_dims(image_np, axis=2)
    cam_overlay = cam_overlay / np.max(cam_overlay)
    cam_overlay_image = Image.fromarray(np.uint8(255 * cam_overlay))

    st.image(image_with_rectangles, caption='Image with highlighted regions.', use_column_width=True)
    st.image(cam_overlay_image, caption='Stacked CAM overlay.', use_column_width=True)
