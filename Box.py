import numpy as np
import operator
import cv2
from PIL import Image

def find_largest_similar_rectangle(heatmap, origin_x, origin_y, threshold=0.4):
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
    
    def process_and_draw(heatmap_half, origin_x, shift_amount_x, shift_amount_y):
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
        x1 = origin_x + left * (original_width // cms)
        y1 = top * (original_height // cms)
        x2 = origin_x + right * (original_width //  cms)
        y2 = bottom * (original_height // cms)
        
        # Shift the rectangle horizontally and vertically by the specified shift amounts
        x1_shifted = int(x1 + shift_amount_x)
        y1_shifted = int(y1 + shift_amount_y)
        x2_shifted = int(x2 + shift_amount_x)
        y2_shifted = int(y2 + shift_amount_y)
        
        # Draw the rectangle on the image
        cv2.rectangle(image_np, (x1_shifted, y1_shifted), (x2_shifted, y2_shifted), color=(255, 0, 0), thickness=2)
    
    # Calculate the amount by which to shift the rectangles
    shift_amount_x_left = original_width // 20  # Example: 5% of the image width
    shift_amount_x_right = original_width // 25
    shift_amount_y = original_height // 10  # Shift down by 10% of the image height
    
    # Process and draw rectangles on the left and right halves
    process_and_draw(heatmap_left, origin_x=0, shift_amount_x=shift_amount_x_left, shift_amount_y=shift_amount_y)
    process_and_draw(heatmap_right, origin_x=midline * original_width // cms, shift_amount_x=shift_amount_x_right, shift_amount_y=shift_amount_y)
    
    # Convert numpy array back to PIL image and return
    return Image.fromarray(image_np)
