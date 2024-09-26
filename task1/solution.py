import cv2
import numpy as np

# Load images
selfie1 = cv2.imread('selfie1.jpg')
selfie2 = cv2.imread('selfie2.jpg')
selfie3 = cv2.imread('selfie3.jpg')
filter1 = cv2.imread('dog2.png', cv2.IMREAD_UNCHANGED)
filter2 = cv2.imread('rainbow.png', cv2.IMREAD_UNCHANGED)
filter3 = cv2.imread('mustache.png', cv2.IMREAD_UNCHANGED)
# cv2.IMREAD_UNCHANGED loads the image including its alpha channel (transparency) if it exists.

# Check if the images are loaded correctly
if any(img is None for img in [selfie1, selfie2, selfie3, filter1, filter2, filter3]):
    print("Error loading one or more images!")
else:
    print("All images loaded successfully!")


def overlay_image(background, overlay, position=(0, 0), size=(300, 300)):
    x, y = position  # Get x and y coordinates to start the filter
    h, w = size  # Get filter width (w) and height (h)

    # Ensure filter dimensions fit within the selfie bounds
    h = min(h, background.shape[0] - y)
    w = min(w, background.shape[1] - x)

    # Resize filter  to fit the area
    overlay_resized = cv2.resize(overlay, (w, h))

    # Extract transparency info from filter and make an inverse
    alpha_mask = overlay_resized[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha_mask

    # Get only color parts of the filter
    overlay_rgb = overlay_resized[:, :, :3]

    # ROI(Region of Interest) selects a part of the selfie where the filter will be applied
    roi = background[y:y + h, x:x + w]

    # Expand alpha_mask to match RGB channels for blending
    alpha_mask_3d = alpha_mask[..., None]

    # Blend filter with ROI using alpha transparency
    roi_new = ((alpha_mask_3d * overlay_rgb) + (alpha_inv[..., None] * roi)).astype(np.uint8)

    # Update selfie with the blended image (roi_new)
    background[y:y + h, x:x + w] = roi_new

    # Finally the function gives back new selfie with filter on it
    return background


# Set filters and positions for each selfie
adjustments = {
    'selfie1': {'filter': filter1, 'position': (105, 100), 'size': (200, 200)},
    'selfie2': {'filter': filter2, 'position': (165, 290), 'size': (150, 85)},
    'selfie3': {'filter': filter3, 'position': (515, 1000), 'size': (200, 580)}
}

# Dictionary to store results
results = {}
for selfie_name, settings in adjustments.items():
    selfie = eval(f"{selfie_name}")

    # Apply the filter
    result = overlay_image(
        background=selfie,
        overlay=settings['filter'],
        position=settings['position'],
        size=settings['size']
    )
    results[selfie_name] = result

    # Display and save the processed image
    cv2.imshow(f'Result {selfie_name}', result)
    cv2.waitKey(0)
    cv2.imwrite(f'{selfie_name}_with_filter.jpg', result)

cv2.destroyAllWindows()
