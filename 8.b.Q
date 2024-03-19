import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image from the COCO dataset
image_path = '/content/bay-california-corps-marines-wallpaper-preview.jpg'  # Replace with the path to your image
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold the image to obtain a binary image
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Perform morphological operations to clean up the image
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Perform morphological operations to detect sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Perform morphological operations to detect sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

# Identify areas where we are uncertain (potential markers)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Label markers
_, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Mark unknown regions with 0
markers[unknown == 255] = 0

# Apply watershed algorithm
markers = cv2.watershed(image, markers)

# Outline the segmented regions
image[markers == -1] = [255, 0, 0]  # Mark watershed boundaries in red

# Display the result using matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Segmented Image')
plt.axis('off')
plt.show()
