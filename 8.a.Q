import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('/content/bay-california-corps-marines-wallpaper-preview.jpg')
print(image)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Noise removal using Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Thresholding to get binary image
_, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Sure background area using morphological operations
kernel = np.ones((3, 3), np.uint8)
sure_bg = cv2.dilate(thresholded, kernel, iterations=3)

# Marker labelling using connected components
_, markers = cv2.connectedComponents(sure_bg)

# Apply watershed algorithm
markers = markers + 1
markers[thresholded == 255] = 0
markers = cv2.watershed(image, markers)

# Outline the segmented regions
image[markers == -1] = [255, 0, 0]  # Mark watershed boundaries in red

# Save the segmented image
cv2.imwrite('segmented_image.jpg', image)

# Read the segmented image
segmented_image = cv2.imread('segmented_image.jpg')

# Display the segmented image using matplotlib
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title('Segmented Image')
plt.axis('off')
plt.show()
