import cv2
import numpy as np

# Load the scene and logo images
scene = cv2.imread("56390036.JPG", 1)
logo = cv2.imread("assassins-creed-logo.jpg", -1)

# Create a mask for the logo by thresholding its blue channel
_, mask = cv2.threshold(logo[:, :, 2], 10, 255, cv2.THRESH_BINARY)

# Create an inverted mask
mask = cv2.bitwise_not(mask)

# resize the mask to match the size of the scene
mask = cv2.resize(mask, (scene.shape[1], scene.shape[0]))

# change mask datatype
mask = np.array(mask, dtype=np.uint8)

# convert the mask to multi-channel
mask = np.dstack([mask] * 3)

# Use the mask to create the masked region of the logo
logo_masked = cv2.bitwise_and(logo, logo, mask=mask)

# Create a masked region for the scene using the inverted mask
scene_masked = cv2.bitwise_and(scene, scene, mask=mask)

# Use the cv2.add function to superimpose the logo over the scene
superimposed_image = cv2.add(scene_masked, logo_masked)

# Show the resulting image
cv2.imshow("Superimposed Image", superimposed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()