import cv2
import numpy as np
from math import atan2, pi

img1 = cv2.imread('sphere1.jpg')
img2 = cv2.imread('sphere2.jpg')

# Resizing the image 
img1 = cv2.resize(img1, (0, 0), fx=2, fy=2)
img2 = cv2.resize(img2, (0, 0), fx=2, fy=2)

# Converting images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Getting the dimensions of image
height = min(gray1.shape[0], gray2.shape[0])
width = min(gray1.shape[1], gray2.shape[1])

print(f"Image dimensions: {height} x {width}")

# Initiliazing empty images
img1_float = np.zeros((height, width))
img2_float = np.zeros((height, width))

# Converting greyscale pixels to floats and storing them
for y in range(height):
    for x in range(width):
        img1_float[y, x] = float(gray1[y, x])
        img2_float[y, x] = float(gray2[y, x])

# Initializing empty images
Ix = np.zeros((height, width))      # X-gradient for first image
Iy = np.zeros((height, width))      # Y-gradient for first image
It = np.zeros((height, width))       # Temporal gradient (difference between exact pixels in both images)
Ix_It = np.zeros((height, width))   # Temporal gradient * X-gradient (first image)
Iy_It = np.zeros((height, width))   # Temporal gradient * Y-gradient (first image)
u = np.zeros((height, width))        # Flow in the X-direction
v = np.zeros((height, width))        # Flow in the Y-direction
mag = np.zeros((height, width))      # Magnitude of flow
angle = np.zeros((height, width))      # Flow angle

# Creating a mask based on where pixels with actual motion are
motion_mask = np.zeros((height, width))

# A threshold value is set to control the sensitivity of motion between images
# and will reduce noise captured in the background where there is little or no motion compared
# to significant motion that should be detected (for example: a rotating sphere or a car speeding by)
#
# Higher threshold -> motion is detected in smaller pixel differences 
# Lower threshold -> motion is detected in greater pixel differences
motion_threshold = 4
for y in range(1, height - 1):
    for x in range(1, width - 1):

        # Check if the pixel is part of the moving object
        if (img2_float[y, x] - img1_float[y, x]) ** 2 > (motion_threshold ** 2):
            motion_mask[y, x] = 1  # Marking a pixel as having motion

        # Computing temporal gradients for the part of the image with detected motion
        if motion_mask[y, x] == 1:
            Ix[y, x] = (img1_float[y, x + 1] - img1_float[y, x - 1]) / 2.0
            Iy[y, x] = (img1_float[y + 1, x] - img1_float[y - 1, x]) / 2.0
            It[y, x] = img2_float[y, x] - img1_float[y, x]

            Ix_It[y, x] = It[y, x] * Ix[y, x]
            Iy_It[y, x] = It[y, x] * Iy[y, x]

        else: # Gradients become zero in the background
            Ix[y, x] = 0
            Iy[y, x] = 0
            It[y, x] = 0
            Ix_It[y, x] = 0
            Iy_It[y, x] = 0

# Images at intermediate stages
cv2.imshow("Gradient in the X direction (first image)", Ix)
cv2.imshow("Gradient in the Y direction (first image)", Iy)
cv2.imshow("Temporal gradient", It)
cv2.imshow("Temporal gradient * Gradient x-direction (first image)", Ix_It)
cv2.imshow("Temporal gradient * Gradient y-direction (first image)", Iy_It)

# Using a 5x5 neighborhood of pixels    
upper_window = 3
half_window = 2

# Setting image boundaries to account for neighborhood coordinates going outside of the image
for y in range(half_window, height - upper_window):    
    for x in range(half_window, width - upper_window):

        # Computing the harris matrix components
        sumIx_Ix = sum(Ix[y+i, x+j] * Ix[y+i, x+j] for i in range(-half_window, upper_window) for j in range(-half_window, upper_window))
        sumIx_Iy = sum(Ix[y+i, x+j] * Iy[y+i, x+j] for i in range(-half_window, upper_window) for j in range(-half_window, upper_window))
        sumIy_Iy = sum(Iy[y+i, x+j] * Iy[y+i, x+j] for i in range(-half_window, upper_window) for j in range(-half_window, upper_window))
        sumIt_Ix = sum(It[y+i, x+j] * Ix[y+i, x+j] for i in range(-half_window, upper_window) for j in range(-half_window, upper_window))
        sumIt_Iy = sum(It[y+i, x+j] * Iy[y+i, x+j] for i in range(-half_window, upper_window) for j in range(-half_window, upper_window))

        # Computing the motion for each pixel
        det = (sumIx_Ix * sumIy_Iy) - (sumIx_Iy * sumIx_Iy)
        if det != 0:    # Pixel does have motion

            # Flow of a pixel in the x-direction
            u[y, x] = (-sumIy_Iy * sumIt_Ix + sumIx_Iy * sumIt_Iy) / det

            # Flow of a pixel in the y-direction
            v[y, x] = (sumIx_Iy * sumIt_Ix - sumIx_Ix * sumIt_Iy) / det

        else:   # Pixel has no motion
            u[y, x] = 0
            v[y, x] = 0

cv2.imshow("Flow in the x-direction (u value)", u)
cv2.imshow("Flow in the y-direction (v value)", v)

# Compute flow magnitude and angle
for y in range(height):
    for x in range(width):
        mag[y, x] = (u[y, x] ** 2 + v[y, x] ** 2) ** 0.5
        angle[y, x] = atan2(v[y, x], u[y, x])

cv2.imshow("Magnitude of flow", mag)

color_img = np.zeros((height, width, 3), dtype=np.uint8)
for y in range(height):
    for x in range(width):
        flow_angle = angle[y, x]
        magnitude = mag[y, x]

        final_color = (flow_angle + pi) / (2 * pi)    # Mapping [-pi, pi] to a range from 0 to 1

        green = int((1.0 - final_color) * 255)
        blue = int(final_color * 255)

        # Intensity multiplier to brighten color channels
        intensity = int(magnitude * 30)

        if green < 0: green = 0
        if green > 255: green = 255

        if blue < 0: blue = 0
        if blue > 255: blue = 255

        if intensity < 0: intensity = 0
        if intensity > 255: intensity = 255

        color_img[y, x, 0] = int(blue * intensity) // 255
        color_img[y, x, 1] = int(green * intensity) // 255

# Final Colored Visualization of Optical Flow
cv2.imshow("colored.jpg", color_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
