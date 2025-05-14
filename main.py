import cv2
import numpy as np

# Load overlay images with alpha channel (must be PNG with transparency)
stop_sign = cv2.imread('stop.png', cv2.IMREAD_UNCHANGED)
go_arrow = cv2.imread('go2.png', cv2.IMREAD_UNCHANGED)

# Resize overlays
stop_sign = cv2.resize(stop_sign, (100, 100))
go_arrow = cv2.resize(go_arrow, (100, 100))

# Function to overlay transparent image
def overlay_image(bg, fg, x, y):
    h, w = fg.shape[:2]

    # Ensure overlay fits inside background
    if y + h > bg.shape[0] or x + w > bg.shape[1]:
        return bg

    alpha = fg[:, :, 3] / 255.0
    for c in range(3):
        bg[y:y+h, x:x+w, c] = (1 - alpha) * bg[y:y+h, x:x+w, c] + alpha * fg[:, :, c]
    return bg

# Define HSV color ranges
RED_LOWER = np.array([0, 120, 70])
RED_UPPER = np.array([10, 255, 255])
GREEN_LOWER = np.array([40, 50, 50])
GREEN_UPPER = np.array([90, 255, 255])

# Load test image (instead of webcam)
frame = cv2.imread('traffic_lights.jpg')

if frame is None:
    print("Image not found! Make sure 'traffic_light.jpg' exists.")
    exit()

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Create masks
red_mask = cv2.inRange(hsv, RED_LOWER, RED_UPPER)
green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

# Find contours
red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
green_contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Detect and overlay
for contour in red_contours:
    if cv2.contourArea(contour) > 500:
        x, y, w, h = cv2.boundingRect(contour)
        y = max(y - 110, 0)  # Prevent overlay from going outside
        frame = overlay_image(frame, stop_sign, x, y)
        cv2.putText(frame, "RED SIGNAL", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        break  # Only detect one

for contour in green_contours:
    if cv2.contourArea(contour) > 500:
        x, y, w, h = cv2.boundingRect(contour)
        y = max(y - 110, 0)
        frame = overlay_image(frame, go_arrow, x, y)
        cv2.putText(frame, "GREEN SIGNAL", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        break

# Show result
cv2.imshow("AR Traffic Signal Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
