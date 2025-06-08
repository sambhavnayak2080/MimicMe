import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip the image for natural interaction
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Create a black canvas
    black_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmark coordinates
            landmark_points = []
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                landmark_points.append((x, y))
                cv2.circle(black_canvas, (x, y), 4, (0, 255, 0), -1)  # dots

            # Draw lines to mimic hand bones
            connections = mp_hands.HAND_CONNECTIONS
            for start_idx, end_idx in connections:
                start = landmark_points[start_idx]
                end = landmark_points[end_idx]
                cv2.line(black_canvas, start, end, (255, 255, 255), 2)

            # Draw a bounding rectangle around the hand
            x_list = [pt[0] for pt in landmark_points]
            y_list = [pt[1] for pt in landmark_points]
            x_min, y_min = min(x_list), min(y_list)
            x_max, y_max = max(x_list), max(y_list)
            cv2.rectangle(black_canvas, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Stack original and black canvas
    stacked = np.hstack((frame, black_canvas))
    cv2.imshow("Live Feed (Left) | Hand Movement Visualization (Right)", stacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
