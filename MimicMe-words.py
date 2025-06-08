import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe modules
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(max_num_hands=2)
face = mp_face.FaceMesh()
mp_draw = mp.solutions.drawing_utils

# Start camera
cap = cv2.VideoCapture(0)

# Exit function
def end_interface():
    cap.release()
    cv2.destroyAllWindows()
    print("Interface ended. Resources released.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    black_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hands
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            label = handedness.classification[0].label  # 'Left' or 'Right'
            landmark_points = []
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                landmark_points.append((x, y))
                cv2.circle(black_canvas, (x, y), 4, (0, 255, 0), -1)  # hand dots
            # Draw hand bones
            for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
                start = landmark_points[start_idx]
                end = landmark_points[end_idx]
                cv2.line(black_canvas, start, end, (255, 255, 255), 2)
            # Bounding box
            x_vals = [pt[0] for pt in landmark_points]
            y_vals = [pt[1] for pt in landmark_points]
            x_min, y_min, x_max, y_max = min(x_vals), min(y_vals), max(x_vals), max(y_vals)
            cv2.rectangle(black_canvas, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            # Label text
            cv2.putText(black_canvas, f"{label} Hand", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)

    # Process face
    face_results = face.process(rgb_frame)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            face_points = []
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                face_points.append((x, y))
                cv2.circle(black_canvas, (x, y), 1, (0, 0, 255), -1)  # red face dots
            # Label face
            x_list = [pt[0] for pt in face_points]
            y_list = [pt[1] for pt in face_points]
            x_avg, y_avg = int(np.mean(x_list)), int(np.mean(y_list))
            cv2.putText(black_canvas, "Face", (x_avg - 30, y_avg - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Stack original and black canvas
    stacked = np.hstack((frame, black_canvas))
    cv2.imshow("Camera Feed (Left) | Tracked View with Labels (Right)", stacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        end_interface()
        break
