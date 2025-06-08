import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
canvas = None
prev_x, prev_y = 0, 0
draw_color = (0, 255, 0)
brush_size = 5
eraser_size = 30
mode = 'draw'

def fingers_up(landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []
    fingers.append(landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x)
    for i in range(1, 5):
        fingers.append(landmarks[tip_ids[i]].y < landmarks[tip_ids[i] - 2].y)
    return fingers

def end_interface():
    cap.release()
    cv2.destroyAllWindows()
    print("Interface ended.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = fingers_up(lm)
            fingers_count = fingers.count(True)
            x, y = int(lm[8].x * w), int(lm[8].y * h)

            # Gesture: Color switch
            if fingers_count == 3:
                draw_color = (0, 0, 255)
                mode = "color: red"
            elif fingers_count == 4:
                draw_color = (0, 255, 0)
                mode = "color: green"
            elif fingers_count == 5:
                draw_color = (255, 0, 0)
                mode = "color: blue"

            # Gesture: Brush size (thumb only)
            thumb_tip = lm[4]
            thumb_base = lm[2]
            if fingers == [True, False, False, False, False]:
                if thumb_tip.y < thumb_base.y:
                    brush_size = min(brush_size + 1, 50)
                    mode = f'brush + ({brush_size})'
                elif thumb_tip.y > thumb_base.y:
                    brush_size = max(brush_size - 1, 1)
                    mode = f'brush - ({brush_size})'

            # Draw
            if fingers[1] and not fingers[2]:
                cv2.circle(frame, (x, y), brush_size, draw_color, -1)
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y
                cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, brush_size)
                prev_x, prev_y = x, y
                mode = f'draw ({brush_size})'

            # Erase
            elif fingers[1] and fingers[2] and not fingers[3]:
                cv2.circle(frame, (x, y), eraser_size, (0, 0, 0), -1)
                cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), eraser_size)
                prev_x, prev_y = x, y
                mode = f'erase ({eraser_size})'
            else:
                prev_x, prev_y = 0, 0

            cv2.putText(frame, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 2)

    # Merge canvas
    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv_mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY_INV)
    inv_mask = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv_mask)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Virtual Drawing", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        end_interface()
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == ord('s'):
        cv2.imwrite("drawing_capture.png", canvas)
        print("Drwaing saved as 'drawing_capture.png'.")