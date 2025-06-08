import cv2
import mediapipe as mp
import numpy as np
import time
import math

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
last_select_time = 0
cooldown = 1

# Radial tool segments
tools = [
    {"label": "RED", "color": (0, 0, 255)},
    {"label": "GREEN", "color": (0, 255, 0)},
    {"label": "BLUE", "color": (255, 0, 0)},
    {"label": "ERASE", "action": "erase"},
    {"label": "CLEAR", "action": "clear"},
    {"label": "SAVE", "action": "save"},
]

def fingers_up(landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []
    fingers.append(landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x)
    for i in range(1, 5):
        fingers.append(landmarks[tip_ids[i]].y < landmarks[tip_ids[i] - 2].y)
    return fingers

def draw_radial_menu(frame, center, radius, segments, active_angle=None):
    angle_step = 360 / len(segments)
    for i, tool in enumerate(segments):
        start_angle = int(i * angle_step - angle_step / 2)
        end_angle = int(start_angle + angle_step)
        color = tool.get("color", (200, 200, 200))
        if active_angle is not None and start_angle <= active_angle < end_angle:
            cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, (255, 255, 255), -1)
        cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, color, 2)

        # Draw label
        angle_rad = math.radians((start_angle + end_angle) / 2)
        label_x = int(center[0] + radius * 0.7 * math.cos(angle_rad))
        label_y = int(center[1] + radius * 0.7 * math.sin(angle_rad))
        cv2.putText(frame, tool['label'], (label_x - 20, label_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def get_hovered_segment(center, finger_pos, num_segments):
    dx, dy = finger_pos[0] - center[0], finger_pos[1] - center[1]
    angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
    return int(angle // (360 / num_segments)), angle

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
            x, y = int(lm[8].x * w), int(lm[8].y * h)
            finger_tip = (x, y)

            # Show radial menu if all fingers up
            if fingers.count(True) == 5:
                menu_center = finger_tip
                menu_radius = 100
                seg_id, seg_angle = get_hovered_segment(menu_center, finger_tip, len(tools))
                draw_radial_menu(frame, menu_center, menu_radius, tools, seg_angle)

                if time.time() - last_select_time > cooldown:
                    selected = tools[seg_id]
                    if "color" in selected:
                        draw_color = selected["color"]
                        mode = f'color: {selected["label"].lower()}'
                    elif selected["action"] == "clear":
                        canvas = np.zeros_like(frame)
                        mode = "cleared"
                    elif selected["action"] == "save":
                        filename = f"radial_drawing_{int(time.time())}.png"
                        cv2.imwrite(filename, canvas)
                        mode = f"saved: {filename}"
                    elif selected["action"] == "erase":
                        mode = "erase"
                    last_select_time = time.time()

            # Draw / Erase when only 1 or 2 fingers
            if fingers[1] and not fingers[2]:
                if mode == 'erase':
                    cv2.circle(frame, (x, y), eraser_size, (0, 0, 0), -1)
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), eraser_size)
                else:
                    cv2.circle(frame, (x, y), brush_size, draw_color, -1)
                    cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, brush_size)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = 0, 0

            cv2.putText(frame, f"Mode: {mode}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, draw_color, 2)

    # Merge canvas
    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv_mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY_INV)
    inv_mask = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv_mask)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Gesture Drawing + Radial Menu", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        end_interface()
        break
