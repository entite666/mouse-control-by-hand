import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # removes delay

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cam_w, cam_h = 640, 480

# smoothing variables
smooth_x, smooth_y = 0, 0
prev_iy = 0
click_cooldown = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False  # speeds up processing
    result = hands.process(rgb)
    rgb.flags.writeable = True

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        lm = result.multi_hand_landmarks[0].landmark

        # index fingertip position
        ix = int(lm[8].x * cam_w)
        iy = int(lm[8].y * cam_h)

        # map to screen
        mouse_x = int(ix * screen_w / cam_w)
        mouse_y = int(iy * screen_h / cam_h)

        # smooth movement
        smooth_x += (mouse_x - smooth_x) / 7
        smooth_y += (mouse_y - smooth_y) / 7
        pyautogui.moveTo(int(smooth_x), int(smooth_y))

        # flick down to click
        if click_cooldown == 0:
            if iy - prev_iy > 15:
                pyautogui.click()
                click_cooldown = 20
                cv2.putText(frame, "CLICK!", (250, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        if click_cooldown > 0:
            click_cooldown -= 1

        prev_iy = iy

        # draw index fingertip
        cv2.circle(frame, (ix, iy), 12, (0, 255, 0), -1)

    cv2.imshow("Hand Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()