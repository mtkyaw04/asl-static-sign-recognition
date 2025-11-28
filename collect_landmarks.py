import cv2
import mediapipe as mp
import json
import os

SAVE_DIR = "dataset_landmarks"
os.makedirs(SAVE_DIR, exist_ok=True)

label = input("Enter label (e.g. 'a', 'b', 'c', ...): ").strip().lower()
output_file = os.path.join(SAVE_DIR, f"{label}.json")

if os.path.exists(output_file):
    with open(output_file, "r") as f:
        data = json.load(f)
else:
    data = []

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

print("Press SPACE to capture hand, ESC to exit")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Collecting Landmarks", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            landmark_data = []
            for pt in lm.landmark:
                landmark_data.extend([pt.x, pt.y])  # Only x and y
            data.append(landmark_data)
            print(f"Captured {len(data)} samples for label '{label}'")

cap.release()
cv2.destroyAllWindows()

with open(output_file, "w") as f:
    json.dump(data, f)

print(f"Saved {len(data)} samples to {output_file}")
