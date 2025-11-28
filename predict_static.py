import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import json

# Load model
checkpoint = torch.load("best_landmark_model.pth")
label_map = {v: k for k, v in checkpoint["label_to_index"].items()}

# Corrected model definition
class HandSignClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

model = HandSignClassifier(input_size=42, num_classes=len(label_map))
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Init webcam
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            coords = []
            for pt in lm.landmark:
                coords.extend([pt.x, pt.y])  # Only x,y coordinates
            
            if len(coords) == 42:  # Ensure 21 landmarks (x,y)
                x = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    pred = model(x)
                    prob = torch.softmax(pred, dim=1)[0]
                    confidence, label_idx = torch.max(prob, 0)
                    label = label_map[int(label_idx)]
                
                # Display prediction with confidence
                cv2.putText(frame, f"{label} ({confidence:.0%})", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            mp.solutions.drawing_utils.draw_landmarks(
                frame, lm, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Sign Prediction", frame)
    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()