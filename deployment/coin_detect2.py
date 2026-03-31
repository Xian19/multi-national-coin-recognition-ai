import torch
import cv2
import numpy as np
from ultralytics import YOLO
from timm import create_model
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from picamera2 import Picamera2
import time
import os
from datetime import datetime
import pyttsx3
import RPi.GPIO as GPIO  # <-- For LED control

# --- LED Setup ---
LED_PINS = [17, 27, 22, 26]
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in LED_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.HIGH)  # Turn ON LEDs at start

# --- Paths ---
annotated_dir = "/home/GacheSalazar/annotated_results"
unlabelled_dir = "/home/GacheSalazar/unlabelled_results"
final_results_dir = "/home/GacheSalazar/final_results"
yolo_model_path = "/home/GacheSalazar/best.pt"
rcnn_model_path = "/home/GacheSalazar/best_model.pth"

# --- Class Labels ---
class_names = [
    'Chinese One Yuan Coin', 'Japanese Fifty Yen Coin',
    'Japanese Five Hundred Coin', 'Japanese Five Yen Coin', 'Japanese One Hundred Yen Coin',
    'Japanese One Yen Coin', 'Japanese Ten Yen Coin', 'Korean Fifty Won Coin',
    'Korean Five Hundred Won Coin', 'Korean One Hundred Won Coin', 'Korean Ten Won Coin',
    'Philippine Five Peso Coin', 'Philippine One Peso Coin', 'Philippine Ten Peso Coin',
    'Thailand Five Baht Coin', 'Thailand One Baht Coin', 'Thailand Ten Baht Coin'
]

# --- Init directories ---
os.makedirs(annotated_dir, exist_ok=True)
os.makedirs(unlabelled_dir, exist_ok=True)
os.makedirs(final_results_dir, exist_ok=True)

# --- YOLO model ---
yolo_model = YOLO(yolo_model_path)

# --- EfficientNet classifier ---
class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # create_model returns features when num_classes=0 and global_pool=''
        self.backbone = create_model("efficientnet_b2", pretrained=False, num_classes=0, global_pool='')
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1408, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

# --- Load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomEfficientNet(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(rcnn_model_path, map_location=device))
model.eval()

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- TTS ---
engine = pyttsx3.init()
engine.setProperty('rate', 180)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# --- Camera ---
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(
    main={"size": (1920, 1080), "format": "RGB888"},
    lores={"size": (640, 480)},
    display="main"
))
picam2.start()
time.sleep(1.5)

# --- Window Setup (Fullscreen) ---
WINDOW_NAME = "Detected Coins"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

MIN_CONFIDENCE = 0.50

# --- Global Flags (use globals to avoid nonlocal syntax error) ---
mouse_clicked = False
capture_pressed = False
exit_pressed = False

# --- Button Definitions (for touchscreen GUI) ---
# Coordinates assume 1920x1080 frame. Adjust if your display differs.
BUTTON_CAPTURE = ((50, 900), (420, 1040))   # left bottom big button
BUTTON_EXIT = ((1500, 900), (1870, 1040))   # right bottom big button

# --- Mouse/Touch Callback ---
def on_mouse(event, x, y, flags, param):
    global mouse_clicked, capture_pressed, exit_pressed
    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
        mouse_clicked = True
        # Check if tap is inside buttons
        if BUTTON_CAPTURE[0][0] <= x <= BUTTON_CAPTURE[1][0] and BUTTON_CAPTURE[0][1] <= y <= BUTTON_CAPTURE[1][1]:
            capture_pressed = True
        elif BUTTON_EXIT[0][0] <= x <= BUTTON_EXIT[1][0] and BUTTON_EXIT[0][1] <= y <= BUTTON_EXIT[1][1]:
            exit_pressed = True

cv2.setMouseCallback(WINDOW_NAME, on_mouse)

try:
    while True:
        # Autofocus
        picam2.set_controls({"AfMode": 1})
        time.sleep(0.2)
        picam2.set_controls({"AfTrigger": 0})
        time.sleep(1.0)

        # Capture frame
        frame = picam2.capture_array()
        if frame is None or frame.size == 0:
            print("❌ Failed to capture image.")
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = os.path.join(unlabelled_dir, f"unlabelled_{timestamp}.jpg")
        annotated_path = os.path.join(annotated_dir, f"result_{timestamp}.jpg")
        final_output_path = os.path.join(final_results_dir, f"final_{timestamp}.jpg")
        cv2.imwrite(raw_path, frame)

        results = yolo_model(frame)[0]
        counts = {}
        output_frame = frame.copy()

        if not results.boxes:
            print("🪙 No coins detected.")
            full_message = "No coin detected."
        else:
            # Iterate all detected boxes
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # skip low-confidence detection
                if conf < MIN_CONFIDENCE:
                    continue

                # crop safely (clamp coordinates)
                h, w = frame.shape[:2]
                x1c = max(0, min(w - 1, x1))
                x2c = max(0, min(w - 1, x2))
                y1c = max(0, min(h - 1, y1))
                y2c = max(0, min(h - 1, y2))

                coin_crop = frame[y1c:y2c, x1c:x2c]
                if coin_crop.size == 0:
                    continue

                # prepare for classifier
                image_rgb = cv2.cvtColor(coin_crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(image_rgb)
                inp = transform(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    out = model(inp)
                    pred = torch.argmax(out, dim=1).item()
                    confidence = torch.softmax(out, dim=1)[0][pred].item()

                if confidence < MIN_CONFIDENCE:
                    continue

                label = class_names[pred]
                counts[label] = counts.get(label, 0) + 1

                # Annotate bounding box and label on output_frame
                cv2.rectangle(output_frame, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)
                cv2.putText(output_frame, f"{label} ({confidence*100:.1f}%)",
                            (x1c, y1c - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)

            if counts:
                full_message = ". ".join([f"{count} {coin}" for coin, count in counts.items()])
            else:
                full_message = "No coin detected."

            print("\n🪙 Coin counts:")
            for coin_type, count in counts.items():
                print(f" - {count} {coin_type}")
            print(f"🗣 Speaking: {full_message}")

        # --- Draw On-Screen Buttons (large, visible for touch) ---
        # Draw semi-opaque background for buttons
        overlay = output_frame.copy()
        cv2.rectangle(overlay, BUTTON_CAPTURE[0], BUTTON_CAPTURE[1], (0, 200, 0), -1)
        cv2.rectangle(overlay, BUTTON_EXIT[0], BUTTON_EXIT[1], (0, 0, 200), -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0, output_frame)

        # Put labels
        cv2.putText(output_frame, "📸 Capture", (BUTTON_CAPTURE[0][0] + 30, BUTTON_CAPTURE[0][1] + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(output_frame, "❌ Exit", (BUTTON_EXIT[0][0] + 60, BUTTON_EXIT[0][1] + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)

        # Save and display results
        cv2.imwrite(annotated_path, output_frame)
        cv2.imwrite(final_output_path, output_frame)
        cv2.imshow(WINDOW_NAME, output_frame)
        cv2.waitKey(1)  # ensure GUI updates before TTS

        # Speak *after* showing the image
        if full_message:
            speak(full_message)

        print("🔁 Tap '📸 Capture' or press any key to take another photo. Tap '❌ Exit' or press ESC to quit.")

        # --- Wait for Input (Touch or Key) ---
        while True:
            key = cv2.waitKey(10)
            # ESC or Exit button pressed
            if key == 27 or exit_pressed:
                print("👋 Exiting.")
                raise KeyboardInterrupt
            # Any key OR touch anywhere OR capture button
            if key != -1 or mouse_clicked or capture_pressed:
                # reset flags for next iteration
                mouse_clicked = False
                capture_pressed = False
                exit_pressed = False
                break

        # Recreate fullscreen window after blocking input (prevents shrinking)
        cv2.destroyWindow(WINDOW_NAME)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)

except KeyboardInterrupt:
    pass

finally:
    GPIO.cleanup()  # Turn off LEDs and release GPIO pins
    picam2.stop()
    cv2.destroyAllWindows()
