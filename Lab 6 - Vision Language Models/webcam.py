import cv2
import time
import base64
import numpy as np
import matplotlib.pyplot as plt
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# ===============================
# CONFIG
# ===============================
VIDEO_PATH = None  # e.g. "clip.mp4"
USE_WEBCAM = True
SAMPLE_EVERY_SECONDS = 8
MODEL_NAME = "llava"

if not USE_WEBCAM and not VIDEO_PATH:
    raise ValueError("You must either set USE_WEBCAM = True or provide a valid VIDEO_PATH.")

# ===============================
# Load Model
# ===============================
llm = ChatOllama(model=MODEL_NAME, temperature=0)

# ===============================
# Helper: Ask LLaVA
# ===============================
def person_in_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    image_b64 = base64.b64encode(buffer).decode("utf-8")

    message = HumanMessage(
        content=[
            {"type": "text", "text": "Is there a person in this scene? Answer yes or no only."},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"}
        ]
    )

    response = llm.invoke([message])
    answer = response.content.lower().strip()
    print("LLaVA answer:", answer)

    return "yes" in answer, answer

#list devices on computer
def list_video_devices():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr
print("Available video devices:", list_video_devices())

# ===============================
# Video Setup
# ===============================
if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Error opening video source")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

frame_interval = int(fps * SAMPLE_EVERY_SECONDS)

# ===============================
# GUI Setup (Matplotlib)
# ===============================
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")

# ===============================
# Detection State Tracking
# ===============================
person_present = False
entry_time = None
frame_count = 0
last_sampled_frame = None
last_answer = "Waiting..."

print("Starting detection...")


# ===============================
# Main Loop
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame_count % frame_interval == 0:
        timestamp = frame_count / fps
        print(f"\nChecking frame at {timestamp:.2f}s")

        detected, answer_text = person_in_frame(frame)
        last_sampled_frame = frame_rgb.copy()
        last_answer = f"Time: {timestamp:.2f}s | LLaVA: {answer_text}"

        if detected and not person_present:
            person_present = True
            entry_time = timestamp
            print(f"Person ENTERED at {timestamp:.2f}s")

        elif not detected and person_present:
            person_present = False
            print(f"Person EXITED at {timestamp:.2f}s")

    # Draw GUI frame
    display_frame = frame_rgb.copy()

    if last_sampled_frame is not None:
        h, w, _ = display_frame.shape
        thumb_h = int(h * 0.25)
        thumb_w = int(w * 0.25)
        thumb = cv2.resize(last_sampled_frame, (thumb_w, thumb_h))

        display_frame[h - thumb_h - 10:h - 10, w - thumb_w - 10:w - 10] = thumb

    ax.clear()
    ax.imshow(display_frame)
    ax.set_title(last_answer, fontsize=12)
    ax.axis("off")
    plt.pause(0.001)

    frame_count += 1

cap.release()

if person_present:
    final_time = frame_count / fps
    print(f"Person EXITED at {final_time:.2f}s (video ended)")

print("Done.")