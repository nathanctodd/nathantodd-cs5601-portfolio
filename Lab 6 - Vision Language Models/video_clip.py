import cv2
import base64
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# ===============================
# CONFIG
# ===============================
VIDEO_PATH = "./video.mov"
SAMPLE_EVERY_SECONDS = 1
MODEL_NAME = "llava"

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
    print("LLaVA raw answer:", answer)

    return "yes" in answer


# ===============================
# Load and Sample Video Frames
# ===============================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video file: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

interval = int(fps * SAMPLE_EVERY_SECONDS)

frames = []
timestamps = []

frame_num = 0

print("Extracting frames...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_num % interval == 0:
        frames.append(frame)
        timestamps.append(frame_num / fps)

    frame_num += 1

cap.release()

print(f"Extracted {len(frames)} frames for analysis.")

# ===============================
# Analyze Frames with LLaVA
# ===============================
person_present = False
entry_time = None

print("Starting person detection...")

for frame, timestamp in zip(frames, timestamps):
    print(f"\nChecking frame at {timestamp:.2f}s")

    detected = person_in_frame(frame)

    if detected and not person_present:
        person_present = True
        entry_time = timestamp
        print(f"Person ENTERED at {timestamp:.2f} seconds")

    elif not detected and person_present:
        person_present = False
        print(f"Person EXITED at {timestamp:.2f} seconds")

# If video ends while person still present
if person_present:
    final_time = timestamps[-1] if timestamps else 0
    print(f"Person EXITED at {final_time:.2f} seconds (video ended)")

print("Done.") 