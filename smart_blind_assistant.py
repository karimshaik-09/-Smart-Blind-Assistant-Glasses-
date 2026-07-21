import sys
import glob
import time
import cv2
import numpy as np
import argparse
from threading import Thread
from ultralytics import YOLO
from gtts import gTTS
from playsound import playsound
import tempfile
import shutil
# -------------------- Safe Non-Blocking TTS --------------------
def speak(text):
"""Speak text using gTTS safely without permission errors."""
def _play():
try:
tmp_dir = tempfile.mkdtemp()
file_path = os.path.join(tmp_dir, "tts_audio.mp3")
tts = gTTS(text=text, lang='en', slow=False, tld='co.in')
tts.save(file_path)
playsound(file_path)
shutil.rmtree(tmp_dir, ignore_errors=True)
except Exception as e:
print(f" Audio error: {e}")
Thread(target=_play, daemon=True).start()
# -------------------- Threaded Camera Stream --------------------
class CameraStream:
def __init__(self, src):
self.cap = cv2.VideoCapture(src)
if not self.cap.isOpened():
print(f" ERROR: Unable to open camera source: {src}")
sys.exit(0)
self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
self.grabbed, self.frame = self.cap.read()
self.stopped = False
Thread(target=self.update, daemon=True).start()
def update(self):
while not self.stopped:
if not self.cap.isOpened():
time.sleep(0.05)
continue
self.grabbed, self.frame = self.cap.read()
if not self.grabbed:
time.sleep(0.01)
def read(self):
return self.frame
def release(self):
    self.stopped = True
time.sleep(0.2)
self.cap.release()
# -------------------- Argument Parser --------------------
ESP32_STREAM_URL = "ipcam:http://10.240.255.132/capture"
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model')
parser.add_argument('--source', required=True, help='Input source (usb0, ipcam, video, etc.)')
parser.add_argument('--thresh', type=float, default=0.5, help='Confidence threshold')
parser.add_argument('--resolution', default=None, help='Display resolution WxH (optional)')
args = parser.parse_args()
# -------------------- Setup --------------------
model_path = args.model
img_source = args.source
conf_thresh = float(args.thresh)
user_res = args.resolution
if not os.path.exists(model_path):
print(' ERROR: YOLO model file not found.')
sys.exit(0)
print(" Loading YOLO model...")
model = YOLO(model_path, task='detect')
labels = model.names
# -------------------- Source --------------------
if 'usb' in img_source:
source_type = 'usb'
cam_index = int(img_source[3:])
cap = CameraStream(cam_index)
elif 'ipcam' in img_source:
source_type = 'ipcam'
ipcam_url = img_source.split(':', 1)[1].strip()
cap = CameraStream(ipcam_url)
elif os.path.isfile(img_source):
source_type = 'video'
cap = CameraStream(img_source)
else:
print(' Invalid input source.')
sys.exit(0)
resize = False
if user_res:
resW, resH = map(int, user_res.split('x'))
resize = True
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133),
(88,159,106), (96,202,231), (159,124,168), (169,162,241), (98,118,150)]
spoken_objects = {}
cooldown_time = 5
reset_time = 4
frame_rate_buffer = []
fps_avg_len = 100
avg_fps = 0
print(" Detection started. Press 'Q' to quit.\n")
# -------------------- Main Loop --------------------
while True:
t_start = time.perf_counter()
frame = cap.read()
if frame is None:
continue
if resize:
frame = cv2.resize(frame, (resW, resH))
height, width = frame.shape[:2]
results = model(frame, verbose=False)
detections = results[0].boxes
current_detected = set()
obj_count = 0
if detections is not None and len(detections) > 0:
for det in detections:
xyxy = det.xyxy.cpu().numpy().squeeze()
if xyxy.ndim != 1 or len(xyxy) != 4:
continue
xmin, ymin, xmax, ymax = xyxy.astype(int)
cls_id = int(det.cls.item())
conf = det.conf.item()
if conf < conf_thresh:
continue
classname = labels.get(cls_id, 'object')
current_detected.add(classname)
# Position Logic
x_center = (xmin + xmax) / 2
if x_center < width / 3:
position = "on the left side"
elif x_center > (2 * width / 3):
position = "on the right side"
else:
position = "ahead"
# Activity for Person
if classname.lower() == "person":
box_height = ymax - ymin
if box_height > height * 0.6:
action = "standing"
elif box_height > height * 0.4:
action = "sitting"
else:
action = "far away"
sentence = f"Person is {action} {position}"
else:
sentence = f"{classname} {position}"
color = bbox_colors[cls_id % len(bbox_colors)]
label = f"{classname}: {conf*100:.0f}%"
cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
cv2.putText(frame, label, (xmin, max(15, ymin-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
obj_count += 1
# Speak logic
now = time.time()
if classname not in spoken_objects or (now - spoken_objects[classname]) > cooldown_time:
print(f" {sentence}")
speak(sentence)
spoken_objects[classname] = now
# FPS & Display
cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 20),
cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
cv2.putText(frame, f"Objects: {obj_count}", (10, 45),
cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
cv2.imshow("YOLO Smart Audio Detection", frame)
key = cv2.waitKey(1)
if key in [ord('q'), ord('Q')]:
break
# FPS Calculation
t_end = time.perf_counter()
fps = 1 / (t_end - t_start)
frame_rate_buffer.append(fps)
if len(frame_rate_buffer) > fps_avg_len:
frame_rate_buffer.pop(0)
avg_fps = np.mean(frame_rate_buffer)
print(f" Average FPS: {avg_fps:.2f}")
cap.release()
cv2.destroyAllWindows()