# main functions for fastapi server
from dataclasses import dataclass
from pathlib import Path
from usbx import usb
import cv2, time, threading, os, random, datetime, json
import numpy as np
from dotenv import load_dotenv
load_dotenv()

CAMERA_RTSP = {
    "Camera1": os.getenv("CAMERA1_RTSP"),
    "Camera2": os.getenv("CAMERA2_RTSP"),
}

CAMERAS_USB = [
    {"type": "usb", "source": 0, "name": "LEFT"},
    {"type": "usb", "source": 1, "name": "RIGHT"}
]

CAMERA_CONFIG = {"WEBCAM1": 0, "WEBCAM2": 1}
OUTPUT_DIR = Path("recordings")
OUTPUT_DIR.mkdir(exist_ok=True)
from utils.usb_device import list_usb_devices


BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

STATE = {
    "Camera1": {"online": True, "motion": False},
    "Camera2": {"online": True, "motion": False},
    "Webcam1": {"online": True, "motion": False},
    "Webcam2": {"online": True, "motion": False},
}

_camera_open_locks = {name: threading.Lock() for name in CAMERA_CONFIG}
_camera_discovery_lock = threading.Lock()
CAMERA_SEARCH_TIMEOUT = 30.0
CAMERA_RETRY_INTERVAL = 5.0

@dataclass
class CameraSearch:
    deadline: float
    next_attempt: float
    stopped: bool = False

_camera_searches: dict[str, CameraSearch] = {}

_cameras: dict[str, cv2.VideoCapture] = {}
# Open your USB webcam
def get_usb_camera_index():
    usb_devices = list_usb_devices()
    if "WEBCAM" in usb_devices:
        return usb_devices.index("WEBCAM")
    else:
        return 0

def _open_camera():
    indices = [get_usb_camera_index()] + [i for i in range(4)]
    for idx in dict.fromkeys(indices):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                return cap
            cap.release()
    return cv2.VideoCapture(indices[0])

# camera = _open_camera()

def open_camera_with_fallback(index: int, deadline: float | None = None):
    # DSHOW를 먼저 시도해야 MSMF 에러를 피함
    backends = [
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_MSMF, "MSMF"),
        (cv2.CAP_ANY, "ANY"),
    ]
    for backend, name in backends:
        if deadline is not None and time.monotonic() >= deadline:
            break
        print(f"Trying {name} for index {index}...")
        cam = cv2.VideoCapture(index, backend)
        if not cam.isOpened():
            cam.release()
            continue
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cam.set(cv2.CAP_PROP_FPS, 30)
        ret, _ = cam.read()
        if ret:
            print(f"Success with {name} on index {index}")
            return cam
        cam.release()
    return None

def get_camera(name: str):
    with _camera_open_locks[name]:
        cam = _cameras.get(name)
        if cam is not None and cam.isOpened():
            return cam

        now = time.monotonic()
        search = _camera_searches.get(name)
        if search is None:
            search = CameraSearch(now + CAMERA_SEARCH_TIMEOUT, now)
            _camera_searches[name] = search
        if search.stopped:
            return None
        if now >= search.deadline:
            search.stopped = True
            print(f"{name} unavailable after {CAMERA_SEARCH_TIMEOUT:g} seconds; restart server to retry.")
            return None
        if now < search.next_attempt:
            return None

        with _camera_discovery_lock:
            if not hasattr(get_camera, "available"):
                get_camera.available = list_available_cameras()
                print(f"Available cameras: {get_camera.available}")

        if time.monotonic() >= search.deadline:
            search.stopped = True
            print(f"{name} unavailable after {CAMERA_SEARCH_TIMEOUT:g} seconds; restart server to retry.")
            return None

        idx = CAMERA_CONFIG[name]
        if idx not in get_camera.available and get_camera.available:
            pos = 0 if name=="WEBCAM1" else 1
            if pos < len(get_camera.available):
                idx = get_camera.available[pos]
                print(f"{name} remapped {CAMERA_CONFIG[name]} -> {idx}")

        cam = open_camera_with_fallback(idx, deadline=search.deadline)
        if cam is not None:
            _cameras[name] = cam
            del _camera_searches[name]
        else:
            search.next_attempt = time.monotonic() + CAMERA_RETRY_INTERVAL
        return cam
def get_usb_camera():
    """Return the shared camera (a second VideoCapture on the same device fails on Windows)."""
    global camera
    if not camera.isOpened():
        camera.release()
        camera = _open_camera()
    return camera

def list_available_cameras(max_test=5):
    available=[]
    for i in range(max_test):
        cap=cv2.VideoCapture(i, cv2.CAP_DSHOW) # DSHOW로 탐색해야 정확
        if cap.isOpened():
            ret,_=cap.read()
            if ret:
                print(f"[FOUND] Camera index {i} available")
                available.append(i)
            cap.release()
    return available

def play_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()   
    return True


        


            



        








