import cv2, datetime, time, os, threading
import numpy as np

from pathlib import Path
from dataclasses import dataclass
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

_cameras: dict[str, cv2.VideoCapture] = {}
_locks: dict[str, threading.Lock] = {
    "WEBCAM1": threading.Lock(),
    "WEBCAM2": threading.Lock(),
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

def to_portrait_wide(frame, width_factor=1.75):
    """width_factor 1.0=360x640, 1.75=630x640 추천, 2.0=720x640 더 와이드 - 비율 유지, 뚱뚱해지지 않음"""
    h, w = frame.shape[:2]
    base_w = int(h * 9 / 16)
    target_w = int(base_w * width_factor)
    target_w = min(target_w, w - 10)
    x1 = (w - target_w)//2
    cropped = frame[:, x1:x1+target_w]
    out_w = int(target_w * 640 / h)
    return cv2.resize(cropped, (out_w, 640))

def to_square(frame):
    h, w = frame.shape[:2]
    size = min(h, w)
    x1 = (w - size)//2
    y1 = (h - size)//2
    square = frame[y1:y1+size, x1:x1+size]
    return cv2.resize(square, (640, 640))


def open_camera_with_fallback(index: int):
    backends = [(cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "ANY"), (cv2.CAP_DSHOW, "DSHOW")]
    for backend, name in backends:
        print(f"Trying {name} for index {index}...")
        cam = cv2.VideoCapture(index, backend)
        if not cam.isOpened():
            cam.release()
            continue
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cam.set(cv2.CAP_PROP_FPS, 30)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, _ = cam.read()
        if ret:
            print(f"Success with {name} on index {index}")
            return cam
        cam.release()
    return None

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

def mjpeg_generator(name: str):
    lock = _locks[name]
    while True:
        with lock:
            cam = get_camera(name)
            if cam is None or not cam.isOpened():
                break
            ret, frame = cam.read()
        if not ret:
            time.sleep(0.05)
            continue
        # Individual stream also portrait wide
        frame = to_portrait_wide(frame, width_factor=1.75)
        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret: continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.03)

def dummy_mjpeg_generator(name: str):
    """Create a dummy MJPEG stream with a placeholder image_frame for the given camera name."""
    lock = _locks[name]
    while True:
        with lock:
            # Create a dummy frame for the given camera name
            f = np.zeros((720, 640, 3), dtype=np.uint8)
            cv2.putText(f, f"NO SIGNAL: {name}", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            frame = f
        # Individual stream also portrait wide
        frame = to_portrait_wide(frame, width_factor=1.75)
        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret: continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.03)
        
        
recording_state = {"active": False, "writer": None}

# --- NEW: Portrait frame adjust (same as Tapo IP version) ---
def get_synced_split_frame(mode="portrait_wide"):
    cams = [get_camera("WEBCAM1"), get_camera("WEBCAM2")]
    frames = []
    
    for cam in cams:
        f = None
        if cam is not None and cam.isOpened():
            ret, f = cam.read()
            if not ret or f is None:
                f = None
        
        # 카메라 하나가 없거나 읽기 실패하면 더미 video 프레임 생성
        if f is None:
            f = np.zeros((720, 640, 3), dtype=np.uint8)
            cv2.putText(f, "NO SIGNAL", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        # 여기서부터는 항상 같은 함수로 변환 -> 높이가 항상 같아짐
        if mode == "portrait_wide":
            f = to_portrait_wide(f, width_factor=1.75) # -> 630x640
        elif mode == "square":
            f = to_square(f) # -> 640x640
        else:
            f = cv2.resize(f, (640, 360))

        frames.append(f)
    
    combined = np.hstack(frames)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(combined, ts, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(combined, "WEBCAM1", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(combined, "WEBCAM2", (frames[0].shape[1] + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    if recording_state["active"]:
        cv2.circle(combined, (combined.shape[1]-20, 20), 8, (0,0,255), -1)
    return combined

def recording_loop():
    print("Recording started")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    filepath = OUTPUT_DIR / f"dual_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    writer = None
    while recording_state["active"]:
        combined = get_synced_split_frame(mode="portrait_wide")
        if writer is None:
            h, w = combined.shape[:2]
            print(f"Writer {w}x{h}")
            writer = cv2.VideoWriter(str(filepath), fourcc, 20.0, (w, h))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(filepath), fourcc, 20.0, (w, h))
        writer.write(combined)
        time.sleep(0.05)
    if writer:
        writer.release()
        print(f"Saved: {filepath}")


    while True:
        frame = get_synced_split_frame(mode="portrait_wide")
        _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.04)
        
def synced_mjpeg_generator():
    while True:
        frame = get_synced_split_frame(mode="portrait_wide")
        _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.04)
        
def _mjpeg_frames(url: str):
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (640, 360))
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
    finally:
        cap.release()  
        
