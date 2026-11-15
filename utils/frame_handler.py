import cv2, datetime, time, os, threading
import numpy as np
from utils.functions import get_usb_camera, get_camera, open_camera_with_fallback
from utils.functions import OUTPUT_DIR, CAMERA_CONFIG
from pathlib import Path
from dataclasses import dataclass

print(f"OUTPUT_DIR:{OUTPUT_DIR}")
_cameras: dict[str, cv2.VideoCapture] = {}
_locks: dict[str, threading.Lock] = {
    "WEBCAM1": threading.Lock(),
    "WEBCAM2": threading.Lock(),
}

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
    """record the combined frames"""
    print("Recording started")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    filepath = OUTPUT_DIR / f"dual_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    writer = None
    while recording_state["active"]:
        frame = get_synced_split_frame(mode="portrait_wide")
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(filepath), fourcc, 20.0, (w, h))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(filepath), fourcc, 20.0, (w, h))
        writer.write(frame)
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
        
def _webcam_mjpeg():
    """Generator for MJPEG frames from USB webcam"""
    failures = 0
    while True:
        ret, frame = get_usb_camera().read()
        if not ret:
            failures += 1
            if failures > 50:
                break
            continue
        failures = 0
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        