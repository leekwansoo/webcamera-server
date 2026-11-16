import os, cv2, time, datetime, threading, random, json, asyncio
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from contextlib import asynccontextmanager
from utils.functions import CAMERA_RTSP, STATE
from utils.frame_handler import CAMERA_CONFIG, OUTPUT_DIR
from utils.frame_handler import to_portrait_wide, to_square
from utils.frame_handler import get_synced_split_frame, _mjpeg_frames, open_camera_with_fallback
from utils.frame_handler import get_camera, synced_mjpeg_generator, dummy_mjpeg_generator
# ====== 설정 ======
CAMERA_RTSP = CAMERA_RTSP
print(CAMERA_RTSP)
# WiFi AP 환경이라 끊김 대비
# RECONNECT_DELAY = 2.0
OUTPUT_DIR = Path("recordings")
OUTPUT_DIR.mkdir(exist_ok=True)

if os.environ.get("OPENCV_LOG_LEVEL", "").upper() == "ERROR":
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

_cameras: dict[str, cv2.VideoCapture] = {}
_locks: dict[str, threading.Lock] = {
    "WEBCAM1": threading.Lock(),
    "WEBCAM2": threading.Lock(),
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting...")
    yield
    print("Server shutting down...")
    for name, cam in _cameras.items():
        try: cam.release()
        except: pass
    _cameras.clear()

app = FastAPI(lifespan=lifespan)

templates = Jinja2Templates(directory="templates")

# ====== 녹화 상태 ======
# ====== 녹화 상태 ======
# ✅ Only ONE recording_state in the whole file - this one
recording_state = {"active": False, "state": "passive", "last_file": None}
_recording_thread = None
       
class CameraReader:
    def __init__(self, url, cam_id, mode="square", start_delay=0):
        self.start_delay = start_delay
        self.url = url
        self.cam_id = cam_id
        self.mode = mode
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        # Stagger startup so cameras don't connect at the same instant
        time.sleep(self.start_delay)
        delay = 2
        while self.running:
            print(f"[CONNECT] {self.cam_id} -> {self.url}")
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            
            if not cap.isOpened():
                cap.release()
                print(f"[FAILED OPEN] {self.cam_id} - retry in {delay}s")
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            delay = 2

            fails = 0
            while self.running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    fails += 1
                    if fails > 100:  # ~5s without frames: reconnect
                        print(f"[RECONNECT] {self.cam_id}")
                        break
                    time.sleep(0.05)
                    continue
                fails = 0
                
                # Convert to square for golf full-body
                if self.mode == "square":
                    h, w = frame.shape[:2]
                    size = min(h, w)
                    x1 = (w - size)//2
                    y1 = (h - size)//2
                    frame = frame[y1:y1+size, x1:x1+size]
                    frame = cv2.resize(frame, (640, 640))
                elif self.mode == "portrait":
                    h, w = frame.shape[:2]
                    target_w = int(h * 9/16)
                    x1 = (w - target_w)//2
                    frame = frame[:, x1:x1+target_w]
                    frame = cv2.resize(frame, (360, 640))
                else:
                    frame = cv2.resize(frame, (640, 360))

                with self.lock:
                    self.frame = frame
            
            cap.release()

    def get_jpeg(self):
        with self.lock:
            if self.frame is None:
                return None
            _, buf = cv2.imencode(".jpg", self.frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return buf.tobytes()


    def to_square(self, frame):
        # 1920x1080 -> center 1080x1080 -> 640x640
        h, w = frame.shape[:2]
        size = min(h, w)
        x1 = (w - size) // 2
        y1 = (h - size) // 2
        square = frame[y1:y1+size, x1:x1+size]
        return cv2.resize(square, (640, 640))

    def to_portrait(self, frame, width_factor=1.0):
        # width_factor = 1.0 -> 360x640 (좁음)
        # width_factor = 1.5 -> 540x640 (이전)
        # width_factor = 1.75 -> 630x640 (추천 - 와이드하면서 비율 정상)
        # width_factor = 2.0 -> 720x640 (더 와이드)
        h, w = frame.shape[:2] # 1080, 1920
        base_w = int(h * 9 / 16) # 607px가 기본 portrait 너비
        target_w = int(base_w * width_factor)
        # 원본 1920보다 넓게 잘라지지 않게 제한
        target_w = min(target_w, w - 20)
        x1 = (w - target_w)//2
        cropped = frame[:, x1:x1+target_w]
        # 비율 유지해서 리사이즈 - 찌그러짐 없음
        # target_w:1080 = out_w:640  비율 그대로
        out_w = int(target_w * 640 / h)
        return cv2.resize(cropped, (out_w, 640))

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
        
# Usage for your golf setup:
managers = {
    "Camera1": CameraReader(CAMERA_RTSP["Camera1"], "Camera1", mode="square"),
    "Camera2": CameraReader(CAMERA_RTSP["Camera2"], "Camera2", mode="square", start_delay=3),
}
print(managers)

print(managers)

# ====== ✅ Part 1: IPCombinedSource Class Definition - ADD THIS ======
class IPCombinedSource:
    """Tapo version of CombinedFrameSource - one source, many consumers"""
    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._ready = threading.Event()

    def start(self):
        if self._running:
            return
        self._running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        print("[IP-COMBINED] Source started")

    def _capture_loop(self):
        print("[IP-COMBINED] _capture_loop STARTED")
        while self._running:
            combined = get_combined_portrait()
            if combined is None:
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = combined.copy()
            time.sleep(0.033)

    def get_frame(self):
        if not self._running:
            print("[IP-COMBINED] AUTO STARTING")
            self.start()
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def wait_until_ready_sync(self, timeout=15):
        if not self._running:
            self.start()
        for _ in range(30):
            if self._frame is not None:
                self._ready.set()
                return True
            time.sleep(0.5)
        self._ready.set()
        return False
# ====== End of Class Definition ======

# ====== ✅ Part 2: Creation - KEEP THIS ONE LINE ======
ip_combined_source = IPCombinedSource()
print(f"[IP-COMBINED] Created: {ip_combined_source}")

# get_combined에서:
def get_combined_portrait():
    # 각 카메라에서 50% 더 넓게
    # 간단하게는 CameraReader 안에서 width_factor를 1.75로 고정:
    f1 = managers["Camera1"].frame # 이제 630x640
    f2 = managers["Camera2"].frame # 630x640
    if f1 is None or f2 is None:
        return None
    # 630+630 = 1260x640 = 16:9 와이드 모니터에 꽉 차면서 사람 비율 정상
    combined = cv2.hconcat([f1, f2])
    return combined

print(managers)
# 카메라 리더 2개 생성
cameras = list(managers.values())

        
def get_synced_split_frame():
    # 두 카메라 최신 프레임 가져와서 좌우 분할
    frames = []
    for cam in cameras:
        f = cam.get_frame()
        if f is None:
            f = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(f, f"CAM {cam.cam_id} NO SIGNAL", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        frames.append(cv2.resize(f, (640, 360)))
    
    # 좌우 합치기
    combined = np.hstack(frames)
    # 상단에 타임스탬프
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(combined, ts, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    if recording_state["active"]:
        cv2.circle(combined, (combined.shape[1]-30, 30), 10, (0,0,255), -1)
    return combined

def recording_loop_ipcamera():
    print("[REC] === IP Recording started ===")
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if not ip_combined_source._running:
            ip_combined_source.start()
        ip_combined_source.wait_until_ready_sync(timeout=15)
        recording_state["state"] = "active"  # ✅ Polling waits for THIS
        print("[REC] state -> active")

        filepath = OUTPUT_DIR / f"dual_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        print(f"[REC] target: {filepath}")

        writer = None
        frames_written = 0
        start_time = time.time()
        TARGET_FPS = 15.0  # ✅ Tapo TC71 is 14-17 fps real - don't use 30, it will look slow
        frame_interval = 1.0 / TARGET_FPS
        next_frame_time = time.time()

        while recording_state["active"]:
            frame = ip_combined_source.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            if writer is None:
                h, w = frame.shape[:2]
                print(f"[REC] Writer {w}x{h} @ {TARGET_FPS}fps -> {filepath}")
                # ✅ Try avc1 first, mp4v second
                writer = None
                for fourcc_str in ['avc1', 'mp4v']:
                    test_writer = cv2.VideoWriter(str(filepath), cv2.VideoWriter_fourcc(*fourcc_str), TARGET_FPS, (w, h))
                    print(f"[REC] Trying {fourcc_str} isOpened={test_writer.isOpened()}")
                    if test_writer.isOpened():
                        writer = test_writer
                        print(f"[REC] Writer OPENED with {fourcc_str}")
                        break
                    test_writer.release()
                # ✅ If BOTH failed, stop recording - don't try to write
                if writer is None:
                    print("[REC] ERROR: All codecs failed! Check OpenH264 DLL for avc1.")
                    print("[REC] Put openh264-2.5.0-win64.dll in same folder as ipcamera-server.py")
                    recording_state["active"] = False
                    break  # ✅ This break is for WHILE loop, exits recording

            # ✅ Safety - never call writer.write if writer is None
            if writer is None:
                print("[REC] writer is None, skipping frame")
                time.sleep(0.01)
                continue

            now = time.time()
            if now < next_frame_time:
                time.sleep(next_frame_time - now)
                continue
            writer.write(frame)
            frames_written += 1
            if frames_written % 75 == 0:
                print(f"[REC] frames={frames_written} elapsed={time.time()-start_time:.1f}s")
            next_frame_time += frame_interval
            if time.time() > next_frame_time + 0.2:
                next_frame_time = time.time()

        print(f"[REC] Loop ended frames={frames_written}")
    except Exception as e:
        print(f"[REC] EXCEPTION: {e}")
        import traceback; traceback.print_exc()
    finally:
        if writer is not None:
            writer.release()
        if filepath.exists():
            print(f"[REC] ✅ SAVED: {filepath} size={filepath.stat().st_size} bytes")
            recording_state["last_file"] = str(filepath)
        else:
            print(f"[REC] ❌ FILE NOT FOUND: {filepath}")
        recording_state["state"] = "passive"
        recording_state["active"] = False
        print("[REC] === IP Recording ended ===")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "ipcamera_index.html")

def mjpeg_generator():
    while True:
        frame = get_synced_split_frame()
        _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.04)

def mjpeg_gen_combined():
    while True:
        combined = ip_combined_source.get_frame()  # ✅ Not get_combined_portrait() directly
        if combined is None:
            time.sleep(0.05)
            continue
        cv2.putText(combined, "FRONT", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(combined, "SIDE", (660, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        _, jpeg = cv2.imencode('.jpg', combined, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
        time.sleep(0.04)

@app.get("/api/combined/stream")
def combined_stream():
    return StreamingResponse(mjpeg_gen_combined(), media_type="multipart/x-mixed-replace; boundary=frame")  
         
@app.get("/ipcamera", response_class=HTMLResponse)
async def root(request: Request):
    """Main dashboard - Integrated Control Center"""
    return templates.TemplateResponse(request, "ipcamera-viewer.html", {"now": datetime.datetime.now()})

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/camera/{camera_id}/stream")
async def api_camera_stream(camera_id: str):
    """[UPLINK] RTSP -> MJPEG stream for browser <img>"""
    url = CAMERA_RTSP.get(camera_id)
    print(url)
    if not url:
        return JSONResponse({"error": f"Set CAMERA1_RTSP/CAMERA2_RTSP env for {camera_id}"}, status_code=404)
    try:
        import cv2  # noqa: F401
    except ImportError:
        return JSONResponse({"error": "opencv-python not installed"}, status_code=500)
    return StreamingResponse(_mjpeg_frames(url), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/camera/start_recording")
async def start_recording():
    global _recording_thread
    print("recording_state:", recording_state)
    if recording_state["active"]:
        return {"status": "already recording", "state": "active"}
    recording_state["active"] = True
    recording_state["state"] = "preparing"
    _recording_thread = threading.Thread(target=recording_loop_ipcamera, daemon=True)
    _recording_thread.start()
    return {"status": "recording", "state": "preparing"}  # ✅ JS needs state field

@app.post("/camera/stop_recording")
async def stop_recording():
    global _recording_thread
    print("[REC] stop requested")
    recording_state["active"] = False
    if _recording_thread is not None:
        _recording_thread.join(timeout=3)
    return {"status": "stopped", "state": "passive", "file_path": recording_state.get("last_file")}

@app.get("/camera/recording_state")
async def get_recording_state():
    return {"state": recording_state["state"], "active": recording_state["active"], "file_path": recording_state.get("last_file")}

@app.get("/api/status")
async def api_all_status():
    """[UPLINK] Terminal -> Server : Get all devices status (Both-way demo)"""
    # Simulate temp drift
    STATE["Temp1"]["temperature"] = round(22.5 + random.random()*3.5, 1)
    STATE["Temp1"]["humidity"] = random.randint(52, 68)
    
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "direction": "Terminal -> Server (UPLINK)",
        "devices": {
            "Temp1": {
                "temperature_c": STATE["Temp1"]["temperature"],
                "humidity": STATE["Temp1"]["humidity"],
                "battery": 87
            },
            "Door1": {"status": STATE["Door1"]},
            "Door2": {"status": STATE["Door2"]},

            "Switch1": {
                **STATE["Switch1"],
                "current_power_w": random.randint(800,1500) if STATE["Switch1"]["state"]=="ON" else 0
            },
            "Switch2": {
                **STATE["Switch2"],
                "current_power_w": random.randint(300,800) if STATE["Switch2"]["state"]=="ON" else 0
            },
            "Camera1": {"online": True, "motion": random.choice([True, False]), "recording": True},
            "Camera2": {"online": True, "motion": random.choice([True, False]), "recording": True},
        }
    }

# WebSocket for real-time Both-way push

  
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ipcamera-server:app", host="0.0.0.0", port=8000)
