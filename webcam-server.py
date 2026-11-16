import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2, datetime, time, threading, asyncio
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from contextlib import asynccontextmanager
from utils.functions import CAMERA_CONFIG, OUTPUT_DIR, get_camera, open_camera_with_fallback
from utils.frame_handler import synced_mjpeg_generator, mjpeg_generator, dummy_mjpeg_generator, get_synced_split_frame, recording_loop


_cameras = {name: get_camera(name) for name in CAMERA_CONFIG}
print(_cameras)

class CombinedFrameSource:
    def __init__(self, mode="portrait_wide"):
        self.mode = mode
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._ready = threading.Event()

    def start(self):
        if self._running:
            return
        self._running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        print("[COMBINED] Source started - _capture_loop should start now")

    def _capture_loop(self):
        print("[COMBINED] _capture_loop STARTED - now making frames")
        loop_count = 0
        while self._running:
            combined = get_synced_split_frame(mode=self.mode)
            loop_count += 1
            if combined is None:
                time.sleep(0.05)
                continue

            with self._lock:
                self._frame = combined.copy()

            if loop_count == 1 or not self._ready.is_set():
                print(f"[COMBINED] first frame -> {combined.shape}")
                self._ready.set()  # <-- AFTER frame exists, not before

            time.sleep(0.033)

    def get_frame(self):
        # ✅ Auto-start - if no one started me, I start myself
        if not self._running:
            print("[COMBINED] get_frame() called but not running - AUTO STARTING")
            self.start()
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def wait_until_ready_sync(self, timeout=15):
        # ✅ Recording is a sync thread, use this - not async version
        if not self._running:
            print("[COMBINED] wait_until_ready_sync() - AUTO STARTING")
            self.start()
        print(f"[COMBINED] wait_until_ready_sync() waiting... _running={self._running}")
        ok = self._ready.wait(timeout=timeout)
        print(f"[COMBINED] wait done ok={ok} _frame exists={self._frame is not None}")
        return ok

   
# 전역으로 하나만 생성
combined_source = CombinedFrameSource(mode="portrait_wide")

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

# --- NEW: Portrait frame adjust (same as Tapo IP version) ---

# --- Camera open (same as yours) ---


# 1. Viewer - combined source에서만 읽음
def generate_frames():
    print("[VIDEO_FEED] waiting for cameras...")
    # ✅ 카메라가 준비될 때까지 기다림 - 바로 yield 안 함
    if not combined_source.wait_until_ready_sync(timeout=10):
        print("[VIDEO_FEED] cameras not ready, still trying...")

    print("[VIDEO_FEED] cameras ready, streaming...")
    while True:
        # frame = combined_source.synced_get_frame()
        frame = get_synced_split_frame(mode="portrait_wide")
        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.04)
        
# 2. Recording - combined source에서만 읽음
def webcam_recording_loop():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / f"dual_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    writer = None
    frames_written = 0
    start_time = time.time()
    TARGET_FPS = 30.0  # ✅ Source is 30fps, writer must be 30fps
    frame_interval = 1.0 / TARGET_FPS
    next_frame_time = time.time()

    while recording_state["active"]:
        frame = combined_source.get_frame()
        if frame is None:
            time.sleep(0.005)
            continue

        if writer is None:
            h, w = frame.shape[:2]
            print(f"[REC] Writer {w}x{h} @ {TARGET_FPS}fps -> {filepath}")
            for fourcc_str in ['avc1', 'mp4v']:
                writer = cv2.VideoWriter(str(filepath), cv2.VideoWriter_fourcc(*fourcc_str), TARGET_FPS, (w, h))
                if writer.isOpened():
                    break
            recording_state["state"] = "active"  # <-- THIS is what frontend waits for
            print(f"[REC] state -> active", flush=True)

        # ✅ Wait until it's time for next frame - not fixed sleep
        now = time.time()
        if now < next_frame_time:
            time.sleep(next_frame_time - now)
            continue

        writer.write(frame)
        frames_written += 1
        next_frame_time += frame_interval

        # If we fell behind, don't accumulate lag - reset
        if time.time() > next_frame_time + 0.2:
            next_frame_time = time.time()

    if writer:
        writer.release()
    elapsed = time.time() - start_time
    real_fps = frames_written / elapsed if elapsed > 0 else 0
    print(f"[REC] ✅ SAVED: {filepath} frames={frames_written} elapsed={elapsed:.1f}s real_fps={real_fps:.1f} writer_fps={TARGET_FPS}")

# 3. Skeleton - 녹화 파일이 아니라 실시간 combined source에서도 가능
def generate_live_skeleton_frames():
    # 실시간 combined frame에 스켈레톤을 그려서 스트리밍
    while True:
        frame = combined_source.synced_get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        # ... draw_skeleton(frame, ...) ...
        # ... yield jpg ...
        time.sleep(0.033)


              
# FastAPI routes for webcam server
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "webcam_index.html")

@app.get("/webcam", response_class=HTMLResponse)
async def webcam(request: Request):
    return templates.TemplateResponse(request, "webcam-viewer.html", {"now": datetime.datetime.now()})

@app.get("/api/webcam/{cam_name}/stream")
async def api_named_stream(cam_name: str):
    cam_name = cam_name.upper()
    if cam_name not in CAMERA_CONFIG:
        #return JSONResponse({"error": "Use WEBCAM1 or WEBCAM2"}, status_code=404)
        return StreamingResponse(dummy_mjpeg_generator(cam_name), media_type="multipart/x-mixed-replace; boundary=frame")
    cam = get_camera(cam_name)
    if cam is None or not cam.isOpened():
        #return JSONResponse({"error": f"{cam_name} failed"}, status_code=404)
        return StreamingResponse(dummy_mjpeg_generator(cam_name), media_type="multipart/x-mixed-replace; boundary=frame")
    return StreamingResponse(mjpeg_generator(cam_name), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_feed")
async def video_feed():
    """feeding video stream from the synced cameras in portrait_wide mode"""
    print("Webcam video feed requested")
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# Keep your existing dict, just add "state"
# passive = idle, preparing = warming up 14-15s, active = really writing file
recording_state = {"active": False, "state": "passive"}

@app.post("/webcam/start_recording")
async def start_recording():
    if not recording_state["active"]:
        recording_state["active"] = True
        recording_state["state"] = "preparing"

        # ✅ THIS BLOCK GOES HERE - in start_recording
        if combined_source._frame is not None and combined_source._running:
            print("[REC] frame already exists, skip wait")
            # start recording immediately, don't call wait_until_ready_sync
            pass
        else:
            combined_source.wait_until_ready_sync(timeout=15)

        threading.Thread(target=webcam_recording_loop, daemon=True).start()

    return {"status": "recording", "state": recording_state["state"]}

@app.post("/webcam/stop_recording")
async def stop_recording():
    recording_state["active"] = False
    recording_state["state"] = "passive"  # <-- added
    # NOTE: don't call Thread._stop() - it doesn't stop the loop safely.
    # Just setting active=False lets webcam_recording_loop exit by itself.
    return {"status": "stopped", "state": recording_state["state"]}

@app.get("/webcam/recording_state")
async def get_recording_state():
    return {"state": recording_state["state"], "active": recording_state["active"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webcam-server:app", host="0.0.0.0", port=8000, reload=False)