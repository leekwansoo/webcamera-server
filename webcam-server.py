import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2, datetime, time, threading
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from contextlib import asynccontextmanager
from utils.frame_handler import CAMERA_CONFIG, OUTPUT_DIR
from utils.frame_handler import to_portrait_wide, to_square
from utils.frame_handler import get_camera, synced_mjpeg_generator, mjpeg_generator, dummy_mjpeg_generator, get_synced_split_frame, open_camera_with_fallback

if os.environ["OPENCV_LOG_LEVEL"].upper() == "ERROR":
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

# --- NEW: Portrait frame adjust (same as Tapo IP version) ---

# --- Camera open (same as yours) ---

recording_state = {"active": False, "writer": None}

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


# FastAPI routes for webcam server
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "webcam_index.html")

@app.get("/webcam/{cam_name}", response_class=HTMLResponse)
async def webcam(request: Request, cam_name: str):
    return templates.TemplateResponse(request, "webcam-viewer.html", {"now": datetime.datetime.now(), "cam_name": cam_name})

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
    return StreamingResponse(synced_mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/start_recording")
async def start_recording():
    if not recording_state["active"]:
        recording_state["active"] = True
        threading.Thread(target=recording_loop, daemon=True).start()
    return {"status": "recording"}

@app.post("/stop_recording")
async def stop_recording():
    recording_state["active"] = False
    return {"status": "stopped"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webcam-server:app", host="0.0.0.0", port=8000, reload=False)