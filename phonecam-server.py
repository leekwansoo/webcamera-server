from os import name

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import numpy as np, cv2, time, asyncio, threading
from pathlib import Path
from contextlib import asynccontextmanager

PHONE_CONFIG = ["PHONE1", "PHONE2"]
CAMERA_CONFIG = {"PHONE1": 0, "PHONE2": 1}
OUTPUT_DIR = Path("recordings")
OUTPUT_DIR.mkdir(exist_ok=True)

_cameras: dict[str, cv2.VideoCapture] = {}
_locks: dict[str, threading.Lock] = {
    "PHONE1": threading.Lock(),
    "PHONE2": threading.Lock(),
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting...")
    yield
    print("Server shutting down...")

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
OUTPUT_DIR = Path("recordings")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Dual Phone Frames ---
phones = {
    "PHONE1": {"jpeg": None, "frame": None, "last": 0},
    "PHONE2": {"jpeg": None, "frame": None, "last": 0},
}
viewers = set()
combined_jpeg = None

recording_state = {"active": False}

def decode_jpeg(jpeg_bytes):
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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
        
def get_combined_frame():
    f1 = phones["PHONE1"]["frame"]
    f2 = phones["PHONE2"]["frame"]
    
    # 360x480 포터레이트
    if f1 is None:
        f1 = np.zeros((480, 360, 3), dtype=np.uint8)
        cv2.putText(f1, "PHONE1 Waiting", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    if f2 is None:
        f2 = np.zeros((480, 360, 3), dtype=np.uint8)
        cv2.putText(f2, "PHONE2 Waiting /phone2", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    # 좌우 합치기 -> 720x480 (16:9 비슷한 와이드)
    combined = np.hstack([f1, f2])
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(combined, ts, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(combined, "TAB1", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(combined, "TAB2", (370, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    if recording_state["active"]:
        cv2.circle(combined, (700, 20), 8, (0,0,255), -1)
    return combined

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

def recording_loop():
    print("Recording started")
    filepath = OUTPUT_DIR / f"dual_phone_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    while recording_state["active"]:
        combined = get_combined_frame()
        if writer is None:
            h,w = combined.shape[:2]
            writer = cv2.VideoWriter(str(filepath), fourcc, 15.0, (w,h))
            print(f"Writer {w}x{h} -> {filepath}")
        writer.write(combined)
        time.sleep(0.066) # 15fps
    if writer:
        writer.release()
        print(f"Saved: {filepath} {filepath.stat().st_size/1024:.1f}KB")

# --- Phone Pages ---
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

def get_camera(name: str):
    cam = _cameras.get(name)
    if cam is None or not cam.isOpened():
        idx = CAMERA_CONFIG[name]
        cam = open_camera_with_fallback(idx)
        if cam:
            _cameras[name] = cam
    return cam



@app.get("/{phone_id}", response_class=HTMLResponse)
async def phone_generic(request: Request, phone_id: str):
    pid = phone_id.upper()
    print(f"pid={pid}") # phone1 -> PHONE1
    if pid not in PHONE_CONFIG:
        return HTMLResponse("Use /phone1 or /phone2", status_code=404)
    return templates.TemplateResponse(request, "phonecam-viewer.html", {
        "phone_id": pid,
        "phone_id_lower": pid.lower()
    })

@app.get("/api/{phone_id}/stream")
async def api_named_stream(phone_id: str):
    cam_name = phone_id.upper()
    if cam_name not in PHONE_CONFIG:
        return JSONResponse({"error": "Use PHONE1 or PHONE2"}, status_code=404)
    cam = get_camera(cam_name)
    if cam is None or not cam.isOpened():
        return JSONResponse({"error": f"{cam_name} failed"}, status_code=404)
    return StreamingResponse(mjpeg_generator(cam_name), media_type="multipart/x-mixed-replace; boundary=frame")


# --- WebSockets for 2 phones ---
@app.websocket("/ws/phone1")
async def ws_phone1(websocket: WebSocket):
    await websocket.accept()
    print("PHONE1 connected")
    try:
        while True:
            data = await websocket.receive_bytes()
            phones["PHONE1"]["jpeg"]=data
            phones["PHONE1"]["frame"]=decode_jpeg(data)
            phones["PHONE1"]["last"]=time.time()
            await broadcast_combined()
    except WebSocketDisconnect:
        print("PHONE1 disconnected")

@app.websocket("/ws/phone2")
async def ws_phone2(websocket: WebSocket):
    await websocket.accept()
    print("PHONE2 connected")
    try:
        while True:
            data = await websocket.receive_bytes()
            phones["PHONE2"]["jpeg"]=data
            phones["PHONE2"]["frame"]=decode_jpeg(data)
            phones["PHONE2"]["last"]=time.time()
            await broadcast_combined()
    except WebSocketDisconnect:
        print("PHONE2 disconnected")

async def broadcast_combined():
    global combined_jpeg
    combined = get_combined_frame()
    _, jpg = cv2.imencode('.jpg', combined, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    combined_jpeg = jpg.tobytes()
    dead=set()
    for vws in viewers:
        try: await vws.send_bytes(combined_jpeg)
        except: dead.add(vws)
    for d in dead: viewers.discard(d)

@app.websocket("/ws/viewer")
async def ws_viewer(websocket: WebSocket):
    await websocket.accept()
    viewers.add(websocket)
    try:
        while True: await asyncio.sleep(10)
    except WebSocketDisconnect:
        viewers.discard(websocket)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "phonecam-index.html", {})

@app.post("/start_recording")
async def start_recording():
    if not recording_state["active"]:
        recording_state["active"]=True
        threading.Thread(target=recording_loop, daemon=True).start()
    return {"status":"recording"}

@app.post("/stop_recording")
async def stop_recording():
    recording_state["active"]=False
    return {"status":"stopped"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("phonecam-server:app", host="0.0.0.0", port=8080, reload=False)