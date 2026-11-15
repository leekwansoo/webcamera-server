from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import ExitStack
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import numpy as np
import cv2, os, time, threading, datetime

app = FastAPI()
templates = Jinja2Templates(directory="templates")
# recordings 폴더 마운트 (한번만)
try:
    app.mount("/recordings_files", StaticFiles(directory="recordings"), name="recordings_files")
except:
    pass # 이미 마운트 되어있으면 패스

VIDEO_DIR = "recordings"

# PoseLandmarker 모델 파일 (최초 1회 다운로드 필요)
MODEL_PATH = "pose_landmarker_heavy.task"
if not os.path.exists(MODEL_PATH):
    import urllib.request
    print("Downloading pose model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        MODEL_PATH
    )

# ===== MediaPipe Pose 33 Landmarks =====

POSE_LANDMARKS = {
    # 0: "NOSE",
    # 1: "LEFT_EYE_INNER",
    # 2: "LEFT_EYE",
    # 3: "LEFT_EYE_OUTER",
    # 4: "RIGHT_EYE_INNER",
    # 5: "RIGHT_EYE",
    # 6: "RIGHT_EYE_OUTER",
    # 7: "LEFT_EAR",
    # 8: "RIGHT_EAR",
    # 9: "MOUTH_LEFT",
    # 10: "MOUTH_RIGHT",
    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",
    16: "RIGHT_WRIST",
    17: "LEFT_PINKY",
    18: "RIGHT_PINKY",
    19: "LEFT_INDEX",
    20: "RIGHT_INDEX",
    21: "LEFT_THUMB",
    22: "RIGHT_THUMB",
    23: "LEFT_HIP",
    24: "RIGHT_HIP",
    25: "LEFT_KNEE",
    26: "RIGHT_KNEE",
    27: "LEFT_ANKLE",
    28: "RIGHT_ANKLE",
    29: "LEFT_HEEL",
    30: "RIGHT_HEEL",
    31: "LEFT_FOOT_INDEX",
    32: "RIGHT_FOOT_INDEX",
}

# 이름으로 인덱스 찾기용 (역방향)
POSE_LANDMARK_INDEX = {v: k for k, v in POSE_LANDMARKS.items()}

# ===== Connections - 스켈레톤 선 연결 =====
POSE_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left Arm - 골프에서 중요
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19),
    # Right Arm
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22), (18, 20),
    # Left Leg
    (23, 25), (25, 27),
    (27, 29), (27, 31), (29, 31),
    # Right Leg
    (24, 26), (26, 28),
    (28, 30), (28, 32), (30, 32),
]

# 골프 스윙 분석용으로 자주 쓰는 그룹
GOLF_KEY_POINTS = {
    "HEAD": [0],
    "SHOULDERS": [11, 12],
    "ELBOWS": [13, 14],
    "WRISTS": [15, 16],
    "HIPS": [23, 24],
    "KNEES": [25, 26],
    "ANKLES": [27, 28],
}

# Face dots는 그리지 않음 - 머리 위치만 원으로 표시
FACE_LANDMARKS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
FINGER_TIPS_TO_SKIP = {17, 18, 21, 22} # PINKY / THUMB - 점도 안 그림

BaseOptions = mp_tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode


def draw_landmarks_on_frame(frame, landmarks, connections):
    h, w = frame.shape[:2]
    # 연결선
    for start, end in connections:
        if start < len(landmarks) and end < len(landmarks):
            p1 = landmarks[start]
            p2 = landmarks[end]
            if p1.visibility > 0.5 and p2.visibility > 0.5:
                cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (255,0,0), 2)
    # 점
    for lm in landmarks:
        if lm.visibility > 0.5:
            cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 2, (0,255,0), -1)
    return frame

# MediaPipe Pose Connections (옛날 POSE_CONNECTIONS와 동일)
# POSE_CONNECTIONS = PoseLandmarker.POSE_CONNECTIONS
def draw_skeleton_1(frame, landmarks):
    h, w = frame.shape[:2]
    # 선: visibility 체크 없이 무조건 그리기 (테스트용)
    for s, e in POSE_CONNECTIONS:
        if s < len(landmarks) and e < len(landmarks):
            p1, p2 = landmarks[s], landmarks[e]
            # 0.3으로 낮춤, presence도 체크
            if p1.visibility > 0.3 and p2.visibility > 0.3:
                cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0,0,255), 3)
    for lm in landmarks:
        if lm.visibility > 0.3:
            cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (0,255,0), -1)
    return frame

def draw_skeleton_2(frame, landmarks):
    h, w = frame.shape[:2]

    # --- 1. Body + Index connections만 그리기 ---
    for s, e in POSE_CONNECTIONS:
        if s < len(landmarks) and e < len(landmarks):
            p1, p2 = landmarks[s], landmarks[e]
            if p1.visibility > 0.3 and p2.visibility > 0.3:
                cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0,0,255), 3)

    # --- 2. Head position - 얼굴 스켈레톤 대신 큰 원 ---
    # NOSE(0) + LEFT_EAR(7) + RIGHT_EAR(8) 중심 = 머리 중심
    if len(landmarks) > 8:
        nose = landmarks[0]
        ear_l = landmarks[7]
        ear_r = landmarks[8]
        shoulder_l = landmarks[11]
        shoulder_r = landmarks[12]
        if nose.visibility > 0.3 and ear_l.visibility > 0.3 and ear_r.visibility > 0.3:
            hx = int((nose.x + ear_l.x + ear_r.x) / 3 * w)
            hy = int((nose.y + ear_l.y + ear_r.y) / 3 * h)
            # 머리 반지름 = 어깨 너비의 1/4 (사람 크기에 따라 자동 조절)
            shoulder_w = abs(shoulder_l.x - shoulder_r.x) * w
            radius = max(int(shoulder_w * 0.35), 15)
            # 머리 원 - 골프에서 머리 움직임 보기 좋게 노란색
            cv2.circle(frame, (hx, hy), radius, (0, 255, 255), 3)
            # 머리 중심점
            cv2.circle(frame, (hx, hy), 4, (0, 255, 255), -1)
            # 머리 중심에 수직선 - 스웨이 체크용
            cv2.line(frame, (hx, 0), (hx, h), (0, 255, 255), 1)

    # --- 3. Dots - 얼굴, 새끼/엄지 제외 ---
    for idx, lm in enumerate(landmarks):
        if idx in FACE_LANDMARKS:
            continue  # 얼굴 점 안 그림
        if idx in FINGER_TIPS_TO_SKIP:
            continue  # 새끼/엄지 점 안 그림
        if lm.visibility > 0.3:
            # Index finger는 파란색으로 강조, 나머지는 초록색
            color = (255, 255, 0) if idx in (19, 20) else (0, 255, 0)
            cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, color, -1)
    return frame

def draw_skeleton(frame, landmarks):
    h, w = frame.shape[:2]

    # --- 1. Body + Index connections ---
    for s, e in POSE_CONNECTIONS:
        if s < len(landmarks) and e < len(landmarks):
            p1, p2 = landmarks[s], landmarks[e]
            if p1.visibility > 0.3 and p2.visibility > 0.3:
                cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0,0,255), 3)

    # --- 2. Virtual points for Torso -> Head ---
    # landmarks: 11=LEFT_SHOULDER, 12=RIGHT_SHOULDER, 23=LEFT_HIP, 24=RIGHT_HIP
    #            0=NOSE, 7=LEFT_EAR, 8=RIGHT_EAR
    if len(landmarks) > 24:
        sl = landmarks[11]; sr = landmarks[12]
        hl = landmarks[23]; hr = landmarks[24]
        nose = landmarks[0]; ear_l = landmarks[7]; ear_r = landmarks[8]

        if all(p.visibility > 0.3 for p in [sl, sr, hl, hr, nose, ear_l, ear_r]):
            # Shoulder center (어깨 중심)
            sc_x = (sl.x + sr.x) / 2 * w
            sc_y = (sl.y + sr.y) / 2 * h
            sc = (int(sc_x), int(sc_y))

            # Hip center (골반 중심)
            hc_x = (hl.x + hr.x) / 2 * w
            hc_y = (hl.y + hr.y) / 2 * h
            hc = (int(hc_x), int(hc_y))

            # Head center (머리 중심)
            head_cx = (nose.x + ear_l.x + ear_r.x) / 3 * w
            head_cy = (nose.y + ear_l.y + ear_r.y) / 3 * h
            head_center = (int(head_cx), int(head_cy))

            # Head radius = shoulder width * 0.35
            shoulder_w = abs(sl.x - sr.x) * w
            radius = max(int(shoulder_w * 0.35), 15)

            # Head TOP = head center에서 반지름만큼 위로
            head_top = (int(head_cx), int(head_cy - radius - 5))

            # --- Torso extended lines ---
            # 1. Spine: Hip center -> Shoulder center (흰색, 굵게)
            cv2.line(frame, hc, sc, (255, 255, 255), 3)
            # 2. Neck: Shoulder center -> Head top (노란색)
            cv2.line(frame, sc, head_top, (0, 255, 255), 3)
            # 3. Head circle
            cv2.circle(frame, head_center, radius, (0, 255, 255), 3)
            cv2.circle(frame, head_center, 4, (0, 255, 255), -1)
            # 4. Points for centers - 분석하기 좋게
            cv2.circle(frame, sc, 5, (255, 255, 255), -1)  # 어깨 중심 = 흰점
            cv2.circle(frame, hc, 5, (255, 255, 255), -1)  # 골반 중심 = 흰점
            cv2.circle(frame, head_top, 5, (0, 255, 255), -1) # 머리 꼭대기

            # 5. Optional: Head top 수직선 (스웨이 체크)
            # cv2.line(frame, (head_top[0], 0), (head_top[0], h), (0, 255, 255), 1)

    # --- 3. Dots (face / pinky / thumb 제외) ---
    FACE_LANDMARKS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    FINGER_TIPS_TO_SKIP = {17, 18, 21, 22}
    for idx, lm in enumerate(landmarks):
        if idx in FACE_LANDMARKS or idx in FINGER_TIPS_TO_SKIP:
            continue
        if lm.visibility > 0.3:
            color = (255, 255, 0) if idx in (19, 20) else (0, 255, 0)
            cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, color, -1)
    return frame

def generate_skeleton_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    print(f"[SKELETON] Open {video_path}")

    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2,  # <-- 합성 프레임은 2명이므로 1 -> 2 로!
        min_pose_detection_confidence=0.3,  # 0.5 -> 0.3 낮춤
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        idx = 0
        detected_frames = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            h, w = frame.shape[:2]
            # 디버그: 30프레임마다 한 번만 로그
            if idx % 30 == 0:
                print(f"[FRAME {idx}] size {w}x{h}")

            # 합성 프레임이면 좌/우 나눔
            if w > 1000:
                mid = w // 2
                halves = [frame[:, :mid].copy(), frame[:, mid:].copy()]
                out = []
                for hi, half in enumerate(halves):
                    rgb = cv2.cvtColor(half, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    res = landmarker.detect_for_video(mp_img, int(idx * 33))
                    # 디버그 로그
                    if idx % 30 == 0:
                        print(f"  half {hi}: poses={len(res.pose_landmarks) if res.pose_landmarks else 0}")
                    if res.pose_landmarks:
                        detected_frames += 1
                        for pose_lm in res.pose_landmarks:
                            # visibility 필터 0.5 -> 0.3으로 낮춤
                            draw_skeleton(half, pose_lm)
                    out.append(half)
                frame = np.hstack(out)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = landmarker.detect_for_video(mp_img, int(idx * 33))
                if idx % 30 == 0:
                    print(f"  single: poses={len(res.pose_landmarks) if res.pose_landmarks else 0}")
                if res.pose_landmarks:
                    detected_frames += 1
                    for pose_lm in res.pose_landmarks:
                        draw_skeleton(frame, pose_lm)

            if idx % 90 == 0:
                print(f"[STATS] detected in {detected_frames}/{idx+1} frames")

            idx += 1
            ret, buf = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    cap.release()
    print(f"[SKELETON] Done. detected {detected_frames} frames")


def generate_skeleton_image_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    print(f"[SKELETON] Open {video_path}")

    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,  # <-- VIDEO -> IMAGE 로 변경!
        num_poses=2,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            h, w = frame.shape[:2]
            if w > 1000:
                mid = w // 2
                halves = [frame[:, :mid].copy(), frame[:, mid:].copy()]
                out = []
                for half in halves:
                    rgb = cv2.cvtColor(half, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    # IMAGE 모드는 timestamp가 필요 없음
                    res = landmarker.detect(mp_img)
                    if res.pose_landmarks:
                        for pose_lm in res.pose_landmarks:
                            draw_skeleton(half, pose_lm)
                    out.append(half)
                frame = np.hstack(out)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = landmarker.detect(mp_img)
                if res.pose_landmarks:
                    for pose_lm in res.pose_landmarks:
                        draw_skeleton(frame, pose_lm)
            idx += 1
            ret, buf = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    cap.release()

def save_skeleton_video(input_path: Path, output_path: Path):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[SAVE] Cannot open {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[SAVE] {input_path} -> {output_path} | {w}x{h} @ {fps}fps, {total} frames")

    # mp4v for compatibility - same as your webcam-server
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        if not writer.isOpened():
                raise Exception("avc1 not available")
    except:
        print("[SAVE] avc1 failed, fallback to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=2,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            h_f, w_f = frame.shape[:2]
            if w_f > 1000:
                mid = w_f // 2
                halves = [frame[:, :mid].copy(), frame[:, mid:].copy()]
                out = []
                for half in halves:
                    rgb = cv2.cvtColor(half, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    res = landmarker.detect(mp_img)
                    if res.pose_landmarks:
                        for pose_lm in res.pose_landmarks:
                            draw_skeleton(half, pose_lm)
                    out.append(half)
                frame = np.hstack(out)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = landmarker.detect(mp_img)
                if res.pose_landmarks:
                    for pose_lm in res.pose_landmarks:
                        draw_skeleton(frame, pose_lm)

            writer.write(frame)
            idx += 1
            if idx % 30 == 0:
                print(f"[SAVE] {idx}/{total} frames...")

    cap.release()
    writer.release()
    print(f"[SAVE] Done: {output_path} ({output_path.stat().st_size/1024:.0f} KB)")
    return True

           
VIDEO_DIR = Path("recordings")
VIDEO_DIR.mkdir(exist_ok=True)

@app.get("/recordings", response_class=HTMLResponse)
async def recordings_viewer(request: Request):
    rec_dir = Path("recordings")
    rec_dir.mkdir(exist_ok=True)
    files = sorted(rec_dir.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    # JS에서 안전하게 쓰기 위해 파일명 리스트를 JSON으로
    file_list_json = [f.name for f in files]
    print(file_list_json)

    return templates.TemplateResponse(request,"record_viewer.html", {"files": file_list_json})

# 2. API - JSON list for JS - THIS IS MISSING NOW
@app.get("/api/recordings")
async def api_list_recordings():
    files = sorted(
        [f.name for f in VIDEO_DIR.glob("*.mp4")],
        reverse=True  # newest first
    )
    return files  # -> ["dual_20261103_222147.mp4", ...]

# 3. Actual video file - for <video src>
@app.get("/api/recordings/{file_name:path}")
async def api_get_recording(file_name: str):
    video_path = VIDEO_DIR / file_name
    if not video_path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return StreamingResponse(
        open(video_path, "rb"),
        media_type="video/mp4"
    )
    
from fastapi.responses import FileResponse

@app.get("/recordings/{file_name:path}")
async def serve_video(file_name: str):
    video_path = VIDEO_DIR / file_name
    if not video_path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    # ✅ FileResponse는 Range를 자동으로 처리하지만, 헤더를 주면 재요청이 빨라짐
    return FileResponse(
        video_path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"}
    )
    
# 2. Endpoint - save + return new filename
@app.post("/api/save_skeleton/{file_name:path}")
async def api_save_skeleton(file_name: str):
    input_path = VIDEO_DIR / file_name
    if not input_path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)

    # skeleton_ + original name
    output_name = f"skeleton_{file_name}"
    output_path = VIDEO_DIR / output_name

    # 이미 저장된 파일이 있으면 다시 만들지 않고 바로 반환
    if output_path.exists():
        return {"status": "already_exists", "saved_file": output_name}

    ok = save_skeleton_video(input_path, output_path)
    if not ok:
        return JSONResponse({"error": "save failed"}, status_code=500)
    return {"status": "saved", "saved_file": output_name}

@app.get("/skeleton/{file_name:path}")
async def play_skeleton(file_name: str):
    # 이미 skeleton_ 파일이면 재추적 없이 원본 스트리밍만
    if file_name.startswith("skeleton_"):
        # 그냥 일반 파일로 취급 - 스켈레톤 재검출 안 함
        video_path = VIDEO_DIR / file_name
        return FileResponse(video_path, media_type="video/mp4")
    video_path = VIDEO_DIR / file_name
    return StreamingResponse(
        generate_skeleton_image_frames(video_path),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
    
# 4. Delete - for 🗑 button
@app.delete("/api/recordings/{file_name:path}")
async def api_delete_recording(file_name: str):
    video_path = VIDEO_DIR / file_name
    if video_path.exists():
        video_path.unlink()
        return {"status": "deleted", "file": file_name}
    return JSONResponse({"error": "not found"}, status_code=404)

@app.get("/skeleton/{file_name:path}")
async def generate_skeleton_frames(file_name: str):
    return StreamingResponse(
        generate_skeleton_image_frames(VIDEO_DIR / file_name),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("video-server:app", host="0.0.0.0", port=8002, reload=True)