
# server.py - Fantasy Server (FastAPI + WebSocket)
# 설치: pip install fastapi uvicorn
# 실행: uvicorn server:app --host 0.0.0.0 --port 5000
# Tab 접속: http://PC_IP:5000/cam
# PC 모니터링: http://localhost:5000/

import json
import asyncio
from urllib import request
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Fantasy Server")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 연결된 Tab 목록 { tab_id: {name, ip, mac} }
cameras: dict[str, dict] = {}
# PC 모니터 클라이언트들
monitors: set[WebSocket] = set()

# ============ Routes ============

@app.get("/", response_class=HTMLResponse)
async def monitor_page(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.get("/cam", response_class=HTMLResponse)
async def cam_page(request: Request):
    return templates.TemplateResponse(request, "cam.html")

@app.get("/api/cams")
async def list_cams():
    return {"count": len(cameras), "cams": cameras}

async def broadcast_to_monitors(payload: dict):
    dead = []
    for ws in list(monitors):
        try:
            await ws.send_text(json.dumps(payload))
        except:
            dead.append(ws)
    for ws in dead:
        monitors.discard(ws)

async def broadcast_list():
    await broadcast_to_monitors({"type": "list", "cams": cameras})

# ============ WebSocket: Tab (카메라 송신) ============

@app.websocket("/ws/tab")
async def ws_tab(ws: WebSocket):
    await ws.accept()
    tab_id = f"tab-{id(ws)}"
    ip = ws.client.host if ws.client else "unknown"
    print(f"✅ Tab 연결: {ip}")
    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            if data.get("type") == "register":
                cameras[tab_id] = {
                    "name": data.get("name", "Galaxy Tab"),
                    "ip": ip,
                    "mac": data.get("mac", "browser에서는 MAC을 읽을 수 없음 - IP로 식별"),
                }
                print(f"✅ Tab Activate! {cameras[tab_id]}")
                await broadcast_list()
            elif data.get("type") == "frame":
                # PC 모니터들에게 바로 중계
                await broadcast_to_monitors({
                    "type": "frame",
                    "tab_id": tab_id,
                    "image": data.get("image", ""),
                })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Tab 에러: {e}")
    finally:
        if tab_id in cameras:
            del cameras[tab_id]
            print(f"❌ Tab 연결 끊김: {tab_id}")
            await broadcast_list()

# ============ WebSocket: PC Monitor (수신) ============

@app.websocket("/ws/monitor")
async def ws_monitor(ws: WebSocket):
    await ws.accept()
    monitors.add(ws)
    # 처음 접속 시 현재 목록 바로 보내주기
    await ws.send_text(json.dumps({"type": "list", "cams": cameras}))
    try:
        while True:
            # 모니터에서 오는 메시지는 무시하고 연결 유지 (ping 등)
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        monitors.discard(ws)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
