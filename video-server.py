from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

app = FastAPI()
templates = Jinja2Templates(directory="templates")
# recordings 폴더 마운트 (한번만)
try:
    app.mount("/recordings_files", StaticFiles(directory="recordings"), name="recordings_files")
except:
    pass # 이미 마운트 되어있으면 패스

@app.get("/recordings", response_class=HTMLResponse)
async def recordings_viewer(request: Request):
    rec_dir = Path("recordings")
    rec_dir.mkdir(exist_ok=True)
    files = sorted(rec_dir.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    # JS에서 안전하게 쓰기 위해 파일명 리스트를 JSON으로
    file_list_json = [f.name for f in files]
    print(file_list_json)

    return templates.TemplateResponse(request,"record_viewer.html", {"__FILES__": file_list_json})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("video-server:app", host="0.0.0.0", port=8002, reload=True)