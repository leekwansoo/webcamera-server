## Procedures after clone this repository
From the Terminal
```text
with uv commands
  uv init
  uv venv
  .venv/Scripts/activate
  uv add -r requirements.txt
```
## Add OpenH264.dll into .venv/ ..../cv2/
  copy "openh264-2.5.0-win64.dll" to \venv\Lib\site-packages\cv2\openh264-2.5.0-win64.dll
  
  add recordings folder 

## For IP Camera 
From Terminal
```text
  python ipcamera-server.py
From Browser
  localhost:8000  : then you will see the 2 combined view frame for 2 IP Cameras
  localhost:8000/ipcamera  : you will see two separate frames for ipcamera1 and ipcamera2
```
## For Web Camera 
From Terminal
```text
  python webcam-server.py
From Browser
  localhost:8000  : then you will see the 2 combined view frame for 2 Webcam Cameras
  localhost:8000/webcam  : you will see two separate frames for webcam1 and webcam2
```
## Recording Combined Views from IPCamera and WebCamera
make sure you created "recordings" dirctory at the root-directory since all generated mp3 files will be stored in that directory

## View Recorded MP3 files with video-server.py
From terminal
```text
  python video-server.py
From Browser
  localhost:8002/recordings  : you will see the Video Viewer with recorded filelist and Viewing Canvas
  you can view the mp3 files with skeletons attached
```
## Webcam availability
```text
The webcam server retries an unavailable camera every five seconds for up to
30 seconds from the first request. It then stops searching that camera until
the server is restarted and continues showing a no-signal placeholder.
The deadline prevents new attempts; an OpenCV camera operation already in
progress may finish after the deadline.
```
## Skeleton playback
MediaPipe 0.10.35 is required for Windows skeleton playback. After updating
dependencies, run `uv sync --inexact` and restart the video server.

## openh264-2.5.0-win64.dll 
```text
copy "openh264-2.5.0-win64.dll" to your venv so OpenCV always finds it:
C:\Users\user_name\Desktop\webcamera-server\venv\Lib\site-packages\cv2\openh264-2.5.0-win64.dll
```
