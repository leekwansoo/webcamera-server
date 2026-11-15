## First if you dont have Git installed on your PC Install Git

Go to Git Download

https://git-scm.com  >install

chose Widows

<img width="1071" height="493" alt="image" src="https://github.com/user-attachments/assets/e09e0723-eaac-4568-9764-e2522efe9548" />



## Procedures after clone this repositoryOpen the Terminal
Open the new terminal and create a venv
```text
with uv commands
  uv init
  uv venv
  .venv/Scripts/activate
  uv add -r requirements.txt
```

## add .env file in the root directory
```text
TAPO_USERNAME=""
TAPO_PASSWORD=""
CAMERA1_RTSP="rtsp://camera_id:password@camera_ipaddress:554/stream2"
CAMERA1_MAC=""
CAMERA2_MAC="1"
CAMERA2_RTSP="rtsp://camera_id:password@camera_ipaddress:554/stream2"
```
you can find IP Address for connected IPCAMERA with "arp -a" or "TAPO APP for IPCAMERA"

## Add OpenH264.dll into .venv/ ..../cv2/
  copy "openh264-2.5.0-win64.dll" to \venv\Lib\site-packages\cv2\openh264-2.5.0-win64.dll
  
  add recordings folder 

## For IP Camera 
```text
From Terminal
  python ipcamera-server.py
From Browser
  localhost:8000  : then you will see the 2 combined view frame for 2 IP Cameras
  localhost:8000/ipcamera  : you will see two separate frames for ipcamera1 and ipcamera2
```
## For Web Camera 
```text
From Terminal
  python webcam-server.py
From Browser
  localhost:8000  : then you will see the 2 combined view frame for 2 Webcam Cameras
  localhost:8000/webcam  : you will see two separate frames for webcam1 and webcam2
```
## Recording Combined Views from IPCamera and WebCamera
make sure you created "recordings" dirctory at the root-directory since all generated mp3 files will be stored in that directory

## View Recorded MP3 files with video-server.py
```text
From terminal
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
