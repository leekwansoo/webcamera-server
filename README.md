# Webcam availability

The webcam server retries an unavailable camera every five seconds for up to
30 seconds from the first request. It then stops searching that camera until
the server is restarted and continues showing a no-signal placeholder.
The deadline prevents new attempts; an OpenCV camera operation already in
progress may finish after the deadline.

## Skeleton playback

MediaPipe 0.10.35 is required for Windows skeleton playback. After updating
dependencies, run `uv sync --inexact` and restart the video server.

## openh264-2.5.0-win64.dll 
copy "openh264-2.5.0-win64.dll" to your venv so OpenCV always finds it:
C:\Users\user_name\Desktop\webcamera-server\venv\Lib\site-packages\cv2\openh264-2.5.0-win64.dll
