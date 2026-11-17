##  1 [Home Camera] Tapo TC71 Setting
### Step 1: 카메라 Tapo 앱에 등록

Tapo 앱에서 TC71 추가: 상단 우측의 +  Circle Click

![1794952590437](image/tapocamera/1794952590437.jpg)

상단 우측 메뉴 Hamburger Mark Click

![1794952649225](image/tapocamera/1794952649225.jpg)

TAPO 실내 카메라 에서 TC71 선택

![1794952735824](image/tapocamera/1794952735824.jpg)
![1794952827903](image/tapocamera/1794952827903.jpg)

 이 곳부터 Instruction 따라하면 됨

![1794952775876](image/tapocamera/1794952775876.jpg)

Network 연결에서 Wifi AP 의 user_id와 password 입력
![1794952839244](image/tapocamera/1794952839244.jpg)

카메라 설정 -> 고급 설정 -> RTSP 활성화

RTSP 계정 생성:  camera_id / 비밀번호 1234abcd 생성

### Step 2: RTSP URL 확인 .env 에 저장
```text
TAPO_USERNAME = ""
TAPO_PASSWORD = ""
CAMERA1_RTSP = "rtsp://camera_id:password@192.168.0.47:554/stream2"  # TAPO APP Camera 고급설정에서 "camera_id" dhk "password" 설정함
CAMERA2_RTSP = "rtsp://camera_id:password.168.0.49:554/stream2"
Tapo 앱 -> 카메라 -> 설정 -> RTSP 주소에서 확인
```

### Step 3: VLC로 먼저 테스트

VLC -> 미디어 -> 네트워크 스트림 열기 -> RTSP URL 붙여넣기

화면 나오면 성공