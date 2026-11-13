import asyncio
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


spec = importlib.util.spec_from_file_location(
    "video_server", Path(__file__).resolve().parents[1] / "video-server.py"
)
video_server = importlib.util.module_from_spec(spec)
with patch("urllib.request.urlretrieve"):
    spec.loader.exec_module(video_server)


class FakeCapture:
    def __init__(self, width):
        self.frames = iter([np.zeros((32, width, 3), dtype=np.uint8)] * 2)

    def isOpened(self):
        return True

    def read(self):
        frame = next(self.frames, None)
        return frame is not None, frame

    def release(self):
        pass


class FakeLandmarker:
    def __init__(self):
        self.timestamps = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def detect_for_video(self, image, timestamp):
        if self.timestamps and timestamp <= self.timestamps[-1]:
            raise ValueError("timestamps must increase")
        self.timestamps.append(timestamp)
        return type("Result", (), {"pose_landmarks": []})()


class SkeletonStreamTests(unittest.TestCase):
    def test_skeleton_endpoint_streams_jpeg_frames(self):
        landmarker = FakeLandmarker()
        landmarker.detect = lambda image: type("Result", (), {"pose_landmarks": []})()

        async def request():
            messages = []
            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.4"},
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": "/skeleton/video.mp4",
                "raw_path": b"/skeleton/video.mp4",
                "query_string": b"",
                "headers": [],
                "server": ("testserver", 80),
                "client": ("testclient", 1234),
            }

            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                messages.append(message)

            await video_server.app(scope, receive, send)
            return messages

        with patch.object(video_server.cv2, "VideoCapture", return_value=FakeCapture(640)), \
             patch.object(video_server.PoseLandmarker, "create_from_options", return_value=landmarker):
            messages = asyncio.run(request())

        self.assertEqual(messages[0]["status"], 200)
        self.assertIn((b"content-type", b"multipart/x-mixed-replace; boundary=frame"), messages[0]["headers"])
        body = b"".join(message["body"] for message in messages[1:])
        parts = body.split(b"--frame\r\n")
        self.assertEqual(len(parts), 3)
        for part in parts[1:]:
            self.assertTrue(part.startswith(b"Content-Type: image/jpeg\r\n\r\n\xff\xd8"))
            self.assertTrue(part.endswith(b"\xff\xd9\r\n"))
        self.assertTrue(landmarker.closed)

    def test_single_and_composite_frames_use_independent_trackers(self):
        for width, expected_trackers in ((640, 1), (1260, 2)):
            with self.subTest(width=width):
                trackers = []

                def create_landmarker(options):
                    tracker = FakeLandmarker()
                    trackers.append(tracker)
                    return tracker

                with patch.object(video_server.cv2, "VideoCapture", return_value=FakeCapture(width)), \
                     patch.object(video_server.PoseLandmarker, "create_from_options", side_effect=create_landmarker):
                    frames = list(video_server.generate_skeleton_frames("video.mp4"))

                self.assertEqual(len(frames), 2)
                self.assertTrue(all(frame.startswith(b"--frame\r\n") for frame in frames))
                self.assertEqual(len(trackers), expected_trackers)
                for tracker in trackers:
                    self.assertEqual(tracker.timestamps, [0, 33])
                    self.assertTrue(tracker.closed)


if __name__ == "__main__":
    unittest.main()
