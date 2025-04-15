import io
import numpy as np
import cv2
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def temp_upload_dir(tmp_path, monkeypatch):
    temp_dir = tmp_path / "uploads"
    temp_dir.mkdir()
    monkeypatch.setattr("main.data_dir", temp_dir)
    yield

def create_test_image_bytes(format="PNG"):
    img = Image.new("RGB", (100, 100), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format)
    img_byte_arr.seek(0)
    return img_byte_arr

def create_test_video_bytes():
    return io.BytesIO(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")

class FakeVideoCapture:
    def __init__(self, filename):
        self.filename = filename
        self.frames = [
            (True, np.full((200, 200, 3), 255, dtype=np.uint8)),  # Первый кадр
            (False, None)  # Конец видео
        ]
        self.current_frame = 0

    def read(self):
        frame = self.frames[self.current_frame]
        self.current_frame = min(self.current_frame + 1, len(self.frames) - 1)
        return frame

    def release(self):
        pass

    def isOpened(self):
        return True

@pytest.fixture
def fake_video(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)

# ---------- PUT /api/upload ----------

def test_upload_image():
    img_bytes = create_test_image_bytes()
    files = {"file": ("img.png", img_bytes, "image/png")}
    response = client.put("/api/upload", files=files)
    assert response.status_code == 200
    json = response.json()
    assert "uuid" in json
    assert json["type"] == "image"
    assert json["filename"] == "img.png"

def test_upload_video(fake_video):
    video = create_test_video_bytes()
    files = {"file": ("vid.mp4", video, "video/mp4")}
    response = client.put("/api/upload", files=files)
    assert response.status_code == 200
    json = response.json()
    assert json["type"] == "video"
    assert json["filename"] == "vid.mp4"

def test_upload_invalid_mime():
    txt = io.BytesIO(b"not an image or video")
    files = {"file": ("test.txt", txt, "text/plain")}
    response = client.put("/api/upload", files=files)
    assert response.status_code == 400
    assert "Only images and videos are allowed" in response.json()["detail"]


def test_upload_corrupted_video(fake_video, monkeypatch):
    fake_video_data = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"corrupted" * 100
    broken = io.BytesIO(fake_video_data)

    files = {"file": ("broken.mp4", broken, "video/mp4")}

    class CorruptedVideoCapture:
        def __init__(self, filename):
            self.filename = filename

        def read(self):
            return False, None

        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", CorruptedVideoCapture)

    response = client.put("/api/upload", files=files)

    assert response.status_code == 400
    assert "not readable or corrupted" in response.json()["detail"]

# ---------- GET /api/{uuid} ----------

def test_get_file():
    img = create_test_image_bytes()
    files = {"file": ("img.png", img, "image/png")}
    upload_response = client.put("/api/upload", files=files)
    file_uuid = upload_response.json()["uuid"]

    response = client.get(f"/api/{file_uuid}")
    assert response.status_code == 200
    assert "attachment; filename=" in response.headers.get("Content-Disposition", "")

def test_get_thumbnail():
    img = create_test_image_bytes()
    files = {"file": ("thumb.png", img, "image/png")}
    upload_response = client.put("/api/upload", files=files)
    file_uuid = upload_response.json()["uuid"]

    response = client.get(f"/api/{file_uuid}?width=50&height=50")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")

def test_get_video_thumbnail(fake_video):
    video = create_test_video_bytes()
    files = {"file": ("video.mp4", video, "video/mp4")}
    upload_response = client.put("/api/upload", files=files)
    file_uuid = upload_response.json()["uuid"]

    response = client.get(f"/api/{file_uuid}?width=100&height=100")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")

def test_get_nonexistent():
    response = client.get("/api/nonexistent")
    assert response.status_code == 404
    assert "File not found" in response.json()["detail"]

# ---------- DELETE /api/{uuid} ----------

def test_delete_file():
    img = create_test_image_bytes()
    files = {"file": ("del.png", img, "image/png")}
    upload_response = client.put("/api/upload", files=files)
    file_uuid = upload_response.json()["uuid"]

    response = client.delete(f"/api/{file_uuid}")
    assert response.status_code == 200
    assert response.json()["deleted_file"]["uuid"] == file_uuid

    # Повторное удаление должно вернуть 404
    response = client.delete(f"/api/{file_uuid}")
    assert response.status_code == 404

def test_delete_nonexistent():
    response = client.delete("/api/nonexistent")
    assert response.status_code == 404
    assert "File not found" in response.json()["detail"]