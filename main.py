from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from uuid import uuid4
from PIL import Image
import shutil
import mimetypes
import cv2
import os
import filetype
app = FastAPI()

data_dir = Path("uploads")
data_dir.mkdir(exist_ok=True)


def is_valid_media_type(file: UploadFile):
    file_content = file.file.read(2048)
    file.file.seek(0)
    kind = filetype.guess(file_content)
    return kind and (kind.mime.startswith("image/") or kind.mime.startswith("video/"))


def get_file_info(file_path: Path, original_filename: str = None, file_uuid: str = None):
    stat = file_path.stat()

    with open(file_path, 'rb') as f:
        file_content = f.read(2048)
        kind = filetype.guess(file_content)

    if kind:
        mime_type = kind.mime
        ext_type = "image" if mime_type.startswith("image/") else "video"
    else:
        mime_type = "unknown"
        ext_type = "unknown"

    return {
        "uuid": file_uuid or file_path.stem.split("_thumb")[0],
        "filename": original_filename or file_path.name,
        "size": stat.st_size,
        "created_at": stat.st_ctime,
        "modified_at": stat.st_mtime,
        "type": ext_type,
        "mime_type": mime_type
    }


@app.put("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    file_content = await file.read()
    await file.seek(0)
    kind = filetype.guess(file_content)

    if not kind or not (kind.mime.startswith("image/") or kind.mime.startswith("video/")):
        raise HTTPException(status_code=400, detail="Only images and videos are allowed")

    file_uuid = str(uuid4())
    file_ext = Path(file.filename).suffix
    file_path = data_dir / f"{file_uuid}{file_ext}"

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if kind.mime.startswith("video/"):
        cap = cv2.VideoCapture(str(file_path))
        success, _ = cap.read()
        cap.release()
        if not success:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Uploaded video is not readable or corrupted")

    return get_file_info(file_path, original_filename=file.filename, file_uuid=file_uuid)


@app.get("/api/{file_uuid}")
async def get_file(file_uuid: str, width: int = None, height: int = None):
    files = list(data_dir.glob(f"{file_uuid}.*"))
    if not files:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = files[0]

    with open(file_path, 'rb') as f:
        file_content = f.read(2048)
        kind = filetype.guess(file_content)

    if not kind:
        raise HTTPException(status_code=400, detail="Unknown file type")

    mime_type = kind.mime

    if width and height:
        if mime_type.startswith("image/"):
            return generate_image_thumbnail(file_path, width, height)
        elif mime_type.startswith("video/"):
            return generate_video_thumbnail(file_path, width, height)

    return FileResponse(file_path, headers={"Content-Disposition": f"attachment; filename={file_path.name}"})


def generate_image_thumbnail(image_path: Path, width: int, height: int):
    img = Image.open(image_path)
    img.thumbnail((width, height))
    thumb_path = image_path.with_name(f"{image_path.stem}_thumb{image_path.suffix}")
    img.save(thumb_path)
    return FileResponse(thumb_path)


def generate_video_thumbnail(video_path: Path, width: int, height: int):
    cap = cv2.VideoCapture(str(video_path))
    success, frame = cap.read()
    cap.release()

    if not success:
        raise HTTPException(status_code=500, detail="Failed to read video frame for thumbnail")

    frame = cv2.resize(frame, (width, height))
    thumb_path = video_path.with_name(f"{video_path.stem}_thumb.jpg")
    cv2.imwrite(str(thumb_path), frame)
    return FileResponse(thumb_path)


@app.delete("/api/{file_uuid}")
async def delete_file(file_uuid: str):
    files = list(data_dir.glob(f"{file_uuid}.*"))
    if not files:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = files[0]
    file_info = get_file_info(file_path, file_uuid=file_uuid)

    try:
        os.remove(file_path)
        thumb = data_dir / f"{file_path.stem}_thumb.jpg"
        if thumb.exists():
            os.remove(thumb)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

    return {"deleted_file": file_info}
