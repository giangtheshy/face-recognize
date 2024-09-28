import os
import threading
import uuid
import shutil
from fastapi import FastAPI, File, Form, UploadFile
from master_helper import  TASKS_DIR,queue,process_upload_file

app = FastAPI()

@app.post("/upload")
async def upload_files(image: UploadFile = File(...), tracking_path: str = Form(...)):
    """
    Endpoint để nhận một hình ảnh và một video, xử lý video bằng các luồng và kết hợp các khuôn mặt.

    :param image: Hình ảnh đầu vào.
    :param video: Video đầu vào.
    :return: task_id và đường dẫn đến ảnh tổng hợp.
    """
    # Tạo task_id duy nhất
    task_id = str(uuid.uuid4())
    task_path = os.path.join(TASKS_DIR, task_id)
    frames_path = os.path.join(task_path, 'frames')
    output_folder = os.path.join("tasks", task_id, "processed_faces")

    # Tạo các thư mục cần thiết
    os.makedirs(frames_path, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Lưu hình ảnh
    image_path = os.path.join(task_path, 'input_image' + os.path.splitext(image.filename)[1])
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    print(f"Đã lưu hình ảnh tại {image_path}")



    queue.put({
        "task_id": task_id,
        "tracking_path": tracking_path,
        "image_path": image_path,
        "frames_path": frames_path,
    })
    # Có thể trả về danh sách các hình ảnh đã xử lý hoặc bất kỳ thông tin nào khác
    return {
        "task_id": task_id,
        "message": "Hoàn thành xử lý video và nhận diện khuôn mặt."
    }


t = threading.Thread(target=process_upload_file, args=())
t.start()