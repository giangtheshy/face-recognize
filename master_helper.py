import datetime
import math
from multiprocessing import Queue
import threading
import time
import cv2
from fastapi import HTTPException
import numpy as np
import requests
from pydantic import BaseModel
from PIL import Image
import os
import dotenv
dotenv.load_dotenv()
NUM_WORKER = int(os.environ.get("WORKER_NUMBER", 1))
NUM_THREAD = int(os.environ.get("THREAD_NUMBER", 1))
SLAVE_URL = os.environ.get("SLAVE_URL", "http://localhost:8001")

# URL của API slave
SLAVE_API_URL_RECOGNIZE_FACES = f"{SLAVE_URL}/recognize_faces"  # Thay đổi nếu cần

# Thư mục lưu trữ các task
TASKS_DIR = "tasks"

queue  = Queue()
# Đảm bảo thư mục tasks tồn tại
os.makedirs(TASKS_DIR, exist_ok=True)

def call_slave_recognize_faces_api(task_id, video_path):
    """
    Hàm để gọi API slave để xử lý nhận diện khuôn mặt.

    :param task_id: ID của task.
    :param image_paths: Danh sách đường dẫn tới các hình ảnh.
    """
    payload = {
        "task_id": task_id,
        "video_path": video_path
    }
    try:
        response = requests.post(SLAVE_API_URL_RECOGNIZE_FACES, json=payload)
        if response.status_code == 200:
            print(f"Nhận diện khuôn mặt thành công cho video {video_path}.")
        else:
            print(f"Nhận diện khuôn mặt thất bại cho video {video_path}. Lỗi: {response.text}")
    except Exception as e:
        print(f"Gặp lỗi khi gọi API slave nhận diện khuôn mặt. Lỗi: {e}")
        

def process_upload_file():
    while True:
        if queue.qsize() == 0:
            time.sleep(2)
            continue
        task = queue.get()
        task_id = task["task_id"]
        tracking_path = f"tracking/{task['tracking_path']}"
        print(f"Đang xử lý video và nhận diện khuôn mặt... Task ID: {task_id}")

        videos = get_all_video_segments(tracking_path)
        threads = []
        for video_path in videos:
            # Lấy thông tin video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Không thể mở video.")
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            cap.release()

            print(f"FPS của video: {fps}")
            print(f"Tổng số khung hình: {total_frames}")
            print(f"Thời lượng video: {duration} giây")
            t = threading.Thread(target=call_slave_recognize_faces_api, args=(task_id, video_path))
            threads.append(t)
            t.start()

        # Chờ tất cả các luồng hoàn thành
        for t in threads:
            t.join()

        print("Waiting for all threads to finish...")


def get_all_video_segments(folder_path):
    print(f"Đang tìm tất cả các video trong thư mục {folder_path}")
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    segments = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in video_extensions:
                video_path = os.path.join(root, file)
                segments.append(video_path)
    return segments
