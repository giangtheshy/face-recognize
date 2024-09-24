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
SLAVE_API_URL_PROCESS_VIDEO = f"{SLAVE_URL}/process_video"  # Thay đổi nếu cần
SLAVE_API_URL_PROCESS_FACES = f"{SLAVE_URL}/process_faces"  # Thay đổi nếu cần

# Thư mục lưu trữ các task
TASKS_DIR = "tasks"

queue  = Queue()
# Đảm bảo thư mục tasks tồn tại
os.makedirs(TASKS_DIR, exist_ok=True)

# Mô hình dữ liệu cho request body (nếu cần)
class CombineImagesRequest(BaseModel):
    task_id: str

def combine_images(image_paths, output_path, columns=3, rows=3, padding=5, background_color=(255, 255, 255), output_width=240, output_height=272):
    """
    Kết hợp một nhóm hình ảnh thành một hình ảnh lớn với bố cục theo lưới.

    :param image_paths: Danh sách đường dẫn tới các hình ảnh.
    :param output_path: Đường dẫn lưu trữ hình ảnh kết quả.
    :param columns: Số cột trong lưới.
    :param rows: Số dòng trong lưới.
    :param padding: Khoảng cách giữa các hình ảnh.
    :param background_color: Màu nền cho hình ảnh kết quả.
    :param output_width: Chiều rộng của ảnh kết quả.
    :param output_height: Chiều cao của ảnh kết quả.
    """
    if not image_paths:
        print("Không có hình ảnh nào để kết hợp.")
        return

    # Mở các ảnh và lưu vào danh sách
    images = [Image.open(path) for path in image_paths]

    # Tính số lượng hình ảnh cần kết hợp
    num_images = len(images)
    max_images = columns * rows

    if num_images > max_images:
        print(f"Số lượng hình ảnh vượt quá giới hạn ({max_images}). Chỉ sử dụng {max_images} hình ảnh đầu tiên.")
        images = images[:max_images]
        num_images = max_images

    # Tính kích thước mỗi ô ảnh
    total_padding_width = (columns - 1) * padding
    total_padding_height = (rows - 1) * padding
    cell_width = (output_width - total_padding_width) // columns
    cell_height = (output_height - total_padding_height) // rows

    if cell_width <= 0 or cell_height <= 0:
        print("Kích thước ô ảnh không hợp lệ. Vui lòng kiểm tra lại kích thước đầu ra và tham số padding.")
        return

    # Tạo ảnh mới với kích thước cố định
    new_image = Image.new('RGB', (output_width, output_height), color=background_color)

    # Dán từng ảnh vào ảnh mới
    for index, img in enumerate(images):
        if index >= max_images:
            break  # Đảm bảo không vượt quá số lượng hình ảnh tối đa

        # Tính vị trí x, y cho mỗi ảnh
        x = (index % columns) * (cell_width + padding)
        y = (index // columns) * (cell_height + padding)

        # Thay đổi kích thước ảnh để phù hợp với ô
        img_resized = img.resize((cell_width, cell_height), Image.LANCZOS)

        # Dán ảnh vào vị trí tương ứng
        new_image.paste(img_resized, (x, y))

    # Lưu ảnh kết quả
    new_image.save(output_path)
    print(f"Đã tạo ảnh tổng hợp tại {output_path}")

def compress_faces(task_id: str):
    """
    Hàm nén các khuôn mặt thành một hình ảnh tổng hợp.

    :param task_id: ID của task.
    """
    image_folder = os.path.join(TASKS_DIR, task_id, 'faces')
    output_folder = os.path.join(TASKS_DIR, task_id, 'combined_faces')
    os.makedirs(output_folder, exist_ok=True)

    # Lấy danh sách đường dẫn các ảnh, sắp xếp để đảm bảo thứ tự nhất định
    image_files = sorted([
        os.path.join(image_folder, img) for img in os.listdir(image_folder)
        if img.lower().endswith(('jpg', 'png', 'jpeg'))
    ])

    # Tham số cho lưới ảnh
    columns = 5  # Số cột
    rows = 5     # Số dòng
    padding = 10  # Khoảng cách giữa các hình ảnh
    output_width = 240
    output_height = 272
    max_images_per_grid = columns * rows

    # Tính số lượng grid cần tạo
    total_images = len(image_files)
    total_grids = math.ceil(total_images / max_images_per_grid)

    print(f"Tổng số hình ảnh: {total_images}")
    print(f"Sẽ tạo {total_grids} hình ảnh lớn với mỗi hình chứa tối đa {max_images_per_grid} hình.")

    for grid_index in range(total_grids):
        start_idx = grid_index * max_images_per_grid
        end_idx = start_idx + max_images_per_grid
        current_batch = image_files[start_idx:end_idx]

        # Tạo tên file cho image lớn
        output_image = os.path.join(output_folder, f"combined_image_{grid_index + 1}.jpg")

        # Gọi hàm để kết hợp ảnh với kích thước cố định
        combine_images(
            current_batch,
            output_image,
            columns=columns,
            rows=rows,
            padding=padding,
            output_width=output_width,
            output_height=output_height
        )

def call_slave_api(segment, thread_index, task_id):
    """
    Hàm để gọi API slave để xử lý đoạn video.

    :param segment: Dict chứa 'start_time' và 'end_time'.
    :param thread_index: Chỉ số của luồng.
    :param task_id: ID của task.
    :param video_path: Đường dẫn tới video.
    """
    payload = {
        "start_time": segment["start_time"],
        "end_time": segment["end_time"],
        "video_path": segment["video_path"],
        "task_id": task_id
    }
    try:
        response = requests.post(SLAVE_API_URL_PROCESS_VIDEO, json=payload)
        if response.status_code == 200:
            print(f"Luồng {thread_index}: Xử lý thành công đoạn từ {segment['start_time']} đến {segment['end_time']} giây.")
        else:
            print(f"Luồng {thread_index}: Xử lý thất bại đoạn từ {segment['start_time']} đến {segment['end_time']} giây. Lỗi: {response.text}")
    except Exception as e:
        print(f"Luồng {thread_index}: Gặp lỗi khi gọi API slave. Lỗi: {e}")

def call_slave_process_faces_api(task_id, image_paths):
    """
    Hàm để gọi API slave để xử lý nhận diện khuôn mặt.

    :param task_id: ID của task.
    :param image_paths: Danh sách đường dẫn tới các hình ảnh.
    """
    payload = {
        "task_id": task_id,
        "image_paths": image_paths
    }
    try:
        response = requests.post(SLAVE_API_URL_PROCESS_FACES, json=payload)
        if response.status_code == 200:
            print(f"Nhận diện khuôn mặt thành công cho {len(image_paths)} hình ảnh.")
        else:
            print(f"Nhận diện khuôn mặt thất bại cho {len(image_paths)} hình ảnh. Lỗi: {response.text}")
    except Exception as e:
        print(f"Gặp lỗi khi gọi API slave nhận diện khuôn mặt. Lỗi: {e}")

# Hàm để gọi API slave /process_faces trong các luồng
def call_process_faces_thread(image_subset, task_id):
    print(f"Đang nhận diện khuôn mặt cho {image_subset}")
    # Chuyển mảng numpy thành list và loại bỏ các phần tử trống nếu có
    image_subset = [img for img in image_subset if img]
    if not image_subset:
        return
    call_slave_process_faces_api(task_id, image_subset)



def get_video_segments(video_path, NUM_WORKER):
    print(f"Đang chia video thành các đoạn nhỏ... {video_path}")
    # Lấy thông tin video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở video {video_path}")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0:
        print(f"Không thể lấy FPS cho video {video_path}")
        return []
    duration = total_frames / fps
    cap.release()

    print(f"Video: {video_path}")
    print(f"FPS: {fps}")
    print(f"Tổng số khung hình: {total_frames}")
    print(f"Thời lượng video: {duration} giây")

    # Chia video thành các segment
    segment_duration = duration / NUM_WORKER
    segments = []
    for i in range(NUM_WORKER):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration if i != NUM_WORKER - 1 else duration
        segments.append({
            "start_time": start_time,
            "end_time": end_time,
            "video_path": video_path
        })
        print(f"Đoạn {i}: Từ {start_time} đến {end_time} giây.")
    return segments

def get_all_video_segments(folder_path, NUM_WORKER):
    print(f"Đang tìm tất cả các video trong thư mục {folder_path}")
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    segments = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in video_extensions:
                video_path = os.path.join(root, file)
                video_segments = get_video_segments(video_path, NUM_WORKER)
                segments.extend(video_segments)
    return segments

def process_upload_file():
    while True:
        if queue.qsize() == 0:
            time.sleep(2)
            continue
        task = queue.get()
        task_id = task["task_id"]
        task_path = task["task_path"]
        tracking_path = f"./tracking/{task["tracking_path"]}"
        image_path = task["image_path"]
        frames_path = task["frames_path"]
        faces_path = task["faces_path"]
        combined_faces_path = task["combined_faces_path"]
        time_start = datetime.datetime.now().timestamp()
        
        print(f"Đang xử lý video và nhận diện khuôn mặt... Task ID: {task_id}")
        
        segments = get_all_video_segments(tracking_path, NUM_WORKER)

        # Tạo và khởi động các luồng để gọi API slave /process_video
        threads = []
        for index, segment in enumerate(segments):
            t = threading.Thread(target=call_slave_api, args=(segment, index, task_id))
            threads.append(t)
            t.start()

        # Chờ tất cả các luồng hoàn thành
        for t in threads:
            t.join()

        print("Đã hoàn thành xử lý tất cả các đoạn video.")

        # Nén các khuôn mặt thành hình ảnh tổng hợp
        compress_faces(task_id)

        # Lấy danh sách các hình ảnh đã được nén
        combined_faces_folder = os.path.join(task_path, 'combined_faces')
        combined_image_files = sorted([
            os.path.join(combined_faces_folder, img) for img in os.listdir(combined_faces_folder)
            if img.lower().endswith(('jpg', 'png', 'jpeg'))
        ])

        if not combined_image_files:
            raise HTTPException(status_code=500, detail="Không tìm thấy hình ảnh tổng hợp sau khi nén.")

        # Chia danh sách hình ảnh đã nén thành NUM_THREADS mảng con
        split_combined_images = np.array_split(combined_image_files, NUM_WORKER)

        # Tạo và khởi động các luồng để gọi API slave /process_faces
        faces_threads = []
        for image_subset in split_combined_images:
            t = threading.Thread(target=call_process_faces_thread, args=(image_subset,task_id))
            faces_threads.append(t)
            t.start()

        # Chờ tất cả các luồng hoàn thành
        for t in faces_threads:
            t.join()

        print("Đã hoàn thành nhận diện khuôn mặt cho tất cả các hình ảnh đã nén.")

        # Đường dẫn đến ảnh tổng hợp đã xử lý (nếu cần, có thể thay đổi tùy vào yêu cầu)
        # Ở đây giả sử các hình ảnh đã xử lý được lưu trong "processed_faces" và bạn có thể tổng hợp lại nếu cần
        processed_faces_folder = os.path.join(task_path, 'processed_faces')
        processed_image_files = sorted([
            os.path.join(processed_faces_folder, img) for img in os.listdir(processed_faces_folder)
            if img.lower().endswith(('jpg', 'png', 'jpeg'))
        ])

        # Tính thời gian xử lý
        time_end = datetime.datetime.now().timestamp()
        print(f"Đã hoàn thành xử lý tất cả các hình ảnh đã nhận diện khuôn mặt : {time_end - time_start} giây.")
        # Trả về đường dẫn đến các hình ảnh đã xử lý hoặc các thông tin cần thiết
        # Ở đây chúng ta trả về đường dẫn đến thư mục processed_faces
        if not processed_image_files:
            raise HTTPException(status_code=500, detail="Không tìm thấy hình ảnh đã xử lý.")