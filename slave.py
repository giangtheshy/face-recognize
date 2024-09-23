from fastapi import FastAPI, HTTPException
from multiprocessing import Process
import os
import numpy as np
import cv2
import datetime
import threading
from slave_helper import process_video_segment, process_list_images, mean_embedding_jack, scaler,\
    mean_embedding_file, scaler_file, NUM_THREADS, ProcessFacesRequest, VideoProcessingRequest, process_video_segment, \
    process_list_images, mean_embedding_jack, scaler, mean_embedding_file, scaler_file

app = FastAPI()


@app.post("/process_video")
def process_video(request: VideoProcessingRequest):
    # Tạo thư mục cho task_id
    task_id = request.task_id

    # Mở video để lấy thông tin
    video_info = cv2.VideoCapture(request.video_path)
    if not video_info.isOpened():
        return {"error": "Không thể mở video."}

    # Lấy FPS của video
    fps = video_info.get(cv2.CAP_PROP_FPS)
    video_info.release()

    # Tính toán thời lượng và phân đoạn cho mỗi luồng
    duration = request.end_time - request.start_time
    segment_duration = duration / NUM_THREADS

    threads = []
    for i in range(NUM_THREADS):
        segment_start_time = request.start_time + i * segment_duration
        segment_end_time = request.start_time + (i + 1) * segment_duration if i != NUM_THREADS - 1 else request.end_time

        # Tạo và khởi động luồng
        t = threading.Thread(target=process_video_segment, args=(
            i, segment_start_time, segment_end_time, request.video_path, task_id, fps))
        threads.append(t)
        t.start()

    # Chờ tất cả các luồng hoàn thành
    for t in threads:
        t.join()

    return {"message": "Hoàn thành xử lý video."}

@app.post("/process_faces")
def process_faces(request: ProcessFacesRequest):
    """
    Endpoint để nhận danh sách đường dẫn hình ảnh và task_id, sau đó thực hiện nhận diện khuôn mặt.
    
    :param request: Yêu cầu chứa task_id và danh sách image_paths.
    :return: Thông báo hoàn thành xử lý.
    """
    global NUM_THREADS
    task_id = request.task_id
    image_paths = request.image_paths

    # Kiểm tra tồn tại của image_paths
    for img_path in image_paths:
        if not os.path.exists(img_path):
            raise HTTPException(status_code=400, detail=f"Không tìm thấy hình ảnh: {img_path}")

    # Định nghĩa thư mục đầu ra
    output_processed_folder = os.path.join("tasks", task_id, "processed_faces")
    os.makedirs(output_processed_folder, exist_ok=True)

    print("Bắt đầu chương trình...")

    # Kiểm tra sự tồn tại của các file cần thiết
    required_files = [mean_embedding_file,  scaler_file]
    for file in required_files:
        if not os.path.exists(file):
            raise HTTPException(status_code=500, detail=f"File {file} không tồn tại.")

    print("Đang tải các tệp mô hình...")

    # Tải mean_embedding và threshold đã được tải ở đầu file

    # Kiểm tra và tải scaler đã được tải ở đầu file

    print(f"Tổng số hình ảnh cần xử lý: {len(image_paths)}")

    # Bắt đầu đo thời gian
    start_time = datetime.datetime.now().timestamp()

    if len(image_paths) < NUM_THREADS:
        NUM_THREADS = len(image_paths)
    split_images = np.array_split(image_paths, NUM_THREADS)
    list_process = []

    for i in range(NUM_THREADS):
        p = Process(target=process_list_images, args=(
            split_images[i],
            mean_embedding_jack,
            scaler,
            output_processed_folder
        ))
        list_process.append(p)

    # Khởi động tất cả các tiến trình
    for p in list_process:
        p.start()

    # Chờ tất cả các tiến trình hoàn thành
    for p in list_process:
        p.join()

    # Kết thúc đo thời gian
    end_time = datetime.datetime.now().timestamp()
    execution_time = end_time - start_time
    print(f"\nThời gian thực thi: {execution_time:.2f} giây")

    return {"message": "Hoàn thành xử lý nhận diện khuôn mặt."}