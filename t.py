import datetime
import cv2
import os
import numpy as np
from threading import Thread
from queue import Queue
import torch
from ultralytics import YOLO  # Import YOLOv8 từ thư viện ultralytics
import logging
import threading
import subprocess  # Import subprocess để thựcconda lis thi lệnh ffmpeg
import time  # Thêm thư viện time để đo thời gian

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Đường dẫn đến mô hình YOLOv8 đã được huấn luyện cho phát hiện khuôn mặt
model_name = "yolov8n-face.pt"  # Đảm bảo bạn đã tải mô hình này

# Hàm để phát hiện khuôn mặt trong mỗi luồng
def detect_faces(detection_queue, model_name,gpu_id):
    """
    Hàm để xử lý các đường dẫn frame từ hàng đợi cho việc phát hiện khuôn mặt.
    Mỗi luồng sẽ lấy một đường dẫn frame từ hàng đợi, đọc frame, thực hiện phát hiện và lưu kết quả.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        if torch.cuda.is_available():
            logging.info(f"{threading.current_thread().name}: Sử dụng GPU để phát hiện khuôn mặt.") 
        # Tải mô hình YOLOv8 trong mỗi luồng để tránh xung đột
        model = YOLO(model_name)  # Nếu mô hình chưa được tải, ultralytics sẽ tự động tải
        logging.info(f"{threading.current_thread().name}: Đã tải mô hình YOLOv8 thành công.")
    except Exception as e:
        logging.error(f"{threading.current_thread().name}: Lỗi khi tải mô hình YOLOv8: {e}")
        return

    while True:
        item = detection_queue.get()
        if item is None:
            # Tín hiệu để kết thúc xử lý cho luồng này
            logging.info(f"{threading.current_thread().name}: Nhận tín hiệu kết thúc.")
            detection_queue.task_done()
            break
        task_id, current_time_sec, frame_path = item

        try:
            # Bắt đầu đo thời gian đọc frame
            start_read_time = time.time()

            # Đọc frame từ đường dẫn
            frame = cv2.imread(frame_path)
            if frame is None:
                logging.warning(f"{threading.current_thread().name}: Không thể đọc frame: {frame_path}")
                detection_queue.task_done()
                continue

            

            # Chuyển đổi frame từ BGR (OpenCV) sang RGB (yêu cầu bởi YOLOv8)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Sử dụng YOLOv8 để phát hiện khuôn mặt
            results = model(frame_rgb, verbose=False)

            # Kết quả chứa danh sách các phát hiện
            detections = results[0]

            # Lặp qua các phát hiện
            faces_dir = os.path.join('tasks', task_id, 'faces')
            os.makedirs(faces_dir, exist_ok=True)
            face_index = 0  # Chỉ số của khuôn mặt trong giây hiện tại

            for det in detections.boxes:
                confidence = det.conf.item()
                class_id = int(det.cls[0])

                # Kiểm tra nếu đối tượng là khuôn mặt (class_id = 0 trong mô hình YOLOv8-Face)
                if class_id == 0 and confidence > 0.5:
                    # Lấy toạ độ bounding box
                    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())

                    # Đảm bảo toạ độ nằm trong frame
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1] - 1, x2)
                    y2 = min(frame.shape[0] - 1, y2)

                    # Cắt khuôn mặt
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue

                    # Lưu khuôn mặt vào /tasks/<task_id>/faces/<second>_<index>.png
                    face_filename = os.path.join(faces_dir, f"{int(current_time_sec)}_{face_index}.png")
                    cv2.imwrite(face_filename, face)
                    face_index += 1

        except Exception as e:
            logging.error(f"Lỗi khi xử lý frame tại giây {current_time_sec} cho task_id {task_id}: {e}")

        finally:
            detection_queue.task_done()
            read_time = time.time() - start_read_time
            logging.info(f"{threading.current_thread().name}: Thời gian đọc frame {frame_path}: {read_time:.4f} giây")

# Hàm để tách frame từ video bằng ffmpeg và đưa đường dẫn vào hàng đợi
def extract_frames(video_path, task_id, detection_queue):
    """
    Hàm để tách frame từ video bằng ffmpeg và đặt đường dẫn frame vào hàng đợi để xử lý.
    Mỗi luồng sẽ xử lý một video riêng biệt.
    """
    try:
        start_time = datetime.datetime.now()

        # Thư mục lưu trữ frame
        frames_dir = os.path.join('tasks', task_id, 'frames')
        os.makedirs(frames_dir, exist_ok=True)

        # Sử dụng ffmpeg để tách frame với tốc độ 1 fps
        ffmpeg_command = [
            'ffmpeg',
            '-hwaccel', 'cuda',  # Sử dụng CUDA cho hardware acceleration
            '-c:v', 'h264_cuvid',  # Chỉ định codec sử dụng CUDA để decode (nếu video là H.264)
            '-i', video_path,  # Đường dẫn đến video input
            '-vf', 'fps=1',  # Tách frame với tốc độ 1 FPS
            '-c:v', 'mjpeg',  # Chỉ định codec cho file output là MJPEG
            os.path.join(frames_dir, 'frame_%d.jpg')  # Đường dẫn lưu frame output
        ]


        # Bắt đầu đo thời gian cắt frame
        start_extract_time = time.time()

        # Thực thi lệnh ffmpeg và ghi lại stderr
        logging.info(f"Bắt đầu tách frame cho video: {video_path}")
        ffmpeg_process = subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if ffmpeg_process.returncode != 0:
            logging.error(f"Lỗi ffmpeg cho video {video_path}:\n{ffmpeg_process.stderr}")
            return
        logging.info(f"Hoàn thành tách frame cho video: {video_path}")

        extract_time = time.time() - start_extract_time
        logging.info(f"Thời gian cắt frame cho video {video_path}: {extract_time:.2f} giây")

        # Bắt đầu đo thời gian đọc và đưa vào hàng đợi
        start_queue_time = time.time()

        # Danh sách các frame đã tách
        frame_files = sorted(os.listdir(frames_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))

        current_time_sec = 0

        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            # Đưa đường dẫn frame vào hàng đợi để xử lý
            detection_queue.put((task_id, current_time_sec, frame_path))
            current_time_sec += 1

        queue_time = time.time() - start_queue_time
        logging.info(f"Thời gian đọc và đưa vào hàng đợi cho video {video_path}: {queue_time:.2f} giây")
        # Ghi lại thời gian xử lý cho video
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logging.info(f"Hoàn thành xử lý video: {video_path}. Thời gian xử lý: {processing_time:.2f} giây.")

    except Exception as e:
        logging.error(f"Lỗi khi xử lý video {video_path}: {e}")

    # Không cần đặt None vào hàng đợi ở đây

# Hàm chính
def main():
    """
    Hàm chính để xử lý tất cả các video trong thư mục.
    """
    video_folder_path = "./yolo-video"
    time_start = datetime.datetime.now().timestamp()

    # Kiểm tra nếu mô hình tồn tại
    if not os.path.exists(model_name):
        logging.error(f"Mô hình YOLOv8 không tồn tại tại: {model_name}")
        logging.info("Vui lòng tải mô hình YOLOv8-Face và đặt nó tại đường dẫn trên.")
        return

    # Lấy danh sách tất cả các video trong thư mục
    video_files = [f for f in os.listdir(video_folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv'))]

    if not video_files:
        logging.warning("Không tìm thấy video nào trong thư mục.")
        return

    num_detection_threads = 5  # Số lượng luồng cho phát hiện khuôn mặt
    detection_queue = Queue()

    # Khởi tạo các luồng phát hiện khuôn mặt
    detection_threads = []
    for i in range(num_detection_threads):
        t = Thread(target=detect_faces, args=(detection_queue, model_name,i%4), name=f"Detection-Thread-{i+1}")
        t.start()
        detection_threads.append(t)
        logging.info(f"Đã khởi động luồng phát hiện khuôn mặt {i+1}/{num_detection_threads}.")

    # Khởi tạo các luồng tách frame (mỗi luồng xử lý một video)
    extraction_threads = []
    for video_file in video_files:
        video_path = os.path.join(video_folder_path, video_file)
        task_id = os.path.splitext(video_file)[0]  # Sử dụng tên file không có phần mở rộng làm task_id

        # Bắt đầu luồng tách frame cho video này
        t = Thread(target=extract_frames, args=(video_path, task_id, detection_queue), name=f"Extractor-{task_id}")
        t.start()
        extraction_threads.append(t)
        logging.info(f"Đã khởi động luồng tách frame cho video: {video_file}")

    # Chờ cho đến khi tất cả các luồng tách frame hoàn thành
    for t in extraction_threads:
        t.join()
        logging.info(f"{t.name} đã hoàn thành.")

    # Chờ cho đến khi tất cả các frame được xử lý
    detection_queue.join()

    # Sau khi tất cả các frame đã được xử lý, gửi tín hiệu kết thúc cho các luồng phát hiện
    for _ in detection_threads:
        detection_queue.put(None)

    # Chờ cho các luồng phát hiện hoàn thành
    for t in detection_threads:
        t.join()
        logging.info(f"{t.name} đã kết thúc.")

    time_end = datetime.datetime.now().timestamp()
    total_processing_time = time_end - time_start

    logging.info(f"Đã hoàn thành xử lý tất cả các video. Tổng thời gian: {total_processing_time:.2f} giây.")

if __name__ == "__main__":
    main()
