from datetime import datetime
import cv2
import os
import numpy as np
import threading

# Đường dẫn tới mô hình DNN của OpenCV
modelFile = "./model/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "./model/deploy.prototxt"

# Mở video để lấy thông tin
video_path = './videoplayback.mp4'  # Đường dẫn tới video của bạn
video_info = cv2.VideoCapture(video_path)

if not video_info.isOpened():
    print("Không thể mở video.")
    exit()

# Lấy FPS và tổng số khung hình của video
fps = video_info.get(cv2.CAP_PROP_FPS)
total_frames = int(video_info.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps
print(f"FPS của video: {fps}")
print(f"Tổng số khung hình: {total_frames}")
print(f"Thời lượng video: {duration} giây")

# Đóng VideoCapture sau khi lấy thông tin
video_info.release()

# Nhập tên và tạo thư mục lưu ảnh khuôn mặt
nameID = str(input("Enter Your Name: ")).lower()
dir = 'test'
path = os.path.join(dir, nameID)

if os.path.exists(path):
    print("Name Already Taken")
    while True:
        nameID = str(input("Enter Your Name Again: ")).lower()
        path = os.path.join(dir, nameID)
        if not os.path.exists(path):
            os.makedirs(path)
            break
        else:
            print("Name Already Taken")
else:
    os.makedirs(path)

# Số luồng xử lý
NUM_THREADS = 50  # Bạn có thể thay đổi số luồng tại đây
FRAME_INTERVAL_SEC = 1  # Khoảng thời gian giữa các khung hình cần lấy (giây)

# Khóa để đồng bộ biến đếm
count_lock = threading.Lock()
global_count = 0  # Biến đếm toàn cục

# Hàm xử lý video trong mỗi luồng
def process_video_segment(thread_index, start_time_sec, end_time_sec):
    # Khởi tạo mạng nơ-ron riêng cho mỗi luồng
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Luồng {thread_index}: Không thể mở video.")
        return

    current_time_sec = start_time_sec

    while current_time_sec < end_time_sec:
        # Tính số khung hình tương ứng với thời gian hiện tại
        frame_number = int(current_time_sec * fps)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = video.read()
        if not ret:
            print(f"Luồng {thread_index}: Không thể đọc khung hình tại {current_time_sec} giây.")
            current_time_sec += FRAME_INTERVAL_SEC
            continue

        (h, w) = frame.shape[:2]

        # Tiền xử lý khung hình để đưa vào mạng DNN
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        # Lặp qua các phát hiện
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Chỉ xét các phát hiện có độ tin cậy cao hơn ngưỡng
            if confidence > 0.5:
                # Tính tọa độ của bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                # Đảm bảo tọa độ nằm trong khung hình
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w - 1, x2)
                y2 = min(h - 1, y2)

                # Trích xuất khuôn mặt
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                # Sử dụng khóa để đảm bảo biến đếm toàn cục được cập nhật an toàn
                with count_lock:
                    global global_count
                    global_count += 1
                    filename = f"{nameID}_{global_count}.jpg"
                    name = os.path.join(path, filename)
                    print(f"Luồng {thread_index}: Đang tạo ảnh {name}")
                    cv2.imwrite(name, face)

        # Tăng thời gian lên FRAME_INTERVAL_SEC giây
        current_time_sec += FRAME_INTERVAL_SEC

    # Giải phóng tài nguyên
    video.release()
    print(f"Luồng {thread_index}: Hoàn thành xử lý.")

# Chia video thành các phân đoạn cho mỗi luồng
segment_duration = duration / NUM_THREADS
threads = []
time_start = datetime.now().timestamp()
for i in range(NUM_THREADS):
    start_time_sec = i * segment_duration
    end_time_sec = (i + 1) * segment_duration if i != NUM_THREADS - 1 else duration
    print(f"Luồng {i}: Xử lý từ {start_time_sec} đến {end_time_sec} giây.")

    # Tạo và khởi động luồng
    t = threading.Thread(target=process_video_segment, args=(i, start_time_sec, end_time_sec))
    threads.append(t)
    t.start()

# Chờ tất cả các luồng hoàn thành
for t in threads:
    t.join()

time_end = datetime.now().timestamp()
print(f"Hoàn thành xử lý tất cả các luồng. : {time_end - time_start} giây")
