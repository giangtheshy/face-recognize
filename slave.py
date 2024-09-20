from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from multiprocessing import Process, current_process
import os
import numpy as np
import cv2
import datetime
import joblib
import tensorflow as tf
import threading
import dotenv
import uuid
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

dotenv.load_dotenv()

app = FastAPI()


NUM_THREADS = int(os.environ.get("THREAD_NUMBER", 1))

BACKEND = 'mtcnn'
MODEL = 'Facenet512'

# Đường dẫn tới mô hình DNN của OpenCV
modelFile = "./model/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "./model/deploy.prototxt"

# Đường dẫn tới các file mô hình cho nhận diện khuôn mặt
mean_embedding_file = './model/mean_embedding_jack.npy'
scaler_file = './model/scaler.pkl'

# Kiểm tra sự tồn tại của các file cần thiết
required_files = [modelFile, configFile, mean_embedding_file, scaler_file]
for file in required_files:
    if not os.path.exists(file):
        print(f"File {file} không tồn tại. Vui lòng kiểm tra lại quá trình cài đặt.")
        exit()

# Tải mean_embedding và threshold
mean_embedding_jack = np.load(mean_embedding_file)

# Tải scaler
scaler = joblib.load(scaler_file)

# Mô hình dữ liệu cho request body
class ProcessFacesRequest(BaseModel):
    task_id: str
    image_paths: List[str]


# Mô hình dữ liệu cho request body
class VideoProcessingRequest(BaseModel):
    start_time: float  # Thời gian bắt đầu xử lý (giây)
    end_time: float    # Thời gian kết thúc xử lý (giây)
    video_path: str    # Đường dẫn tới video
    task_id: str       # ID của task (sử dụng để đặt tên thư mục)

# Đường dẫn tới mô hình DNN của OpenCV
modelFile = "./model/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "./model/deploy.prototxt"


def compute_cosine_similarity(embedding, mean_embedding):
    """
    Tính toán cosine similarity giữa embedding của khuôn mặt hiện tại và mean_embedding.
    
    :param embedding: Embedding của khuôn mặt hiện tại.
    :param mean_embedding: Embedding trung bình (mean_embedding_jack).
    :return: Giá trị cosine similarity.
    """
    return cosine_similarity([embedding], [mean_embedding])[0][0]

# Hàm xử lý video trong mỗi luồng
def process_video_segment(thread_index, start_time_sec, end_time_sec, video_path, task_id, fps):
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
            current_time_sec += 1  # Chuyển sang giây tiếp theo
            continue

        (h, w) = frame.shape[:2]

        # Lưu frame vào thư mục /task_id/frames/<second>.jpg
        frames_dir = os.path.join('tasks',task_id, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        frame_filename = os.path.join(frames_dir, f"{int(current_time_sec)}.jpg")
        cv2.imwrite(frame_filename, frame)

        # Tiền xử lý khung hình để đưa vào mạng DNN
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        # Lặp qua các phát hiện
        faces_dir = os.path.join('tasks',task_id, 'faces')
        os.makedirs(faces_dir, exist_ok=True)
        face_index = 0  # Chỉ số khuôn mặt trong giây hiện tại
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

                # Lưu khuôn mặt vào thư mục /task_id/faces/<second>_<index>.png
                face_filename = os.path.join(faces_dir, f"{int(current_time_sec)}_{face_index}.png")
                cv2.imwrite(face_filename, face)
                face_index += 1

        # Tăng thời gian lên 1 giây
        current_time_sec += 1

    # Giải phóng tài nguyên
    video.release()
    print(f"Luồng {thread_index}: Hoàn thành xử lý.")

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

def process_image(image_path, mean_embedding_jack,  scaler, output_image=None):
    """
    Xử lý một hình ảnh để phát hiện và nhận diện khuôn mặt.
    
    :param image_path: Đường dẫn tới hình ảnh cần xử lý.
    :param mean_embedding_jack: Embedding trung bình của "Jack".
    :param threshold_cosine: Ngưỡng cosine similarity để xác định "Jack".
    :param scaler: Scaler đã được huấn luyện.
    :param output_image: Đường dẫn để lưu hình ảnh đã xử lý. Nếu None, không lưu.
    :return: Hình ảnh đã xử lý (cv2 image).
    """
    print(f"[{current_process().name}] Đang đọc hình ảnh từ: {image_path}")
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"[{current_process().name}] Không thể đọc hình ảnh: {image_path}. Bỏ qua.")
        return None

    print(f"[{current_process().name}] Chuyển đổi hình ảnh từ BGR sang RGB...")
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Tạo thư mục tạm thời để lưu các khung hình
    temp_dir = './temp_frames'
    os.makedirs(temp_dir, exist_ok=True)
    print(f"[{current_process().name}] Thư mục tạm thời được tạo tại: {temp_dir}")

    # Lưu khung hình vào tệp tạm thời
    temp_img_path = os.path.join(temp_dir, f"{uuid.uuid4()}.jpg")
    print(f"[{current_process().name}] Đang lưu hình ảnh tạm thời tại: {temp_img_path}")
    cv2.imwrite(temp_img_path, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))

    try:
        print(f"[{current_process().name}] Đang sử dụng DeepFace để phát hiện và trích xuất embedding...")
        # Sử dụng DeepFace để phát hiện và trích xuất embedding
        start_time = datetime.datetime.now().timestamp()
        results = DeepFace.represent(img_path=temp_img_path, enforce_detection=True, model_name=MODEL, detector_backend=BACKEND)
        end_time = datetime.datetime.now().timestamp()
        print(f"[{current_process().name}] DeepFace.represent hoàn thành trong {end_time - start_time:.2f} giây")
    except Exception as e:
        print(f"[{current_process().name}] Lỗi khi phát hiện khuôn mặt trong {image_path}: {e}")
        results = []

    # Xóa tệp tạm sau khi sử dụng
    if os.path.exists(temp_img_path):
        print(f"[{current_process().name}] Đang xóa tệp tạm: {temp_img_path}")
        os.remove(temp_img_path)

    # 'results' có thể là list hoặc dict tùy thuộc vào số khuôn mặt
    if isinstance(results, dict):
        results = [results]  # Chuyển đổi thành list nếu chỉ có một khuôn mặt

    print(f"[{current_process().name}] Số khuôn mặt được phát hiện: {len(results)}")

    if results:
        for idx, embedding_dict in enumerate(results):
            print(f"[{current_process().name}] Đang xử lý khuôn mặt thứ {idx + 1}")
            embedding = embedding_dict.get('embedding', None)
            if embedding is None:
                print(f"[{current_process().name}] Không có embedding, bỏ qua khuôn mặt này.")
                continue

            # Chuẩn hóa và normalize embedding
            embedding_scaled = scaler.transform([embedding])[0]
            embedding_normalized = embedding_scaled / np.linalg.norm(embedding_scaled)

            # Tính cosine similarity
            similarity = compute_cosine_similarity(embedding_normalized, mean_embedding_jack)
            print(f"[{current_process().name}] Cosine Similarity với Jack: {similarity}")

            # Quyết định nhãn và màu sắc
            label = ""
            color = (0, 0, 255)  # Đỏ
            if similarity > 0:
                label = f"Jack ({similarity:.2f})"
                color = (0, 255, 0)  # Xanh lá
            else:
                label = f"Unknown ({similarity:.2f})"
                color = (0, 0, 255)  # Đỏ

            # Lấy vị trí khuôn mặt từ embedding_dict nếu có
            region = None
            if 'region' in embedding_dict:
                region = embedding_dict['region']
            elif 'facial_area' in embedding_dict:
                region = embedding_dict['facial_area']

            if region:
                # Kiểm tra xem 'x','y','w','h' hoặc 'left','top','right','bottom' có tồn tại
                if all(k in region for k in ('x', 'y', 'w', 'h')):
                    top = int(region['y'])
                    left = int(region['x'])
                    bottom = int(region['y'] + region['h'])
                    right = int(region['x'] + region['w'])
                elif all(k in region for k in ('left', 'top', 'right', 'bottom')):
                    top = int(region['top'])
                    left = int(region['left'])
                    bottom = int(region['bottom'])
                    right = int(region['right'])
                else:
                    print(f"[{current_process().name}] Thông tin vị trí khuôn mặt không đầy đủ.")
                    top, left, bottom, right = 0, 0, frame.shape[0], frame.shape[1]
            else:
                # Nếu không có thông tin vị trí, đặt mặc định toàn bộ khung hình
                top, left, bottom, right = 0, 0, frame.shape[0], frame.shape[1]

            # Đảm bảo các tọa độ nằm trong phạm vi khung hình
            top = max(0, top)
            left = max(0, left)
            bottom = min(frame.shape[0], bottom)
            right = min(frame.shape[1], right)

            print(f"[{current_process().name}] Vẽ rectangle tại: Top={top}, Left={left}, Bottom={bottom}, Right={right}")

            # Vẽ khung hình và nhãn
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    else:
        print(f"[{current_process().name}] Không phát hiện khuôn mặt trong hình ảnh này.")

    # Nếu có đường dẫn lưu hình ảnh đã xử lý, lưu lại
    if output_image:
        cv2.imwrite(output_image, frame)
        print(f"[{current_process().name}] Đã lưu hình ảnh đã xử lý tại: {output_image}")

    return frame

def process_list_images(image_files, mean_embedding_jack, scaler, output_folder):
    """
    Xử lý danh sách hình ảnh trong một tiến trình riêng biệt.
    
    :param image_files: Danh sách đường dẫn tới các hình ảnh cần xử lý.
    :param mean_embedding_jack: Embedding trung bình của "Jack".
    :param threshold_cosine: Ngưỡng cosine similarity để xác định "Jack".
    :param scaler: Đối tượng scaler đã được tải từ file.
    :param output_folder: Thư mục để lưu hình ảnh đã xử lý.
    """
    # Cấu hình GPU trong mỗi tiến trình
    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"[{current_process().name}] Tìm thấy {len(physical_devices)} GPU.")
    if len(physical_devices) > 0:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f'[{current_process().name}] Sử dụng GPU:', physical_devices[0])
        except Exception as e:
            print(f"[{current_process().name}] Lỗi khi cấu hình GPU: {e}")
    else:
        print(f"[{current_process().name}] Không tìm thấy GPU, sử dụng CPU.")

    for idx, image_path in enumerate(image_files, start=1):
        print(f"\n[{current_process().name}] Xử lý hình ảnh {idx}/{len(image_files)}: {image_path}")
        # Tạo tên file cho image đã xử lý
        image_name = os.path.basename(image_path)
        name, ext = os.path.splitext(image_name)
        output_image = os.path.join(output_folder, f"{name}_processed{ext}")

        # Gọi hàm để xử lý hình ảnh
        processed_frame = process_image(image_path, mean_embedding_jack,  scaler, output_image=output_image)

        if processed_frame is not None:
            # Không cần hiển thị hình ảnh trong đa tiến trình
            pass

@app.post("/process_faces")
def process_faces(request: ProcessFacesRequest):
    """
    Endpoint để nhận danh sách đường dẫn hình ảnh và task_id, sau đó thực hiện nhận diện khuôn mặt.
    
    :param request: Yêu cầu chứa task_id và danh sách image_paths.
    :return: Thông báo hoàn thành xử lý.
    """
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

    PROCESSES = 40
    if len(image_paths) < PROCESSES:
        PROCESSES = len(image_paths)
    split_images = np.array_split(image_paths, PROCESSES)
    list_process = []

    for i in range(PROCESSES):
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