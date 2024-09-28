import logging
from queue import Queue
import subprocess
import threading
import time
from pydantic import BaseModel
import uuid
from deepface import DeepFace
import joblib
import dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import cv2
import datetime
from multiprocessing import current_process

dotenv.load_dotenv()

# Configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
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
        logging.info(f"File {file} không tồn tại. Vui lòng kiểm tra lại quá trình cài đặt.")
        exit()

# Tải mean_embedding và threshold
mean_embedding_jack = np.load(mean_embedding_file)

# Tải scaler
scaler = joblib.load(scaler_file)


class RecognitionRequest(BaseModel):
    task_id: str
    video_path: str


recognize_queue = Queue()
extract_queue = Queue()


def compute_cosine_similarity(embedding, mean_embedding):
    """
    Tính toán cosine similarity giữa embedding của khuôn mặt hiện tại và mean_embedding.
    
    :param embedding: Embedding của khuôn mặt hiện tại.
    :param mean_embedding: Embedding trung bình (mean_embedding_jack).
    :return: Giá trị cosine similarity.
    """
    return cosine_similarity([embedding], [mean_embedding])[0][0]


def process_image(frame, mean_embedding_jack,  scaler, output_image=None):
    """
    Xử lý một hình ảnh để phát hiện và nhận diện khuôn mặt.
    
    :param image_path: Đường dẫn tới hình ảnh cần xử lý.
    :param mean_embedding_jack: Embedding trung bình của "Jack".
    :param threshold_cosine: Ngưỡng cosine similarity để xác định "Jack".
    :param scaler: Scaler đã được huấn luyện.
    :param output_image: Đường dẫn để lưu hình ảnh đã xử lý. Nếu None, không lưu.
    :return: Hình ảnh đã xử lý (cv2 image).
    """

    # logging.info(f"[{current_process().name}] Chuyển đổi hình ảnh từ BGR sang RGB...")
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Tạo thư mục tạm thời để lưu các khung hình
    temp_dir = './temp_frames'
    os.makedirs(temp_dir, exist_ok=True)
    # logging.info(f"[{current_process().name}] Thư mục tạm thời được tạo tại: {temp_dir}")

    # Lưu khung hình vào tệp tạm thời
    temp_img_path = os.path.join(temp_dir, f"{uuid.uuid4()}.jpg")
    # logging.info(f"[{current_process().name}] Đang lưu hình ảnh tạm thời tại: {temp_img_path}")
    cv2.imwrite(temp_img_path, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))

    try:
        # logging.info(f"[{current_process().name}] Đang sử dụng DeepFace để phát hiện và trích xuất embedding...")
        # Sử dụng DeepFace để phát hiện và trích xuất embedding
        start_time = datetime.datetime.now().timestamp()
        results = DeepFace.represent(img_path=temp_img_path, enforce_detection=True, model_name=MODEL, detector_backend=BACKEND)
        end_time = datetime.datetime.now().timestamp()
        logging.info(f"[{current_process().name}] DeepFace.represent hoàn thành trong {end_time - start_time:.2f} giây")
    except Exception as e:
        logging.info(f"[{current_process().name}] Lỗi khi phát hiện khuôn mặt trong: {e}")
        results = []

    # Xóa tệp tạm sau khi sử dụng
    if os.path.exists(temp_img_path):
        logging.info(f"[{current_process().name}] Đang xóa tệp tạm: {temp_img_path}")
        os.remove(temp_img_path)

    # 'results' có thể là list hoặc dict tùy thuộc vào số khuôn mặt
    if isinstance(results, dict):
        results = [results]  # Chuyển đổi thành list nếu chỉ có một khuôn mặt

    logging.info(f"[{current_process().name}] Số khuôn mặt được phát hiện: {len(results)}")

    if results:
        for idx, embedding_dict in enumerate(results):
            # logging.info(f"[{current_process().name}] Đang xử lý khuôn mặt thứ {idx + 1}")
            embedding = embedding_dict.get('embedding', None)
            if embedding is None:
                logging.info(f"[{current_process().name}] Không có embedding, bỏ qua khuôn mặt này.")
                continue

            # Chuẩn hóa và normalize embedding
            embedding_scaled = scaler.transform([embedding])[0]
            embedding_normalized = embedding_scaled / np.linalg.norm(embedding_scaled)

            # Tính cosine similarity
            similarity = compute_cosine_similarity(embedding_normalized, mean_embedding_jack)
            # logging.info(f"[{current_process().name}] Cosine Similarity với Jack: {similarity}")

            # Quyết định nhãn và màu sắc
            label = ""
            color = (0, 0, 255)  # Đỏ
            if similarity > 0:
                label = f""
                color = (0, 255, 0)  # Xanh lá
            else:
                label = f""
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
                    logging.info(f"[{current_process().name}] Thông tin vị trí khuôn mặt không đầy đủ.")
                    top, left, bottom, right = 0, 0, frame.shape[0], frame.shape[1]
            else:
                # Nếu không có thông tin vị trí, đặt mặc định toàn bộ khung hình
                top, left, bottom, right = 0, 0, frame.shape[0], frame.shape[1]
                logging.info(f"[{current_process().name}] Không có thông tin vị trí khuôn mặt, sử dụng toàn bộ khung hình.")

            # Đảm bảo các tọa độ nằm trong phạm vi khung hình
            top = max(0, top)
            left = max(0, left)
            bottom = min(frame.shape[0], bottom)
            right = min(frame.shape[1], right)

            # logging.info(f"[{current_process().name}] Vẽ rectangle tại: Top={top}, Left={left}, Bottom={bottom}, Right={right}")

            # Vẽ khung hình và nhãn
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    else:
        logging.info(f"[{current_process().name}] Không phát hiện khuôn mặt trong hình ảnh.")

    # Nếu có đường dẫn lưu hình ảnh đã xử lý, lưu lại
    if output_image:
        cv2.imwrite(output_image, frame)
        logging.info(f"[{current_process().name}] Đã lưu hình ảnh đã xử lý tại: {output_image}")

    return frame

def process_extracted_frame():
    logging.info(f"[{current_process().name}] Bắt đầu xử lý các frame đã được trích xuất.")
    while True:

        item = extract_queue.get()
        if item is None:
            time.sleep(2)
            continue
        extract_frames(item.video_path, item.task_id, recognize_queue)

def process_recognize_frame(mean_embedding_jack, scaler):
    try:
    # Load the cv2 DNN face detection model in each thread to avoid conflicts
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        logging.info(f"{threading.current_thread().name}: Loaded face detection model successfully.")
    except Exception as e:
        logging.info(f"{threading.current_thread().name}: Error loading face detection model: {e}")
    while True:
        item = recognize_queue.get()
        if item is None:
            time.sleep(2)
            continue
        task_id, current_time_sec, frame_path = item
            # Định nghĩa thư mục đầu ra
        output_folder = os.path.join("tasks", task_id, "processed_faces")
        image_name = os.path.basename(frame_path)
        name, ext = os.path.splitext(image_name)
        output_image = os.path.join(output_folder, f"{name}_processed{ext}")

        frame = cv2.imread(frame_path)
        if frame is None:
            logging.info(f"[{current_process().name}] Không thể đọc frame: {frame_path}. Bỏ qua.")
            continue

        # Check if the frame has face
        has_face = is_has_face(net, frame)
        if has_face:
            logging.info(f"[{current_process().name}] Frame {frame_path} có khuôn mặt.")
            process_image(frame, mean_embedding_jack, scaler,output_image=output_image)
        else:
            logging.info(f"[{current_process().name}] Frame {frame_path} không có khuôn mặt.")


# Function to extract frames from video using ffmpeg and put paths into the queue
def extract_frames(video_path, task_id, recognize_queue):
    """
    Function to extract frames from video using ffmpeg and put frame paths into the queue for processing.
    Each thread will process a separate video.
    """
    try:
        start_time = datetime.datetime.now()

        # Directory to store frames
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frames_dir = os.path.join('tasks', task_id, 'frames',video_name)
        os.makedirs(frames_dir, exist_ok=True)

        # Use ffmpeg to extract frames at 1 fps
        ffmpeg_command = [
            'ffmpeg',
            '-i', video_path,
            '-vf', 'fps=1',
            os.path.join(frames_dir, 'frame_%d.jpg')
        ]

        # Execute ffmpeg command and capture stderr
        logging.info(f"Starting frame extraction for video: {video_path}")
        ffmpeg_process = subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if ffmpeg_process.returncode != 0:
            logging.info(f"ffmpeg error for video {video_path}:\n{ffmpeg_process.stderr}")
            return

        # List of extracted frames
        frame_files = sorted(os.listdir(frames_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))

        current_time_sec = 0

        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            # Put the frame path into the queue for processing
            recognize_queue.put((task_id, current_time_sec, frame_path))
            current_time_sec += 1

        # Record the processing time for the video
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logging.info(f"Completed processing video: {video_path}. Processing time: {processing_time:.2f} seconds.")

    except Exception as e:
        logging.info(f"Error processing video {video_path}: {e}")



# Function to detect faces in each thread
def is_has_face(net,frame):
    """
    Function to process frame paths from the queue for face detection.
    Each thread will take a frame path from the queue, read the frame, perform detection, and save the results.
    """

    has_face = False

    try:
        # Start timing for reading the frame
        start_read_time = time.time()

        # Prepare the frame for the DNN model
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                        (300, 300), (104.0, 177.0, 123.0))

        # Set the input to the net
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Only consider detections with confidence greater than the threshold
            if confidence > 0.5:
                has_face = True
                break

    except Exception as e:
        logging.info(f"Error processing frame : {e}")
        has_face = False

    finally:
        read_time = time.time() - start_read_time
        logging.info(f"{threading.current_thread().name}: Time to end detect frame: {read_time:.4f} seconds")
        return has_face
