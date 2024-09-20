import os
import datetime
from multiprocessing import Process, current_process
import cv2
import numpy as np
import joblib
import uuid  # Thư viện để tạo tên file duy nhất

from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def compute_cosine_similarity(embedding, mean_embedding):
    """
    Tính toán cosine similarity giữa embedding của khuôn mặt hiện tại và mean_embedding.
    
    :param embedding: Embedding của khuôn mặt hiện tại.
    :param mean_embedding: Embedding trung bình (mean_embedding_jack).
    :return: Giá trị cosine similarity.
    """
    return cosine_similarity([embedding], [mean_embedding])[0][0]

BACKEND = 'mtcnn'
MODEL = 'Facenet512'

def combine_images(image_paths, output_path, columns=3, rows=3, padding=5, background_color=(255, 255, 255)):
    """
    Kết hợp một nhóm hình ảnh thành một hình ảnh lớn với bố cục theo lưới.
    
    :param image_paths: Danh sách đường dẫn tới các hình ảnh.
    :param output_path: Đường dẫn lưu trữ hình ảnh kết quả.
    :param columns: Số cột trong lưới.
    :param rows: Số dòng trong lưới.
    :param padding: Khoảng cách giữa các hình ảnh.
    :param background_color: Màu nền cho hình ảnh kết quả.
    """
    if not image_paths:
        print("Không có hình ảnh nào để kết hợp.")
        return

    # Mở các ảnh và lưu vào danh sách
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            images.append(img)
        except Exception as e:
            print(f"Không thể mở hình ảnh {path}: {e}")

    if not images:
        print("Không có hình ảnh hợp lệ để kết hợp.")
        return

    # Tính toán kích thước tối đa cho mỗi ô ảnh
    widths, heights = zip(*(img.size for img in images))
    max_width = max(widths)
    max_height = max(heights)

    # Tính số lượng hình ảnh cần kết hợp
    num_images = len(images)
    max_images = columns * rows

    if num_images > max_images:
        print(f"Số lượng hình ảnh vượt quá giới hạn ({max_images}). Chỉ sử dụng {max_images} hình ảnh đầu tiên.")
        images = images[:max_images]
        num_images = max_images

    # Tính số cột và dòng dựa trên tham số đã cho
    total_width = columns * max_width + (columns - 1) * padding
    total_height = rows * max_height + (rows - 1) * padding
    new_image = Image.new('RGB', (total_width, total_height), color=background_color)

    # Dán từng ảnh vào ảnh mới
    for index, img in enumerate(images):
        if index >= max_images:
            break  # Đảm bảo không vượt quá số lượng hình ảnh tối đa

        # Tính vị trí x, y cho mỗi ảnh
        x = (index % columns) * (max_width + padding)
        y = (index // columns) * (max_height + padding)

        # Thay đổi kích thước ảnh để phù hợp với ô
        img_resized = img.resize((max_width, max_height), Image.LANCZOS)

        # Dán ảnh vào vị trí tương ứng
        new_image.paste(img_resized, (x, y))

    # Lưu ảnh kết quả
    new_image.save(output_path)
    print(f"Đã tạo ảnh tổng hợp tại {output_path}")

def process_image(image_path, mean_embedding_jack, threshold_cosine, scaler, output_image=None):
    """
    Xử lý một hình ảnh để phát hiện và nhận diện khuôn mặt.
    
    :param image_path: Đường dẫn tới hình ảnh cần xử lý.
    :param mean_embedding_jack: Embedding trung bình của "Jack".
    :param threshold_cosine: Ngưỡng cosine similarity để xác định "Jack".
    :param scaler: Scaler đã được huấn luyện.
    :param output_image: Đường dẫn để lưu hình ảnh đã xử lý. Nếu None, không lưu.
    :return: Hình ảnh đã xử lý (cv2 image).
    """
    from deepface import DeepFace
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
        results = DeepFace.represent(img_path=temp_img_path, enforce_detection=True, model_name=MODEL, detector_backend=BACKEND, max_faces=50)
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
                # label = f"Jack ({similarity:.2f})"
                color = (0, 255, 0)  # Xanh lá
            else:
                # label = f"Unknown ({similarity:.2f})"
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

def create_video_from_images(image_folder, output_video, frame_rate=1):
    """
    Tạo video MP4 từ các hình ảnh trong thư mục.
    
    :param image_folder: Thư mục chứa các hình ảnh.
    :param output_video: Đường dẫn lưu trữ video kết quả.
    :param frame_rate: Tốc độ khung hình (frames per second).
    """
    print(f"Đang tạo video từ các hình ảnh trong: {image_folder}")

    # Lấy danh sách các hình ảnh, sắp xếp theo thứ tự tên file
    image_files = sorted([
        os.path.join(image_folder, img) for img in os.listdir(image_folder)
        if img.lower().endswith(('jpg', 'png', 'jpeg'))
    ])

    if not image_files:
        print(f"Không tìm thấy hình ảnh nào trong thư mục: {image_folder}")
        return

    # Đọc hình ảnh đầu tiên để lấy kích thước
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print(f"Không thể đọc hình ảnh đầu tiên: {image_files[0]}")
        return

    height, width, layers = first_frame.shape
    size = (width, height)

    # Khởi tạo VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec cho MP4
    out = cv2.VideoWriter(output_video, fourcc, frame_rate, size)

    for idx, image_path in enumerate(image_files, start=1):
        print(f"Đang thêm hình ảnh {idx}/{len(image_files)} vào video: {image_path}")
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Không thể đọc hình ảnh: {image_path}. Bỏ qua.")
            continue

        # Kiểm tra kích thước hình ảnh, nếu khác, thay đổi kích thước
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size)
            print(f"Thay đổi kích thước hình ảnh: {image_path}")

        out.write(frame)

    out.release()
    print(f"Đã tạo video tại: {output_video}")

def process_list_images(image_files, mean_embedding_jack, threshold_cosine, scaler_file_path, output_folder):
    """
    Xử lý danh sách hình ảnh trong một tiến trình riêng biệt.
    
    :param image_files: Danh sách đường dẫn tới các hình ảnh cần xử lý.
    :param mean_embedding_jack: Embedding trung bình của "Jack".
    :param threshold_cosine: Ngưỡng cosine similarity để xác định "Jack".
    :param scaler_file_path: Đường dẫn tới file scaler.pkl.
    :param output_folder: Thư mục để lưu hình ảnh đã xử lý.
    """
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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

    # Tải lại scaler từ file
    scaler = joblib.load(scaler_file_path)

    for idx, image_path in enumerate(image_files, start=1):
        print(f"\n[{current_process().name}] Xử lý hình ảnh {idx}/{len(image_files)}: {image_path}")
        # Tạo tên file cho image đã xử lý
        image_name = os.path.basename(image_path)
        name, ext = os.path.splitext(image_name)
        output_image = os.path.join(output_folder, f"{name}_processed{ext}")

        # Gọi hàm để xử lý hình ảnh
        processed_frame = process_image(image_path, mean_embedding_jack, threshold_cosine, scaler, output_image=output_image)

        if processed_frame is not None:
            # Không cần hiển thị hình ảnh trong đa tiến trình
            pass

def main():
    print("Bắt đầu chương trình...")

    # Đường dẫn đến các file cần thiết
    model_dir = './model'
    mean_embedding_file = os.path.join(model_dir, 'mean_embedding_jack.npy')
    threshold_file = os.path.join(model_dir, 'threshold_cosine.txt')
    scaler_file = os.path.join(model_dir, 'scaler.pkl')

    # Kiểm tra sự tồn tại của các file cần thiết
    required_files = [mean_embedding_file, threshold_file, scaler_file]
    for file in required_files:
        if not os.path.exists(file):
            print(f"File {file} không tồn tại. Vui lòng kiểm tra lại quá trình huấn luyện.")
            return

    print("Đang tải các tệp mô hình...")

    # Tải mean_embedding và threshold
    mean_embedding_jack = np.load(mean_embedding_file)
    with open(threshold_file, 'r') as f:
        threshold_cosine = float(f.read())

    # Không cần tải scaler ở đây
    scaler_file_path = scaler_file  # Đường dẫn tới file scaler.pkl

    # Normalize mean_embedding_jack nếu chưa normalize
    if not np.isclose(np.linalg.norm(mean_embedding_jack), 1.0):
        mean_embedding_jack = mean_embedding_jack / np.linalg.norm(mean_embedding_jack)

    print(f"Threshold Cosine Similarity: {threshold_cosine}")

    # Đường dẫn đến thư mục chứa các image_combined
    input_folder = './videos/3x'  # Thay đường dẫn tới thư mục chứa các combined images của bạn
    output_processed_folder = './output'  # Thay đường dẫn tới thư mục lưu hình ảnh đã xử lý
    os.makedirs(output_processed_folder, exist_ok=True)

    # Lấy danh sách các image_combined từ thư mục
    image_files = sorted([
        os.path.join(input_folder, img) for img in os.listdir(input_folder)
        if img.lower().endswith(('jpg', 'png', 'jpeg'))
    ])

    if not image_files:
        print(f"Không tìm thấy hình ảnh nào trong thư mục: {input_folder}")
        return

    print(f"Tổng số hình ảnh cần xử lý: {len(image_files)}")

    # Bắt đầu đo thời gian
    start_time = datetime.datetime.now().timestamp()
    PROCESSES = 40
    split_images = np.array_split(image_files, PROCESSES)
    list_process = []
    for i in range(PROCESSES):
        p = Process(target=process_list_images, args=(split_images[i], mean_embedding_jack, threshold_cosine, scaler_file_path, output_processed_folder))
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

    # Giải phóng tài nguyên
    cv2.destroyAllWindows()

    print("Đã hoàn tất xử lý tất cả hình ảnh.")

    # Tạo video từ các hình ảnh đã xử lý
    output_video = './face_recognition.mp4'  # Đường dẫn tới video đầu ra
    frame_rate = 1  # Thay đổi tốc độ khung hình nếu cần (frames per second)
    create_video_from_images(output_processed_folder, output_video, frame_rate=frame_rate)

if __name__ == "__main__":
    main()
