import cv2
import os
from ultralytics import YOLO

def process_video(input_folder, output_file, model_path, device, faces_folder, video_files, lock):
    model = YOLO(model_path)  # Khởi tạo mô hình YOLOv8

    # Tạo thư mục faces nếu chưa tồn tại
    if not os.path.exists(faces_folder):
        os.makedirs(faces_folder)

    for filename in video_files:
        video_path = os.path.join(input_folder, filename)
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps)  # Số frame mỗi giây

        current_frame = 0
        face_count = 0  # Đếm số khuôn mặt đã lưu

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Kiểm tra kết thúc video

            if current_frame % frame_interval == 0:
                results = model(frame, device=device)

                # Tính thời gian giây
                time_in_seconds = current_frame / fps

                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ bounding box

                        # Ghi thông tin vào file output
                        with lock:  # Sử dụng khóa để đồng bộ hóa ghi file
                            with open(output_file, 'a') as f:
                                f.write(f'Video: {filename}, Time: {time_in_seconds:.2f} s, BBox: ({x1}, {y1}), ({x2}, {y2})\n')

                        # Lưu khuôn mặt vào thư mục faces
                        if (y2 - y1) > 0 and (x2 - x1) > 0:
                            face_image = frame[y1:y2, x1:x2]
                            face_filename = os.path.join(faces_folder, f'face_{face_count}_{filename}')
                            cv2.imwrite(face_filename + '.jpg', face_image)
                            face_count += 1

            current_frame += 1

        cap.release()  # Giải phóng video

    print(f'Thông số khuôn mặt đã được lưu vào {output_file}')
    print(f'Khuôn mặt đã được lưu vào thư mục {faces_folder}')

