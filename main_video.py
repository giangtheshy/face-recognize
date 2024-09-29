import argparse
import time
import os
from multiprocessing import Process, Lock
from video import process_video

def run_process(input_folder, output_file, model, device, faces_folder, video_files, lock):
    process_video(input_folder, output_file, model, device, faces_folder, video_files, lock)

def main():
    # Cấu hình các tham số từ dòng lệnh
    parser = argparse.ArgumentParser(description='Process video files for face detection using YOLOv8.')
    parser.add_argument('--input_folder', type=str, default='yolo-video', help='Thư mục chứa video đầu vào')
    parser.add_argument('--output_file', type=str, default='output.txt', help='File để lưu thông số')
    parser.add_argument('--model', type=str, default='yolov8n-face.pt', help='Đường dẫn đến mô hình YOLOv8')
    parser.add_argument('--faces_folder', type=str, default='faces', help='Thư mục lưu các khuôn mặt đã phát hiện')

    args = parser.parse_args()

    # Bắt đầu đếm thời gian
    start_time = time.time()

    # Tạo danh sách các quá trình
    processes = []
    gpus = ['cuda:0', 'cuda:1', 'cuda:2']
    lock = Lock()  # Khóa để đồng bộ hóa ghi file

    # Lấy danh sách video trong thư mục input
    video_files = [f for f in os.listdir(args.input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    num_videos = len(video_files)
    videos_per_gpu = num_videos // len(gpus) + (num_videos % len(gpus) > 0)  # Phân chia video cho mỗi GPU

    # Tạo các quá trình cho từng GPU
    for i, gpu in enumerate(gpus):
        start_index = i * videos_per_gpu
        end_index = start_index + videos_per_gpu
        assigned_videos = video_files[start_index:end_index]

        if assigned_videos:
            p = Process(target=run_process, args=(args.input_folder, args.output_file, args.model, gpu, args.faces_folder, assigned_videos, lock))
            processes.append(p)
            p.start()

    # Đợi tất cả các quá trình hoàn thành
    for p in processes:
        p.join()

    # Kết thúc đếm thời gian
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Tổng thời gian xử lý: {total_time:.2f} giây')


if __name__ == "__main__":
	main()
