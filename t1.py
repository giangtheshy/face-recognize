import datetime
import cv2
import os
import numpy as np
from threading import Thread
from queue import Queue
import logging
import threading
import subprocess  # Import subprocess to execute ffmpeg command
import time  # Import time to measure time

# Configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths to the Caffe model files for face detection
configFile = "./model/deploy.prototxt"
modelFile = "./model/res10_300x300_ssd_iter_140000.caffemodel"

# Function to detect faces in each thread
def detect_faces(detection_queue):
    """
    Function to process frame paths from the queue for face detection.
    Each thread will take a frame path from the queue, read the frame, perform detection, and save the results.
    """
    try:
        # Load the cv2 DNN face detection model in each thread to avoid conflicts
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        logging.info(f"{threading.current_thread().name}: Loaded face detection model successfully.")
    except Exception as e:
        logging.error(f"{threading.current_thread().name}: Error loading face detection model: {e}")
        return

    while True:
        item = detection_queue.get()
        if item is None:
            # Signal to terminate processing for this thread
            logging.info(f"{threading.current_thread().name}: Received termination signal.")
            detection_queue.task_done()
            break
        task_id, current_time_sec, frame_path = item

        try:
            # Start timing for reading the frame
            start_read_time = time.time()

            # Read the frame from the path
            frame = cv2.imread(frame_path)
            if frame is None:
                logging.warning(f"{threading.current_thread().name}: Unable to read frame: {frame_path}")
                detection_queue.task_done()
                continue

            # Get frame dimensions
            (h, w) = frame.shape[:2]

            # Prepare the frame for the DNN model
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))

            # Set the input to the net
            net.setInput(blob)
            detections = net.forward()

            # Directory to save faces
            faces_dir = os.path.join('tasks', task_id, 'faces')
            os.makedirs(faces_dir, exist_ok=True)
            face_index = 0  # Index of the face in the current second

            # Loop over the detections
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # Only consider detections with confidence greater than the threshold
                if confidence > 0.5:
                    # Compute the coordinates of the bounding box
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")

                    # Ensure the coordinates are within the frame
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w - 1, x2)
                    y2 = min(h - 1, y2)

                    # Extract the face
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue

                    # Save the face image
                    face_filename = os.path.join(faces_dir, f"{int(current_time_sec)}_{face_index}.png")
                    cv2.imwrite(face_filename, face)
                    face_index += 1

        except Exception as e:
            logging.error(f"Error processing frame at second {current_time_sec} for task_id {task_id}: {e}")

        finally:
            detection_queue.task_done()
            read_time = time.time() - start_read_time
            logging.info(f"{threading.current_thread().name}: Time to end detect frame {frame_path}: {read_time:.4f} seconds")

# Function to extract frames from video using ffmpeg and put paths into the queue
def extract_frames(video_path, task_id, detection_queue):
    """
    Function to extract frames from video using ffmpeg and put frame paths into the queue for processing.
    Each thread will process a separate video.
    """
    try:
        start_time = datetime.datetime.now()

        # Directory to store frames
        frames_dir = os.path.join('tasks', task_id, 'frames')
        os.makedirs(frames_dir, exist_ok=True)

        # Use ffmpeg to extract frames at 1 fps
        ffmpeg_command = [
            'ffmpeg',
            '-i', video_path,
            '-vf', 'fps=1',
            os.path.join(frames_dir, 'frame_%d.jpg')
        ]

        # Start timing for frame extraction
        start_extract_time = time.time()

        # Execute ffmpeg command and capture stderr
        logging.info(f"Starting frame extraction for video: {video_path}")
        ffmpeg_process = subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if ffmpeg_process.returncode != 0:
            logging.error(f"ffmpeg error for video {video_path}:\n{ffmpeg_process.stderr}")
            return
        logging.info(f"Completed frame extraction for video: {video_path}")

        extract_time = time.time() - start_extract_time
        logging.info(f"Time to extract frames for video {video_path}: {extract_time:.2f} seconds")

        # Start timing for reading and queuing
        start_queue_time = time.time()

        # List of extracted frames
        frame_files = sorted(os.listdir(frames_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))

        current_time_sec = 0

        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            # Put the frame path into the queue for processing
            detection_queue.put((task_id, current_time_sec, frame_path))
            current_time_sec += 1

        queue_time = time.time() - start_queue_time
        logging.info(f"Time to read and queue frames for video {video_path}: {queue_time:.2f} seconds")

        # Record the processing time for the video
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logging.info(f"Completed processing video: {video_path}. Processing time: {processing_time:.2f} seconds.")

    except Exception as e:
        logging.error(f"Error processing video {video_path}: {e}")

# Main function
def main():
    """
    Main function to process all videos in the directory.
    """
    video_folder_path = "./tracking/v2"
    time_start = datetime.datetime.now().timestamp()

    # Check if the model files exist
    if not os.path.exists(configFile) or not os.path.exists(modelFile):
        logging.error(f"Face detection model files do not exist at: {configFile} and {modelFile}")
        logging.info("Please download the Caffe face detection model and place it in the specified path.")
        return

    # Get the list of all videos in the directory
    video_files = [f for f in os.listdir(video_folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv'))]

    if not video_files:
        logging.warning("No videos found in the directory.")
        return

    num_detection_threads = 5  # Number of threads for face detection
    detection_queue = Queue()

    # Initialize face detection threads
    detection_threads = []
    for i in range(num_detection_threads):
        t = Thread(target=detect_faces, args=(detection_queue,), name=f"Detection-Thread-{i+1}")
        t.start()
        detection_threads.append(t)
        logging.info(f"Started face detection thread {i+1}/{num_detection_threads}.")

    # Initialize frame extraction threads (each thread processes one video)
    extraction_threads = []
    for video_file in video_files:
        video_path = os.path.join(video_folder_path, video_file)
        task_id = os.path.splitext(video_file)[0]  # Use file name without extension as task_id

        # Start a frame extraction thread for this video
        t = Thread(target=extract_frames, args=(video_path, task_id, detection_queue), name=f"Extractor-{task_id}")
        t.start()
        extraction_threads.append(t)
        logging.info(f"Started frame extraction thread for video: {video_file}")

    # Wait until all frame extraction threads complete
    for t in extraction_threads:
        t.join()
        logging.info(f"{t.name} has completed.")

    # Wait until all frames have been processed
    detection_queue.join()

    # After all frames have been processed, send termination signal to detection threads
    for _ in detection_threads:
        detection_queue.put(None)

    # Wait for detection threads to complete
    for t in detection_threads:
        t.join()
        logging.info(f"{t.name} has terminated.")

    time_end = datetime.datetime.now().timestamp()
    total_processing_time = time_end - time_start

    logging.info(f"Completed processing all videos. Total time: {total_processing_time:.2f} seconds.")

if __name__ == "__main__":
    main()
