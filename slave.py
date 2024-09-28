from fastapi import FastAPI
import threading
from slave_helper import RecognitionRequest, mean_embedding_jack, scaler,\
     NUM_THREADS, mean_embedding_jack, scaler,extract_queue,process_extracted_frame,process_recognize_frame

app = FastAPI()


@app.post("/recognize_faces")
def recognize_faces(request: RecognitionRequest):
    extract_queue.put(request)



for i in range(3):
    t = threading.Thread(target=process_extracted_frame, args=(),name=f"Extraction-Thread-{i+1}")
    t.start()

for i in range(NUM_THREADS):
    t = threading.Thread(target=process_recognize_frame, args=(mean_embedding_jack, scaler),name=f"Recognition-Thread-{i+1}")
    t.start()