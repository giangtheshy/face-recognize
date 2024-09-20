import os
import numpy as np
import cv2
from deepface import DeepFace
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

MODEL='Facenet512'
BACKEND='opencv'
def extract_embeddings(directory, model_name='Facenet'):
    embeddings = []
    for subdir in os.listdir(directory):
        print(f"Đang trích xuất embeddings cho: {subdir}")
        path = os.path.join(directory, subdir)
        if not os.path.isdir(path):
            continue

        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Không thể đọc ảnh: {img_path}")
                continue

            try:
                # Sử dụng DeepFace để phát hiện khuôn mặt và trích xuất embedding
                result = DeepFace.represent(img_path=img_path, model_name=model_name, enforce_detection=True,detector_backend=BACKEND)

                if isinstance(result, list) and len(result) > 0:
                    for embedding_dict in result:
                        embedding = embedding_dict['embedding']
                        embeddings.append(embedding)
                elif isinstance(result, dict):
                    embedding = result['embedding']
                    embeddings.append(embedding)
                else:
                    print(f"Không thể trích xuất embedding cho ảnh: {img_path}")
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {img_path}: {e}")

    return np.array(embeddings)

def main():
    # Đường dẫn đến dữ liệu huấn luyện
    data_dir = 'test'  # Thay đổi đường dẫn nếu cần

    # Trích xuất embeddings cho Jack
    embeddings_jack = extract_embeddings(data_dir, model_name=MODEL)  # Bạn có thể chọn các mô hình khác như 'VGG-Face', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace'

    if embeddings_jack.size == 0:
        raise ValueError("Không có embeddings hợp lệ cho Jack. Kiểm tra dữ liệu huấn luyện.")

    print(f"Số embeddings được trích xuất: {len(embeddings_jack)}")

    # Chuẩn hóa embeddings
    scaler = StandardScaler()
    embeddings_jack_scaled = scaler.fit_transform(embeddings_jack)

    # Normalize các embeddings sau khi chuẩn hóa
    embeddings_jack_normalized = embeddings_jack_scaled / np.linalg.norm(embeddings_jack_scaled, axis=1, keepdims=True)


    # Tính trung bình vector đặc trưng của Jack (sau khi chuẩn hóa)
    mean_embedding_jack = np.mean(embeddings_jack_normalized, axis=0)
    mean_embedding_jack_normalized = mean_embedding_jack / np.linalg.norm(mean_embedding_jack)

    # Tính Cosine Similarity cho dữ liệu huấn luyện
    similarities = cosine_similarity(embeddings_jack_normalized, mean_embedding_jack_normalized.reshape(1, -1)).flatten()

    # Xác định ngưỡng dựa trên phân vị 40% cho Cosine similarity
    threshold_cosine = np.percentile(similarities, 40)
    print(f"Threshold Cosine Similarity (40th percentile): {threshold_cosine}")

    # Lưu mean_embedding và threshold
    os.makedirs('./model', exist_ok=True)
    np.save('./model/mean_embedding_jack.npy', mean_embedding_jack_normalized)
    with open('./model/threshold_cosine.txt', 'w') as f:
        f.write(str(threshold_cosine))    
    joblib.dump(scaler, './model/scaler.pkl')
    print("Đã lưu mean_embedding, threshold và scaler thành công.")

if __name__ == "__main__":
    main()
