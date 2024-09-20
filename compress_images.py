import datetime
from PIL import Image
import math
import os

def combine_images(image_paths, output_path, columns=3, rows=3, padding=5, background_color=(255, 255, 255), output_width=240, output_height=272):
    """
    Kết hợp một nhóm hình ảnh thành một hình ảnh lớn với bố cục theo lưới.

    :param image_paths: Danh sách đường dẫn tới các hình ảnh.
    :param output_path: Đường dẫn lưu trữ hình ảnh kết quả.
    :param columns: Số cột trong lưới.
    :param rows: Số dòng trong lưới.
    :param padding: Khoảng cách giữa các hình ảnh.
    :param background_color: Màu nền cho hình ảnh kết quả.
    :param output_width: Chiều rộng của ảnh kết quả.
    :param output_height: Chiều cao của ảnh kết quả.
    """
    if not image_paths:
        print("Không có hình ảnh nào để kết hợp.")
        return

    # Mở các ảnh và lưu vào danh sách
    images = [Image.open(path) for path in image_paths]

    # Tính số lượng hình ảnh cần kết hợp
    num_images = len(images)
    max_images = columns * rows

    if num_images > max_images:
        print(f"Số lượng hình ảnh vượt quá giới hạn ({max_images}). Chỉ sử dụng {max_images} hình ảnh đầu tiên.")
        images = images[:max_images]
        num_images = max_images

    # Tính kích thước mỗi ô ảnh
    total_padding_width = (columns - 1) * padding
    total_padding_height = (rows - 1) * padding
    cell_width = (output_width - total_padding_width) // columns
    cell_height = (output_height - total_padding_height) // rows

    if cell_width <= 0 or cell_height <= 0:
        print("Kích thước ô ảnh không hợp lệ. Vui lòng kiểm tra lại kích thước đầu ra và tham số padding.")
        return

    # Tạo ảnh mới với kích thước cố định
    new_image = Image.new('RGB', (output_width, output_height), color=background_color)

    # Dán từng ảnh vào ảnh mới
    for index, img in enumerate(images):
        if index >= max_images:
            break  # Đảm bảo không vượt quá số lượng hình ảnh tối đa

        # Tính vị trí x, y cho mỗi ảnh
        x = (index % columns) * (cell_width + padding)
        y = (index // columns) * (cell_height + padding)

        # Thay đổi kích thước ảnh để phù hợp với ô
        img_resized = img.resize((cell_width, cell_height), Image.LANCZOS)

        # Dán ảnh vào vị trí tương ứng
        new_image.paste(img_resized, (x, y))

    # Lưu ảnh kết quả
    new_image.save(output_path)
    print(f"Đã tạo ảnh tổng hợp tại {output_path}")

def main():
    # Sử dụng hàm
    image_folder = './test/faces'
    output_folder = './videos/3x'
    os.makedirs(output_folder, exist_ok=True)

    # Lấy danh sách đường dẫn các ảnh, sắp xếp để đảm bảo thứ tự nhất định
    image_files = sorted([
        os.path.join(image_folder, img) for img in os.listdir(image_folder)
        if img.lower().endswith(('jpg', 'png', 'jpeg'))
    ])

    # Tham số cho lưới ảnh
    columns = 5  # Số cột
    rows = 5     # Số dòng
    padding = 10  # Khoảng cách giữa các hình ảnh
    output_width = 240
    output_height = 272
    max_images_per_grid = columns * rows

    # Tính số lượng grid cần tạo
    total_images = len(image_files)
    total_grids = math.ceil(total_images / max_images_per_grid)

    print(f"Tổng số hình ảnh: {total_images}")
    print(f"Sẽ tạo {total_grids} hình ảnh lớn với mỗi hình chứa tối đa {max_images_per_grid} hình.")

    # Bắt đầu đo thời gian
    start_time = datetime.datetime.now().timestamp()

    for grid_index in range(total_grids):
        start_idx = grid_index * max_images_per_grid
        end_idx = start_idx + max_images_per_grid
        current_batch = image_files[start_idx:end_idx]

        # Tạo tên file cho image lớn
        output_image = os.path.join(output_folder, f"combined_image_{grid_index + 1}.jpg")

        # Gọi hàm để kết hợp ảnh với kích thước cố định
        combine_images(
            current_batch,
            output_image,
            columns=columns,
            rows=rows,
            padding=padding,
            output_width=output_width,
            output_height=output_height
        )

    # Kết thúc đo thời gian
    end_time = datetime.datetime.now().timestamp()
    print(f"Thời gian thực thi: {end_time - start_time} giây")

if __name__ == "__main__":
    main()
