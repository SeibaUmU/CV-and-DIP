import cv2
from ultralytics import YOLO

# 1. Load the YOLO model
# Sử dụng đường dẫn tuyệt đối mà bạn đã cấu hình thành công trước đó
model = YOLO(r"D:\TaiLieu\XLA\XLA_TH\CV-and-DIP\TH\runs\detect\train6\weights\best.pt")

# 2. Danh sách đường dẫn tới 2 tấm ảnh của bạn
# Hãy thay đổi tên file 'image1.jpg' và 'image2.jpg' cho đúng với thực tế trong thư mục của bạn
image_paths = [
    r"D:\TaiLieu\XLA\XLA_TH\CV-and-DIP\TH\Buoi1\YOLO\dataset\frame19.jpg", 
    r"D:\TaiLieu\XLA\XLA_TH\CV-and-DIP\TH\Buoi1\YOLO\dataset\frame35.jpg"
]

# 3. Lặp qua từng ảnh trong danh sách
for path in image_paths:
    # Đọc ảnh từ đường dẫn
    frame = cv2.imread(path)

    if frame is not None:
        # Chạy YOLO inference trên tấm ảnh
        # Bạn có thể giữ device=0 nếu muốn dùng GPU trên máy MSI Katana của mình
        results = model(frame, device=0)

        # Vẽ kết quả lên ảnh
        annotated_frame = results[0].plot()

        # Hiển thị ảnh đã nhận diện
        cv2.imshow("YOLO Detection - Image", annotated_frame)

        print(f"Đang hiển thị ảnh: {path}")
        print("Nhấn phím bất kỳ để xem ảnh tiếp theo...")
        
        # Đợi nhấn phím bất kỳ để chuyển sang ảnh sau hoặc đóng cửa sổ
        cv2.waitKey(0) 
    else:
        print(f"Không thể tìm thấy hoặc mở ảnh tại: {path}")

# Đóng tất cả cửa sổ khi hoàn thành
cv2.destroyAllWindows()