import cv2
import os

def FrameCapture(path, target_frames=2000):
    # 1. Mở video và kiểm tra
    vidObj = cv2.VideoCapture(path)
    if not vidObj.isOpened():
        print("Không thể mở video. Kiểm tra lại đường dẫn!")
        return

    # 2. Tính toán tổng số frame và bước nhảy (step)
    total_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // target_frames)
    
    print(f"Tổng số frame trong video: {total_frames}")
    print(f"Bước nhảy dự kiến: {step} (Cứ {step} frame lấy 1 ảnh)")

    # 3. Tạo thư mục lưu ảnh nếu chưa có
    output_folder = "dataset\\anh"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    count = 0
    saved_count = 0

    # 4. Vòng lặp cắt frame
    while True:
        success, image = vidObj.read()
        
        if not success:
            break
            
        # Kiểm tra nếu đúng vị trí bước nhảy thì lưu
        if count % step == 0 and saved_count < target_frames:
            # Lưu ảnh vào thư mục 'anh' với định dạng 4 chữ số (0001, 0002...)
            file_name = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(file_name, image)
            saved_count += 1
            
        count += 1

    vidObj.release()
    print(f"Hoàn thành! Đã lưu {saved_count} ảnh vào {output_folder}")

if __name__ == '__main__':
    # Thay 'test.mp4' bằng đường dẫn video của bạn
    FrameCapture("test.mp4")