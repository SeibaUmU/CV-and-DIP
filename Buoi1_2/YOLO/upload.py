from roboflow import Roboflow
import os
import time

# Khởi tạo Roboflow
rf = Roboflow(api_key="eaY9bZRzBEFO7LSSW31U")
project = rf.workspace("yolo-mj3sk").project("gan-nhan-hp0ej")

# Đường dẫn folder ảnh 2000 frames của bạn
image_dir = r"D:\TaiLieu\XLA\XLA_TH\CV-and-DIP\Buoi1_2\YOLO\dataset\anh"

all_images = os.listdir(image_dir)

# 2. Bạn đã xong 482 ảnh, vậy chúng ta sẽ bắt đầu từ ảnh thứ 483
# Trong lập trình, index bắt đầu từ 0 nên ta lấy từ 482 trở đi
images_to_upload = all_images[482:] 

print(f"Bắt đầu upload tiếp từ ảnh thứ 483 (Tổng còn lại: {len(images_to_upload)})")

for img_name in images_to_upload:
    img_path = os.path.join(image_dir, img_name)
    
    try:
        project.single_upload(img_path)
        print(f"Đã tải lên: {img_name}")
        
        # 3. Thêm 0.5 giây nghỉ giữa mỗi lần upload để tránh bị Rate Limit
        time.sleep(0.5) 
        
    except Exception as e:
        print(f"Lỗi ở ảnh {img_name}: {e}")
        print("Đang nghỉ 5 giây rồi thử lại...")
        time.sleep(5)
        continue