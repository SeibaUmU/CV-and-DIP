import os
import random
import shutil

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Nguồn (Chỗ 1k6 ảnh vừa tải về)
src_img = r'D:\TaiLieu\XLA\XLA_TH\CV-and-DIP\Buoi1_2\train\images'
src_lbl = r'D:\TaiLieu\XLA\XLA_TH\CV-and-DIP\Buoi1_2\train\labels'

# Đích (Cấu trúc dataset bạn vừa tạo)
dst_root = r'D:\TaiLieu\XLA\XLA_TH\CV-and-DIP\Buoi1_2\YOLO\dataset'

# --- THIẾT LẬP TỶ LỆ ---
train_ratio = 0.8  # 80% cho train, 20% cho val

def move_files(files, mode):
    for filename in files:
        # Move ảnh
        shutil.move(os.path.join(src_img, filename), 
                    os.path.join(dst_root, mode, 'images', filename))
        
        # Move nhãn tương ứng
        label_name = filename.rsplit('.', 1)[0] + '.txt'
        src_label_path = os.path.join(src_lbl, label_name)
        if os.path.exists(src_label_path):
            shutil.move(src_label_path, 
                        os.path.join(dst_root, mode, 'labels', label_name))

# Lấy danh sách ảnh và xáo trộn ngẫu nhiên
all_images = [f for f in os.listdir(src_img) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.seed(42) # Để kết quả chia không đổi mỗi lần chạy
random.shuffle(all_images)

# Chia danh sách
split_idx = int(len(all_images) * train_ratio)
train_files = all_images[:split_idx]
val_files = all_images[split_idx:]

# Thực hiện di chuyển
print(f"🚀 Đang bắt đầu di chuyển {len(all_images)} ảnh...")
move_files(train_files, 'train')
move_files(val_files, 'val')

print(f"✅ Hoàn thành!")
print(f"📂 Train: {len(train_files)} ảnh")
print(f"📂 Val: {len(val_files)} ảnh")