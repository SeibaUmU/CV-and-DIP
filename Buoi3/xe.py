import RPi.GPIO as GPIO
import time

# Sử dụng chuẩn BCM
M1_T_PIN = 4   # Chân điều khiển M1 Thuận (Board 7)
M1_N_PIN = 17  # Chân điều khiển M1 Nghịch (Board 11)
M2_T_PIN = 22  # Chân điều khiển M2 Thuận (Board 15)
M2_N_PIN = 18  # Chân điều khiển M2 Nghịch (Board 12)
# 1->run, 0->off

def setup():
    #BCM
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Thiết lập các chân là OUTPUT
    pins = [M1_T_PIN, M1_N_PIN, M2_T_PIN, M2_N_PIN]
    for pin in pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW) # Tắt tất cả khi mới khởi động
        
def set_motors(m1_t, m1_n, m2_t, m2_n):
    """Hàm gán trạng thái cho 4 chân điều khiển dựa theo bảng logic"""
    GPIO.output(M1_T_PIN, m1_t)
    GPIO.output(M1_N_PIN, m1_n)
    GPIO.output(M2_T_PIN, m2_t)
    GPIO.output(M2_N_PIN, m2_n)

def main():
    setup()
    print("=== CHƯƠNG TRÌNH ĐIỀU KHIỂN ĐỘNG CƠ JETSON NANO ===")
    print("Nhập các phím: w, r, s, f, e, d, q, t, o để điều khiển.")
    print("Nhập 'exit' hoặc bấm Ctrl+C để thoát.")
    
    try:
        while True:
            cmd = input("Nhập lệnh: ").strip().lower()
            
            # Khớp lệnh với bảng logic (M1_T, M1_N, M2_T, M2_N)
            if cmd == 'q':
                set_motors(1, 0, 0, 0)
                print("-> M1 thuận")
            elif cmd == 'w':
                set_motors(0, 0, 1, 0)
                print("-> M2 thuận")
            elif cmd == 'a':
                set_motors(0, 1, 0, 0)
                print("-> M1 nghịch")
            elif cmd == 's':
                set_motors(0, 0, 0, 1)
                print("-> M2 nghịch")
            elif cmd == 'e':
                set_motors(1, 0, 1, 0)
                print("-> Cả 2 thuận")
            elif cmd == 'd':
                set_motors(0, 1, 0, 1)
                print("-> Cả 2 nghịch")
            elif cmd == 'r':
                set_motors(1, 0, 0, 1)
                print("-> M1 thuận + M2 nghịch")
            elif cmd == 'f':
                set_motors(0, 1, 1, 0)
                print("-> M1 nghịch + M2 thuận")
            elif cmd == 'o':
                set_motors(0, 0, 0, 0)
                print("-> Stop")
            elif cmd == 'exit':
                break
            else:
                print("Lệnh không hợp lệ! Vui lòng nhập lại.")
                
    except KeyboardInterrupt:
        print("\nĐã dừng chương trình Ctrl+C")
    finally:
        GPIO.cleanup()
        print("Đã dọn dẹp GPIO và thoát.")

if __name__ == "__main__":
    main()