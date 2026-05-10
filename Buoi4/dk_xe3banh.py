import time
import serial
import threading
import RPi.GPIO as GPIO

# ===== SERVO =====
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
channel = 8
servo_pulse = 1500 # mặc định giữa

def send_cmd(pulse, t=200, d=0):
    cmd = f"#{channel}P{pulse}T{t}D{d}\r\n"
    ser.write(cmd.encode())

def servo_thread():
    global servo_pulse
    while True:
        send_cmd(servo_pulse)
        time.sleep(0.02) # gửi liên tục để servo mượt

# ===== MOTOR =====
M1_T_PIN = 4 #7
M1_N_PIN = 17 #11
M2_T_PIN = 22 #15
M2_N_PIN = 8 #24

def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in [M1_T_PIN, M1_N_PIN, M2_T_PIN, M2_N_PIN]:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, 0)

def set_motors(m1_t, m1_n, m2_t, m2_n):
    GPIO.output(M1_T_PIN, m1_t)
    GPIO.output(M1_N_PIN, m1_n)
    GPIO.output(M2_T_PIN, m2_t)
    GPIO.output(M2_N_PIN, m2_n)

# ===== SERVO MƯỢT =====
def smooth_servo_move(start, end, step=10, delay=0.02):
    global servo_pulse
    if start < end:
        for p in range(start, end+1, step):
            servo_pulse = p
            time.sleep(delay)
    else:
        for p in range(start, end-1, -step):
            servo_pulse = p
            time.sleep(delay)

# ===== LOGIC ĐỔI TRẠNG THÁI =====
def change_direction(forward=True, turn_dir=None):
    """
    forward=True: đi thẳng
    forward=False: đi lùi
    turn_dir: None/left/right
    """
    global servo_pulse
    
    # 1. Dừng bánh sau trước khi đổi trạng thái
    set_motors(0,0,0,0)
    time.sleep(0.1)
    
    # 2. Nếu cần rẽ, điều khiển servo mượt
    if turn_dir == "left":
        target_pulse = 1000
        smooth_servo_move(servo_pulse, target_pulse, step=10, delay=0.02)
        time.sleep(0.3)
        smooth_servo_move(servo_pulse, 1500, step=10, delay=0.02)
    elif turn_dir == "right":
        target_pulse = 2000
        smooth_servo_move(servo_pulse, target_pulse, step=10, delay=0.02)
        time.sleep(0.3)
        smooth_servo_move(servo_pulse, 1500, step=10, delay=0.02)
        
    # 3. Chạy bánh sau theo chiều mong muốn
    if forward:
        set_motors(1,0,1,0)
    else:
        set_motors(0,1,0,1)

# ===== MAIN =====
def main():
    global servo_pulse
    setup()
    
    t = threading.Thread(target=servo_thread, daemon=True)
    t.start()
    
    print("Nhập lệnh: w-thẳng, a-tiến+trái, d-tiến+phải, s-lùi thẳng, z-lùi+trái, c-lùi+phải, 0-stop, exit-thoát")
    
    try:
        while True:
            cmd = input("Cmd: ").strip().lower()
            
            if cmd == 'w':      # đi thẳng
                change_direction(forward=True, turn_dir=None)
                print("Di thang")
                
            elif cmd == 'a':    # tiến + rẽ trái
                change_direction(forward=True, turn_dir="left")
                print("Re Trai")
                
            elif cmd == 'd':    # tiến + rẽ phải
                change_direction(forward=True, turn_dir="right")
                print("Re Phai")
                
            elif cmd == 's':    # lùi thẳng
                change_direction(forward=False, turn_dir=None)
                print("Lui thang")
                
            elif cmd == 'z':    # lùi + rẽ trái (trong code ảnh ghi re phai)
                change_direction(forward=False, turn_dir="right")
                print("Lui trai")
                
            elif cmd == 'c':    # lùi + rẽ phải (trong code ảnh ghi re trai)
                change_direction(forward=False, turn_dir="left")
                print("Lui phai")
                
            elif cmd == '0':    # stop
                set_motors(0,0,0,0)
                print("Dung xe")
                
            elif cmd == 'exit':
                break
            else:
                print("Lệnh không hợp lệ!")
                
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()
        ser.close()

if __name__ == "__main__":
    main()