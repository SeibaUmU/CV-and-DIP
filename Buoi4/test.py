import time
import threading
import cv2
import os
from ultralytics import YOLO
import RPi.GPIO as GPIO
import serial
from RPLCD.i2c import CharLCD

# ===== LCD =====
lcd = CharLCD(i2c_expander='PCF8574', address=0x3F, port=1, cols=20, rows=4)

# Biến lưu trạng thái cũ của LCD để tránh xóa/vẽ lại liên tục
last_lcd_text = []

def update_lcd(l1="", l2="", l3="", l4=""):
    global last_lcd_text
    current = [l1, l2, l3, l4]
    # Chỉ xóa và in lại nếu nội dung thực sự thay đổi
    if current != last_lcd_text:
        lcd.clear()
        lcd.write_string(f"{l1[:20]:20}\n{l2[:20]:20}\n{l3[:20]:20}\n{l4[:20]:20}")
        last_lcd_text = current

# ===== LED =====
LED_RED = 7
LED_YELLOW = 23
LED_GREEN = 24

# ===== MOTOR =====
M1_T = 4
M1_N = 17
M2_T = 22
M2_N = 18

# ===== SERVO =====
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
channel = 5
servo_pulse = 1500

def send_cmd(pulse):
    cmd = f"#{channel}P{pulse}T200\r\n"
    ser.write(cmd.encode())

def servo_thread():
    global servo_pulse
    while True:
        send_cmd(servo_pulse)
        time.sleep(0.02)

# ===== MOTOR FUNC =====
def set_motor(a,b,c,d):
    GPIO.output(M1_T,a)
    GPIO.output(M1_N,b)
    GPIO.output(M2_T,c)
    GPIO.output(M2_N,d)

def stop():
    set_motor(0,0,0,0)

def forward():
    set_motor(1,0,1,0)

# ===== YOLO =====
model = YOLO(os.path.expanduser("~/Documents/best.pt"))

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# FIX LAG: Ép độ phân giải thấp ở cấp độ phần cứng camera thay vì thu nhỏ bằng phần mềm
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# FIX POPUP NHỎ: Tạo cửa sổ có thể thay đổi kích thước và phóng to nó lên 800x600
cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO", 800, 600)

# ===== SETUP =====
GPIO.setwarnings(False) # Tắt cảnh báo GPIO
GPIO.setmode(GPIO.BCM)
for pin in [LED_RED, LED_YELLOW, LED_GREEN, M1_T, M1_N, M2_T, M2_N]:
    GPIO.setup(pin, GPIO.OUT)

t = threading.Thread(target=servo_thread, daemon=True)
t.start()

# ===== TRAFFIC LIGHT VARIABLES =====
light_state = "RED" # Trạng thái đèn ban đầu
last_switch_time = time.time()
RED_DURATION = 10
YELLOW_DURATION = 3
GREEN_DURATION = 20

# ===== MAIN LOOP =====
try:
    while True:
        current_time = time.time()
        elapsed = current_time - last_switch_time
        
        # --- 1. KIỂM TRA VÀ CHUYỂN TRẠNG THÁI ĐÈN ---
        if light_state == "RED" and elapsed >= RED_DURATION:
            light_state = "YELLOW"
            last_switch_time = current_time
        elif light_state == "YELLOW" and elapsed >= YELLOW_DURATION:
            light_state = "GREEN"
            last_switch_time = current_time
        elif light_state == "GREEN" and elapsed >= GREEN_DURATION:
            light_state = "RED"
            last_switch_time = current_time
            
        # --- 2. XỬ LÝ PHẦN CỨNG THEO ĐÈN ---
        if light_state == "RED":
            GPIO.output(LED_RED,1)
            GPIO.output(LED_YELLOW,0)
            GPIO.output(LED_GREEN,0)
            stop()
            update_lcd("DEN DO", "Dung xe", "", "")
            
        elif light_state == "YELLOW":
            GPIO.output(LED_RED,0)
            GPIO.output(LED_YELLOW,1)
            GPIO.output(LED_GREEN,0)
            stop()
            update_lcd("DEN VANG", "Dung xe", "", "")
            
        elif light_state == "GREEN":
            GPIO.output(LED_RED,0)
            GPIO.output(LED_YELLOW,0)
            GPIO.output(LED_GREEN,1)
        
        # --- 3. CAMERA LUÔN CHẠY XUYÊN SUỐT ---
        ret, frame = cap.read()
        if not ret:
            continue
            
        # LƯU Ý MẠNH: Nếu bạn dùng Jetson Nano/Xavier, hãy xóa "device='cpu'" để mạch dùng GPU, sẽ hết lag hoàn toàn!
        results = model(frame, imgsz=320, device='cpu') 
        annotated = results[0].plot()
        
        label = ""
        if len(results[0].boxes) > 0:
            cls = int(results[0].boxes.cls[0])
            # FIX NHẬN DIỆN: Ép tất cả label về chữ thường (lowercase) để không bị lỗi viết hoa viết thường
            label = model.names[cls].lower() 
        
        # --- 4. NẾU ĐÈN XANH THÌ XE MỚI HÀNH ĐỘNG THEO BIỂN BÁO ---
        if light_state == "GREEN":
            if "50" in label:
                servo_pulse = 1500
                forward()
                update_lcd("GREEN", "Speed 50", "Di thang", "")
                
            elif "re phai" in label:  # <--- Đổi "right" thành "re phai" ở đây
                servo_pulse = 2000
                forward()
                update_lcd("GREEN", "Re phai", "", "")
                
            elif "stop" in label:
                stop()
                update_lcd("GREEN", "STOP", "", "")
                
            else:
                forward()
                update_lcd("GREEN", "Khong ro", "", "")
        
        cv2.imshow("YOLO", annotated)
        
        if cv2.waitKey(1) == 27: # Bấm phím ESC để thoát
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    ser.close()