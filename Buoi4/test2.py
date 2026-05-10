import RPi.GPIO as GPIO
import cv2
import time
from ultralytics import YOLO
from RPLCD.i2c import CharLCD 

# ==========================================
# 1. KHỞI TẠO VÀ CẤU HÌNH CƠ BẢN
# ==========================================
lcd = CharLCD(
    i2c_expander='PCF8574',
    address=0x3F,
    port=1,
    cols=20,
    rows=4
)
lcd.clear()
lcd.cursor_pos = (0, 0)
lcd.write_string("He thong khoi dong..")

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Chân Motor (M1: Trái | M2: Phải)
M1_T_PIN = 4
M1_N_PIN = 17
M2_T_PIN = 22
M2_N_PIN = 8
motor_pins = [M1_T_PIN, M1_N_PIN, M2_T_PIN, M2_N_PIN]

# Chân LED (7 chân)
led_pins = [7, 23, 24, 18, 25, 12, 16]

GPIO.setup(motor_pins, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(led_pins, GPIO.OUT, initial=GPIO.HIGH) 

# ==========================================
# 2. NGƯỠNG NHẬN DIỆN
# ==========================================
class_thresholds = {0: 0.3, 1: 0.3, 2: 0.3}
MIN_AREA = 2000

# ==========================================
# 3. HÀM CẬP NHẬT TRẠNG THÁI (GPIO & LCD)
# ==========================================
def update_gpio(stop_sign, re_phai, bien_50):
    motor_states = [0, 0, 0, 0]
    led_states = [1, 1, 1, 1, 1, 1, 1] 
    motor_text = "Dung"

    if stop_sign and re_phai and bien_50:
        motor_states = [0, 0, 0, 0]
        led_states = [1, 1, 1, 1, 1, 1, 0]
        motor_text = "Dung (3 bien)"
    elif re_phai and stop_sign:
        motor_states = [1, 0, 1, 0]
        led_states = [1, 1, 1, 1, 1, 0, 1]
        motor_text = "T-Tien, P-Tien"
    elif bien_50 and stop_sign:
        motor_states = [0, 1, 0, 1]
        led_states = [1, 1, 1, 1, 0, 1, 1]
        motor_text = "T-Lui, P-Lui"
    elif re_phai and bien_50:
        motor_states = [1, 0, 1, 1]
        led_states = [1, 1, 1, 0, 1, 1, 1]
        motor_text = "T-Tien, P-Lui"
    elif stop_sign:
        motor_states = [0, 0, 1, 0]
        led_states = [1, 1, 0, 1, 1, 1, 1]
        motor_text = "Phai Tien"
    elif re_phai:
        motor_states = [0, 1, 1, 0]
        led_states = [1, 0, 1, 1, 1, 1, 1]
        motor_text = "T-Lui, P-Tien"
    elif bien_50:
        motor_states = [1, 0, 1, 0]
        led_states = [0, 1, 1, 1, 1, 1, 1]
        motor_text = "Ca Hai Tien"

    # Cập nhật GPIO Motor & LED
    GPIO.output(motor_pins, motor_states)
    GPIO.output(led_pins, led_states)

    # --- CẬP NHẬT TRẠNG THÁI LÊN LCD ---
    objs = []
    if stop_sign: objs.append("Stop")
    if re_phai: objs.append("Re Phai")
    if bien_50: objs.append("50")
    
    obj_text = ",".join(objs) if objs else "Khong"
    
    lcd.clear()
    lcd.cursor_pos = (0, 0)
    lcd.write_string(f"Bien: {obj_text}")
    lcd.cursor_pos = (1, 0)
    lcd.write_string(f"Motor: {motor_text}")
    lcd.cursor_pos = (2, 0)
    so_led_sang = led_states.count(0) 
    lcd.write_string(f"LED sang: {so_led_sang}")

# ==========================================
# 4. VÒNG LẶP CHÍNH
# ==========================================
try:
    prev_state = (False, False, False)
    lcd.clear()
    lcd.cursor_pos = (0, 0)
    lcd.write_string("Dang quet camera...")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        stop_sign, re_phai, bien_50 = False, False, False
        
        results = model(frame, verbose=False)
        boxes = results[0].boxes
        annotated_frame = frame.copy()
        
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                
                if conf >= class_thresholds.get(cls, 0.3) and area >= MIN_AREA:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Lấy tên hiển thị theo class
                    label = "Stop" if cls == 0 else ("Re Phai" if cls == 1 else "50")
                    cv2.putText(annotated_frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if cls == 0:
                        stop_sign = True
                    elif cls == 1:
                        re_phai = True
                    elif cls == 2:
                        bien_50 = True

        current_state = (stop_sign, re_phai, bien_50)
        if current_state != prev_state:
            update_gpio(stop_sign, re_phai, bien_50)
            prev_state = current_state

        cv2.imshow("Detection System", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    try:
        lcd.clear()
        lcd.cursor_pos = (0, 0)
        lcd.write_string("He thong da tat!")
    except:
        pass