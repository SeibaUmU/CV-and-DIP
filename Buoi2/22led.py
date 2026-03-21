import RPi.GPIO as GPIO
import time

# Danh sách các chân GPIO (22 chân)
raw_pins = [[4,17],[27,22],[10,9],[11,5],[6,13],[19,26],[21,20],[16,12],[7,8],[25,24],[23,18]]
#led_pins = [4,17,27,22,10,9,11,5,6,13,19,26,21,20,16,12,7,8,25,24,23,18] #BCM
#BCM
#7,11,13,15,19,21,23,29,31,33,35,37,40,38,36,32,26,24,22,18,16,12 Board
#1 = tat, 0 = sang => cathode
led_pins =[pin for sublist in raw_pins for pin in sublist]

def main():
    # Thiết lập chế độ BCM (đánh số theo tên GPIO)
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Cấu hình tất cả các chân là OUTPUT và mặc định là HIGH (tắt)
    for pin in led_pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.HIGH)
        
    try:
        while True:
            # print(Sang duoi 1 led")
            # for pin in led_pins:
            #     GPIO.output(pin, GPIO.LOW)
            #     time.sleep(0.1) # Điều chỉnh tốc độ chạy ở đây
            #     GPIO.output(pin, GPIO.HIGH)
            #     time.sleep(0.5) # Đợi một chút khi tất cả 22 LED đã sáng
                
            # # Đợi một chút khi tất cả 22 LED đã sáng
            # time.sleep(0.5)
            # print("Bắt đầu chu kỳ mới: sáng dần...")
            # # 1. Bật sáng dần từng LED và giữ nguyên các LED trước đó
            # for pin in led_pins:
            #     GPIO.output(pin, GPIO.HIGH)
            #     time.sleep(0.1) # Điều chỉnh tốc độ chạy ở đây
                
            # # Đợi một chút khi tất cả 22 LED đã sáng
            # time.sleep(0.5)
            
            # # 2. Tắt toàn bộ LED để bắt đầu chu kỳ mới
            # print("Tắt tất cả và reset...")
            # for pin in led_pins:
            #     GPIO.output(pin, GPIO.LOW)
                
            # # Đợi một chút trước khi lặp lại
            # time.sleep(0.5)

            print("Sang duoi 2 led 1 luc")
            for i in range(0,len(led_pins),2):
                idx1 = i
                idx2 = (i+1)
                
                GPIO.output(led_pins[idx1], GPIO.LOW)
                GPIO.output(led_pins[idx2], GPIO.LOW)
                
                time.sleep(0.3)
                
                GPIO.output(led_pins[idx1], GPIO.HIGH)
                GPIO.output(led_pins[idx2], GPIO.HIGH)
                
    except KeyboardInterrupt:
        print("\nDang dung chuong trinh...")
    finally:
        # Giải phóng tài nguyên GPIO
        GPIO.cleanup()

if __name__ == "__main__":
    main()