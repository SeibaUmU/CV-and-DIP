import RPi.GPIO as GPIO
import time

# Pin Definitions (BCM)
led1 = 7
led2 = 23
led3 = 24
led4 = 18
led5 = 25
led6 = 12

def main():
    # Pin Setup
    GPIO.setmode(GPIO.BCM)
    
    GPIO.setup(led1, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(led2, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(led3, GPIO.OUT, initial=GPIO.HIGH)
    
    print("Starting demo now! Press CTRL+C to exit")
    
    curr_value = GPIO.HIGH
    
    try:
        while True:
            time.sleep(1)
            
            print("Outputting {} to pins {}, {}, {}".format(curr_value, led1, led2, led3))
            
            GPIO.output(led1, curr_value)
            GPIO.output(led2, curr_value)
            GPIO.output(led3, curr_value)
            
            curr_value ^= GPIO.HIGH  # đảo trạng thái
            
    finally:
        GPIO.cleanup()

if __name__ == '__main__':
    main()