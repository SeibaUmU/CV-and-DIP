import time
import serial

ser = serial.Serial(
    port='/dev/ttyACM0',
    #port='COM3',
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)

channel = 8

def send_cmd(pulse, t=500, d=500):
    cmd = f'#{channel}P{pulse}T{t}D{d}\r\n'
    ser.write(cmd.encode())

try:
    while 1: # 10xung: 50 1501 10
        #for pulse in range(500, 1501, 10):
            #send_cmd(pulse, t=500, d=500)
            #time.sleep(0.5)
        #for pulse in range(1500, 499, -10):
            #send_cmd(pulse, t=500, d=500)
            #time.sleep(0.5)
            
        # 8buoc
        for pulse in range(500, 1501, 125):
            send_cmd(pulse, t=500, d=0)
            time.sleep(2)
        
        for pulse in range(1500, 499, -125):
            send_cmd(pulse, t=500, d=0)
            time.sleep(2)

except KeyboardInterrupt:
    print("STOP")
    ser.close()