import time
from RPLCD.i2c import CharLCD

lcd = CharLCD(
    i2c_expander='PCF8574',
    address=0x3F,
    port=1,
    cols=20,
    rows=4
)

lines = [
    {"text": "Nhom 6", "align": "center"},
    {"text": "Nguyen Manh Thang", "align": "left"},
    {"text": "Cao Huynh Phi", "align": "center"},
    {"text": "Nguyen Hoai An", "align": "right"}
]

def format_text(text, align, width=20):
    if align == "left":
        return text.ljust(width)
    elif align == "right":
        return text.rjust(width)
    elif align == "center":
        return text.center(width)

# Hiển thị từng ký tự từ phải sang trái (cố định vị trí)
def show_line_rtl(row, text, align):
    formatted = format_text(text, align)
    
    # tạo buffer rỗng
    display = [" "] * 20
    
    # fill từ phải qua trái
    for i in range(19, -1, -1):
        display[i] = formatted[i]
        
        lcd.cursor_pos = (row, 0)
        lcd.write_string("".join(display))
        
        time.sleep(0.1)

lcd.clear()

# chạy từng dòng
for i, line in enumerate(lines):
    show_line_rtl(i, line["text"], line["align"])