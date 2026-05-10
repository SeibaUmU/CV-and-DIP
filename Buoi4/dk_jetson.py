import sys
import select
import termios
import tty
import rclpy
from rclpy.node import import Node
from my_interfaces.srv import Jetson # Sử dụng đúng service 'Jetson'

MOVE_BINDINGS = {
    'd': 'dc',
    's': 'servo',
    'a': 'lcd',
    'c': 'cam',
    'e': 'led',
    ' ': 'stop',
}

def get_key(settings, timeout=0.1):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

class KeyboardClient(Node):
    def __init__(self):
        super().__init__('keyboard_client')
        self.cli = self.create_client(Jetson, 'nhom7') # Đúng service name ở đây
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Đang chờ Server...')

    def send_request(self, cm):
        req = Jetson.Request() # Đúng kiểu request cho service 'Jetson'
        req.a = cm
        future = self.cli.call_async(req)
        return future # Bạn cần lưu future để kiểm tra kết quả trả về

def main():
    rclpy.init()
    settings = termios.tcgetattr(sys.stdin)
    client = KeyboardClient()
    cm = 'stop' # Default command

    try:
        while rclpy.ok():
            key = get_key(settings)
            if key in MOVE_BINDINGS:
                cm = MOVE_BINDINGS[key] # Get the full command
                print(f"Command: {cm}")
                future = client.send_request(cm)
                rclpy.spin_until_future_complete(client, future) # Đợi cho đến khi request hoàn thành
                
                if future.result() is not None:
                    # Log kết quả từ response
                    if future.result().success:
                        print(f"Command {cm} executed successfully.")
                    else:
                        print(f"Command {cm} failed.")
            
            elif key == '\x03': # Ctrl+C
                break
            
            rclpy.spin_once(client, timeout_sec=0)

    except Exception as e:
        print(e)

    finally:
        client.send_request('stop') # Send stop on exit
        client.destroy_node()
        rclpy.shutdown()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

if __name__ == '__main__':
    main()