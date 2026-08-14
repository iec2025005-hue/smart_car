import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import time

class Esp32BridgeNode(Node):
    def __init__(self):
        super().__init__('esp32_bridge_node')
        
        # We are using hardware UART on GPIO pins (ttyAMA0 or serial0)
        serial_port = '/dev/serial0'
        baud_rate = 115200
        
        try:
            self.ser = serial.Serial(serial_port, baud_rate, timeout=1)
            self.get_logger().info(f'Connected to ESP32 on {serial_port}')
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to connect to {serial_port}: {e}')
            self.ser = None

        self.subscription = self.create_subscription(
            String,
            'swerve_commands',
            self.command_callback,
            10)
            
    def command_callback(self, msg):
        if self.ser and self.ser.is_open:
            command = msg.data + '\n'
            self.ser.write(command.encode('utf-8'))
            self.get_logger().debug(f'Sent to ESP32: {command.strip()}')
        else:
            self.get_logger().warn('Serial is not open, cannot send command')

def main(args=None):
    rclpy.init(args=args)
    node = Esp32BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.ser and node.ser.is_open:
            node.ser.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
