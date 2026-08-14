import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point
import math

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')
        # Subscribes to the target point from vision
        self.subscription = self.create_subscription(
            Point,
            'target_point',
            self.target_callback,
            10)
        
        # Publishes Swerve string commands to ESP32 bridge
        self.publisher_ = self.create_publisher(String, 'swerve_commands', 10)
        
        self.get_logger().info('Navigation Node initialized.')
        
        # Kinematics constants (using existing logic)
        self.frame_center_x = 320 # Assuming 640x480
        self.max_speed = 150.0

    def target_callback(self, msg):
        # We receive msg.x, msg.y as target center
        # If msg.z < 0 it means target lost
        if msg.z < 0:
            self.publish_stop()
            return
            
        error_x = msg.x - self.frame_center_x
        # Simple proportional steering
        angle = max(min(error_x * 0.1, 45.0), -45.0)
        speed = self.max_speed
        
        self.publish_swerve(angle, speed, angle, speed, angle, speed, angle, speed)

    def publish_swerve(self, fl_a, fl_s, fr_a, fr_s, bl_a, bl_s, br_a, br_s):
        cmd = f"SWERVE:{fl_a:.1f},{fl_s:.1f},{fr_a:.1f},{fr_s:.1f},{bl_a:.1f},{bl_s:.1f},{br_a:.1f},{br_s:.1f}"
        msg = String()
        msg.data = cmd
        self.publisher_.publish(msg)
        
    def publish_stop(self):
        self.publish_swerve(0, 0, 0, 0, 0, 0, 0, 0)

def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
