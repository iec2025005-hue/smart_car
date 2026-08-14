import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import cv2
from ultralytics import YOLO

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        # Publish the target's center coordinates
        self.target_publisher = self.create_publisher(Point, 'target_point', 10)
        
        # Load YOLO model
        self.get_logger().info('Loading YOLO model...')
        self.model = YOLO('yolov8n.pt')
        self.target_class_id = 0 # Default to Person
        
        # Initialize camera (using OpenCV as fallback for all platforms)
        # Note: on Pi you might want to use picamera2 directly if latency is high
        self.cap = cv2.VideoCapture(0)
        
        # Timer for processing frames
        self.timer = self.create_timer(0.05, self.timer_callback) # ~20 FPS

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to capture frame')
            return
            
        results = self.model.predict(frame, verbose=False, classes=[self.target_class_id])
        
        msg = Point()
        msg.z = -1.0 # Default to 'not found'
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Get highest confidence detection
            boxes = results[0].boxes
            best_box = max(boxes, key=lambda b: b.conf[0].item())
            
            if best_box.conf[0].item() > 0.5:
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                msg.x = float(cx)
                msg.y = float(cy)
                msg.z = 1.0 # Found
                
        self.target_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.cap:
            node.cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
