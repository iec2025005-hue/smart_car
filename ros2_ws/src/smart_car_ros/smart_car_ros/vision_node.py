import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        # Publish the target's center coordinates and the camera image
        self.target_publisher = self.create_publisher(Point, 'target_point', 10)
        self.image_publisher = self.create_publisher(Image, 'camera/image_raw', 10)
        
        self.bridge = CvBridge()
        
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
                
                # Draw bounding box and center dot for Foxglove video stream
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                
        self.target_publisher.publish(msg)
        
        # Publish the image frame to ROS 2 (so we can view it in Foxglove!)
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.image_publisher.publish(img_msg)

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
