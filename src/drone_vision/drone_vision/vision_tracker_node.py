import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import message_filters
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker


class VisionTrackerNode(Node):
    def __init__(self):
        super().__init__('vision_tracker_node')
        self.bridge = CvBridge()

        # 1. Declare and retrieve the TensorRT model path parameter
        default_model_path = os.path.expanduser(
            '~/Utkarsh-Sandbox/drone_stack_unified/models/16_k_and_drone_studio_realsense_images_model.engine'
        )
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('conf_threshold', 0.5)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value

        # 2. Load the Quantized TensorRT Engine
        self.get_logger().info(f'Loading TensorRT engine from: {model_path}')
        self.model = YOLO(model_path, task='detect')

        # 3. Initialize Norfair 3D Tracker
        self.tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=1.5,
            hit_counter_max=15,
            initialization_delay=2
        )

        # 4. Camera Intrinsics (Using sensor_data QoS to match ZED2i)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/zed/zed_node/rgb/color/rect/camera_info',
            self.info_callback,
            qos_profile_sensor_data
        )

        # 5. Synchronized ZED2i RGB & Depth Subscribers (Using sensor_data QoS)
        self.rgb_sub = message_filters.Subscriber(
            self, Image, '/zed/zed_node/rgb/color/rect/image', qos_profile=qos_profile_sensor_data
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/zed/zed_node/depth/depth_registered', qos_profile=qos_profile_sensor_data
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.05
        )
        self.ts.registerCallback(self.image_callback)

        # 6. Publisher for Tracked Drones & Velocity Vectors
        self.marker_pub = self.create_publisher(MarkerArray, '/tracked_drones', 10)
        self.get_logger().info('Vision Tracker Node ready.')

    def info_callback(self, msg):
        if self.fx is None:
            self.fx = msg.k[0]
            self.cx = msg.k[2]
            self.fy = msg.k[4]
            self.cy = msg.k[5]
            self.get_logger().info(f'ZED2i intrinsics locked: fx={self.fx:.1f}, fy={self.fy:.1f}')

    def pixel_to_3d(self, u, v, depth_img):
        h, w = depth_img.shape[:2]
        u_min, u_max = max(0, u - 2), min(w, u + 3)
        v_min, v_max = max(0, v - 2), min(h, v + 3)

        patch = depth_img[v_min:v_max, u_min:u_max]
        valid_depths = patch[np.isfinite(patch) & (patch > 0.1)]

        if len(valid_depths) == 0:
            return None

        z = float(np.median(valid_depths))
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.array([x, y, z])

    def image_callback(self, rgb_msg, depth_msg):
        if self.fx is None:
            return

        frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')

        # Run TensorRT YOLO Inference
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)

        norfair_detections = []
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = box
                u = int((x1 + x2) / 2.0)
                v = int((y1 + y2) / 2.0)
                pt_3d = self.pixel_to_3d(u, v, depth)

                if pt_3d is not None:
                    norfair_detections.append(
                        Detection(points=np.array([pt_3d]), scores=np.array([float(conf)]))
                    )

        # Update Norfair Tracker
        tracked_objects = self.tracker.update(detections=norfair_detections)
        self.publish_tracks(tracked_objects, rgb_msg.header)

    def publish_tracks(self, tracked_objects, header):
        marker_array = MarkerArray()

        for obj in tracked_objects:
            pos = obj.estimate[0]          # [x, y, z] in optical frame
            vel = obj.estimate_velocity[0] # [vx, vy, vz] in optical frame

            # Sphere Marker for Drone Position
            marker = Marker()
            marker.header = header
            marker.ns = "tracked_drones"
            marker.id = obj.id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = float(pos[2])
            marker.scale.x = 0.4
            marker.scale.y = 0.4
            marker.scale.z = 0.4
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.2
            marker.color.b = 0.2
            marker_array.markers.append(marker)

            # Arrow Marker for Velocity Vector
            arrow = Marker()
            arrow.header = header
            arrow.ns = "drone_velocities"
            arrow.id = obj.id + 1000
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.scale.x = 0.08
            arrow.scale.y = 0.16
            arrow.scale.z = 0.2
            arrow.color.a = 1.0
            arrow.color.r = 0.1
            arrow.color.g = 0.9
            arrow.color.b = 0.2

            p_start = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            p_end = Point(
                x=float(pos[0] + vel[0]),
                y=float(pos[1] + vel[1]),
                z=float(pos[2] + vel[2])
            )
            arrow.points = [p_start, p_end]
            marker_array.markers.append(arrow)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = VisionTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

