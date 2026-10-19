"""YOLOE target selector in front of the original complete geometric scene server."""
import threading
import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener
from robot_task_interfaces.srv import GetPlanningScene

from .ai_selection import registered_mask_points, select_scene_object, transform_points
from .yoloe_detector import YOLOEDetector


class AISceneServer(Node):
    def __init__(self):
        super().__init__('robot_ai_perception_server')
        defaults = {'service_name': '/perception/get_planning_scene',
                    'geometry_service_name': '/perception/get_planning_scene/geometry',
                    'target_frame': 'base_link', 'model': 'yoloe-26n-seg.pt',
                    'target_class': 'drink can', 'device': 'cpu', 'confidence': 0.05,
                    'image_size': 960, 'backend_timeout': 15.0}
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.value = lambda name: self.get_parameter(name).value
        if self.value('service_name') == self.value('geometry_service_name'):
            raise ValueError('AI and geometry services must have distinct names')
        self.bridge = CvBridge()
        self.tf = Buffer(node=self)
        self.listener = TransformListener(self.tf, self)
        self.client = self.create_client(GetPlanningScene, self.value('geometry_service_name'),
                                         callback_group=ReentrantCallbackGroup())
        self.overlay = self.create_publisher(Image, '/perception/ai/detections', 1)
        self.get_logger().info(f"[AI] Loading YOLOE model={self.value('model')}, class={self.value('target_class')}, device={self.value('device')}")
        self.detector = YOLOEDetector(self.value('model'), [self.value('target_class')],
                                      self.value('device'), self.value('confidence'), self.value('image_size'))
        # Advertise only after model + text prompts are ready, so readiness includes model initialization.
        self.service = self.create_service(GetPlanningScene, self.value('service_name'), self.handle_scene)
        self.get_logger().info('[AI] Ready; target selection uses YOLOE masks and registered 3D points')

    def handle_scene(self, request, response):
        response.success = False
        future = None
        try:
            label = request.target_label.strip() or self.value('target_class')
            self.detector.set_classes([label])
            self.get_logger().info(f'[AI] Request target={label!r}; acquiring a new scene')
            timeout = self.value('backend_timeout')
            if not self.client.wait_for_service(timeout_sec=timeout):
                raise RuntimeError('Geometric scene service is unavailable')
            event = threading.Event()
            future = self.client.call_async(request)
            future.add_done_callback(lambda _: event.set())
            if not event.wait(timeout):
                self.client.remove_pending_request(future)
                future.cancel()
                raise RuntimeError('Geometric scene response timed out')
            scene = future.result()
            if not scene or not scene.support_surface_id or not scene.scene_world.collision_objects:
                raise RuntimeError('Geometric backend returned no complete scene/support')
            if not scene.rgb_image.data or not scene.full_cloud.data:
                raise RuntimeError('Geometric backend returned no RGB/depth data')
            image = self.bridge.imgmsg_to_cv2(scene.rgb_image, desired_encoding='bgr8')
            result = self.detector.predict(image)
            annotated = self.bridge.cv2_to_imgmsg(result.plot(), encoding='bgr8')
            annotated.header = scene.rgb_image.header
            self.overlay.publish(annotated)
            if len(result.boxes) != 1 or result.masks is None:
                raise RuntimeError(f"No segmented instance of {label!r}; no geometric target fallback")
            mask = result.masks.data[0].cpu().numpy() >= 0.5
            # Remove one pixel of uncertain silhouette before sampling depth.
            mask = cv2.erode(mask.astype('uint8'), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) > 0
            points = registered_mask_points(scene.full_cloud, scene.rgb_image, mask)
            frame = self.value('target_frame')
            transform = self.tf.lookup_transform(frame, scene.full_cloud.header.frame_id,
                                                  Time.from_msg(scene.full_cloud.header.stamp),
                                                  timeout=Duration(seconds=2.0))
            points = transform_points(points, transform.transform)
            score, selected = select_scene_object(points, scene.scene_world.collision_objects,
                                                   scene.support_surface_id, frame)
            ids = {obj.id for obj in scene.scene_world.collision_objects}
            if scene.support_surface_id not in ids:
                raise RuntimeError('Support object missing from reconstructed scene')
            response.scene_world = scene.scene_world
            response.full_cloud = scene.full_cloud
            response.rgb_image = scene.rgb_image
            response.support_surface_id = scene.support_surface_id
            response.target_object_id = selected
            response.success = True
            confidence = float(result.boxes[0].conf.item())
            self.get_logger().info(f"[AI] Selected class={label} confidence={confidence:.3f}, object={selected}, depth_overlap={score:.3f}; retained {len(ids)} collision objects")
        except Exception as error:
            self.get_logger().error(f'[AI] Target selection failed; planning stopped: {error}')
        return response


def main():
    rclpy.init()
    node = None
    executor = MultiThreadedExecutor(num_threads=3)
    try:
        node = AISceneServer()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f'[AI] Startup failed: {error}', flush=True)
        return 1
    finally:
        executor.shutdown()
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0
