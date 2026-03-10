# v0.610
"""
SRD Pose Emergency ROS2 Node
===========================

이 파일은 실제 TurtleBot4 환경에서 사용하는 ROS2 노드 파일이다.
역할은 명확하다.

1) compressed image 토픽을 구독한다.
2) JPEG 이미지를 OpenCV 프레임으로 디코딩한다.
3) PoseEmergencyEngine을 호출한다.
4) 분석 결과(단일 emergency_level string)와 시각화 이미지(compressed)를 publish 한다.

즉, 이 파일은 ROS2 입출력 담당이고,
실제 포즈 분석 로직 자체는 srd_pose_emergency_core.py 에 있다.
"""
from typing import List

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

try:
    from .srd_pose_emergency_core import AnalyzerConfig, PoseEmergencyEngine
except ImportError:
    from srd_pose_emergency_core import AnalyzerConfig, PoseEmergencyEngine


class SrdPoseEmergencyNode(Node):
    """SRD Pose Emergency ROS2 Node.

    구독 토픽:
    - input_image_topic (CompressedImage)

    발행 토픽:
    - result_topic (String JSON)
    - image_result_topic (CompressedImage)
    """

    def __init__(self):
        super().__init__("srd_pose_emergency_node")

        # --------------------------------------------------------------
        # ROS2 파라미터 선언
        # 실제 로봇 환경에서는 launch 파일 또는 ros2 param 으로 덮어쓸 수 있다.
        # --------------------------------------------------------------
        self.declare_parameter("model_path", "yolo11n-pose.pt")
        self.declare_parameter("input_image_topic", "/camera/image_raw/compressed")
        # 최종 emergency_level 문자열 publish 토픽
        self.declare_parameter("emergency_level_topic", "/robot6/srd/emergency_level")
        self.declare_parameter("image_result_topic", "/robot6/srd/image_result/compressed")
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("show_debug", True)

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        input_image_topic = self.get_parameter("input_image_topic").get_parameter_value().string_value
        emergency_level_topic = self.get_parameter("emergency_level_topic").get_parameter_value().string_value
        image_result_topic = self.get_parameter("image_result_topic").get_parameter_value().string_value
        self.publish_annotated = self.get_parameter("publish_annotated").get_parameter_value().bool_value
        show_debug = self.get_parameter("show_debug").get_parameter_value().bool_value

        # --------------------------------------------------------------
        # 분석 코어 설정 / 생성
        # --------------------------------------------------------------
        cfg = AnalyzerConfig(model_path=model_path, show_debug=show_debug)
        self.engine = PoseEmergencyEngine(cfg)

        # --------------------------------------------------------------
        # Publisher / Subscriber 생성
        # --------------------------------------------------------------
        self.emergency_level_pub = self.create_publisher(String, emergency_level_topic, 10)
        self.image_result_pub = self.create_publisher(CompressedImage, image_result_topic, 10)

        self.image_sub = self.create_subscription(
            CompressedImage,
            input_image_topic,
            self.image_callback,
            10,
        )

        self.get_logger().info(
            f"SRD Pose Emergency Node started. input={input_image_topic}, emergency_level={emergency_level_topic}, annotated={image_result_topic}"
        )

    # ------------------------------------------------------------------
    # 이미지 디코딩 / 인코딩 보조 함수
    # ------------------------------------------------------------------
    @staticmethod
    def _decode_compressed_image(msg: CompressedImage) -> np.ndarray:
        """ROS2 CompressedImage 메시지를 OpenCV BGR 이미지로 디코딩한다."""
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode compressed image.")
        return frame

    @staticmethod
    def _encode_compressed_image(frame: np.ndarray, header_stamp, format_str: str = "jpeg") -> CompressedImage:
        """OpenCV BGR 이미지를 ROS2 CompressedImage 메시지로 인코딩한다."""
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError("Failed to encode annotated image.")

        msg = CompressedImage()
        msg.header.stamp = header_stamp
        msg.format = format_str
        msg.data = encoded.tobytes()
        return msg

    # ------------------------------------------------------------------
    # 메인 callback
    # ------------------------------------------------------------------
    def image_callback(self, msg: CompressedImage) -> None:
        """compressed image를 받아 분석하고 결과를 publish 한다."""
        try:
            # 1) compressed image -> OpenCV frame 디코딩
            frame = self._decode_compressed_image(msg)

            # 2) 분석 코어 호출
            annotated, emergency_level = self.engine.analyze_frame_with_emergency_level(frame)

            # 3) 최종 emergency_level 하나만 publish
            # 사람이 검출되지 않은 프레임은 운영 의미가 약하므로 publish 생략
            if emergency_level is not None:
                em_msg = String()
                em_msg.data = emergency_level
                self.emergency_level_pub.publish(em_msg)

            # 4) 필요 시 annotated image도 compressed로 발행
            if self.publish_annotated:
                annotated_msg = self._encode_compressed_image(annotated, msg.header.stamp)
                self.image_result_pub.publish(annotated_msg)

        except Exception as exc:
            self.get_logger().error(f"image_callback failed: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = SrdPoseEmergencyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
