# ============================================================================
# 파일명: robot6_control_node_v0_100.py
# 버전: v0.210
# 날짜: 2026-03-11
# 변경사항:
# - v0.100: robot6 관제노드 1차 뼈대 작성
# - v0.110: 상태머신(SEARCH/FRAME/VERIFY/MEASURING/RESULT_LOCKED/WAIT_TTS_DONE) 정리
# - v0.120: body 타입별 집계(full/upper/partial) 및 최종 result JSON 추가
# - v0.130: victim_position(depth + K + TF map 변환) 로직 추가
# - v0.140: target selection / yaw align / backoff / verify 흐름 정리
# - v0.150: session reset / TTS request / final result publish 추가
# - v0.160: latest raw bytes만 저장하고 step()에서 디코딩하도록 변경
# - v0.170: target lost patience(timeout) 로직 추가
# - v0.180: MultiThreadedExecutor + callback group 구조 반영
# - v0.190: FULL_BODY too_close도 backoff 후보에 포함되도록 제어 보강
# - v0.200: 실행형 main, 예외 처리, 상태 publish, 결과 집계 함수 정리
# - v0.210: core export `rep_point_px` 최우선 사용 / _select_position_pixel fallback 적용
# ============================================================================

from __future__ import annotations

import json
import math
import threading
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
try:
    from .rescue_vision_core import AnalyzerConfig, PoseEmergencyEngine
except ImportError:
    from rescue_vision_core import AnalyzerConfig, PoseEmergencyEngine


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class Robot6ControlNode(Node):
    def __init__(self) -> None:
        super().__init__("robot6_control_node")

        # ------------------------------------------------------------------
        # ROS2 parameters
        # ------------------------------------------------------------------
        self.declare_parameter("model_path", "yolo11n-pose.pt")
        self.declare_parameter("rgb_topic", "/robot6/oakd/rgb/preview/image_raw")
        self.declare_parameter("depth_topic", "/robot6/oakd/stereo/image_raw")
        self.declare_parameter("camera_info_topic", "/robot6/oakd/stereo/camera_info")
        self.declare_parameter("arrived_topic", "/robot6/mission/arrived")
        self.declare_parameter("tts_done_topic", "/robot6/tts/done")
        self.declare_parameter("cmd_vel_topic", "/robot6/cmd_vel")
        self.declare_parameter("result_topic", "/robot6/session/result")
        self.declare_parameter("status_topic", "/robot6/session/status")
        self.declare_parameter("tts_req_topic", "/robot6/tts/request")
        self.declare_parameter("image_result_topic", "/robot6/image_result")
        self.declare_parameter("victim_pose_topic", "/rescue/victim_pose_stamped")
        self.declare_parameter("timer_period_sec", 0.10)
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("show_debug", True)

        self.declare_parameter("center_tol_px", 100)
        self.declare_parameter("edge_margin_px", 5)
        self.declare_parameter("verify_n", 10)
        self.declare_parameter("measure_min_sec", 10.0)
        self.declare_parameter("measure_min_stable_frames", 50)
        self.declare_parameter("yaw_kp", 0.002)
        self.declare_parameter("max_yaw_speed", 0.5)
        self.declare_parameter("search_yaw_speed", 0.10)
        self.declare_parameter("backoff_speed", -0.05)
        self.declare_parameter("backoff_h_ratio", 0.95)
        self.declare_parameter("backoff_w_ratio", 0.72)
        self.declare_parameter("depth_too_close_m", 0.90)
        self.declare_parameter("target_lost_timeout_sec", 0.50)
        self.declare_parameter("depth_patch_r", 3)
        self.declare_parameter("depth_offset_m", -0.04)
        self.declare_parameter("robot_id", "robot6")

        model_path = str(self.get_parameter("model_path").value)
        rgb_topic = str(self.get_parameter("rgb_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        arrived_topic = str(self.get_parameter("arrived_topic").value)
        tts_done_topic = str(self.get_parameter("tts_done_topic").value)
        cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        result_topic = str(self.get_parameter("result_topic").value)
        status_topic = str(self.get_parameter("status_topic").value)
        tts_req_topic = str(self.get_parameter("tts_req_topic").value)
        image_result_topic = str(self.get_parameter("image_result_topic").value)
        victim_pose_topic = str(self.get_parameter("victim_pose_topic").value)
        self.timer_period_sec = float(self.get_parameter("timer_period_sec").value)
        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)
        show_debug = bool(self.get_parameter("show_debug").value)

        self.CENTER_TOL_PX = int(self.get_parameter("center_tol_px").value)
        self.EDGE_MARGIN_PX = int(self.get_parameter("edge_margin_px").value)
        self.VERIFY_N = int(self.get_parameter("verify_n").value)
        self.MEASURE_MIN_SEC = float(self.get_parameter("measure_min_sec").value)
        self.MEASURE_MIN_STABLE_FRAMES = int(self.get_parameter("measure_min_stable_frames").value)
        self.YAW_KP = float(self.get_parameter("yaw_kp").value)
        self.MAX_YAW_SPEED = float(self.get_parameter("max_yaw_speed").value)
        self.SEARCH_YAW_SPEED = float(self.get_parameter("search_yaw_speed").value)
        self.BACKOFF_SPEED = float(self.get_parameter("backoff_speed").value)
        self.BACKOFF_H_RATIO = float(self.get_parameter("backoff_h_ratio").value)
        self.BACKOFF_W_RATIO = float(self.get_parameter("backoff_w_ratio").value)
        self.DEPTH_TOO_CLOSE_M = float(self.get_parameter("depth_too_close_m").value)
        self.TARGET_LOST_TIMEOUT_SEC = float(self.get_parameter("target_lost_timeout_sec").value)
        self.DEPTH_PATCH_R = int(self.get_parameter("depth_patch_r").value)
        self.DEPTH_OFFSET_M = float(self.get_parameter("depth_offset_m").value)
        self.robot_id = str(self.get_parameter("robot_id").value)

        # ------------------------------------------------------------------
        # Core engine
        # ------------------------------------------------------------------
        cfg = AnalyzerConfig(model_path=model_path, show_debug=show_debug)
        self.engine = PoseEmergencyEngine(cfg)

        # ------------------------------------------------------------------
        # TF / threading / callback groups
        # ------------------------------------------------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.sensor_lock = threading.Lock()

        self.sensor_group = MutuallyExclusiveCallbackGroup()
        self.control_group = MutuallyExclusiveCallbackGroup()
        self.timer_group = MutuallyExclusiveCallbackGroup()

        # ------------------------------------------------------------------
        # Latest sensor cache (raw bytes only)
        # ------------------------------------------------------------------
        self.latest_rgb_msg: Optional[Image] = None
        self.latest_rgb_stamp = None
        self.latest_depth_msg: Optional[Image] = None
        self.latest_depth_frame_id: Optional[str] = None
        self.K: Optional[np.ndarray] = None

        # ------------------------------------------------------------------
        # Session/runtime variables
        # ------------------------------------------------------------------
        self.state = "WAIT_ARRIVAL"
        self.session_id: Optional[str] = None
        self.session_start_t: Optional[float] = None
        self.session_start_iso: Optional[str] = None
        self.measure_start_t: Optional[float] = None
        self.last_target_x = 320
        self.measure_end_t: Optional[float] = None

        self.session_active = False
        self.result_locked = False
        self.tts_requested = False
        self.tts_done = False
        self.framing_forced = False
        self.state_start_t = time.time()

        self.current_track_id: Optional[int] = None
        self.current_target: Optional[Dict[str, Any]] = None
        self.last_valid_target: Optional[Dict[str, Any]] = None
        self.last_seen_t = 0.0
        self.last_target_x = 320

        self.verify_ok_count = 0
        self.stable_frame_count = 0
        self.low_conf_count = 0
        self.no_person_count = 0
        self.framing_forced = False

        self.bucket_overall: List[Dict[str, Any]] = []
        self.bucket_full_body: List[Dict[str, Any]] = []
        self.bucket_upper_body: List[Dict[str, Any]] = []
        self.bucket_partial: List[Dict[str, Any]] = []

        self.victim_map_points: List[Dict[str, float]] = []
        self.victim_method_hist: List[str] = []
        self.valid_position_samples = 0

        self.result_snapshot: Optional[Dict[str, Any]] = None
        self.last_annotated: Optional[np.ndarray] = None
        self.bridge = CvBridge()

        # ------------------------------------------------------------------
        # ROS pubs/subs/timer
        # ------------------------------------------------------------------
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.result_pub = self.create_publisher(String, result_topic, 10)
        self.status_pub = self.create_publisher(String, status_topic, 10)
        self.tts_req_pub = self.create_publisher(String, tts_req_topic, 10)
        self.image_pub = self.create_publisher(Image, image_result_topic, 10)
        self.image_compressed_pub = self.create_publisher(CompressedImage, image_result_topic + "/compressed", 10)
        self.victim_pose_pub = self.create_publisher(PoseStamped, victim_pose_topic, 10)

        self.create_subscription(
            Bool,
            arrived_topic,
            self.arrived_callback,
            10,
            callback_group=self.control_group,
        )
        self.create_subscription(
            Bool,
            tts_done_topic,
            self.tts_done_callback,
            10,
            callback_group=self.control_group,
        )
        self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10,
            callback_group=self.sensor_group,
        )
        self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            qos_profile_sensor_data,
            callback_group=self.sensor_group,
        )
        self.create_subscription(
            Image,
            rgb_topic,
            self.rgb_callback,
            qos_profile_sensor_data,
            callback_group=self.sensor_group,
        )

        self.timer = self.create_timer(self.timer_period_sec, self.step, callback_group=self.timer_group)
        self.get_logger().info("robot6_control_node started")

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def arrived_callback(self, msg: Bool) -> None:
        self.get_logger().info(f"Received Arrived signal: {msg.data}")
        if not msg.data:
            return
        if self.state != "WAIT_ARRIVAL":
            self.get_logger().warn(f"Ignoring Arrived signal. Current state is {self.state}, not WAIT_ARRIVAL")
            return

        self.get_logger().info(">>> SESSION STARTING: Target Arrived successfully! <<<")
        self.reset_session()
        self.session_id = self._make_session_id()
        self.session_start_t = time.time()
        self.session_start_iso = datetime.now().isoformat(timespec="seconds")
        self.session_active = True
        self.state = "SEARCH_PERSON"
        self.get_logger().info(f"Created Session ID: {self.session_id}")

    def tts_done_callback(self, msg: Bool) -> None:
        if msg.data and self.state == "WAIT_TTS_DONE":
            self.tts_done = True
            self.state = "SESSION_END"

    def camera_info_callback(self, msg: CameraInfo) -> None:
        with self.sensor_lock:
            self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)

    def rgb_callback(self, msg: Image) -> None:
        with self.sensor_lock:
            self.latest_rgb_msg = msg
            self.latest_rgb_stamp = msg.header.stamp

    def depth_callback(self, msg: Image) -> None:
        with self.sensor_lock:
            self.latest_depth_msg = msg
            self.latest_depth_frame_id = msg.header.frame_id

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def step(self) -> None:
        try:
            # 1. 센서 데이터 스냅샷 가져오기 (항상 수행)
            snapshot = self._get_sensor_snapshot()
            
            # 주기적인 상태 로그 (데이터가 없어도 현재 상태 확인 가능하게)
            now_t = time.time()
            if not hasattr(self, '_last_state_log_t'): self._last_state_log_t = 0
            if (now_t - self._last_state_log_t) > 2.0:
                self.get_logger().info(f"[Step] Current State: {self.state} | Person In View: {self.current_target is not None}")
                self._last_state_log_t = now_t

            if snapshot is None:
                self._publish_status()
                return

            # 2. YOLO 분석 및 타겟팅 (항상 수행 - Always-on Video)
            # WAIT_TTS_DONE이나 SESSION_END 상태일 때는 이전 어노테이션 유지하거나 원본 출력
            if self.state in ("WAIT_TTS_DONE", "SESSION_END"):
                annotated = self.last_annotated if self.last_annotated is not None else snapshot["rgb"]
                target = None
            else:
                annotated, results = self.engine.analyze_frame_with_results(snapshot["rgb"])
                self.last_annotated = annotated.copy()
                target = self._select_target(results, snapshot["rgb"].shape)
                self.current_target = target
                # 주행 중(WAIT_ARRIVAL)에는 트래킹 북키핑 생략하여 노이즈 방지 가능
                if self.state != "WAIT_ARRIVAL":
                    self._update_tracking_bookkeeping(target)

            # 3. 상태별 로직 처리
            if self.state == "WAIT_ARRIVAL":
                self.stop_motion()
            elif self.state == "SEARCH_PERSON":
                self._handle_search_person(target)
            elif self.state == "FRAME_PERSON":
                self._handle_frame_person(target, snapshot)
            elif self.state == "VERIFY_FRAME":
                self._handle_verify_frame(target, snapshot)
            elif self.state == "MEASURING":
                self._handle_measuring(target, snapshot)
            elif self.state == "RESULT_LOCKED":
                self._handle_result_locked()
            elif self.state == "WAIT_TTS_DONE":
                self.stop_motion()
            elif self.state == "SESSION_END":
                self._publish_final_result()
                self.reset_session()

            # 4. 상태 및 영상 퍼블리시 (항상 수행)
            self._publish_status()
            if self.publish_annotated:
                self._publish_annotated(annotated, snapshot.get("rgb_stamp"))

        except Exception as exc:
            self.get_logger().error(f"step failed: {exc}")

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------
    def _handle_search_person(self, target: Optional[Dict[str, Any]]) -> None:
        now = time.time()
        if target is None:
            if self._target_recently_seen(now):
                self.stop_motion()
                return
            self.search_rotate()
            return

        self.stop_motion()
        self.state = "FRAME_PERSON"

    def _handle_frame_person(self, target: Optional[Dict[str, Any]], snapshot: Dict[str, Any]) -> None:
        now = time.time()
        if target is None:
            if self._target_recently_seen(now):
                self.stop_motion()
                return
            self.state = "SEARCH_PERSON"
            return

        # 프레이밍 타임아웃 (10초 이상 조준 실패 시 강제 진행)
        if (now - self.state_start_t) > 10.0:
            self.get_logger().warn("Framing timeout reached (10s). Forcing transition to VERIFY_FRAME with framing_forced=True.")
            self.framing_forced = True
            self.stop_motion()
            self.verify_ok_count = 0
            self.state = "VERIFY_FRAME"
            self.state_start_t = now
            return

        action = self._decide_frame_action(target, snapshot)
        if action == "SEARCH":
            if self._target_recently_seen(now):
                self.stop_motion()
            else:
                self.state = "SEARCH_PERSON"
                self.state_start_t = now
        elif action == "YAW_ALIGN":
            self.cmd_vel_pub.publish(self._make_yaw_cmd(target, snapshot["rgb"].shape[1]))
        elif action == "BACKOFF":
            self.cmd_vel_pub.publish(self._make_backoff_cmd())
        else:
            self.stop_motion()
            self.verify_ok_count = 0
            self.state = "VERIFY_FRAME"
            self.state_start_t = now

    def _handle_verify_frame(self, target: Optional[Dict[str, Any]], snapshot: Dict[str, Any]) -> None:
        now = time.time()
        if target is None:
            if self._target_recently_seen(now):
                self.stop_motion()
                return
            self.verify_ok_count = 0
            self.state = "SEARCH_PERSON"
            return

        if not self.framing_forced and self._decide_frame_action(target, snapshot) != "HOLD":
            self.verify_ok_count = 0
            # FRAME_PERSON으로 돌아가더라도 timer를 reset하지 않음 (전체 framing timeout 10s 유지)
            self.state = "FRAME_PERSON"
            return
        self.verify_ok_count += 1
        self.stop_motion()
        if self.verify_ok_count >= self.VERIFY_N:
            self.measure_start_t = time.time()
            self.measure_end_t = None
            self.stable_frame_count = 0
            self.state = "MEASURING"

    def _handle_measuring(self, target: Optional[Dict[str, Any]], snapshot: Dict[str, Any]) -> None:
        now = time.time()
        if target is None:
            if self._target_recently_seen(now):
                self.stop_motion()
                return
            self.state = "SEARCH_PERSON"
            self.state_start_t = now
            return

        if not self.framing_forced and self._decide_frame_action(target, snapshot) != "HOLD":
            # MEASURING 중이라도 Jitter 발생 시 돌아가되 timer reset 안함
            self.state = "FRAME_PERSON"
            return

        self.stop_motion()
        self._accumulate_sample(target, snapshot)
        if self.measure_start_t is None:
            self.measure_start_t = now

        if (
            (now - self.measure_start_t) >= self.MEASURE_MIN_SEC
            and self.stable_frame_count >= self.MEASURE_MIN_STABLE_FRAMES
        ):
            self.get_logger().info("Analysis finished. Locking results...")
            self.measure_end_t = now
            self.state = "RESULT_LOCKED"

    def _handle_result_locked(self) -> None:
        self.stop_motion()
        if self.result_snapshot is None:
            self.result_snapshot = self.build_session_result()
            self.result_locked = True
            
            # Fix KeyError: 'observation' -> access via nested structure
            obs_majority = self.result_snapshot.get("overall", {}).get("dominant_observation", "UNKNOWN")
            self.get_logger().info(f"Result locked: Dominant Observation = {obs_majority}")

        if not self.tts_requested:
            # STT Node expects a simple uppercase status string (e.g. "CRITICAL")
            # fallback to "NORMAL" only if data is truly missing
            overall = self.result_snapshot.get("overall", {})
            peak = overall.get("emergency_peak")
            status_to_send = str(peak if peak else "NORMAL").upper()
            
            self.get_logger().info(f"Requesting TTS announcement with status: {status_to_send}")
            msg = String()
            msg.data = status_to_send
            self.tts_req_pub.publish(msg)
            self.tts_requested = True

        self.state = "WAIT_TTS_DONE"

    # ------------------------------------------------------------------
    # Sensor snapshot / decoding
    # ------------------------------------------------------------------
    def _get_sensor_snapshot(self) -> Optional[Dict[str, Any]]:
        with self.sensor_lock:
            rgb_msg = self.latest_rgb_msg
            rgb_stamp = self.latest_rgb_stamp
            depth_msg = self.latest_depth_msg
            frame_id = self.latest_depth_frame_id
            K = None if self.K is None else self.K.copy()

        if rgb_msg is None:
            return None

        rgb = self._decode_rgb(rgb_msg)
        if rgb is None:
            return None

        depth = self._decode_depth(depth_msg)
        return {
            "rgb": rgb,
            "rgb_stamp": rgb_stamp,
            "depth": depth,
            "depth_frame_id": frame_id,
            "K": K,
        }

    def _decode_rgb(self, msg: Image) -> Optional[np.ndarray]:
        try:
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"decode_rgb failed: {exc}")
            return None

    def _decode_depth(self, msg: Optional[Image]) -> Optional[np.ndarray]:
        if msg is None:
            return None
        try:
            # Ignition Gazebo bridges depth as 32FC1 (meters)
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as exc:
            self.get_logger().error(f"decode_depth failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Target selection / tracking bookkeeping
    # ------------------------------------------------------------------
    def _select_target(self, results: List[Dict[str, Any]], frame_shape: Tuple[int, ...]) -> Optional[Dict[str, Any]]:
        if not results:
            return None

        frame_h, frame_w = frame_shape[:2]
        frame_cx = frame_w * 0.5
        vis_rank = {
            "FULL_BODY": 3.0,
            "UPPER_BODY": 2.0,
            "PARTIAL": 1.0,
            "LOW_CONF": 0.0,
        }

        best: Optional[Dict[str, Any]] = None
        best_score = -1e18

        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            bw = max(1.0, float(x2 - x1))
            bh = max(1.0, float(y2 - y1))
            area = bw * bh
            box_cx = 0.5 * (x1 + x2)

            vis_score = 1000.0 * vis_rank.get(r.get("observation"), 0.0)
            center_score = -abs(box_cx - frame_cx)
            area_score = 0.001 * area
            track_bonus = 10000.0 if r.get("track_id") == self.current_track_id else 0.0

            score = vis_score + center_score + area_score + track_bonus
            if score > best_score:
                best_score = score
                best = r

        return best

    def _update_tracking_bookkeeping(self, target: Optional[Dict[str, Any]]) -> None:
        if target is None:
            return
        self.current_track_id = int(target["track_id"])
        self.last_valid_target = dict(target)
        x1, _, x2, _ = target["bbox"]
        self.last_target_x = int((x1 + x2) * 0.5)
        self.last_seen_t = time.time()

    def _target_recently_seen(self, now_t: float) -> bool:
        return (now_t - self.last_seen_t) <= self.TARGET_LOST_TIMEOUT_SEC

    # ------------------------------------------------------------------
    # Framing control
    # ------------------------------------------------------------------
    def _decide_frame_action(self, target: Dict[str, Any], snapshot: Dict[str, Any]) -> str:
        obs = str(target.get("observation", "LOW_CONF"))
        if obs == "LOW_CONF":
            return "SEARCH"

        frame_h, frame_w = snapshot["rgb"].shape[:2]
        x1, y1, x2, y2 = target["bbox"]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx_box = 0.5 * (x1 + x2)
        err_x = cx_box - (0.5 * frame_w)

        if abs(err_x) > self.CENTER_TOL_PX:
            return "YAW_ALIGN"

        w_ratio = bw / frame_w
        h_ratio = bh / frame_h
        touch_left = x1 <= self.EDGE_MARGIN_PX
        touch_right = x2 >= frame_w - self.EDGE_MARGIN_PX
        touch_top = y1 <= self.EDGE_MARGIN_PX
        touch_bottom = y2 >= frame_h - self.EDGE_MARGIN_PX

        depth_m = self._estimate_target_depth(target, snapshot)
        too_close_by_box = (
            h_ratio >= self.BACKOFF_H_RATIO
            or w_ratio >= self.BACKOFF_W_RATIO
            or touch_top
            or touch_bottom
        )
        too_close_by_depth = depth_m is not None and depth_m < self.DEPTH_TOO_CLOSE_M

        # FULL_BODY도 정말 가깝게 붙었으면 backoff 후보로 인정
        if too_close_by_box or too_close_by_depth:
            return "BACKOFF"

        if touch_left or touch_right:
            return "YAW_ALIGN"

        return "HOLD"

    def _make_yaw_cmd(self, target: Dict[str, Any], frame_w: int) -> Twist:
        x1, _, x2, _ = target["bbox"]
        err_x = 0.5 * (x1 + x2) - (0.5 * frame_w)

        twist = Twist()
        yaw = -self.YAW_KP * err_x
        twist.angular.z = float(clamp(yaw, -self.MAX_YAW_SPEED, self.MAX_YAW_SPEED))
        twist.linear.x = 0.0
        return twist

    def _make_backoff_cmd(self) -> Twist:
        twist = Twist()
        twist.linear.x = float(self.BACKOFF_SPEED)
        twist.angular.z = 0.0
        return twist

    def stop_motion(self) -> None:
        self.cmd_vel_pub.publish(Twist())

    def search_rotate(self) -> None:
        twist = Twist()
        direction = 1.0 if self.last_target_x < 320 else -1.0
        twist.angular.z = float(direction * self.SEARCH_YAW_SPEED)
        self.cmd_vel_pub.publish(twist)

    # ------------------------------------------------------------------
    # Measurement / accumulation
    # ------------------------------------------------------------------
    def _accumulate_sample(self, target: Dict[str, Any], snapshot: Dict[str, Any]) -> None:
        obs = str(target.get("observation", "LOW_CONF"))
        sample = {
            "track_id": target.get("track_id"),
            "observation": obs,
            "posture": target.get("posture"),
            "motion": target.get("motion"),
            "emergency_level": target.get("emergency_level"),
            "trapped": bool(target.get("trapped", False)),
            "shoulder_tilt": target.get("shoulder_tilt"),
            "head_drop_ratio": target.get("head_drop_ratio"),
            "torso_angle": target.get("torso_angle"),
            "motion_smooth": target.get("motion_smooth"),
            "motion_upper": target.get("motion_upper"),
            "motion_core": target.get("motion_core"),
        }

        self.bucket_overall.append(sample)

        if obs == "FULL_BODY":
            self.bucket_full_body.append(sample)
        elif obs == "UPPER_BODY":
            self.bucket_upper_body.append(sample)
        elif obs == "PARTIAL":
            self.bucket_partial.append(sample)
        else:
            self.low_conf_count += 1
            return

        pos = self._estimate_victim_position(target, snapshot)
        if pos is not None:
            self.victim_map_points.append(pos["point"])
            self.victim_method_hist.append(pos["method"])
            self.valid_position_samples += 1

        self.stable_frame_count += 1

    # ------------------------------------------------------------------
    # Depth / position helpers
    # ------------------------------------------------------------------
    def sample_depth(self, depth_img: np.ndarray, x: int, y: int) -> Optional[float]:
        if depth_img is None:
            return None
        h, w = depth_img.shape[:2]
        x = int(clamp(x, 0, w - 1))
        y = int(clamp(y, 0, h - 1))
        r = self.DEPTH_PATCH_R

        x1 = max(0, x - r)
        x2 = min(w, x + r + 1)
        y1 = max(0, y - r)
        y2 = min(h, y + r + 1)

        patch = depth_img[y1:y2, x1:x2].astype(np.float32) / 1000.0
        patch += self.DEPTH_OFFSET_M
        patch[patch <= 0.0] = np.nan

        valid = patch[np.isfinite(patch)]
        if valid.size == 0:
            return None

        z = float(np.nanmedian(valid))
        if not math.isfinite(z) or z <= 0.0:
            return None
        return z

    def _estimate_target_depth(self, target: Dict[str, Any], snapshot: Dict[str, Any]) -> Optional[float]:
        depth = snapshot.get("depth")
        rgb = snapshot.get("rgb")
        if depth is None or rgb is None:
            return None
        u, v, _ = self._select_position_pixel(target)
        if u is None or v is None:
            return None

        # RGB와 Depth 해상도가 다를 경우 좌표 스케일링 (실기체 대응)
        rgb_h, rgb_w = rgb.shape[:2]
        depth_h, depth_w = depth.shape[:2]
        u_scaled = u * (depth_w / rgb_w)
        v_scaled = v * (depth_h / rgb_h)

        return self.sample_depth(depth, int(u_scaled), int(v_scaled))

    def _select_position_pixel(self, target: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[str]]:
        # 1) core가 준 대표점이 있으면 최우선 사용
        rep = target.get("rep_point_px")
        rep_method = target.get("rep_point_method")
        if (
            isinstance(rep, (list, tuple))
            and len(rep) == 2
            and rep[0] is not None
            and rep[1] is not None
        ):
            return int(rep[0]), int(rep[1]), str(rep_method or "REP_POINT")

        # 2) 보조 중심점 fallback
        shoulder = target.get("shoulder_center_px")
        if (
            isinstance(shoulder, (list, tuple))
            and len(shoulder) == 2
            and shoulder[0] is not None
            and shoulder[1] is not None
        ):
            return int(shoulder[0]), int(shoulder[1]), "SHOULDER_CENTER"

        face = target.get("face_anchor_px")
        if (
            isinstance(face, (list, tuple))
            and len(face) == 2
            and face[0] is not None
            and face[1] is not None
        ):
            return int(face[0]), int(face[1]), "FACE_ANCHOR"

        # 3) 마지막 fallback: 기존 bbox 근사
        x1, y1, x2, y2 = target["bbox"]
        obs = str(target.get("observation", "LOW_CONF"))

        if obs == "FULL_BODY":
            return int((x1 + x2) * 0.5), int(y1 + (y2 - y1) * 0.52), "BBOX_APPROX_SHOULDER_HIP_MID"
        if obs == "UPPER_BODY":
            return int((x1 + x2) * 0.5), int(y1 + (y2 - y1) * 0.35), "BBOX_APPROX_SHOULDER_CENTER"
        if obs == "PARTIAL":
            return int((x1 + x2) * 0.5), int((y1 + y2) * 0.5), "BBOX_CENTER"
        return None, None, None

    def _estimate_victim_position(self, target: Dict[str, Any], snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        depth = snapshot.get("depth")
        K = snapshot.get("K")
        frame_id = snapshot.get("frame_id")
        if depth is None or K is None or frame_id is None:
            return None

        u, v, method = self._select_position_pixel(target)
        if u is None or v is None or method is None:
            return None

        # RGB와 Depth 해상도가 다를 경우 좌표 스케일링 (실기체 대응)
        rgb = snapshot.get("rgb")
        if rgb is not None:
            rgb_h, rgb_w = rgb.shape[:2]
            depth_h, depth_w = depth.shape[:2]
            u_scaled = u * (depth_w / rgb_w)
            v_scaled = v * (depth_h / rgb_h)
        else:
            u_scaled, v_scaled = u, v

        z = self.sample_depth(depth, int(u_scaled), int(v_scaled))
        if z is None:
            return None

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        if fx == 0.0 or fy == 0.0:
            return None

        X = (float(u) - cx) * z / fx
        Y = (float(v) - cy) * z / fy

        pt_camera = PointStamped()
        pt_camera.header.stamp = self.get_clock().now().to_msg()
        pt_camera.header.frame_id = frame_id
        pt_camera.point.x = float(X)
        pt_camera.point.y = float(Y)
        pt_camera.point.z = float(z)

        try:
            pt_map = self.tf_buffer.transform(pt_camera, "map", timeout=Duration(seconds=0.3))
        except Exception as exc:
            self.get_logger().warn(f"TF map transform failed: {exc}")
            return None

        return {
            "method": method,
            "point": {
                "x": float(pt_map.point.x),
                "y": float(pt_map.point.y),
                "z": float(pt_map.point.z),
            },
        }

    # ------------------------------------------------------------------
    # Result / summary helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _priority(level: Optional[str]) -> int:
        return {
            "ANALYZING": 0,
            "NORMAL": 1,
            "CAUTION": 2,
            "WARNING": 3,
            "CRITICAL": 4,
        }.get(level or "", -1)

    @staticmethod
    def _majority(values: List[Any], default: Any = None) -> Any:
        vals = [v for v in values if v is not None]
        if not vals:
            return default
        return Counter(vals).most_common(1)[0][0]

    def _peak_level(self, values: List[Optional[str]], default: Optional[str] = None) -> Optional[str]:
        vals = [v for v in values if v is not None]
        if not vals:
            return default
        return max(vals, key=self._priority)

    @staticmethod
    def _mean(values: List[Any], ndigits: int = 3) -> Optional[float]:
        vals = [float(v) for v in values if v is not None]
        if not vals:
            return None
        return round(sum(vals) / len(vals), ndigits)

    @staticmethod
    def _ratio_true(values: List[Any], ndigits: int = 3) -> Optional[float]:
        vals = [bool(v) for v in values]
        if not vals:
            return None
        return round(sum(vals) / len(vals), ndigits)

    def _summarize_bucket(self, bucket: List[Dict[str, Any]], bucket_name: str) -> Dict[str, Any]:
        if not bucket:
            return {
                "count": 0,
                "posture_majority": None,
                "motion_majority": None,
                "emergency_peak": None,
                "head_drop_ratio_mean": None,
                "shoulder_tilt_mean": None,
                "torso_angle_mean": None,
                "trapped_ratio": None,
            }

        return {
            "count": len(bucket),
            "posture_majority": self._majority([x.get("posture") for x in bucket]),
            "motion_majority": self._majority([x.get("motion") for x in bucket]),
            "emergency_peak": self._peak_level([x.get("emergency_level") for x in bucket]),
            "head_drop_ratio_mean": self._mean([x.get("head_drop_ratio") for x in bucket]),
            "shoulder_tilt_mean": self._mean([x.get("shoulder_tilt") for x in bucket]),
            "torso_angle_mean": self._mean([x.get("torso_angle") for x in bucket]) if bucket_name == "full_body" else None,
            "trapped_ratio": self._ratio_true([x.get("trapped") for x in bucket]),
        }

    def _summarize_position(self) -> Dict[str, Any]:
        if not self.victim_map_points:
            return {
                "frame_id": "map",
                "x": None,
                "y": None,
                "z": None,
                "sample_count": 0,
                "method_majority": None,
                "position_confidence": 0.0,
            }

        xs = [p["x"] for p in self.victim_map_points]
        ys = [p["y"] for p in self.victim_map_points]
        zs = [p.get("z", 0.0) for p in self.victim_map_points]

        mx = float(sum(xs) / len(xs))
        my = float(sum(ys) / len(ys))
        mz = float(sum(zs) / len(zs))
        dists = [math.hypot(x - mx, y - my) for x, y in zip(xs, ys)]
        spread = float(sum(dists) / len(dists)) if dists else 999.0
        confidence = max(0.0, min(1.0, 1.0 - spread / 1.5))

        return {
            "frame_id": "map",
            "x": round(mx, 3),
            "y": round(my, 3),
            "z": round(mz, 3),
            "sample_count": len(self.victim_map_points),
            "method_majority": self._majority(self.victim_method_hist),
            "position_confidence": round(confidence, 3),
        }

    def build_session_result(self) -> Dict[str, Any]:
        overall = self.bucket_overall
        full_body = self.bucket_full_body
        upper_body = self.bucket_upper_body
        partial = self.bucket_partial

        start_iso = self.session_start_iso
        end_iso = datetime.now().isoformat(timespec="seconds")
        duration = 0.0
        if self.measure_start_t is not None:
            end_t = self.measure_end_t if self.measure_end_t is not None else time.time()
            duration = max(0.0, end_t - self.measure_start_t)

        return {
            "session_id": self.session_id,
            "robot_id": self.robot_id,
            "start_time": start_iso,
            "end_time": end_iso,
            "overall": {
                "emergency_peak": self._peak_level([x.get("emergency_level") for x in overall], default=None),
                "emergency_majority": self._majority([x.get("emergency_level") for x in overall]),
                "dominant_observation": self._majority([x.get("observation") for x in overall]),
                "frame_counts": {
                    "full_body": len(full_body),
                    "upper_body": len(upper_body),
                    "partial": len(partial),
                    "low_conf": self.low_conf_count,
                },
                "measurement_duration_sec": round(duration, 3),
            },
            "body_summary": {
                "full_body": self._summarize_bucket(full_body, "full_body"),
                "upper_body": self._summarize_bucket(upper_body, "upper_body"),
                "partial": self._summarize_bucket(partial, "partial"),
            },
            "victim_position": self._summarize_position(),
            "tts": {
                "requested": self.tts_requested,
                "done": self.tts_done,
            },
            "quality": {
                "stable_frame_count": self.stable_frame_count,
                "valid_position_samples": self.valid_position_samples,
            },
        }

    def _build_tts_text(self, result: Dict[str, Any]) -> str:
        overall = result.get("overall", {})
        victim_position = result.get("victim_position", {})
        peak = overall.get("emergency_peak")
        majority = overall.get("emergency_majority")
        x = victim_position.get("x")
        y = victim_position.get("y")
        if x is not None and y is not None:
            return f"요구조자 상태는 최대 {peak}, 주요 판정은 {majority}입니다. 위치는 맵 기준 x {x}, y {y} 입니다."
        return f"요구조자 상태는 최대 {peak}, 주요 판정은 {majority}입니다."

    # ------------------------------------------------------------------
    # Publish helpers
    # ------------------------------------------------------------------
    # 결과 발행 기능 (JSON 리포트 및 네비게이션용 PoseStamped)
    # ------------------------------------------------------------------
    def _publish_final_result(self) -> None:
        if self.result_snapshot is None:
            return

        # 1. 전체 JSON 리포트 발행
        self.get_logger().info("최종 미션 리포트를 발행합니다...")
        msg = String()
        msg.data = json.dumps(self.result_snapshot, ensure_ascii=False)
        self.result_pub.publish(msg)
        self.get_logger().info("JSON 리포트 전송 완료.")

        # 2. 요구조자 위치(PoseStamped) 발행 (네비게이션 노드 연동용)
        v_pos = self.result_snapshot.get("victim_position", {})
        vx = v_pos.get("x")
        vy = v_pos.get("y")
        vz = v_pos.get("z", 0.0)

        if vx is not None and vy is not None:
            self.get_logger().info(f"요구조자 위치를 발행합니다: x={vx}, y={vy}")
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "map"
            pose_msg.pose.position.x = float(vx)
            pose_msg.pose.position.y = float(vy)
            pose_msg.pose.position.z = float(vz)
            # 회전값은 기본값(정면)으로 설정
            pose_msg.pose.orientation.w = 1.0
            
            self.victim_pose_pub.publish(pose_msg)
            self.get_logger().info("요구조자 위치(PoseStamped) 발행 완료.")
        else:
            self.get_logger().warn("요구조자 좌표를 찾을 수 없어 PoseStamped를 발행하지 않습니다.")

    def _publish_status(self) -> None:
        payload = {
            "session_id": self.session_id,
            "state": self.state,
            "session_active": self.session_active,
            "tts_requested": self.tts_requested,
            "tts_done": self.tts_done,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)

    def _publish_annotated(self, frame: np.ndarray, stamp=None) -> None:
        try:
            if stamp:
                ros_stamp = stamp
            else:
                ros_stamp = self.get_clock().now().to_msg()

            # 1. Raw Image Publish (for RQT)
            raw_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            raw_msg.header.stamp = ros_stamp
            raw_msg.header.frame_id = "oakd_rgb_camera_optical_frame"
            self.image_pub.publish(raw_msg)

            # 2. Compressed Image Publish (for Web/Efficiency)
            comp_msg = CompressedImage()
            comp_msg.header.stamp = ros_stamp
            comp_msg.header.frame_id = "oakd_rgb_camera_optical_frame"
            comp_msg.format = "jpeg"
            success, encoded_image = cv2.imencode(".jpg", frame)
            if success:
                comp_msg.data = encoded_image.tobytes()
                self.image_compressed_pub.publish(comp_msg)

        except Exception as exc:
            self.get_logger().error(f"failed to publish image: {exc}")

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------
    def _make_session_id(self) -> str:
        return f"{self.robot_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def reset_session(self) -> None:
        if hasattr(self.engine, "reset"):
            try:
                self.engine.reset()
            except Exception as exc:
                self.get_logger().warn(f"engine.reset() failed: {exc}")

        self.stop_motion()

        self.state = "WAIT_ARRIVAL"
        self.session_id = None
        self.session_start_t = None
        self.session_start_iso = None
        self.measure_start_t = None
        self.measure_end_t = None

        self.session_active = False
        self.result_locked = False
        self.tts_requested = False
        self.tts_done = False
        self.framing_forced = False
        self.state_start_t = time.time()
        
        self.get_logger().info('[Reset] Session has been reset. Ready for next target.')

        self.get_logger().info("Session reset. Waiting for next target arrival...")

        self.current_track_id = None
        self.current_target = None
        self.last_valid_target = None
        self.last_seen_t = 0.0
        self.last_target_x = 320

        self.verify_ok_count = 0
        self.stable_frame_count = 0
        self.low_conf_count = 0
        self.no_person_count = 0

        self.bucket_overall.clear()
        self.bucket_full_body.clear()
        self.bucket_upper_body.clear()
        self.bucket_partial.clear()

        self.victim_map_points.clear()
        self.victim_method_hist.clear()
        self.valid_position_samples = 0

        self.result_snapshot = None
        self.bridge = CvBridge()
        self.last_annotated = None


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(args=None) -> None:
    rclpy.init(args=args)
    node = Robot6ControlNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_motion()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
