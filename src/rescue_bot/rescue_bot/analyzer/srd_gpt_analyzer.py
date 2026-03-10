# v0.520
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class AnalyzerConfig:
    # model / input (초기 인식 모델 및 카메라 환경 설정)
    model_path: str = "yolo11n-pose.pt" # 관절/포즈 추정을 위한 YOLOv11 모델 파일 경로
    camera_index: int = 0               # 사용할 카메라 장치 번호 (웹캠 기본값: 0)
    det_conf: float = 0.35              # 객체(사람) 탐지를 위한 최소 신뢰도 임계값 (35% 이상 시 인식)
    kp_conf_th: float = 0.45            # 각 신체 관절(Keypoint) 인식에 대한 최소 신뢰도
    kp_margin_px: int = 2               # 관절 포인트가 화면 가장자리에 너무 붙어있지 않도록 줄 마진(픽셀)

    # observation (신체 부위 관찰 유효성 검증 기준)
    low_conf_min_kps: int = 3           # 전체 신체에서 인식되어야 할 최소 관절 수 (이 이하면 인식 불가로 판정)
    upper_body_min_kps: int = 3         # '상체(Upper Body)'로 인정하기 위한 상반신 최소 관절 인식 수
    full_body_extra_lower_kps: int = 1  # '전신(Full Body)'으로 인정하기 위한 골반 이외의 하체 추가 관절 요구 수

    # posture thresholds (신체 기울기 및 자세 불량 기준 임계값 정의)
    leaning_shoulder_tilt_deg: float = 25.0
    collapsed_shoulder_tilt_deg: float = 45.0
    leaning_head_drop_ratio: float = 0.22
    collapsed_head_drop_ratio: float = 0.38
    collapsed_torso_angle_deg: float = 60.0    # 몸통(어깨-골반 선)이 수직선 기준 이 각도 이상 누우면 쓰러짐 판정
    lying_torso_angle_deg: float = 72.0        # 몸통 각도가 거의 눕다시피 한 수평(Lying) 상태 기준
    lying_aspect_ratio: float = 2.00           # 바운딩 박스 가로/세로 비율 (가로가 세로보다 2.0배 이상 길면 누움)
    upper_body_min_shoulder_span_ratio: float = 0.25

    # motion thresholds (대상 움직임 활동량 평가 기준)
    motion_window: int = 12                    # 평균 움직임을 도출하기 위해 저장할 이전 프레임 개수 (버퍼 크기)
    motion_active_smooth: float = 0.025        # 전신에 걸친 부드러운/일반적인 움직임(Active) 판정 임계수치
    motion_active_upper: float = 0.030         # 상체의 격렬한/뚜렷한 움직임 판정 임계수치
    motion_local_only_upper: float = 0.020     # 팔, 머리 등 일부 관절만 움직이는 국소적(Local Only) 움직임 판단 기준
    motion_local_only_core: float = 0.010      # 국소적 움직임 시, 코어(몸통)는 이 수치 이하로 고정되어 있어야 함
    motion_low: float = 0.010                  # 간헐적이고 매우 미세한 움직임(Low Motion) 기준치 (이 이하면 정지/None 상태)

    # time thresholds (상태의 지속성에 따른 위급도 승급 시간 기준 - 초 단위)
    analyzing_sec: float = 1.5   # 모델이 첫 인식 후 최소한 관찰하며 상태를 파악하는 대기 시간
    caution_sec: float = 4.5     # 특정 비정상 패턴이 이 시간 이상 지속되면 요주의(Caution) 알람
    warning_sec: float = 5.5     # 상태가 악화되거나 요주의 상태 초과 시 경고(Warning) 알람으로 격상
    critical_sec: float = 7.0    # 호흡 멎음, 의식 잃은 쓰러짐 등이 장기화될 때 위급(Critical)으로 최종 선고

    # display (화면 출력 및 시각화 옵션)
    show_debug: bool = True      # 화면 텍스트에 포즈 점수, 움직임 수치 등의 튜닝/디버깅 정보를 띄울지 여부


class SeverityAnalyzer:
    UPPER_IDS = [0, 5, 6, 7, 8, 9, 10]
    LOWER_IDS = [11, 12, 13, 14, 15, 16]
    CORE_IDS = [5, 6, 11, 12]

    UPPER_LINKS = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)]
    FULL_LINKS = UPPER_LINKS + [
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
    ]

    COLORS = {
        "ANALYZING": (255, 255, 255),
        "NORMAL": (0, 200, 0),
        "CAUTION": (0, 220, 255),
        "WARNING": (0, 140, 255),
        "CRITICAL": (0, 0, 255),
    }

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.cfg = config or AnalyzerConfig()
        self.model = YOLO(self.cfg.model_path)
        self.history: Dict[int, dict] = {}

    @staticmethod
    def _safe_mean(points: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
        valid = [p for p in points if p is not None]
        return np.mean(valid, axis=0) if valid else None

    def _is_valid_kp(
        self,
        point: np.ndarray,
        conf: float,
        w: int,
        h: int,
    ) -> bool:
        x, y = float(point[0]), float(point[1])
        m = self.cfg.kp_margin_px
        return (
            m <= x < (w - m)
            and m <= y < (h - m)
            and float(conf) >= self.cfg.kp_conf_th
        )

    def _valid_indices(
        self,
        keypoints: np.ndarray,
        kp_conf: np.ndarray,
        ids: List[int],
        shape: Tuple[int, int, int],
    ) -> List[int]:
        h, w = shape[:2]
        return [i for i in ids if self._is_valid_kp(keypoints[i], kp_conf[i], w, h)]

    def _get_point(
        self,
        keypoints: np.ndarray,
        kp_conf: np.ndarray,
        idx: int,
        shape: Tuple[int, int, int],
    ) -> Optional[np.ndarray]:
        h, w = shape[:2]
        return keypoints[idx] if self._is_valid_kp(keypoints[idx], kp_conf[idx], w, h) else None

    @staticmethod
    def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])
        angle = abs(math.degrees(math.atan2(dy, dx)))
        if angle > 90.0:
            angle = 180.0 - angle
        return angle

    def _classify_observation(
        self,
        keypoints: np.ndarray,
        kp_conf: np.ndarray,
        shape: Tuple[int, int, int],
    ) -> str:
        upper_valid = self._valid_indices(keypoints, kp_conf, self.UPPER_IDS, shape)
        lower_valid = self._valid_indices(keypoints, kp_conf, self.LOWER_IDS, shape)
        total_count = len(upper_valid) + len(lower_valid)

        if total_count < self.cfg.low_conf_min_kps:
            return "LOW_CONF"

        hips_ok = 11 in lower_valid or 12 in lower_valid
        extra_lower = len([i for i in lower_valid if i not in (11, 12)])

        if (
            len(upper_valid) >= self.cfg.upper_body_min_kps
            and hips_ok
            and extra_lower >= self.cfg.full_body_extra_lower_kps
        ):
            return "FULL_BODY"

        if len(upper_valid) >= self.cfg.upper_body_min_kps:
            return "UPPER_BODY"

        return "PARTIAL"

    def _classify_posture(
        self,
        keypoints: np.ndarray,
        kp_conf: np.ndarray,
        box: np.ndarray,
        obs: str,
        shape: Tuple[int, int, int],
    ) -> Tuple[str, float, float]:
        x1, y1, x2, y2 = box.astype(float)
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        aspect = bw / bh

        nose = self._get_point(keypoints, kp_conf, 0, shape)
        leye = self._get_point(keypoints, kp_conf, 1, shape)
        reye = self._get_point(keypoints, kp_conf, 2, shape)
        lear = self._get_point(keypoints, kp_conf, 3, shape)
        rear = self._get_point(keypoints, kp_conf, 4, shape)
        ls = self._get_point(keypoints, kp_conf, 5, shape)
        rs = self._get_point(keypoints, kp_conf, 6, shape)

        # 전신일 때만 하체 사용
        lh = self._get_point(keypoints, kp_conf, 11, shape) if obs == "FULL_BODY" else None
        rh = self._get_point(keypoints, kp_conf, 12, shape) if obs == "FULL_BODY" else None

        shoulder_center = self._safe_mean([ls, rs])
        hip_center = self._safe_mean([lh, rh])
        face_anchor = self._safe_mean([nose, leye, reye, lear, rear])

        shoulder_tilt = self._angle_deg(ls, rs) if ls is not None and rs is not None else 0.0

        head_drop = 0.0
        if face_anchor is not None and shoulder_center is not None:
            head_drop = max(float(face_anchor[1] - shoulder_center[1]), 0.0) / bh

        torso_angle = 0.0
        if shoulder_center is not None and hip_center is not None:
            torso_angle = abs(90.0 - self._angle_deg(shoulder_center, hip_center))

        if obs == "FULL_BODY":
            if aspect >= self.cfg.lying_aspect_ratio or torso_angle >= self.cfg.lying_torso_angle_deg:
                return "LYING", shoulder_tilt, head_drop

            if (
                torso_angle >= self.cfg.collapsed_torso_angle_deg
                or shoulder_tilt >= self.cfg.collapsed_shoulder_tilt_deg
                or head_drop >= self.cfg.collapsed_head_drop_ratio
            ):
                return "COLLAPSED", shoulder_tilt, head_drop

            if (
                shoulder_tilt >= self.cfg.leaning_shoulder_tilt_deg
                or head_drop >= self.cfg.leaning_head_drop_ratio
            ):
                return "LEANING", shoulder_tilt, head_drop

            return "NORMAL", shoulder_tilt, head_drop

        if obs == "UPPER_BODY":
            # 상반신은 어깨 두 점이 너무 좁으면 기울기 계산을 신뢰하지 않음
            if ls is not None and rs is not None:
                shoulder_span = abs(float(rs[0] - ls[0]))
                if shoulder_span < bw * self.cfg.upper_body_min_shoulder_span_ratio:
                    shoulder_tilt = 0.0

            # 상반신에서는 tilt/head_drop 둘 중 하나가 충분히 크면 자세 이상으로 본다.
            # 단, COLLAPSED는 기준을 더 높게 둤서 쉽게 뜨지 않게 함.
            if (
                shoulder_tilt >= self.cfg.collapsed_shoulder_tilt_deg
                or head_drop >= self.cfg.collapsed_head_drop_ratio
            ):
                return "COLLAPSED", shoulder_tilt, head_drop

            if (
                shoulder_tilt >= self.cfg.leaning_shoulder_tilt_deg
                or head_drop >= self.cfg.leaning_head_drop_ratio
            ):
                return "LEANING", shoulder_tilt, head_drop

            return "NORMAL", shoulder_tilt, head_drop

        return "UNKNOWN", shoulder_tilt, head_drop

    def _motion_value(
        self,
        track_id: int,
        keypoints: np.ndarray,
        kp_conf: np.ndarray,
        box: np.ndarray,
        shape: Tuple[int, int, int],
    ) -> Tuple[float, float, float]:
        hist = self.history.setdefault(
            track_id,
            {
                "first_seen": time.time(),
                "prev_kps": None,
                "prev_conf": None,
                "motion_buf": deque(maxlen=self.cfg.motion_window),
                "last_signature": None,
                "state_since": time.time(),
            },
        )

        prev_kps = hist["prev_kps"]
        prev_conf = hist["prev_conf"]
        hist["prev_kps"] = keypoints.copy()
        hist["prev_conf"] = kp_conf.copy()

        x1, y1, x2, y2 = box.astype(float)
        bh = max(y2 - y1, 1.0)
        h, w = shape[:2]

        if prev_kps is None or prev_conf is None:
            hist["motion_buf"].append(0.0)
            return 0.0, 0.0, 0.0

        def avg_disp(ids: List[int]) -> float:
            vals = []
            for i in ids:
                p1, p0 = keypoints[i], prev_kps[i]
                c1, c0 = kp_conf[i], prev_conf[i]
                if self._is_valid_kp(p1, c1, w, h) and self._is_valid_kp(p0, c0, w, h):
                    vals.append(float(np.linalg.norm(p1 - p0) / bh))
            return float(np.mean(vals)) if vals else 0.0

        upper = avg_disp(self.UPPER_IDS)
        core = avg_disp(self.CORE_IDS)
        smooth = max(upper, core)
        hist["motion_buf"].append(smooth)
        smooth = float(np.mean(hist["motion_buf"])) if hist["motion_buf"] else 0.0
        return smooth, upper, core

    def _classify_motion(self, smooth: float, upper: float, core: float) -> str:
        if upper >= self.cfg.motion_local_only_upper and core <= self.cfg.motion_local_only_core:
            return "LOCAL_ONLY"
        if smooth >= self.cfg.motion_active_smooth or upper >= self.cfg.motion_active_upper:
            return "ACTIVE"
        if smooth >= self.cfg.motion_low:
            return "LOW"
        return "NONE"

    @staticmethod
    def _possible_trapped(obs: str, posture: str, motion: str) -> bool:
        if obs == "PARTIAL" and motion in ("LOCAL_ONLY", "NONE"):
            return True
        if obs == "UPPER_BODY" and posture == "COLLAPSED" and motion == "NONE":
            return True
        return False

    def _state_duration(self, track_id: int, signature: str) -> Tuple[float, float]:
        hist = self.history[track_id]
        now = time.time()
        if hist["last_signature"] != signature:
            hist["last_signature"] = signature
            hist["state_since"] = now
        return now - hist["first_seen"], now - hist["state_since"]

    def _decide(
        self,
        obs: str,
        posture: str,
        motion: str,
        trapped: bool,
        seen_sec: float,
        state_sec: float,
    ) -> str:
        if seen_sec < self.cfg.analyzing_sec:
            return "ANALYZING"

        if obs == "FULL_BODY":
            if posture in ("LYING", "COLLAPSED") and motion == "NONE" and state_sec >= self.cfg.critical_sec:
                return "CRITICAL"
            if posture in ("LYING", "COLLAPSED") and motion in ("LOW", "NONE") and state_sec >= self.cfg.warning_sec:
                return "WARNING"
            if (posture == "LEANING" and motion == "NONE" and state_sec >= self.cfg.caution_sec) or (
                posture == "NORMAL" and motion == "NONE" and state_sec >= self.cfg.warning_sec
            ):
                return "CAUTION"
            return "NORMAL"

        if obs == "UPPER_BODY":
            if motion in ("ACTIVE", "LOCAL_ONLY"):
                return "NORMAL"
            if posture == "COLLAPSED" and motion == "NONE" and state_sec >= self.cfg.warning_sec:
                return "WARNING"
            return "CAUTION"

        if obs == "PARTIAL":
            if motion in ("ACTIVE", "LOCAL_ONLY"):
                return "NORMAL"
            if trapped and motion == "NONE" and state_sec >= self.cfg.critical_sec:
                return "WARNING"
            return "CAUTION"

        return "CAUTION"

    def _draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        kp_conf: np.ndarray,
        obs: str,
        color: Tuple[int, int, int],
    ) -> None:
        h, w = frame.shape[:2]

        if obs == "FULL_BODY":
            links = self.FULL_LINKS
            draw_ids = set(self.UPPER_IDS + self.LOWER_IDS)
        else:
            links = self.UPPER_LINKS
            draw_ids = set(self.UPPER_IDS)

        for a, b in links:
            pa, pb = keypoints[a], keypoints[b]
            ca, cb = kp_conf[a], kp_conf[b]
            if self._is_valid_kp(pa, ca, w, h) and self._is_valid_kp(pb, cb, w, h):
                cv2.line(frame, tuple(pa.astype(int)), tuple(pb.astype(int)), color, 2)

        for i, p in enumerate(keypoints):
            if i not in draw_ids:
                continue
            if not self._is_valid_kp(p, kp_conf[i], w, h):
                continue
            cv2.circle(frame, tuple(p.astype(int)), 4, (255, 255, 255), -1)
            cv2.circle(frame, tuple(p.astype(int)), 3, color, -1)

    def analyze_frame(self, frame: np.ndarray) -> np.ndarray:
        annotated = frame.copy()
        results = self.model.track(frame, persist=True, verbose=False, conf=self.cfg.det_conf)

        if not results or results[0].boxes is None or results[0].keypoints is None:
            return annotated

        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        keypoints_xy = results[0].keypoints.xy.cpu().numpy()

        if results[0].keypoints.conf is not None:
            keypoints_conf = results[0].keypoints.conf.cpu().numpy()
        else:
            keypoints_conf = np.ones((len(keypoints_xy), keypoints_xy.shape[1]), dtype=np.float32)

        ids = results[0].boxes.id
        track_ids = ids.int().cpu().tolist() if ids is not None else list(range(len(boxes_xyxy)))

        for track_id, box, kps, kp_conf in zip(track_ids, boxes_xyxy, keypoints_xy, keypoints_conf):
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1] - 1, x2)
            y2 = min(frame.shape[0] - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            clipped_box = np.array([x1, y1, x2, y2], dtype=np.float32)

            obs = self._classify_observation(kps, kp_conf, frame.shape)
            posture, shoulder_tilt, head_drop = self._classify_posture(kps, kp_conf, clipped_box, obs, frame.shape)
            smooth, upper, core = self._motion_value(track_id, kps, kp_conf, clipped_box, frame.shape)
            motion = self._classify_motion(smooth, upper, core)
            trapped = self._possible_trapped(obs, posture, motion)
            seen_sec, state_sec = self._state_duration(track_id, f"{obs}|{posture}|{motion}|{trapped}")
            severity = self._decide(obs, posture, motion, trapped, seen_sec, state_sec)

            color = self.COLORS[severity]
            self._draw_skeleton(annotated, kps, kp_conf, obs, color)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            line1 = f"ID {track_id} | {severity}"
            line2 = f"{obs} | {posture} | {motion}"
            line3 = f"tilt:{shoulder_tilt:.1f} hd:{head_drop:.2f} m:{smooth:.3f}"

            ty = max(20, y1 - 10)
            cv2.putText(annotated, line1, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            if self.cfg.show_debug:
                cv2.putText(
                    annotated,
                    line2,
                    (x1, min(frame.shape[0] - 25, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    line3,
                    (x1, min(frame.shape[0] - 5, y2 + 42)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        return annotated


def main() -> None:
    cfg = AnalyzerConfig()
    analyzer = SeverityAnalyzer(cfg)
    cap = cv2.VideoCapture(cfg.camera_index)

    if not cap.isOpened():
        print("Failed to open camera.")
        return

    print("SRD severity analyzer started. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        annotated = analyzer.analyze_frame(frame)
        cv2.imshow("SRD Severity Analyzer", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Analyzer stopped.")


if __name__ == "__main__":
    main()