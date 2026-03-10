# v0.610
"""
SRD Pose Severity Core
======================

이 파일은 ROS2 의존성이 없는 "분석 엔진" 파일이다.
입력으로 OpenCV BGR 프레임(np.ndarray)을 받고,
사람 포즈를 분석해서 다음 두 가지를 반환한다.

1) 시각화가 그려진 annotated frame
2) 구조화된 결과(result list)

즉, 이 파일은 카메라/토픽/네트워크를 몰라도 된다.
오직 "프레임 -> 포즈 분석 -> 상태 판정"만 담당한다.

실제 TurtleBot4 환경에서는 별도의 ROS2 노드가 이 코어를 import 해서 사용한다.
"""

import json
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
    """분석기에 필요한 모든 파라미터를 한곳에 모아둔 설정 클래스.

    코드 리뷰/튜닝 시 가장 먼저 보는 영역이다.
    현장에서 임계값을 바꾸고 싶을 때 이 블록만 수정하면 된다.
    """

    # ---------------------------------------------------------------------
    # Model / Input 관련 파라미터
    # ---------------------------------------------------------------------
    model_path: str = "yolo11n-pose.pt"  # YOLO Pose 모델 파일 경로
    det_conf: float = 0.35               # 사람 탐지 최소 confidence
    kp_conf_th: float = 0.45             # keypoint를 유효하다고 볼 최소 confidence
    kp_margin_px: int = 2                # 프레임 가장자리 keypoint 무시용 margin

    # ---------------------------------------------------------------------
    # Observation(관측 상태) 판단 파라미터
    # ---------------------------------------------------------------------
    low_conf_min_kps: int = 3            # 전체 keypoint가 이 개수보다 적으면 LOW_CONF
    upper_body_min_kps: int = 3          # 상체 keypoint가 이 개수 이상이면 UPPER_BODY 후보
    full_body_extra_lower_kps: int = 1   # hip 외 하체 keypoint가 이 개수 이상이면 FULL_BODY 후보

    # ---------------------------------------------------------------------
    # Posture(자세) 판단 파라미터
    # ---------------------------------------------------------------------
    # shoulder_tilt는 0~90도 범위의 예각으로 정규화된 어깨선 기울기다.
    leaning_shoulder_tilt_deg: float = 25.0
    collapsed_shoulder_tilt_deg: float = 45.0

    # head_down_score는 0~1 범위를 갖는 값이다.
    # 0.0에 가까울수록 얼굴이 어깨보다 충분히 위에 있다.
    # 1.0에 가까울수록 얼굴이 어깨에 가까워지거나 아래로 내려온 상태다.
    leaning_head_down_score: float = 0.55
    collapsed_head_down_score: float = 0.78

    # 전신이 보일 때만 사용하는 torso / lying 관련 기준
    collapsed_torso_angle_deg: float = 60.0
    lying_torso_angle_deg: float = 72.0
    lying_aspect_ratio: float = 2.00

    # 상반신만 보일 때 어깨 간 가로폭이 너무 좁으면 tilt를 신뢰하지 않는다.
    upper_body_min_shoulder_span_ratio: float = 0.25

    # ---------------------------------------------------------------------
    # Motion(움직임) 판단 파라미터
    # ---------------------------------------------------------------------
    motion_window: int = 12              # 최근 N프레임 평균으로 움직임 smoothing
    motion_active_smooth: float = 0.025  # 전체적인 움직임이 이 값 이상이면 ACTIVE
    motion_active_upper: float = 0.030   # 상체 움직임이 이 값 이상이면 ACTIVE
    motion_local_only_upper: float = 0.020
    motion_local_only_core: float = 0.010
    motion_low: float = 0.010            # ACTIVE 미만이지만 이 값 이상이면 LOW

    # ---------------------------------------------------------------------
    # 상태 지속 시간 기준
    # ---------------------------------------------------------------------
    analyzing_sec: float = 1.5
    caution_sec: float = 4.5
    warning_sec: float = 5.5
    critical_sec: float = 7.0

    # ---------------------------------------------------------------------
    # 시각화 옵션
    # ---------------------------------------------------------------------
    show_debug: bool = True              # 디버그 텍스트(tilt/hds/m) 표시 여부
    draw_skeleton: bool = True           # skeleton 시각화 여부
    draw_box: bool = True                # bbox 시각화 여부


class PoseSeverityEngine:
    """사람 포즈 기반 상태 판정 엔진.

    이 클래스는 다음 역할을 담당한다.
    1) YOLO Pose 추론 실행
    2) keypoint 유효성 검사
    3) 관측 상태(FULL_BODY / UPPER_BODY / PARTIAL / LOW_CONF) 분류
    4) 자세(NORMAL / LEANING / COLLAPSED / LYING / UNKNOWN) 분류
    5) 움직임(ACTIVE / LOCAL_ONLY / LOW / NONE) 계산
    6) 시간 지속성을 반영한 최종 severity 결정
    7) annotated frame 및 structured result 생성
    """

    # COCO 17 keypoint index 기준
    UPPER_IDS = [0, 5, 6, 7, 8, 9, 10]
    LOWER_IDS = [11, 12, 13, 14, 15, 16]
    CORE_IDS = [5, 6, 11, 12]

    # 상반신 skeleton 연결
    UPPER_LINKS = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)]

    # 전신 skeleton 연결
    FULL_LINKS = UPPER_LINKS + [
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
    ]

    # severity 시각화 색상(BGR)
    COLORS = {
        "ANALYZING": (255, 255, 255),
        "NORMAL": (0, 200, 0),
        "CAUTION": (0, 220, 255),
        "WARNING": (0, 140, 255),
        "CRITICAL": (0, 0, 255),
    }

    # severity 우선순위. 여러 사람이 잡히더라도 가장 높은 위험도를 대표값으로 사용.
    SEVERITY_PRIORITY = {
        "CRITICAL": 4,
        "WARNING": 3,
        "CAUTION": 2,
        "NORMAL": 1,
        "ANALYZING": 0,
    }

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.cfg = config or AnalyzerConfig()
        self.model = YOLO(self.cfg.model_path)

        # track_id 별 히스토리 저장소
        # - first_seen: 처음 본 시간
        # - prev_kps / prev_conf: 이전 프레임 keypoint, confidence
        # - motion_buf: 최근 움직임 평균 버퍼
        # - last_signature / state_since: 같은 상태가 얼마나 지속되었는지 계산용
        self.history: Dict[int, dict] = {}

    # ------------------------------------------------------------------
    # 기본 유틸리티 함수
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_mean(points: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
        """None이 아닌 점들만 평균 내어 대표점을 만든다.

        예: 양 어깨 평균 -> shoulder_center
        예: 눈/귀/코 평균 -> face_anchor
        """
        valid = [p for p in points if p is not None]
        return np.mean(valid, axis=0) if valid else None

    def _is_valid_kp(self, point: np.ndarray, conf: float, w: int, h: int) -> bool:
        """하나의 keypoint가 유효한지 판단한다.

        조건:
        1) 프레임 내부에 있어야 한다.
        2) confidence가 최소 threshold 이상이어야 한다.
        """
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
        """주어진 index 목록 중 실제로 유효한 keypoint 인덱스만 반환한다."""
        h, w = shape[:2]
        return [i for i in ids if self._is_valid_kp(keypoints[i], kp_conf[i], w, h)]

    def _get_point(
        self,
        keypoints: np.ndarray,
        kp_conf: np.ndarray,
        idx: int,
        shape: Tuple[int, int, int],
    ) -> Optional[np.ndarray]:
        """유효한 keypoint면 해당 좌표를 반환하고, 아니면 None을 반환한다."""
        h, w = shape[:2]
        return keypoints[idx] if self._is_valid_kp(keypoints[idx], kp_conf[idx], w, h) else None

    @staticmethod
    def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
        """두 점을 이은 선의 기울기 절댓값을 0~90도 범위로 정규화해 반환한다.

        주의:
        COCO의 left/right는 사람 기준이다.
        이미지 좌표에서는 left shoulder x > right shoulder x 인 경우가 흔하다.
        그래서 단순 atan2 결과는 177도 같은 값이 나올 수 있다.
        우리는 '방향'이 아니라 '기울기 크기'만 필요하므로 예각(0~90도)으로 정규화한다.
        """
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])
        angle = abs(math.degrees(math.atan2(dy, dx)))
        if angle > 90.0:
            angle = 180.0 - angle
        return angle

    def _new_track_state(self) -> dict:
        """새 track_id가 들어왔을 때 초기 상태를 생성한다."""
        now = time.time()
        return {
            "first_seen": now,
            "prev_kps": None,
            "prev_conf": None,
            "motion_buf": deque(maxlen=self.cfg.motion_window),
            "last_signature": None,
            "state_since": now,
        }

    # ------------------------------------------------------------------
    # 1) 관측 상태 분류
    # ------------------------------------------------------------------
    def _classify_observation(
        self,
        keypoints: np.ndarray,
        kp_conf: np.ndarray,
        shape: Tuple[int, int, int],
    ) -> str:
        """사람이 얼마나 보이는지 관측 상태를 분류한다.

        반환값:
        - LOW_CONF : 전체 keypoint가 너무 적음
        - FULL_BODY: 상체 + 하체가 충분히 보임
        - UPPER_BODY: 상체는 충분히 보이지만 하체는 부족
        - PARTIAL : 일부만 보임
        """
        upper_valid = self._valid_indices(keypoints, kp_conf, self.UPPER_IDS, shape)
        lower_valid = self._valid_indices(keypoints, kp_conf, self.LOWER_IDS, shape)
        total_count = len(upper_valid) + len(lower_valid)

        if total_count < self.cfg.low_conf_min_kps:
            return "LOW_CONF"

        # 현재 규칙:
        # - 상체가 충분히 보이고
        # - hip 중 하나라도 보이고
        # - 추가 하체 관절이 일정 수 이상 보이면 FULL_BODY
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

    # ------------------------------------------------------------------
    # 2) 자세 분류
    # ------------------------------------------------------------------
    def _classify_posture(
        self,
        keypoints: np.ndarray,
        kp_conf: np.ndarray,
        box: np.ndarray,
        obs: str,
        shape: Tuple[int, int, int],
    ) -> Tuple[str, float, float, float]:
        """자세를 분류하고, 디버깅에 필요한 수치도 함께 반환한다.

        반환값:
        - posture: NORMAL / LEANING / COLLAPSED / LYING / UNKNOWN
        - shoulder_tilt: 어깨 기울기 각도
        - head_down_score: 얼굴-어깨 상대 거리 기반 고개 숙임 점수
        - torso_angle: 어깨-골반 축의 기울기 각도 (전신일 때 의미 있음)
        """
        x1, y1, x2, y2 = box.astype(float)
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        aspect = bw / bh

        # 얼굴 / 상체 / 하체 주요 keypoint 불러오기
        nose = self._get_point(keypoints, kp_conf, 0, shape)
        leye = self._get_point(keypoints, kp_conf, 1, shape)
        reye = self._get_point(keypoints, kp_conf, 2, shape)
        lear = self._get_point(keypoints, kp_conf, 3, shape)
        rear = self._get_point(keypoints, kp_conf, 4, shape)
        ls = self._get_point(keypoints, kp_conf, 5, shape)
        rs = self._get_point(keypoints, kp_conf, 6, shape)

        # 전신일 때만 hip를 posture 계산에 사용한다.
        lh = self._get_point(keypoints, kp_conf, 11, shape) if obs == "FULL_BODY" else None
        rh = self._get_point(keypoints, kp_conf, 12, shape) if obs == "FULL_BODY" else None

        shoulder_center = self._safe_mean([ls, rs])
        hip_center = self._safe_mean([lh, rh])
        face_anchor = self._safe_mean([nose, leye, reye, lear, rear])

        shoulder_tilt = self._angle_deg(ls, rs) if ls is not None and rs is not None else 0.0

        shoulder_span = 0.0
        if ls is not None and rs is not None:
            shoulder_span = abs(float(rs[0] - ls[0]))

        # 측면(옆보기) 방어 로직 (전신/상반신 공통 적용)
        if shoulder_span < bw * self.cfg.upper_body_min_shoulder_span_ratio:
            shoulder_tilt = 0.0

        # head_down_score 계산
        # - 기존 head_drop처럼 "얼굴이 어깨 아래로 내려가야" 커지는 방식이 아님
        # - 얼굴이 어깨에 얼마나 가까워졌는지를 0~1 점수로 만든다.
        #   * 얼굴이 충분히 위에 있으면 0에 가까움
        #   * 얼굴이 어깨에 가까워질수록 1에 가까움
        head_down_score = 0.0
        if face_anchor is not None and shoulder_center is not None and shoulder_span > 1.0:
            head_gap = max(float(shoulder_center[1] - face_anchor[1]), 0.0)
            head_down_score = 1.0 - min(head_gap / shoulder_span, 1.0)

        torso_angle = 0.0
        if shoulder_center is not None and hip_center is not None:
            # shoulder_center -> hip_center 선이 수직에서 얼마나 기울어졌는지
            torso_angle = abs(90.0 - self._angle_deg(shoulder_center, hip_center))

        # ------------------------------
        # FULL_BODY posture rule
        # ------------------------------
        if obs == "FULL_BODY":
            # 누움: bbox가 가로로 눕거나, torso가 거의 수평에 가깝게 누운 경우
            if aspect >= self.cfg.lying_aspect_ratio or torso_angle >= self.cfg.lying_torso_angle_deg:
                return "LYING", shoulder_tilt, head_down_score, torso_angle

            # 붕괴: torso / shoulder tilt / head down 중 하나가 충분히 큰 경우
            if (
                torso_angle >= self.cfg.collapsed_torso_angle_deg
                or shoulder_tilt >= self.cfg.collapsed_shoulder_tilt_deg
                or head_down_score >= self.cfg.collapsed_head_down_score
            ):
                return "COLLAPSED", shoulder_tilt, head_down_score, torso_angle

            # 기울어짐: 위 collapsed는 아니지만, shoulder tilt 또는 head_down이 어느 정도 큰 경우
            if (
                shoulder_tilt >= self.cfg.leaning_shoulder_tilt_deg
                or head_down_score >= self.cfg.leaning_head_down_score
            ):
                return "LEANING", shoulder_tilt, head_down_score, torso_angle

            return "NORMAL", shoulder_tilt, head_down_score, torso_angle

        # ------------------------------
        # UPPER_BODY posture rule
        # ------------------------------
        if obs == "UPPER_BODY":
            # 상반신은 단순 규칙:
            # - shoulder tilt 또는 head_down_score 중 큰 쪽을 기준으로 posture 판정
            if (
                shoulder_tilt >= self.cfg.collapsed_shoulder_tilt_deg
                or head_down_score >= self.cfg.collapsed_head_down_score
            ):
                return "COLLAPSED", shoulder_tilt, head_down_score, torso_angle

            if (
                shoulder_tilt >= self.cfg.leaning_shoulder_tilt_deg
                or head_down_score >= self.cfg.leaning_head_down_score
            ):
                return "LEANING", shoulder_tilt, head_down_score, torso_angle

            return "NORMAL", shoulder_tilt, head_down_score, torso_angle

        return "UNKNOWN", shoulder_tilt, head_down_score, torso_angle

    # ------------------------------------------------------------------
    # 3) 움직임 계산 / 분류
    # ------------------------------------------------------------------
    def _motion_value(
        self,
        track_id: int,
        keypoints: np.ndarray,
        kp_conf: np.ndarray,
        box: np.ndarray,
        shape: Tuple[int, int, int],
    ) -> Tuple[float, float, float]:
        """이전 프레임과 현재 프레임 keypoint를 비교해 움직임 크기를 계산한다.

        반환값:
        - smooth: 최근 N프레임 평균 이동량
        - upper : 상체 keypoint 평균 이동량
        - core  : 몸통(core) keypoint 평균 이동량
        """
        hist = self.history.setdefault(track_id, self._new_track_state())

        prev_kps = hist["prev_kps"]
        prev_conf = hist["prev_conf"]
        hist["prev_kps"] = keypoints.copy()
        hist["prev_conf"] = kp_conf.copy()

        x1, y1, x2, y2 = box.astype(float)
        bh = max(y2 - y1, 1.0)
        h, w = shape[:2]

        # 이전 프레임이 없으면 움직임 0으로 시작
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
        """움직임 값을 ACTIVE / LOCAL_ONLY / LOW / NONE 로 분류한다."""
        if upper >= self.cfg.motion_local_only_upper and core <= self.cfg.motion_local_only_core:
            return "LOCAL_ONLY"
        if smooth >= self.cfg.motion_active_smooth or upper >= self.cfg.motion_active_upper:
            return "ACTIVE"
        if smooth >= self.cfg.motion_low:
            return "LOW"
        return "NONE"

    # ------------------------------------------------------------------
    # 4) 보조 판정
    # ------------------------------------------------------------------
    @staticmethod
    def _possible_trapped(obs: str, posture: str, motion: str) -> bool:
        """부분 노출/무움직임 등을 바탕으로 매몰 의심 여부를 계산한다."""
        if obs == "PARTIAL" and motion in ("LOCAL_ONLY", "NONE"):
            return True
        if obs == "UPPER_BODY" and posture == "COLLAPSED" and motion == "NONE":
            return True
        return False

    def _state_duration(self, track_id: int, signature: str) -> Tuple[float, float]:
        """전체 관측 시간과 현재 상태 지속 시간을 계산한다.

        - seen_sec : 이 사람을 처음 본 이후 경과 시간
        - state_sec: 현재 (obs/posture/motion/trapped) 조합이 유지된 시간
        """
        hist = self.history.setdefault(track_id, self._new_track_state())
        now = time.time()
        if hist["last_signature"] != signature:
            hist["last_signature"] = signature
            hist["state_since"] = now
        return now - hist["first_seen"], now - hist["state_since"]

    # ------------------------------------------------------------------
    # 5) 최종 severity 결정
    # ------------------------------------------------------------------
    def _decide(
        self,
        obs: str,
        posture: str,
        motion: str,
        trapped: bool,
        seen_sec: float,
        state_sec: float,
    ) -> str:
        """관측 상태, 자세, 움직임, 시간 지속성을 바탕으로 최종 severity를 결정한다."""
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

    # ------------------------------------------------------------------
    # 6) 시각화
    # ------------------------------------------------------------------
    def _draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        kp_conf: np.ndarray,
        obs: str,
        color: Tuple[int, int, int],
    ) -> None:
        """관측 상태에 맞춰 skeleton과 keypoint를 그린다.

        - FULL_BODY: 상체 + 하체 전체 skeleton
        - 그 외: 상체 skeleton만
        """
        if not self.cfg.draw_skeleton:
            return

        h, w = frame.shape[:2]
        links = self.FULL_LINKS if obs == "FULL_BODY" else self.UPPER_LINKS
        draw_ids = set(self.UPPER_IDS + self.LOWER_IDS) if obs == "FULL_BODY" else set(self.UPPER_IDS)

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
            center = tuple(p.astype(int))
            cv2.circle(frame, center, 4, (255, 255, 255), -1)
            cv2.circle(frame, center, 3, color, -1)

    @staticmethod
    def _pack_result(
        track_id: int,
        bbox: Tuple[int, int, int, int],
        obs: str,
        posture: str,
        motion: str,
        severity: str,
        trapped: bool,
        seen_sec: float,
        state_sec: float,
        shoulder_tilt: float,
        head_down_score: float,
        torso_angle: float,
        motion_smooth: float,
        motion_upper: float,
        motion_core: float,
    ) -> dict:
        """노드/대시보드에서 쓰기 쉬운 결과 dict를 생성한다."""
        return {
            "track_id": int(track_id),
            "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            "observation": obs,
            "posture": posture,
            "motion": motion,
            "severity": severity,
            "trapped": bool(trapped),
            "seen_sec": round(float(seen_sec), 3),
            "state_sec": round(float(state_sec), 3),
            "shoulder_tilt": round(float(shoulder_tilt), 3),
            "head_down_score": round(float(head_down_score), 3),
            "torso_angle": round(float(torso_angle), 3),
            "motion_smooth": round(float(motion_smooth), 5),
            "motion_upper": round(float(motion_upper), 5),
            "motion_core": round(float(motion_core), 5),
        }

    # ------------------------------------------------------------------
    # 외부 호출용 메인 API
    # ------------------------------------------------------------------
    def analyze_frame_with_results(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """프레임 하나를 분석해 annotated image와 structured results를 함께 반환한다.

        이 함수가 이 엔진의 핵심 public API다.
        ROS2 노드에서는 이 함수만 호출하면 된다.
        """
        annotated = frame.copy()
        results: List[dict] = []

        yolo_results = self.model.track(frame, persist=True, verbose=False, conf=self.cfg.det_conf)
        if not yolo_results or yolo_results[0].boxes is None or yolo_results[0].keypoints is None:
            return annotated, results

        boxes_xyxy = yolo_results[0].boxes.xyxy.cpu().numpy()
        keypoints_xy = yolo_results[0].keypoints.xy.cpu().numpy()
        keypoints_conf = (
            yolo_results[0].keypoints.conf.cpu().numpy()
            if yolo_results[0].keypoints.conf is not None
            else np.ones((len(keypoints_xy), keypoints_xy.shape[1]), dtype=np.float32)
        )

        ids = yolo_results[0].boxes.id
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

            # 1) observation
            obs = self._classify_observation(kps, kp_conf, frame.shape)

            # 2) posture
            posture, shoulder_tilt, head_down_score, torso_angle = self._classify_posture(
                kps, kp_conf, clipped_box, obs, frame.shape
            )

            # 3) motion
            smooth, upper, core = self._motion_value(track_id, kps, kp_conf, clipped_box, frame.shape)
            motion = self._classify_motion(smooth, upper, core)

            # 4) trapped / time / severity
            trapped = self._possible_trapped(obs, posture, motion)
            signature = f"{obs}|{posture}|{motion}|{trapped}"
            seen_sec, state_sec = self._state_duration(track_id, signature)
            severity = self._decide(obs, posture, motion, trapped, seen_sec, state_sec)

            # 5) visualization
            color = self.COLORS[severity]
            self._draw_skeleton(annotated, kps, kp_conf, obs, color)

            if self.cfg.draw_box:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            line1 = f"ID {track_id} | {severity}"
            line2 = f"{obs} | {posture} | {motion}"
            line3 = f"tilt:{shoulder_tilt:.1f} hds:{head_down_score:.2f} m:{smooth:.3f}"

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

            # 6) structured result 생성
            results.append(
                self._pack_result(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    obs=obs,
                    posture=posture,
                    motion=motion,
                    severity=severity,
                    trapped=trapped,
                    seen_sec=seen_sec,
                    state_sec=state_sec,
                    shoulder_tilt=shoulder_tilt,
                    head_down_score=head_down_score,
                    torso_angle=torso_angle,
                    motion_smooth=smooth,
                    motion_upper=upper,
                    motion_core=core,
                )
            )

        return annotated, results

    def analyze_frame(self, frame: np.ndarray) -> np.ndarray:
        """annotated frame만 필요한 경우 사용하는 단순 API."""
        annotated, _ = self.analyze_frame_with_results(frame)
        return annotated

    @staticmethod
    def results_to_json(results: List[dict]) -> str:
        """결과 리스트를 JSON 문자열로 변환한다.

        ROS2 String topic, 로깅, 디버깅 용도로 사용 가능하다.
        """
        return json.dumps({"detections": results}, ensure_ascii=False)

    def extract_frame_severity(self, results_list: List[dict]) -> Optional[str]:
        """
        프레임 단위 최종 severity 하나를 만든다.
        현재 시나리오는 한 화면 1명 전제지만, 혹시 여러 개가 잡혀도
        가장 높은 severity를 대표값으로 사용한다.

        사람이 없으면 None 반환.
        """
        if not results_list:
            return None

        return max(
            (r["severity"] for r in results_list),
            key=lambda x: self.SEVERITY_PRIORITY.get(x, -1),
        )

    def analyze_frame_with_severity(self, frame: np.ndarray):
        """
        ROS2 노드에서 쓰기 쉽게
        1) 시각화 프레임
        2) 프레임 대표 severity 하나
        를 반환한다.
        """
        annotated, results_list = self.analyze_frame_with_results(frame)
        severity = self.extract_frame_severity(results_list)
        return annotated, severity
