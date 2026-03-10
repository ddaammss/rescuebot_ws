import cv2
import numpy as np
import time
import os
import json
from collections import defaultdict, deque

# [추가] 라이브러리 버전 충돌 및 Qt 관련 경고 메시지 억제
os.environ['QT_LOGGING_RULES'] = 'qt.qpa.fonts.warning=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # TensorFlow 관련 로그 최소화
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# ROS 2 관련 라이브러리
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge

from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image

class SrdAdvancedAnalyzer(Node):
    """
    신체 일부가 가려진 상황에서 객체를 통합 분석하고
    가중치 기반으로 위급도를 판별하는 재난 특화 분석 노드입니다.
    """
    def __init__(self):
        super().__init__('srd_advanced_analyzer_node')
        
        self.get_logger().info("🚨 [SRD] 재난 특화 분석 모델 로딩 중...")
        self.model = YOLO("yolo11n-pose.pt")
        
        # ROS 2 파라미터 설정
        # ---------------------------------------------------------
        self.declare_parameter('weight_lying', 1.0)       
        self.declare_parameter('weight_tilt', 0.8)        
        self.declare_parameter('pose_score_threshold', 40.0) 
        self.declare_parameter('motion_critical', 2.0)    
        self.declare_parameter('motion_caution', 1.0)     
        self.declare_parameter('depth_std_threshold', 1.5) 
        self.declare_parameter('kpt_conf_threshold', 0.5)  
        self.declare_parameter('min_observation_time', 2.0) 
        self.declare_parameter('buffer_size', 30)           
        
        # [신규] 신체 부위 결합 관련 파라미터
        self.declare_parameter('part_merge_dist', 150.0)    # 객체 간 통합 거리 임계값 (픽셀)
        # ---------------------------------------------------------

        self.data_pub = self.create_publisher(String, '/srd/severity_data', 10)
        self.image_pub = self.create_publisher(ROSImage, '/srd/processed_image', 10)
        self.bridge = CvBridge()
        
        self.font = self._load_font()
        self.patient_history = {} 
        
        self.skeleton_links = [
            (5, 6), (5, 11), (6, 12), (11, 12),
            (5, 7), (7, 9), (6, 8), (8, 10),    
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        self.get_logger().info("✅ 재난 특화 로직 가동 (신체 부위 결합 기능 포함)")

    def _get_patient_data(self, track_id):
        if track_id not in self.patient_history:
            buf_size = self.get_parameter('buffer_size').value
            self.patient_history[track_id] = {
                'prev_gray_roi': None,
                'motion_buffer': deque(maxlen=buf_size),
                'depth_buffer': deque(maxlen=buf_size),
                'first_seen': time.time(),
                'is_part_of_group': False, # 다른 객체와 결합되었는지 여부
                'merged_with': None        # 결합된 상대 track_id
            }
        return self.patient_history[track_id]

    def _load_font(self):
        paths = ["malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
        for path in paths:
            try: return ImageFont.truetype(path, 20)
            except: continue
        return None

    def draw_korean_text(self, img, text, pos, color):
        if self.font is None:
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            return img
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos, text, font=self.font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def draw_skeleton(self, img, keypoints, confidences, color):
        h, w = img.shape[:2]
        conf_th = self.get_parameter('kpt_conf_threshold').value
        for s, e in self.skeleton_links:
            pt1, pt2 = keypoints[s], keypoints[e]
            c1, c2 = confidences[s], confidences[e]
            if (c1 > conf_th and c2 > conf_th):
                if (0 < pt1[0] < w and 0 < pt1[1] < h) and (0 < pt2[0] < w and 0 < pt2[1] < h):
                    cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 2)

    def process_frame(self, frame):
        # 파라미터 로드
        merge_dist_th = self.get_parameter('part_merge_dist').value
        w_lying = self.get_parameter('weight_lying').value
        w_tilt = self.get_parameter('weight_tilt').value
        p_threshold = self.get_parameter('pose_score_threshold').value
        m_crit = self.get_parameter('motion_critical').value
        m_caut = self.get_parameter('motion_caution').value
        d_std_th = self.get_parameter('depth_std_threshold').value
        min_obs = self.get_parameter('min_observation_time').value

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.model.track(frame, persist=True, conf=0.5, verbose=False)
        annotated_frame = frame.copy()
        depth_frame = np.random.normal(1500, 10, gray_frame.shape)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            keypoints_data = results[0].keypoints.xy.cpu().numpy()
            keypoints_conf = results[0].keypoints.conf.cpu().numpy()

            # [Module 0] 신체 부위 결합(Merge) 분석 전처리
            # 각 객체의 중심점과 주요 부위(상체/하체) 보유 여부 파악
            detected_objs = []
            for i, tid in enumerate(track_ids):
                kp, kc = keypoints_data[i], keypoints_conf[i]
                center = np.mean(boxes[i].reshape(2, 2), axis=0)
                # 상체 포인트(0~10)와 하체 포인트(11~16)의 신뢰도 합계로 부위 판단
                has_upper = np.sum(kc[0:11] > 0.5) > 3
                has_lower = np.sum(kc[11:17] > 0.5) > 2
                detected_objs.append({'id': tid, 'idx': i, 'center': center, 'upper': has_upper, 'lower': has_lower})

            # 결합 쌍 찾기
            merge_pairs = {}
            for i in range(len(detected_objs)):
                for j in range(i + 1, len(detected_objs)):
                    obj1, obj2 = detected_objs[i], detected_objs[j]
                    dist = np.linalg.norm(obj1['center'] - obj2['center'])
                    
                    # 서로 다른 부위가 가깝게 붙어있는 경우 (예: 한쪽은 상체만, 한쪽은 하체만 보임)
                    if dist < merge_dist_th:
                        if (obj1['upper'] and not obj1['lower'] and obj2['lower'] and not obj2['upper']) or \
                           (obj2['upper'] and not obj2['lower'] and obj1['lower'] and not obj1['upper']):
                            merge_pairs[obj1['id']] = obj2['id']
                            merge_pairs[obj2['id']] = obj1['id']

            # 개별 객체 분석 루프
            for box, track_id, kp, kc in zip(boxes, track_ids, keypoints_data, keypoints_conf):
                patient = self._get_patient_data(track_id)
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                # 결합 정보 업데이트
                patient['is_part_of_group'] = track_id in merge_pairs
                patient['merged_with'] = merge_pairs.get(track_id)

                # [Module 1] 자세 분석
                nose_y = kp[0][1]
                hip_y = (kp[11][1] + kp[12][1]) / 2 if (kc[11] > 0.5 and kc[12] > 0.5) else 0
                lying_score = max(0, nose_y - hip_y + 100) if (kc[0] > 0.5 and hip_y > 0) else 0
                shoulder_tilt = abs(kp[5][1] - kp[6][1]) if (kc[5] > 0.5 and kc[6] > 0.5) else 0
                
                pose_total_score = (lying_score * w_lying) + (shoulder_tilt * w_tilt)
                is_abnormal_pose = pose_total_score > p_threshold

                # [Module 2] 움직임 분석
                motion_score = 0
                if patient['prev_gray_roi'] is not None:
                    try:
                        prev_roi = cv2.resize(patient['prev_gray_roi'], (x2-x1, y2-y1))
                        diff = cv2.absdiff(prev_roi, gray_frame[y1:y2, x1:x2])
                        motion_score = np.sum(diff) / (diff.size)
                    except: pass
                patient['prev_gray_roi'] = gray_frame[y1:y2, x1:x2]
                patient['motion_buffer'].append(motion_score)
                avg_motion = np.mean(patient['motion_buffer'])

                # [Module 3] 호흡 검증
                chest_y, chest_x = int((kp[5][1]+kp[6][1])/2), int((x1+x2)/2)
                depth_std = 10.0
                if (kc[5] > 0.5 and kc[6] > 0.5) and (0 < chest_y < frame.shape[0]):
                    d_val = np.mean(depth_frame[max(0, chest_y-2):chest_y+3, max(0, chest_x-2):chest_x+3])
                    patient['depth_buffer'].append(d_val)
                    depth_std = np.std(patient['depth_buffer']) if len(patient['depth_buffer']) > 10 else 10.0

                # [Module 4] 최종 판별 (재난 특화 로직 반영)
                severity, status_text, color = "ANALYZING", "판별 중...", (255, 255, 255)
                
                if (time.time() - patient['first_seen']) > min_obs:
                    severity, status_text, color = "NORMAL", "양호 (의식 있음)", (0, 255, 0)
                    
                    # 1. 신체 분리 매몰 상황인 경우 (최우선 위급)
                    if patient['is_part_of_group']:
                        severity, status_text, color = "CRITICAL", "위급 (파편 매몰/신체 분리 감지)", (0, 0, 255)
                    # 2. 일반 자세/움직임 기반 판별
                    elif is_abnormal_pose:
                        if avg_motion < m_crit and depth_std < d_std_th:
                            severity, status_text, color = "CRITICAL", "위급 (의식 불명/호흡 미약)", (0, 0, 255)
                        else:
                            severity, status_text, color = "WARNING", "주의 (거동 불가/부상 의심)", (255, 255, 0)
                    elif avg_motion < m_caut:
                        severity, status_text, color = "CAUTION", "주의 (장시간 정지/상태 관찰)", (0, 165, 255)

                # 데이터 발행
                payload = {"track_id": int(track_id), "severity": severity, "status_msg": status_text, "motion_score": float(avg_motion), "is_lying": bool(is_abnormal_pose)}
                self.data_pub.publish(String(data=json.dumps(payload)))

                # 시각화
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                self.draw_skeleton(annotated_frame, kp, kc, color)
                annotated_frame = self.draw_korean_text(annotated_frame, status_text, (x1, y1 - 35), color)
                
                # 결합 상태 표시 (선으로 연결)
                if patient['is_part_of_group'] and patient['merged_with'] in track_ids:
                    # 상대방 객체 찾기
                    other_idx = np.where(track_ids == patient['merged_with'])[0][0]
                    other_center = detected_objs[other_idx]['center']
                    cv2.line(annotated_frame, (int(patient['center'][0]), int(patient['center'][1])), 
                             (int(other_center[0]), int(other_center[1])), (0, 0, 255), 1, cv2.LINE_AA)

                # 디버깅 정보
                debug_p = f"POSTURE: {pose_total_score:.1f}"
                debug_m = f"MOTION: {avg_motion:.1f}"
                box_width, box_height = 160, 45
                cv2.rectangle(annotated_frame, (x2, y1), (x2 + box_width, y1 + box_height), (0,0,0), -1)
                cv2.putText(annotated_frame, debug_p, (x2 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(annotated_frame, debug_m, (x2 + 5, y1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            self.image_pub.publish(img_msg)
        except: pass
            
        return annotated_frame

def main(args=None):
    rclpy.init(args=args)
    node = SrdAdvancedAnalyzer()
    cap = cv2.VideoCapture(0)
    print("🚑 [SRD] 신체 분리 매몰 감지 로직이 활성화되었습니다.")
    try:
        while rclpy.ok():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (800, int(frame.shape[0] * (800 / frame.shape[1]))))
            display_frame = node.process_frame(frame)
            cv2.imshow("SRD Disaster Specialist View", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            rclpy.spin_once(node, timeout_sec=0)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()