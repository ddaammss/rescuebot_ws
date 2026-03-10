# 재난구조 로봇(RR:Rescue Robot)

## 1. 프로젝트 개요

본 프로젝트는 **2대의 TurtleBot4(`robot5`, `robot6`)와 4대의 PC**를 이용하여,
미지 환경에서 요구조자를 탐지하고 구조 대응을 수행하는 **다중 로봇 협업형 구조 지원 시스템**을 구현하는 것을 목표로 한다.

전체 시나리오는 다음과 같다.

1. `robot5`가 먼저 미지 환경을 주행하며 **SLAM 기반 맵핑**을 수행한다.
2. `robot5`가 맵 생성을 완료하고, 저장된 맵 또는 안정화된 `map` 기준으로 localization 가능한 상태를 만든다.
3. 이후 `robot5`가 RGB/Depth 카메라와 YOLO를 이용해 **요구조자를 탐지**하고, 해당 위치를 **완성된 맵 좌표계 기준으로 추정**한다.
4. 탐지된 요구조자 위치를 `robot6`에게 전달한다.
5. `robot6`는 전달받은 좌표로 이동한 뒤, **YOLO-Pose 기반 상태 분석**을 수행한다.
6. `robot6`는 **STT/TTS 기반 음성 상호작용**을 통해 요구조자와 간단한 의사소통을 수행하고, 구호 물품을 제공한다.
7. 전체 과정은 메인 PC의 **대시보드**에서 통합 모니터링한다.

이 시스템은 단순 객체 탐지를 넘어, **SLAM, 다중 로봇 협업, 사람 상태 인식, 음성 인터랙션, 관제 대시보드**를 하나의 구조 시나리오로 연결하는 것을 목표로 한다.

---

## 2. 시스템 구성

### 2.1 하드웨어 구성

- **TurtleBot4 2대**
  - `robot5`
  - `robot6`
- **웹캠 2대**
- **PC 4대**
  - `robot5` 담당 PC
  - `robot6` 담당 PC
  - 메인 대시보드 PC
  - 웹캠 입력 및 토픽 퍼블리시 PC

### 2.2 네트워크 구성

- 모든 장비는 **같은 Wi-Fi**에 연결
- 동일한 **ROS 2 Domain ID** 사용
- 멀티머신 ROS 2 통신 환경 구성
- 권장 사항
  - 고정 IP 사용
  - 시간 동기화(NTP/chrony)
  - 영상 토픽은 필요 시 압축 전송
  - rosbridge는 메인 PC에서 운영

---

## 3. 장비별 역할 분담

| 장비 | 역할 |
|---|---|
| `robot5` + 담당 PC | SLAM 기반 맵핑, 맵 저장/안정화, 저장 맵 기준 localization, 요구조자 1차 탐지, 요구조자 위치 추정 및 전송 |
| `robot6` + 담당 PC | 전달받은 위치로 이동, 요구조자 정밀 상태 분석, STT/TTS 상호작용, 구호 물품 제공 |
| 메인 PC | 대시보드 운영, 맵/이벤트/탐지 상태/로봇 상태 통합 시각화 |
| 웹캠 PC | 웹캠 2대 영상 수집, ROS 이미지 토픽 publish, 외부 시점 제공 |

---

## 4. 전체 시나리오

### 단계 1. `robot5`의 선행 탐색 및 맵 생성
- `robot5`가 미지 환경을 자율 주행한다.
- SLAM을 통해 환경 지도를 생성한다.
- 생성된 맵을 저장하거나, 더 이상 크게 변하지 않는 안정화된 상태로 만든다.

### 단계 2. 저장 맵 기준 재로컬라이제이션
- `robot5`는 완성된 맵 기준으로 자신의 위치를 다시 안정적으로 추정한다.
- 이후 탐지 결과는 진행 중인 SLAM 좌표가 아니라, **완성된 `map` 기준 좌표계**에 기록된다.

### 단계 3. 요구조자 탐지 및 위치 추정
- `robot5`가 RGB/Depth 카메라와 YOLO를 이용해 요구조자를 탐지한다.
- Depth 정보와 TF 변환을 이용하여, 요구조자의 위치를 **이미지 좌표가 아닌 완성된 `map` 좌표계 기준 위치**로 변환한다.

### 단계 4. 구조 임무 전달
- `robot5`는 탐지된 요구조자의 위치를 `robot6`에게 전달한다.
- 전달 방식은 토픽, 서비스, 액션, 또는 중앙 DB/서버 연동 중 하나로 설계할 수 있다.

### 단계 5. `robot6`의 현장 접근
- `robot6`는 전달받은 좌표를 목표로 설정한다.
- `robot5`가 생성한 맵 또는 공유된 공통 맵을 기반으로 localization 후 목표 위치까지 이동한다.

### 단계 6. 요구조자 상태 정밀 분석
- `robot6`는 YOLO-Pose를 이용해 요구조자의 자세를 분석한다.
- 필요 시 표정, 움직임, lying 상태 등을 종합하여 요구조자의 상태를 분류한다.

### 단계 7. 음성 상호작용
- STT를 이용해 요구조자의 발화를 인식한다.
- TTS를 이용해 질문, 안내, 응답을 수행한다.
- 예시:
  - “괜찮으세요?”
  - “움직일 수 있으신가요?”
  - “구호 물품을 제공하겠습니다.”

### 단계 8. 구호 물품 제공
- 로봇 팔이 없으므로, 실제 구현은 “직접 손에 쥐여주는 전달”보다는 아래 방식에 가깝다.
  - 상단 적재함/트레이 제공
  - 특정 위치에 물품 놓기
  - 음성 안내를 통한 사용 유도

---

## 5. 핵심 기술 요소

### 5.1 공간 인식 및 자율주행
- SLAM
- `map` 생성 및 공유
- localization
- goal navigation

### 5.2 요구조자 탐지 및 상태 판단
- 저장 맵 기준 요구조자 탐지 및 좌표화
- YOLO 기반 요구조자 탐지
- YOLO-Pose 기반 자세 분석
- Motion / posture / facial expression 기반 상태 추정

### 5.3 다중 로봇 협업
- `robot5` → `robot6` 임무 전달
- 공통 좌표계 기반 목표 위치 공유
- 역할 분업
  - `robot5`: 탐색/발견
  - `robot6`: 접근/판단/대응

### 5.4 음성 인터랙션
- STT 기반 음성 인식
- TTS 기반 안내/응답
- 단순 구조 대화 흐름 관리

### 5.5 관제 시스템
- rosbridge 기반 웹 대시보드
- 맵 시각화
- 탐지 상태 및 이벤트 로그 표시
- 외부 웹캠 연동

---

## 6. 전체 아키텍처 개념도

```text
[robot5 PC]
  ├─ SLAM
  ├─ 탐색 주행
  ├─ 맵 저장/안정화
  ├─ localization
  ├─ YOLO 요구조자 탐지
  └─ victim pose 추정
          │
          ▼
   [요구조자 위치/임무 전달]
          │
          ▼
[robot6 PC]
  ├─ localization / navigation
  ├─ goal 이동
  ├─ YOLO-Pose 상태 분석
  ├─ STT / TTS
  └─ 구호 물품 제공
          │
          ├───────────────┐
          ▼               ▼
    [DB / 로그]      [메인 PC 대시보드]
                           ├─ map 시각화
                           ├─ 탐지 요약
                           ├─ 이벤트 로그
                           └─ 외부 웹캠 표시

[웹캠 PC]
  ├─ webcam 1 capture
  └─ webcam 2 capture
          │
          ▼
     ROS image topics
```

---

## 7. ROS 2 권장 노드 구성

### 7.1 `robot5` 측 노드
- SLAM 노드
- map 저장/관리 노드
- localization 노드
- 카메라 드라이버 노드
- YOLO 탐지 노드
- victim localization 노드
- victim report publisher

예시 토픽:
- `/robot5/map`
- `/robot5/camera/rgb/image_raw`
- `/robot5/camera/depth/image_raw`
- `/robot5/victim_detection`
- `/robot5/victim_pose`

### 7.2 `robot6` 측 노드
- localization 노드
- navigation 노드
- mission subscriber
- YOLO-Pose 분석 노드
- speech interface 노드
- mission result publisher

예시 토픽:
- `/robot6/goal_pose`
- `/robot6/pose_assessment`
- `/robot6/tts/text`
- `/robot6/stt/result`
- `/robot6/mission_status`

### 7.3 메인 PC 노드
- dashboard server
- rosbridge server
- event aggregator
- logger / database interface

예시 구독 토픽:
- `/map`
- `/robot5/victim_pose`
- `/robot6/mission_status`
- `/robot6/pose_assessment`

### 7.4 웹캠 PC 노드
- webcam1 capture publisher
- webcam2 capture publisher

예시 토픽:
- `/external_cam1/image_raw`
- `/external_cam2/image_raw`

---

## 8. 프로젝트에서 가장 중요한 설계 포인트

### 8.1 공통 좌표계 사용
이 프로젝트에서 가장 중요한 것은 **좌표계 통일**이다.

`robot5`가 탐지한 요구조자의 위치를 `robot6`가 정확히 찾아가기 위해서는,
두 로봇이 모두 **같은 `map` 기준 좌표계**를 공유해야 한다.

즉, 아래 조건이 반드시 만족되어야 한다.
- `robot5`가 생성하고 저장한 맵을 `robot6`가 사용할 수 있어야 함
- `robot5`와 `robot6` 모두 공통 `map`에서 localization 되어야 함
- camera frame → base_link → map 변환이 일관되어야 함

### 8.2 요구조자 위치는 완성된 맵 기준 좌표여야 함
YOLO는 bounding box나 keypoint를 제공하지만, 이는 픽셀 좌표에 불과하다.
로봇이 실제로 이동하려면 최종적으로 다음 정보가 필요하다.
- 완성된 `map` 좌표계 기준 `(x, y)` 또는 `(x, y, z)`

이를 위해 필요하다.
- Depth 카메라
- 카메라 내부 파라미터
- TF 변환
- 거리 기반 투영 계산
- 저장 맵 또는 안정화된 맵 기준 localization

### 8.3 멀티머신 통신 안정성
같은 Wi-Fi와 같은 Domain ID를 사용하더라도, 실제 멀티머신 환경에서는 아래 문제가 자주 발생한다.
- discovery 불안정
- 멀티캐스트 이슈
- 영상 토픽 대역폭 부족
- 시간 차이로 인한 데이터 mismatch

따라서 네트워크와 QoS 설계가 매우 중요하다.

---

## 9. 권장 구현 순서

### 1단계. `robot5`의 SLAM 안정화
- 맵 생성
- 탐색 주행
- 맵 저장 및 공유 검증

### 2단계. 저장 맵 기준 localization 구성
- `robot5`가 완성된 맵 기준으로 localization 가능한 상태인지 검증
- 탐지/좌표화가 안정화된 `map` 기준으로 기록되는지 확인

### 3단계. `robot5`의 요구조자 위치 추정
- YOLO 탐지
- Depth 기반 거리 추정
- TF를 이용한 `map` 좌표 변환
- `/robot5/victim_pose` publish

### 4단계. `robot6`의 맵 공유 및 localization
- 동일 맵 사용
- AMCL 또는 대응 localization 구성
- goal pose 기반 이동 테스트

### 5단계. `robot5` → `robot6` 임무 전달
- topic/service/action 중 하나 선택
- 구조 임무 프로토콜 정의

### 6단계. `robot6`의 YOLO-Pose 상태 판단 통합
- 현재 `rescue_bot` 분석 모듈을 ROS 이미지 토픽 기반으로 전환
- 위급도 판단 및 로깅 연동

### 7단계. STT/TTS 대화 모듈 통합
- 간단한 상태 확인 문장부터 시작
- 대화 실패 시 재시도 로직 설계

### 8단계. 대시보드 통합
- `map`
- `robot5` / `robot6` 현재 위치
- victim marker
- 상태 로그
- 웹캠 영상

---

## 10. 실행 개요

> 아래 내용은 프로젝트 목표 구조를 기준으로 한 실행 개요이며, 현재 저장소만으로 전체 시나리오가 완전히 실행되지는 않는다.

### 10.1 현재 저장소 기준 실행 대상
- SRD 분석 노드
- DB 저장 노드
- Flask 대시보드

### 10.2 현재 저장소 기준 실행 예시

```bash
colcon build --packages-select rescue_bot
source install/setup.bash
ros2 launch rescue_bot srd_system.launch.py
```

웹 대시보드는 별도 실행이 필요하다.

```bash
python3 rescue_bot/web/srd_flask_server.py
```

> 주의:
> 현재 저장소의 `setup.py`, `package.xml`, 모델 경로, Python 의존성 선언은 실제 배포 수준으로 완성되어 있지 않으므로,
> 환경에 따라 추가 수정이 필요하다.

---

## 11. 기대 효과

- 단일 로봇이 아닌 **다중 로봇 협업 구조 시나리오** 구현
- SLAM과 인명 탐지를 결합한 **공간 기반 구조 지원 시스템** 구현
- 요구조자 탐지 후 단순 발견을 넘어, **접근-판단-상호작용-지원**까지 이어지는 시나리오 구성
- 발표 및 시연 관점에서 높은 전달력 확보

---

## 12. 결론

본 프로젝트는 **TurtleBot4 두 대를 이용한 협업형 구조 지원 시스템**을 목표로 하며,
`robot5`는 **탐색과 발견**, `robot6`는 **접근과 대응**을 맡는 역할 분담 구조를 가진다.
