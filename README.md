# rokey_ws

ROS2 기반 구조/감지 워크스페이스입니다. 현재 워크스페이스는 아래 3개 패키지로 구성됩니다.

- `camera_system`: 카메라 입력, 사람/로봇 감지, 오버레이, 붕괴 감지
- `robot5_person_search`: robot5 탐색 및 사람 검출 이벤트 발행
- `rescue_bot`: robot6 구조 미션 제어, 비전 분석, 내비게이션, STT/TTS, 웹 UI

## Workspace Layout

```text
/home/gom/rokey_ws
├── requirements.txt
└── src
    ├── camera_system
    ├── rescue_bot
    └── robot5_person_search
```

## Environment

이 워크스페이스는 현재 아래 환경을 기준으로 사용하는 것을 전제로 합니다.

- OS: Ubuntu 22.04
- ROS 2: Humble
- Python: 3.10 계열
- Build tool: `colcon`
- Package install: `rosdep`, `pip`

추가 런타임 전제:

- GPU: 선택 사항이지만 `camera_system`, `rescue_bot`의 YOLO/Whisper 추론은 CUDA 가능 GPU가 있으면 유리합니다.
- CUDA: 현재 로컬 환경은 CUDA 12.4 경로가 잡혀 있습니다.
- Audio: `rescue_stt_node` 실행 시 마이크 입력, 스피커 출력, PortAudio/ALSA 계열 시스템 라이브러리가 필요합니다.
- Robot navigation: `robot5_person_search`, `rescue_bot`는 Nav2 및 TurtleBot4 관련 패키지가 준비되어 있어야 합니다.

환경 준비 예시:

```bash
source /opt/ros/humble/setup.bash
cd /home/gom/rokey_ws
python3 -m pip install -r requirements.txt
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## Package Summary

### 1. camera_system
- USB 카메라 영상을 퍼블리시합니다.
- YOLO 기반 감지 결과를 생성합니다.
- 감지 결과를 시각화하고 붕괴 이벤트를 감지합니다.
- 대표 실행 노드:
  - `camera_publisher`
  - `detection_node`
  - `overlay_node`
  - `collapse_detector`

### 2. robot5_person_search
- robot5가 탐색 중 사람을 찾으면 이벤트를 발생시킵니다.
- 검출 위치와 피해자 방향 정보를 후속 시스템에 넘깁니다.
- 대표 실행 노드:
  - `person_event_detector`
  - `explore_detect_supervisor`

### 3. rescue_bot
- robot6 도착 이후 구조 세션을 시작합니다.
- `rescue_vision_core`로 자세/위급도 분석을 수행합니다.
- `rescue_nav_node`가 목표 이동과 도킹을 관리합니다.
- `rescue_stt_node`가 구조 대화와 음성 입출력을 처리합니다.
- 웹 UI도 포함합니다.
- 대표 실행 노드:
  - `rescue_control_node`
  - `rescue_nav_node`
  - `rescue_stt_node`
  - `rescue_ui`

## Runtime Flow

실로봇 기준 구조 흐름은 다음 순서입니다.

1. `robot5_person_search`가 사람 검출 이벤트를 발생시킵니다.
2. `rescue_nav_node`가 `/robot5/robot_pose_at_detection`, `/robot5/victim_point`를 받아 목표 주행을 수행합니다.
3. 도착 시 `/robot6/mission/arrived`를 발행합니다.
4. `rescue_control_node`가 세션을 시작하고 비전 분석 후 `/robot6/tts/request`에 상태 문자열을 발행합니다.
5. `rescue_stt_node`가 대화 시나리오를 수행하고 `/robot6/tts/done`을 발행합니다.
6. `rescue_nav_node`가 다음 목표 또는 도킹 단계로 진행합니다.

## Python Requirements

공용 Python 패키지는 루트 [requirements.txt](/home/gom/rokey_ws/requirements.txt) 에 정리했습니다.

설치:

```bash
python3 -m pip install -r /home/gom/rokey_ws/requirements.txt
```

포함되는 대표 항목:
- `numpy`
- `opencv-python`
- `ultralytics`
- `torch`
- `torchvision`
- `Flask`
- `pygame`
- `SpeechRecognition`
- `openai-whisper`
- `gTTS`
- `PyAudio`

## ROS / System Dependencies

아래 항목은 `pip`가 아니라 ROS 패키지 또는 시스템 패키지로 준비해야 합니다.

- ROS2 기본: `rclpy`, `sensor_msgs`, `std_msgs`, `geometry_msgs`, `nav_msgs`
- TF / 브리지: `tf2_ros`, `tf2_geometry_msgs`, `cv_bridge`
- Navigation: `nav2_simple_commander`, `turtlebot4_navigation`, `explore_lite_msgs`
- Launch / Web: `launch`, `launch_ros`, `rosbridge_server`, `web_video_server`
- 오디오 장치: 마이크, 스피커, PortAudio/ALSA 관련 시스템 라이브러리

권장 설치 방식:

```bash
cd /home/gom/rokey_ws
rosdep install --from-paths src --ignore-src -r -y
```

## Build

```bash
cd /home/gom/rokey_ws
colcon build --symlink-install
source install/setup.bash
```

## Run

### camera_system

```bash
ros2 launch camera_system camera_system.launch.py
```

### robot5_person_search

```bash
ros2 launch robot5_person_search robot5_person_search.launch.py
```

### rescue_bot real runtime

```bash
ros2 launch rescue_bot rescue_real.launch.py
```

### rescue_bot simulator runtime

```bash
ros2 launch rescue_bot rescue_sim.launch.py
```

## Verification

기본 확인 순서는 아래를 권장합니다.

1. Python 의존성 설치
2. `rosdep install`
3. `colcon build --symlink-install`
4. `source install/setup.bash`
5. 각 launch 실행
6. `ros2 node list`, `ros2 topic list`로 노드/토픽 확인

예시:

```bash
ros2 node list
ros2 topic list
ros2 topic echo /robot6/mission/arrived
ros2 topic echo /robot6/tts/request
ros2 topic echo /robot6/tts/done
```

## Notes

- `rescue_bot`의 YOLO pose 모델 기본값은 `yolo11n-pose.pt` 입니다.
- 현재 `rescue_bot`의 `nav` 입력 기준은 `/rescue/victim_pose_stamped`가 아니라 robot5 토픽 체인입니다.
- STT 노드는 Python 패키지 외에도 실제 오디오 입출력 장치 상태에 영향을 받습니다.
