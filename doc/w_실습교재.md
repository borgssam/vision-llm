# Physical AI의 Vision-LLM 융합 시청각 멀티모달 시스템
## Windows 노트북 실습 교재 (`w_` 소스 전용)

> 이 교재는 Jetson Orin Nano 용으로 작성된 원본 실습 자료를
> **Windows 노트북 + NVIDIA GPU** 환경에서 그대로 따라할 수 있도록 이식한
> `w_` 접두사 소스 일체에 대한 설명서입니다.

---

## 목차

- [0. 이 교재에 대하여](#0-이-교재에-대하여)
  - [0.1 대상 환경](#01-대상-환경)
  - [0.2 파일 구성](#02-파일-구성)
  - [0.3 원본과 `w_` 소스의 관계](#03-원본과-w_-소스의-관계)
- [1. 개발 환경 구축](#1-개발-환경-구축)
  - [1.1 사전 준비물](#11-사전-준비물)
  - [1.2 Python 3.11 설치 (중요)](#12-python-311-설치-중요)
  - [1.3 가상 환경 생성 및 활성화](#13-가상-환경-생성-및-활성화)
  - [1.4 패키지 일괄 설치](#14-패키지-일괄-설치)
  - [1.5 GPU / CUDA 동작 확인](#15-gpu--cuda-동작-확인)
  - [1.6 Jupyter 커널 등록](#16-jupyter-커널-등록)
  - [1.7 설치 검증 체크리스트](#17-설치-검증-체크리스트)
- [2. Windows 웹캠 다루기 — 이식의 핵심](#2-windows-웹캠-다루기--이식의-핵심)
  - [2.1 GStreamer 파이프라인이 사라진 이유](#21-gstreamer-파이프라인이-사라진-이유)
  - [2.2 표준 카메라 오픈 코드](#22-표준-카메라-오픈-코드)
  - [2.3 각 설정의 의미](#23-각-설정의-의미)
  - [2.4 카메라 인덱스 찾기](#24-카메라-인덱스-찾기)
  - [2.5 화면 반전 방향이 바뀐 이유](#25-화면-반전-방향이-바뀐-이유)
  - [2.6 `q` 키로 종료가 안 될 때](#26-q-키로-종료가-안-될-때)
- [3. 실습 소스 코드 해설](#3-실습-소스-코드-해설)
  - [3.1 `w_object_detection.py` — LAB 색공간 객체 검출](#31-w_object_detectionpy--lab-색공간-객체-검출)
  - [3.2 `w_object_tracking.py` — Kalman Filter 직접 구현](#32-w_object_trackingpy--kalman-filter-직접-구현)
  - [3.3 `w_object_tracking_cv.py` — `cv2.KalmanFilter` 활용](#33-w_object_tracking_cvpy--cv2kalmanfilter-활용)
  - [3.4 세 소스의 비교](#34-세-소스의-비교)
- [4. 노트북 실습 가이드](#4-노트북-실습-가이드)
  - [4.1 `w_00_Initial-Setup`](#41-w_00_initial-setup)
  - [4.2 `w_01_Linux-and-Python`](#42-w_01_linux-and-python)
  - [4.3 `w_02_Computer-Vision`](#43-w_02_computer-vision)
  - [4.4 `w_03_DL-and-GPU`](#44-w_03_dl-and-gpu)
  - [4.5 `w_04_DL-Object-Detection`](#45-w_04_dl-object-detection)
- [5. Jetson ↔ Windows 명령어 대조표](#5-jetson--windows-명령어-대조표)
  - [5.1 셸 명령어](#51-셸-명령어)
  - [5.2 카메라](#52-카메라)
  - [5.3 성능 모니터링](#53-성능-모니터링)
  - [5.4 패키지 설치](#54-패키지-설치)
- [6. 트러블슈팅](#6-트러블슈팅)
- [부록 A. `w_requirements.txt` 해설](#부록-a-w_requirementstxt-해설)
- [부록 B. 알려진 제약](#부록-b-알려진-제약)

---

## 0. 이 교재에 대하여

### 0.1 대상 환경

| 항목 | 값 |
| --- | --- |
| OS | Windows 11 x64 |
| Python | **3.11** (3.12 이상 불가 — [1.2절](#12-python-311-설치-중요) 참고) |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU |
| CUDA | 12.6 (PyTorch 휠에 포함, 별도 설치 불필요) |
| 카메라 | 노트북 내장 웹캠 또는 USB 웹캠 (UVC 표준) |

### 0.2 파일 구성

```
vision-llm/
├── doc/
│   └── w_실습교재.md               ← 이 문서
│
├── w_requirements.txt              ← Windows 패키지 목록
│
├── w_00_Initial-Setup.ipynb        ← 환경 구축
├── w_01_Linux-and-Python.ipynb     ← PowerShell CLI + Python 기초
├── w_02_Computer-Vision.ipynb      ← OpenCV / 웹캠 / MediaPipe
├── w_03_DL-and-GPU.ipynb           ← PyTorch / CuPy
├── w_04_DL-Object-Detection.ipynb  ← YOLO / TensorRT
├── w_[ANSWER]_01_Linux-and-Python.ipynb
├── w_[ANSWER]_02_Computer-Vision.ipynb
│
├── w_object_detection.py           ← LAB 색공간 객체 검출
├── w_object_tracking.py            ← Kalman Filter 직접 구현
├── w_object_tracking_cv.py         ← cv2.KalmanFilter 활용
│
└── src/
    ├── images/                     ← 실습 이미지
    ├── models/                     ← YOLO / MediaPipe 모델
    └── datasets/calibration/       ← INT8 캘리브레이션 데이터
```

원본 Jetson 파일(`00_Initial-Setup.ipynb`, `object_detection.py`, `requirements.txt` 등)은
**그대로 보존**되어 있습니다. 두 환경을 비교하며 학습할 수 있습니다.

### 0.3 원본과 `w_` 소스의 관계

`w_` 소스는 원본의 **교육 내용과 셀 순서를 그대로 유지**합니다.
바뀐 것은 실행 환경에 종속된 부분뿐입니다.

| 바뀐 것 | 그대로인 것 |
| --- | --- |
| 카메라 입력 방식 (GStreamer → DirectShow) | 이미지 처리 알고리즘 전부 |
| 패키지 설치 명령 (`--no-deps` → 일반 설치) | Kalman Filter 수식과 구현 |
| 셸 명령어 (bash → PowerShell) | YOLO 추론 / 학습 코드 |
| 성능 모니터링 (`tegrastats` → `nvidia-smi`) | CuPy / PyTorch 실습 코드 |
| TensorRT 변환 방법 (`trtexec` → `model.export()`) | 실습 문제와 정답 |

각 `w_` 노트북에는 **`> Jetson 에서는 ... 였지만 Windows 에서는 ...`** 형태의
인용 블록이 들어 있어, 왜 바뀌었는지를 그 자리에서 확인할 수 있습니다.

---

## 1. 개발 환경 구축

### 1.1 사전 준비물

- [ ] 최신 NVIDIA 그래픽 드라이버 ([다운로드](https://www.nvidia.com/download/index.aspx))
- [ ] Visual Studio Code + Python / Jupyter 확장
- [ ] Python **3.11**
- [ ] 충전기 (배터리만으로는 GPU 성능이 크게 제한됩니다)

### 1.2 Python 3.11 설치 (중요)

먼저 설치된 버전을 확인합니다.

```powershell
py --list
```

목록에 `-V:3.11` 이 **없다면** [Python 3.11.9](https://www.python.org/downloads/release/python-3119/)
의 *Windows installer (64-bit)* 를 설치하세요.

> ### ⚠️ 3.12 이상을 쓰면 안 되는 이유
>
> `torch` 와 `mediapipe` 는 Python 3.12/3.13/3.14 용 Windows 휠을 제공하지 않습니다.
> 3.14 로 만든 가상 환경에서는 `pip install` 이 다음과 같이 실패합니다.
>
> ```
> ERROR: Could not find a version that satisfies the requirement torch==2.8.0+cu126
> ```
>
> 이미 다른 버전으로 `.venv` 를 만들었다면 폴더를 지우고 3.11 로 다시 만드세요.
>
> ```powershell
> Remove-Item -Recurse -Force .venv
> ```

### 1.3 가상 환경 생성 및 활성화

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

`이 시스템에서 스크립트를 실행할 수 없으므로` 오류가 나면 한 번만 실행합니다.

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

프롬프트 앞에 `(.venv)` 가 붙으면 활성화된 것입니다.

> Jetson 에서는 JetPack 이 제공하는 OpenCV·TensorRT 를 쓰기 위해
> `--system-site-packages` 옵션이 필요했지만, Windows 에서는 모든 패키지를
> pip 로 직접 설치하므로 이 옵션을 **쓰지 않습니다**.

### 1.4 패키지 일괄 설치

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r w_requirements.txt
```

총 70여 개 패키지가 설치되며, `torch` 만 약 2.5 GB 이므로 시간이 걸립니다.

> ### 💡 한글 주석과 pip 인코딩
>
> `w_requirements.txt` 의 첫 줄에 있는 `# -*- coding: utf-8 -*-` 를 **지우지 마세요.**
> pip 23.x 는 requirements 파일을 시스템 로케일(한글 Windows = cp949)로 읽기 때문에,
> 이 선언이 없으면 한글 주석에서 다음 오류가 납니다.
>
> ```
> UnicodeDecodeError: 'cp949' codec can't decode byte 0xec in position 93
> ```

### 1.5 GPU / CUDA 동작 확인

```powershell
nvidia-smi
```

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

기대 출력:

```text
2.8.0+cu126 True NVIDIA GeForce RTX 3050 Ti Laptop GPU
```

`False` 가 나오면 CPU 전용 torch 가 설치된 것입니다. 재설치하세요.

```powershell
pip uninstall -y torch torchvision
pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
```

> ### 💡 `+cu126` 을 붙이는 이유
>
> PyPI 에도 `torch 2.8.0` 이 있지만 Windows 용은 **CPU 전용 빌드**입니다.
> `+cu126` 이라는 로컬 버전 표기는 `download.pytorch.org` 에만 존재하므로,
> 이를 명시해야 CUDA 빌드가 확실히 선택됩니다.
> 별도의 CUDA Toolkit 설치는 필요 없습니다 — 휠에 런타임이 포함되어 있습니다.

### 1.6 Jupyter 커널 등록

```powershell
python -m ipykernel install --user --name=vision_llm --display-name "Vision-LLM (Python3.11)"
```

VS Code 를 재시작한 뒤, 노트북 우측 상단
`Select Kernel` → `Jupyter Kernel...` → `Vision-LLM (Python3.11)` 을 선택합니다.

### 1.7 설치 검증 체크리스트

아래를 한 번에 실행해 모두 통과하면 준비 완료입니다.

```python
import sys, cv2, numpy, torch, mediapipe, cupy, ultralytics

print("Python     :", sys.version.split()[0])          # 3.11.x
print("OpenCV     :", cv2.__version__)                 # 4.10.0
print("NumPy      :", numpy.__version__)               # 1.26.4
print("PyTorch    :", torch.__version__)               # 2.8.0+cu126
print("CUDA 사용   :", torch.cuda.is_available())       # True
print("GPU        :", torch.cuda.get_device_name(0))
print("MediaPipe  :", mediapipe.__version__)           # 0.10.35
print("CuPy       :", cupy.__version__)                # 13.6.0
print("Ultralytics:", ultralytics.__version__)         # 8.3.40
```

---

## 2. Windows 웹캠 다루기 — 이식의 핵심

이 장이 `w_` 소스에서 **가장 크게 바뀐 부분**입니다.

### 2.1 GStreamer 파이프라인이 사라진 이유

> ### ⚠️ 흔한 오해: "GStreamer 는 Jetson 전용이다"
>
> **아닙니다.** GStreamer 는 Linux·Windows·macOS 를 모두 지원하는 **범용 멀티미디어
> 프레임워크**입니다. Jetson 전용인 것은 GStreamer 가 아니라, 그 안의
> **`nvarguscamerasrc` 라는 요소(element) 하나**입니다.
> 이것이 NVIDIA Argus API 로 CSI 카메라의 ISP 를 제어하는 부분이라 Jetson 에만 존재합니다.
>
> Windows 에서 GStreamer 를 쓰지 않는 진짜 이유는 따로 있습니다.
> **PyPI 로 배포되는 Windows 용 OpenCV 가 GStreamer 지원 없이 빌드되어 있기 때문**입니다.
> 직접 확인해 볼 수 있습니다.
>
> ```python
> import cv2
> print(cv2.getBuildInformation())
> ```
>
> ```text
> Video I/O:
>     GStreamer:                   NO      ← 빌드에서 제외됨
>     DirectShow:                  YES
>     Media Foundation:            YES
>     FFMPEG:                      YES (prebuilt binaries)
> ```
>
> Jetson 의 OpenCV 는 JetPack 이 GStreamer 를 **켜서** 빌드해 준 것이라
> 파이프라인 문자열이 동작했던 것입니다.
>
> 참고로 `cv2.videoio_registry.getCameraBackends()` 에는 `GSTREAMER (id=1800)` 가
> 목록에 나타납니다. 하지만 이는 OpenCV 가 아는 백엔드 번호 목록일 뿐이며,
> 실제 빌드가 `NO` 이므로 `cv2.CAP_GSTREAMER` 로 열면 예외 없이 조용히 실패합니다.
> 디버깅할 때 헷갈리기 쉬운 지점이니 주의하세요.
>
> 그리고 `FFMPEG: YES` 는 켜져 있지만 이는 **동영상 파일** 입출력용이며
> 카메라 입력과는 무관합니다.

Jetson 의 CSI 카메라는 센서가 내보내는 **원본 Bayer 데이터**를 ISP(Image Signal Processor)로
변환해야 했습니다. 그래서 `nvarguscamerasrc` 라는 전용 GStreamer 소스를 거쳐야 했고,
아래처럼 긴 파이프라인 문자열이 필요했습니다.

```python
# ❌ Jetson 전용 — Windows 에서는 동작하지 않습니다
pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
```

반면 **Windows 노트북의 웹캠은 UVC(USB Video Class) 표준 장치**입니다.
카메라 내부에서 이미 MJPG 또는 YUY2 로 변환된 영상을 내보내므로,
OpenCV 가 DirectShow 백엔드를 통해 곧바로 BGR 이미지로 받아올 수 있습니다.

따라서 Windows 에서는 **DirectShow(`cv2.CAP_DSHOW`)** 백엔드를 사용합니다.
`cv2.CAP_GSTREAMER` 를 쓰면 카메라가 열리지 않고 `cap.isOpened()` 가 `False` 를 반환합니다.

두 환경의 처리 계층을 비교하면 다음과 같습니다.

```text
Jetson   : OpenCV → GStreamer → nvarguscamerasrc → Argus/ISP → CSI 센서
Windows  : OpenCV → DirectShow → UVC 드라이버 → USB 웹캠
```

Windows 웹캠은 UVC 표준 장치이므로 원본 Bayer 데이터를 현상하는 ISP 단계가 없습니다.
그래서 중간 파이프라인을 조립할 필요 자체가 사라지고, **인덱스 번호 하나**로 끝납니다.

파이프라인의 각 요소는 아래와 같이 대응됩니다.

| GStreamer 요소 | Windows 대응 |
| --- | --- |
| `nvarguscamerasrc sensor-id=0` | `cv2.VideoCapture(0, cv2.CAP_DSHOW)` |
| `width=1280, height=720` | `cap.set(cv2.CAP_PROP_FRAME_WIDTH / HEIGHT, ...)` |
| `framerate=30/1` | `cap.set(cv2.CAP_PROP_FPS, 30)` |
| `image/jpeg` (압축 포맷) | `cap.set(cv2.CAP_PROP_FOURCC, ...MJPG)` |
| `queue leaky=downstream max-size-buffers=1` | `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` |
| `nvvidconv` / `videoconvert` (BGR 변환) | **OpenCV 가 자동 처리** |
| `appsink` | `cap.read()` |

마지막 두 줄이 핵심입니다. 색공간 변환과 프레임 수신을 DirectShow 백엔드가 처리해 주므로,
`cap.read()` 가 곧바로 BGR NumPy 배열을 돌려줍니다.

### 2.2 표준 카메라 오픈 코드

모든 `w_` 소스는 아래 형태로 통일되어 있습니다.

```python
# ✅ Windows 용
CAMERA_INDEX = 0  # 내장 캠 0, USB 캠은 1 또는 2

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
```

`.py` 파일에는 여기에 더해 친절한 오류 안내가 포함되어 있습니다.

```python
if not cap.isOpened():
    raise RuntimeError(
        f"카메라({CAMERA_INDEX})를 열 수 없습니다.\n"
        "  1) 다른 앱(Zoom, Teams, 카메라 앱)이 웹캠을 쓰고 있는지 확인\n"
        "  2) 설정 > 개인 정보 및 보안 > 카메라 에서 데스크톱 앱 접근 허용\n"
        "  3) CAMERA_INDEX 값을 1, 2 로 바꿔서 재시도"
    )
```

### 2.3 각 설정의 의미

| 설정 | 의미 | Jetson 파이프라인에서 대응하던 부분 |
| --- | --- | --- |
| `cv2.CAP_DSHOW` | DirectShow 백엔드 명시 | `cv2.CAP_GSTREAMER` |
| `CAP_PROP_FOURCC` = `MJPG` | 압축 포맷으로 대역폭 절감 | `image/jpeg` caps |
| `FRAME_WIDTH/HEIGHT` | 해상도 요청 | `width=1280, height=720` |
| `CAP_PROP_FPS` | 프레임 속도 요청 | `framerate=30/1` |
| `CAP_PROP_BUFFERSIZE` = 1 | 지연(latency) 최소화 | `queue leaky=downstream max-size-buffers=1` |

> ### 💡 `MJPG` 를 지정하는 이유
>
> 대부분의 웹캠은 기본 포맷이 **YUY2(무압축)** 입니다. 무압축 720p 30fps 는 USB 대역폭을
> 초과하기 때문에, 지정하지 않으면 실제로는 5~10 FPS 밖에 나오지 않습니다.
> `MJPG` 로 바꾸면 720p 30fps 가 정상적으로 나옵니다.

> ### 💡 `CAP_DSHOW` 를 명시하는 이유
>
> 백엔드를 지정하지 않고 `cv2.VideoCapture(0)` 만 쓰면 OpenCV 가 Media Foundation(MSMF)을
> 고르는데, 일부 노트북에서 카메라가 열리는 데 **수 초가 걸리거나** 해상도 설정이
> 무시됩니다. `CAP_DSHOW` 가 더 빠르고 안정적입니다.
>
> 그래도 열리지 않는 최신 노트북이라면 `cv2.CAP_MSMF` 로 바꿔서 시도해 보세요.

### 2.4 카메라 인덱스 찾기

PowerShell 로 장치 목록 확인:

```powershell
Get-PnpDevice -Class Camera -Status OK | Select-Object FriendlyName, Status
```

Python 으로 실제 사용 가능한 인덱스 탐색:

```python
import cv2

for index in range(5):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"인덱스 {index}: 사용 가능  (해상도 {frame.shape[1]}x{frame.shape[0]})")
        cap.release()
    else:
        print(f"인덱스 {index}: 사용 불가")
```

### 2.5 화면 반전 방향이 바뀐 이유

원본은 상하 반전이었습니다.

```python
frame = cv2.flip(frame, 0)   # Jetson: CSI 카메라가 거꾸로 장착됨
```

`w_` 소스는 좌우 반전으로 바뀌었습니다.

```python
frame = cv2.flip(frame, 1)   # Windows: 노트북 웹캠은 거울처럼 보는 것이 자연스러움
```

Jetson CSI 카메라 모듈은 물리적으로 뒤집혀 장착되는 경우가 많아 **상하 반전(0)** 이
필요했습니다. 노트북 내장 웹캠은 정방향이므로 상하 반전이 필요 없고, 대신 손으로 물체를
움직이며 실습할 때 **좌우 반전(1)** 을 해야 화면과 손의 움직임이 일치합니다.

`.py` 소스에서는 상단 상수로 조절할 수 있습니다.

```python
FLIP_CODE = 1     # 0=상하, 1=좌우, -1=상하좌우, None=반전 없음
```

### 2.6 `q` 키로 종료가 안 될 때

원본은 `waitKey` 가 `imshow` **앞**에 있었습니다.

```python
if cv2.waitKey(1) & 0xFF == ord("q"):   # 아직 창이 없는 상태
    break
cv2.imshow("...", frame)
```

`w_` 소스는 순서를 바로잡았습니다.

```python
cv2.imshow("...", frame)
if cv2.waitKey(1) & 0xFF == ord("q"):   # 창이 만들어진 뒤 키 입력 수신
    break
```

`cv2.waitKey()` 는 **OpenCV 창이 포커스를 가진 상태**에서만 키 입력을 받습니다.
따라서 종료하려면 반드시 **영상 창을 클릭한 뒤** `q` 를 누르세요.
터미널에 포커스가 있으면 반응하지 않습니다.

창이 멈춰서 닫히지 않으면 터미널에서 `Ctrl` + `C` 를 누릅니다.

---

## 3. 실습 소스 코드 해설

세 소스 모두 **초록색 물체**를 검출/추적합니다. 초록색 공이나 포스트잇을 준비하세요.

실행 방법은 동일합니다.

```powershell
python w_object_detection.py
```

### 3.1 `w_object_detection.py` — LAB 색공간 객체 검출

가장 기본이 되는 소스입니다. 매 프레임마다 다음을 수행합니다.

```
프레임 획득
 → 가우시안 블러 (노이즈 제거)
 → BGR → LAB 색공간 변환
 → inRange 로 초록색 마스크 생성
 → Opening ×2, Dilation ×2 (잡음 제거 및 구멍 메움)
 → findContours 로 윤곽선 검출
 → 가장 큰 윤곽선 선택
 → 최소 외접원 + 무게중심 계산
 → 파란 원으로 오버레이
```

**왜 HSV 가 아니라 LAB 인가?**
LAB 색공간은 밝기(L)와 색상(a, b)이 분리되어 있어 조명 변화에 강합니다.
실내 형광등/자연광이 섞이는 실습 환경에서 HSV 보다 안정적입니다.

핵심 파라미터:

```python
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)
```

검출이 잘 안 되면 이 범위를 조정합니다. 마스크를 눈으로 보려면 주석을 해제하세요.

```python
cv2.imshow("LAB Mask", mask)
```

### 3.2 `w_object_tracking.py` — Kalman Filter 직접 구현

3.1의 검출 결과에 **Kalman Filter** 를 얹어 위치를 추정합니다.
NumPy 로 예측(Prediction)과 갱신(Update) 수식을 직접 구현한 버전입니다.

```python
def KalmanFilter(mu_prev, sigma_prev, z):
    mu_bar    = A_t.dot(mu_prev)                                    # 예측
    sigma_bar = A_t.dot(sigma_prev).dot(A_t.T) + R_t
    if z is None:                                                   # 측정값 없음 (가려짐)
        return mu_bar, sigma_bar
    K_t   = sigma_bar.dot(C_t.T).dot(inv(C_t.dot(sigma_bar).dot(C_t.T) + Q_t))
    mu    = mu_bar + K_t.dot(z - C_t.dot(mu_bar))                   # 갱신
    sigma = (np.identity(2) - K_t.dot(C_t)).dot(sigma_bar)
    return mu, sigma
```

화면 표시:

| 색 | 의미 |
| --- | --- |
| 🔵 파란 원 | 실제 검출된 위치 (measurement) |
| 🟡 노란 원 | Kalman Filter 추정 위치 (belief) |

**실습 포인트**: 초록 물체를 손으로 잠깐 가려보세요.
파란 원은 사라지지만 노란 원은 예측만으로 계속 움직입니다.

### 3.3 `w_object_tracking_cv.py` — `cv2.KalmanFilter` 활용

같은 기능을 OpenCV 내장 `cv2.KalmanFilter` 로 구현한 버전입니다.

상태 벡터에 **속도**가 포함되는 것이 3.2와의 결정적 차이입니다.

```python
kalman = cv2.KalmanFilter(4, 2)   # 상태 [x, y, vx, vy], 측정 [x, y]
```

주요 행렬:

| 행렬 | 역할 | 튜닝 효과 |
| --- | --- | --- |
| `transitionMatrix` | 상태 전이 (등속 운동 모델) | — |
| `measurementMatrix` | 측정에서 x, y만 관측 | — |
| `processNoiseCov` | 프로세스 노이즈 | 크게 → 측정값 변화에 민감 |
| `measurementNoiseCov` | 측정 노이즈 | 크게 → 측정을 덜 신뢰, 예측 중시 |
| `errorCovPost` | 초기 추정 오차 | — |

### 3.4 세 소스의 비교

| | `w_object_detection` | `w_object_tracking` | `w_object_tracking_cv` |
| --- | :---: | :---: | :---: |
| 검출 | ✅ | ✅ | ✅ |
| 추적 | ❌ | ✅ | ✅ |
| 상태 변수 | — | 위치만 | 위치 + 속도 |
| 가려짐 대응 | ❌ | 제한적 | 자연스러움 |
| 구현 | — | NumPy 직접 | OpenCV 내장 |

---

## 4. 노트북 실습 가이드

### 4.1 `w_00_Initial-Setup`

Windows 개발 환경 구축 전 과정. [1장](#1-개발-환경-구축)과 동일한 내용입니다.

원본의 **F. Chromium 삭제** 섹션은 Jetson 의 메모리 확보용이었으므로,
Windows 에서는 **F. GPU 및 CUDA 동작 확인** 으로 대체되었습니다.

### 4.2 `w_01_Linux-and-Python`

**A. Windows PowerShell 기초 CLI** + **B. 기초 Python 문법 복습**

원본의 Linux CLI 섹션이 PowerShell 로 전면 재작성되었습니다.
Python 문법 부분은 운영체제와 무관하므로 변경이 없습니다.

특히 주의할 차이:

- `cd` 만 입력하면 홈으로 가는 Linux 와 달리, PowerShell 은 `cd ~` 를 써야 합니다
- `ls -a` 가 아니라 `ls -Force` 입니다
- `mkdir` 이 중간 경로를 자동 생성하므로 `-p` 옵션이 없습니다 (붙이면 오류)
- `rm` 은 휴지통을 거치지 않고 **즉시 영구 삭제**됩니다

### 4.3 `w_02_Computer-Vision`

이미지 전처리 → 특징 추출 → 객체 추적 → MediaPipe.

**가장 많이 바뀐 노트북**입니다. 카메라 관련 셀 전체가 재작성되었습니다.

MediaPipe 모델은 PowerShell 로 내려받습니다.

```powershell
mkdir -Force src\models\MediaPipe

Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "src\models\MediaPipe\hand_landmarker.task"
```

> `wget` 은 Windows 에 없습니다. PowerShell 의 `curl` 은 `Invoke-WebRequest` 의 별칭이라
> Linux 의 curl 과 옵션이 다르니, 굳이 쓰려면 `curl.exe` 를 명시하세요.

### 4.4 `w_03_DL-and-GPU`

PyTorch 기초 → 모델 시각화 → **CuPy GPU 가속** → MNIST/CIFAR10 학습.

> ### 💡 Jetson 과 결과가 달라지는 지점
>
> Jetson Orin Nano 는 CPU 와 GPU 가 메모리 칩을 공유하는
> **UMA(Unified Memory Architecture)** 입니다. 반면 노트북의 외장 GPU 는
> RAM 과 VRAM 이 물리적으로 분리되어 있고, 데이터가 **PCI Express 버스**를 건너야 합니다.
>
> 따라서 CuPy 실습에서 **"연산만의 speedup"** 과 **"전송 시간을 포함한 speedup"** 의
> 차이가 Jetson 보다 훨씬 크게 나타납니다. 이 차이 자체가 좋은 학습 포인트입니다.

### 4.5 `w_04_DL-Object-Detection`

YOLO11n 실시간 객체 탐지 → 성능 모니터링 → TensorRT 최적화.

**TensorRT 파트(H 섹션)는 선택 사항입니다.**

Windows 용 TensorRT 는 pip 로 설치할 수 있지만 용량이 수 GB이고,
Jetson 에 기본 포함되던 `trtexec` CLI 도구가 **포함되어 있지 않습니다.**

따라서 `w_` 버전은 `trtexec` 대신 Ultralytics 의 `model.export()` 로 변환하고,
성능은 실제 추론 시간 측정으로 비교합니다.

```python
engine_path = model.export(format="engine", imgsz=640, half=True, device=0)
```

시간이 부족하거나 설치가 실패하면 **H 섹션을 건너뛰고** 앞의 `.pt` 모델 실습 결과를
그대로 사용해도 학습 목표에는 지장이 없습니다.

---

## 5. Jetson ↔ Windows 명령어 대조표

### 5.1 셸 명령어

| 목적 | Jetson (bash) | Windows (PowerShell) |
| --- | --- | --- |
| 홈으로 이동 | `cd` | `cd ~` |
| 숨김 파일 포함 목록 | `ls -a` | `ls -Force` |
| 폴더 생성 (상위 포함) | `mkdir -p a/b` | `mkdir a\b` |
| 빈 파일 생성 | `touch f.txt` | `New-Item -ItemType File f.txt` |
| 폴더 삭제 | `rm -r dir` | `rm -Recurse dir` |
| 환경 변수 | `$USER` | `$env:USERNAME` |
| 권한 변경 | `chmod 644 f` | `icacls f /grant user:R` |
| 텍스트 편집 | `nano f` | `notepad f` 또는 `code f` |
| IP 확인 | `hostname -I` | `ipconfig` |
| 문자열 필터 | `grep "RAM"` | `Select-String "RAM"` |
| 관리자 권한 | `sudo <명령>` | `Start-Process powershell -Verb RunAs` |
| 패키지 설치 | `sudo apt install` | `winget install` |
| 파일 다운로드 | `wget <url>` | `Invoke-WebRequest -Uri <url> -OutFile <path>` |

### 5.2 카메라

| 목적 | Jetson | Windows |
| --- | --- | --- |
| 장치 확인 | `ls /dev/video*` | `Get-PnpDevice -Class Camera -Status OK` |
| 포맷 확인 | `v4l2-ctl --list-formats-ext -d /dev/video0` | OpenCV 로 해상도 요청 후 실제값 비교 |
| 미리보기 | `gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! autovideosink` | `start microsoft.windows.camera:` |
| 코드에서 열기 | `cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)` | `cv2.VideoCapture(0, cv2.CAP_DSHOW)` |
| 인터페이스 설정 | `sudo /opt/nvidia/jetson-io/jetson-io.py` | 설정 > 개인 정보 및 보안 > 카메라 |

### 5.3 성능 모니터링

| 목적 | Jetson | Windows |
| --- | --- | --- |
| 실시간 모니터링 | `tegrastats` | `nvidia-smi --loop=1` |
| GUI 모니터링 | Power Mode 메뉴 | 작업 관리자 > 성능 > GPU (**Cuda** 그래프 선택) |
| 특정 지표만 | `tegrastats \| grep "RAM"` | `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv --loop=1` |
| 전력 모드 확인 | `sudo nvpmodel -q` | `powercfg /getactivescheme` |
| 전력 모드 변경 | `sudo nvpmodel -m <id>` | 설정 > 시스템 > 전원 및 배터리 > 전원 모드 |

> 작업 관리자에서 GPU 그래프 제목을 클릭해 **Cuda** 를 선택해야 딥러닝 사용률이 보입니다.
> 기본값인 `3D` 그래프는 CUDA 작업에서 거의 움직이지 않습니다.

### 5.4 패키지 설치

| 패키지 | Jetson | Windows |
| --- | --- | --- |
| PyTorch | `--index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-deps` | `torch==2.8.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126` |
| OpenCV | JetPack 기본 제공 (설치 안 함) | `opencv-contrib-python==4.10.0.84` |
| MediaPipe | `--no-deps` + 의존성 수동 설치 | `mediapipe==0.10.35` (의존성 자동) |
| CuPy | whl 다운로드 후 `--no-deps` | `cupy-cuda12x==13.6.0` |
| TensorRT | JetPack 기본 제공 | `pip install tensorrt==10.7.0` (선택) |
| 전체 | `pip install --no-deps -r requirements.txt` | `pip install -r w_requirements.txt` |

> ### 💡 왜 Jetson 은 `--no-deps` 가 필수였나
>
> Jetson 에서 pip 가 의존성을 자동 해결하면, JetPack 이 제공하는 하드웨어 가속
> OpenCV 나 CUDA 빌드를 PyPI 의 일반 빌드로 **덮어써서** GPU·카메라가 동작하지 않게 됩니다.
> Windows 에는 이런 시스템 제공 패키지가 없으므로 `--no-deps` 가 필요 없고,
> 오히려 쓰면 의존성이 빠져 import 가 실패합니다.

---

## 6. 트러블슈팅

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| `Could not find a version that satisfies torch==2.8.0+cu126` | Python 3.12 이상 사용 | `.venv` 삭제 후 `py -3.11` 로 재생성 |
| `UnicodeDecodeError: 'cp949' codec` (pip) | requirements 인코딩 선언 누락 | 첫 줄 `# -*- coding: utf-8 -*-` 확인 |
| `torch.cuda.is_available()` 가 `False` | CPU 전용 torch 설치됨 | [1.5절](#15-gpu--cuda-동작-확인) 재설치 명령 실행 |
| `cap.isOpened()` 가 `False` | 카메라 권한 / 점유 / 인덱스 | 설정에서 **데스크톱 앱** 카메라 접근 허용, 다른 앱 종료, `CAMERA_INDEX` 변경 |
| 영상이 5~10 FPS 로 느림 | YUY2 무압축 포맷 | `CAP_PROP_FOURCC` 를 `MJPG` 로 설정 |
| 카메라 여는 데 수 초 소요 | MSMF 백엔드 선택됨 | `cv2.CAP_DSHOW` 명시 |
| `q` 키가 안 먹음 | 터미널에 포커스 있음 | **영상 창을 클릭**한 뒤 `q`, 또는 터미널에서 `Ctrl`+`C` |
| `ImportError: DLL load failed` (cv2) | opencv 중복 설치 | `pip uninstall -y opencv-python opencv-contrib-python` 후 contrib 만 재설치 |
| MediaPipe import 실패 | Python 3.11 아님 | 3.11 확인 (0.10.30+ 만 3.11 휠 제공) |
| `Activate.ps1` 실행 거부 | 실행 정책 | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| GPU FPS 가 시간이 갈수록 하락 | 발열 Throttling / 배터리 | 충전기 연결, 전원 모드 `최고의 성능` |
| YOLO 가 CPU 로 도는 듯 느림 | 외장 GPU 미선택 | 설정 > 디스플레이 > 그래픽에서 Python 을 `고성능` 지정 |

---

## 부록 A. `w_requirements.txt` 해설

| 그룹 | 패키지 | 비고 |
| --- | --- | --- |
| Jupyter | `jupyter`, `ipykernel`, `ipywidgets` | 노트북 실행 환경 |
| 수치 연산 | `numpy==1.26.4` | **1.x 고정** — mediapipe/cupy 가 2.x 에서 불안정 |
| 영상 | `opencv-contrib-python==4.10.0.84` | contrib **하나만** 설치 (기본판과 충돌) |
| 시각화 | `matplotlib`, `scikit-image` | |
| 카메라 유틸 | `imutils` | `imutils.video.VideoStream` 용 |
| 랜드마크 | `mediapipe==0.10.35` | 0.10.30+ 부터 3.11 지원 |
| 딥러닝 | `torch==2.8.0+cu126`, `torchvision==0.23.0+cu126` | CUDA 빌드 |
| 모델 시각화 | `Pillow`, `aggdraw`, `visualtorch` | |
| GPU 배열 | `cupy-cuda12x==13.6.0` | |
| 객체 탐지 | `ultralytics==8.3.40` | 의존성 자동 설치 |
| 모델 변환 | `onnx`, `onnxruntime`, `onnxslim` | |
| (주석 처리) | `tensorrt`, `onnxruntime-gpu` | 필요 시 해제 |

원본 `requirements.txt` 는 Jetson 전용이며 그대로 보존되어 있습니다.

## 부록 B. 알려진 제약

1. **TensorRT `trtexec` 미제공**
   pip 로 설치한 Windows TensorRT 에는 CLI 도구가 없습니다.
   `model.export()` + 추론 시간 측정으로 대체했습니다.

2. **GStreamer 미지원**
   PyPI 배포 OpenCV 에는 GStreamer 가 빠져 있습니다.
   원본의 GStreamer 파이프라인 교육 내용은 DirectShow 백엔드 설명으로 대체했습니다.

3. **`numpy` 2.x 미적용**
   MediaPipe·CuPy 안정성을 위해 1.26.4 로 고정했습니다.

4. **INT8 캘리브레이션 데이터**
   웹캠으로 500장을 직접 촬영하는 과정은 원본과 동일하며,
   생성된 `.engine` 파일은 **변환한 그 GPU 에서만** 동작합니다.

---

<div align="right"><i>

원본 강의자료 © 2026, 김규래 (Kyu Rae Kim), All rights reserved.<br>
본 문서는 원본 실습 자료의 Windows 이식 버전에 대한 설명서입니다.

</i></div>
