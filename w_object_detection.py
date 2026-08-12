"""
LAB 색공간 기반 초록색 객체 검출 (Windows 노트북 웹캠 버전)

원본: object_detection.py (Jetson + CSI 카메라 / GStreamer nvarguscamerasrc)
변경: Windows 내장/USB 웹캠 (DirectShow 백엔드)

실행:
    python w_object_detection.py
종료:
    영상 창을 클릭한 뒤 'q' 키
"""

import numpy as np
import cv2
import time


# --------------------------- 카메라 설정 ---------------------------
# Windows 웹캠 인덱스. 노트북 내장 캠은 보통 0,
# USB 캠을 추가로 꽂았다면 1 또는 2 로 바꿔가며 시도하세요.
CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 25

# 노트북 웹캠은 Jetson CSI 카메라와 달리 상하 반전이 없습니다.
# 대신 거울처럼 좌우 반전(1)을 하면 실습 시 조작이 직관적입니다.
# 반전을 원하지 않으면 None 으로 두세요. (0=상하, 1=좌우, -1=상하좌우)
FLIP_CODE = 1
# -------------------------------------------------------------------

# 초록 LAB lower/upper range
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

# Morphological operation용 타원형 kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Windows 에서는 GStreamer 파이프라인 대신 DirectShow 백엔드로 카메라를 엽니다.
# CAP_DSHOW: Windows 기본 백엔드(MSMF)보다 열기 속도가 빠르고 호환성이 좋습니다.
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError(
        f"카메라({CAMERA_INDEX})를 열 수 없습니다.\n"
        "  1) 다른 앱(Zoom, Teams, 카메라 앱)이 웹캠을 쓰고 있는지 확인\n"
        "  2) 설정 > 개인 정보 및 보안 > 카메라 에서 데스크톱 앱 접근 허용\n"
        "  3) CAMERA_INDEX 값을 1, 2 로 바꿔서 재시도"
    )

# MJPG 로 받아야 대부분의 웹캠에서 720p 30fps 가 나옵니다. (기본 YUY2 는 매우 느림)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)
# 지연(latency)을 줄이기 위해 내부 버퍼를 최소화
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    start = time.time()
    ret, frame = cap.read()

    if not ret:
        break

    if FLIP_CODE is not None:
        frame = cv2.flip(frame, FLIP_CODE)

    # 가우시안 블러
    blr = cv2.GaussianBlur(frame, (11, 11), 0)

    # LAB 색공간 변환
    lab = cv2.cvtColor(blr, cv2.COLOR_BGR2LAB)

    # LAB color segmentation
    mask = cv2.inRange(lab, green_lower, green_upper)

    # Opening 2회, Dilation 2회
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Contour detection
    contour_lst, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contour_lst) > 0:
        # 가장 큰 contour 선택
        contour = max(contour_lst, key=cv2.contourArea)
        # 최소 외접원 반지름
        _, radius = cv2.minEnclosingCircle(contour)
        # 무게중심
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            center = (0, 0)

        # 검출된 객체에 파란 원 overlay
        cv2.circle(frame, center, int(radius), (255, 0, 0), 2)
        cv2.circle(frame, center, 5, (255, 0, 0), -1)

    cv2.imshow("Object Detection", frame)
    # cv2.imshow("LAB Mask", mask)

    # imshow 로 창이 만들어진 뒤에 키 입력을 받아야 'q' 가 확실히 동작합니다.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # while loop rate (FPS) 설정
    time.sleep(max(1. / FPS - (time.time() - start), 0))

cap.release()
cv2.destroyAllWindows()
